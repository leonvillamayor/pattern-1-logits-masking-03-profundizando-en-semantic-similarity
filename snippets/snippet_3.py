"""
Pattern 1: Logits Masking — Semantic Similarity como función de scoring
Episodio 3 / 10

Caso de uso: forzar al modelo a generar texto relacionado con
            "bebida de proteína" usando cosine similarity entre el prompt
            y los embeddings de los tokens candidatos.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
    set_seed,
)

# sentence-transformers es opcional: sólo necesario si queremos
# embeddings fuera del propio LLM. Aquí lo usamos para el "concept".
try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    SentenceTransformer = None


# ---------------------------------------------------------------------------
# 1. Núcleo: cosine similarity robusta
# ---------------------------------------------------------------------------
def cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Cosine similarity numéricamente estable. Shapes broadcasts."""
    a = F.normalize(a, p=2, dim=-1, eps=eps)
    b = F.normalize(b, p=2, dim=-1, eps=eps)
    return a @ b.transpose(-2, -1)


# ---------------------------------------------------------------------------
# 2. LogitsProcessor basado en similitud semántica
# ---------------------------------------------------------------------------
@dataclass
class SemanticSimilarityLogitsProcessor(LogitsProcessor):
    """
    Sube (o baja) los logits de los tokens cuyo *embedding* es similar
    (o lejano) a un vector de concepto.

    Args:
        concept_embedding: tensor (D,) con el embedding del concepto guía.
        token_embeddings:  tensor (V, D) con los embeddings de la
                           columna de input del LM head (mismo espacio).
        vocab_size:        tamaño real del vocabulario del modelo.
        boost:             factor multiplicativo para tokens similares
                           (1.0 = neutro, 1.3 = +30 % al top similar).
        penalty:           factor multiplicativo para tokens lejanos
                           (por defecto 0.5 para empujar hacia abajo).
        threshold:         similitud mínima para recibir boost.
        target_token_ids:  opcional, ids a los que SÍ se les aplica
                           el reescalado (resto queda intacto).
    """

    concept_embedding: torch.Tensor
    token_embeddings: torch.Tensor
    vocab_size: int
    boost: float = 1.3
    penalty: float = 0.5
    threshold: float = 0.2
    target_token_ids: Optional[List[int]] = field(default=None)

    def __post_init__(self) -> None:
        # (V, D) en float32 para precisión
        self.token_embeddings = self.token_embeddings.float()
        self.concept_embedding = self.concept_embedding.float()

    # ------------------------------------------------------------------
    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """
        input_ids: (batch, seq_len)  — no se usa aquí.
        scores:    (batch, vocab)   — logits del siguiente token.
        """
        if scores.shape[-1] != self.vocab_size:
            # El modelo ya podría haber recortado logits; en ese caso,
            # trabajamos sólo con los disponibles.
            return scores

        # 1) similitud coseno entre el concepto y TODA la columna de embeddings
        sims = cosine_similarity(
            self.concept_embedding.unsqueeze(0),        # (1, D)
            self.token_embeddings,                     # (V, D)
        ).squeeze(0)                                   # (V,)

        # 2) máscara opcional: si nos pasan ids diana, el resto no se toca
        if self.target_token_ids is not None:
            mask = torch.zeros_like(sims, dtype=torch.bool)
            mask[self.target_token_ids] = True
            sims = torch.where(mask, sims, torch.full_like(sims, math.nan))

        # 3) factor por token: boost si sim >= threshold, penalty si no
        factor = torch.where(
            sims >= self.threshold,
            torch.full_like(sims, self.boost),
            torch.full_like(sims, self.penalty),
        )
        # Sustituimos NaN por 1.0 (tokens fuera de la máscara)
        factor = torch.where(torch.isnan(factor), torch.ones_like(factor), factor)

        # 4) aplicamos el factor a los logits (broadcast batch)
        return scores * factor.unsqueeze(0)


# ---------------------------------------------------------------------------
# 3. Helpers: embeddings desde sentence-transformers o desde el LM head
# ---------------------------------------------------------------------------
def get_concept_embedding(
    phrase: str,
    st_model: Optional["SentenceTransformer"],
    fallback_embed: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Embedding del concepto, priorizando sentence-transformers si está."""
    if st_model is not None:
        with torch.inference_mode():
            emb = st_model.encode(phrase, convert_to_tensor=True)
        return emb.to(device)

    # Fallback: media de los embeddings de los tokens de la frase
    # (sólo útil si no tenemos sentence-transformers instalado).
    tok = AutoTokenizer.from_pretrained("gpt2")
    ids = tok(phrase, return_tensors="pt").input_ids.to(device)
    return fallback_embed[ids].mean(dim=1).squeeze(0)


def get_lm_head_embeddings(model: AutoModelForCausalLM) -> torch.Tensor:
    """Devuelve (V, D) = la matriz del LM head (transpose si está tied)."""
    head = model.get_output_embeddings()
    w = head.weight  # (V, D) en modelos tied, (D, V) si no
    # Aseguramos la forma (V, D)
    if w.shape[0] == head.out_features:
        return w.detach().clone()
    return w.detach().T.clone()


# ---------------------------------------------------------------------------
# 4. Generación guiada: bebida de proteína
# ---------------------------------------------------------------------------
def generar_bebida_proteina(
    prompt: str,
    model_name: str = "gpt2",
    concepto: str = "a high-protein beverage drink",
    max_new_tokens: int = 40,
    boost: float = 1.4,
    penalty: float = 0.5,
    threshold: float = 0.1,
    seed: int = 42,
) -> str:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Carga segura del modelo causal
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        model.eval()
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"No se pudo cargar {model_name}: {exc}") from exc

    # Embeddings
    with torch.inference_mode():
        token_emb = get_lm_head_embeddings(model).to(device)

        st_model = (
            SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            if SentenceTransformer is not None
            else None
        )

        concept_emb = get_concept_embedding(
            concepto,
            st_model,
            fallback_embed=token_emb,
            device=device,
        )

        # Si el espacio del ST no coincide con el LM head, proyectamos a la
        # dimensión D del modelo de manera naïve (padding/trunc).
        D = token_emb.shape[-1]
        if concept_emb.shape[-1] != D:
            proj = torch.zeros(D, device=device, dtype=concept_emb.dtype)
            n = min(concept_emb.shape[-1], D)
            proj[:n] = concept_emb[:n]
            concept_emb = proj

        # Instanciamos el processor
        processor = SemanticSimilarityLogitsProcessor(
            concept_embedding=concept_emb,
            token_embeddings=token_emb,
            vocab_size=token_emb.shape[0],
            boost=boost,
            penalty=penalty,
            threshold=threshold,
        )

        # Generación
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.9,
            temperature=0.8,
            logits_processor=LogitsProcessorList([processor]),
            pad_token_id=tokenizer.pad_token_id,
        )

    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return text


# ---------------------------------------------------------------------------
# 5. Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    prompt = "Today I will recommend you"

    print("=== Sin guía semántica ===")
    # Para "sin guía" basta con no añadir el processor:
    print(generar_bebida_proteina.__wrapped__ if False else "")

    print("\n=== Con SemanticSimilarityLogitsProcessor (protein beverage) ===")
    resultado = generar_bebida_proteina(
        prompt=prompt,
        concepto="a high-protein beverage drink for athletes",
        boost=1.6,
        penalty=0.3,
        threshold=0.05,
        max_new_tokens=30,
    )
    print(resultado)