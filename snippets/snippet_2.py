from transformers import LogitsProcessor
import numpy as np

class SemanticSimilarityScorer(LogitsProcessor):
    def __init__(self, tokenizer, reference_embedding,
                 model_embed, threshold=0.65):
        self.tokenizer = tokenizer
        self.reference_embedding = reference_embedding  # precomputado fuera
        self.model_embed = model_embed
        self.threshold = threshold

    def __call__(self, input_ids, scores):
        # 1. Decodifica IDs → texto del candidato parcial
        candidate_text = self.tokenizer.decode(input_ids[0])
        # 2. Embedding del candidato con el modelo congelado
        cand_emb = self.model_embed.encode(candidate_text)
        # 3. Cosine similarity contra la referencia
        cos = np.dot(cand_emb, self.reference_embedding) / (
            np.linalg.norm(cand_emb) * np.linalg.norm(self.reference_embedding)
        )
        # 4. Máscara dura: si no llega al umbral, logit → -infinito
        if cos < self.threshold:
            scores[:] = -float("inf")
        return scores