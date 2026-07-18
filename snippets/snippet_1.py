from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")
emb = model.encode(["Whey protein with nutrients",
                    "Premium protein blend with amino acids"])

score = cosine_similarity([emb[0]], [emb[1]])[0][0]
print(f"Similarity: {score:.3f}")