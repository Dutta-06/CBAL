from scipy.spatial.distance import cosine

def cosine_similarity(emb1, emb2):
    if emb1 is None or emb2 is None:
        return 0.0
    return 1.0 - cosine(emb1, emb2)