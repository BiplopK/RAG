import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def similarity_search(query,chunk_texts, embeddings, model,k=5):
    query_embedding=model.encode([query],convert_to_numpy=True)
    similarities=cosine_similarity(query_embedding,embeddings)[0]
    top_results=np.argsort(similarities)[::-1][:k]
    results=[]
    for idx in top_results:
        results.append({
            'score':similarities[idx],
            'text':chunk_texts[idx]
        })
    
    return results