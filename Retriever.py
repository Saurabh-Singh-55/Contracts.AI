


def retrieve_top_documents(query_embedding, index, top_k=3):
    """Retrieve top K similar documents from FAISS index."""
    distances, indices = index.search(np.array([query_embedding]), top_k)
    return indices.flatten().tolist(), distances.flatten().tolist()