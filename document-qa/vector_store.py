import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import streamlit as st
from typing import List, Dict, Tuple
import torch

@st.cache_resource
def load_embedding_model():
    """
    Load the sentence transformer model for creating embeddings.
    Using a multilingual model to support multiple languages.
    """
    model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    return model

class VectorStore:
    def __init__(self, embedding_dim: int = 768):
        """
        Initialize the vector store with FAISS index.
        Args:
            embedding_dim: Dimension of the embeddings (default: 768 for most transformer models)
        """
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.texts = []
        self.metadata = []
        
    def add_texts(self, texts: List[str], metadata: List[Dict] = None):
        """
        Add texts and their metadata to the vector store.
        Args:
            texts: List of text chunks to add
            metadata: List of metadata dictionaries for each text chunk
        """
        embedding_model = load_embedding_model()
        embeddings = embedding_model.encode(texts, convert_to_tensor=True)
        
        if torch.cuda.is_available():
            embeddings = embeddings.cpu()
        
        embeddings_np = embeddings.numpy()
        self.index.add(embeddings_np)
        
        self.texts.extend(texts)
        if metadata:
            self.metadata.extend(metadata)
    
    def similarity_search(self, query: str, k: int = 3) -> List[Tuple[str, Dict, float]]:
        """
        Search for similar texts in the vector store.
        Args:
            query: Query text
            k: Number of results to return
        Returns:
            List of (text, metadata, score) tuples
        """
        embedding_model = load_embedding_model()
        query_embedding = embedding_model.encode([query], convert_to_tensor=True)
        
        if torch.cuda.is_available():
            query_embedding = query_embedding.cpu()
        
        query_embedding_np = query_embedding.numpy()
        
        # Search the index
        distances, indices = self.index.search(query_embedding_np, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.texts):
                text = self.texts[idx]
                metadata = self.metadata[idx] if self.metadata else {}
                distance = distances[0][i]
                results.append((text, metadata, distance))
        
        return results
