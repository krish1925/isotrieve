
"""
02_collaborative_rag.py
-----------------------
Simulates a more complex scenario:
- Agent A (Researcher) finds documents in its index.
- Agent A sends the *Vectors* of those documents to Agent B.
- Agent B (Analyst) clusters them without ever seeing the text/re-embedding.
"""

import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "isotrieve-python"))

from isotrieve import Isotrieve
from isotrieve.adapters import LocalModelAdapter
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

def main():
    print("--- Multi-Agent RAG Pipeline ---")
    
    # Setup
    researcher = Isotrieve(LocalModelAdapter(SentenceTransformer('all-MiniLM-L6-v2')))
    analyst = Isotrieve(LocalModelAdapter(SentenceTransformer('all-mpnet-base-v2')))
    
    print("Calibrating agents...")
    researcher.calibrate_with(analyst)
    
    # 1. Researcher finds 'documents' (Simulated)
    # In reality, these come from a vector DB search.
    docs = [
        "Database connection timeout in production",
        "Redis latency skewing high on write",
        "Postgres connection pool exhaustion",
        "Frontend styling issue on login page",
        "CSS z-index bug in navbar",
        "React component memory leak"
    ]
    
    print(f"\nResearcher found {len(docs)} documents.")
    
    # 2. Researcher transmits VECTORS to Analyst
    # NOTE: We are NOT sending text. Only vectors.
    print("Researcher transferring vectors to Analyst...")
    
    doc_vectors_a = researcher.embed(docs)
    
    # Bulk Transfer
    doc_vectors_b = [
        researcher.transfer_to(analyst, v) 
        for v in doc_vectors_a
    ]
    
    # 3. Analyst works on the vectors directly
    # Analyst wants to cluster these to find "Topics"
    print("Analyst clustering vectors (No Text Access)...")
    
    kmeans = KMeans(n_clusters=2, random_state=42)
    clusters = kmeans.fit_predict(doc_vectors_b)
    
    # 4. Reveal Results
    print("\n--- Analyst's Clustering Results ---")
    
    # Check if it grouped DB stuff vs Frontend stuff
    cluster_0 = []
    cluster_1 = []
    
    for i, cluster_id in enumerate(clusters):
        if cluster_id == 0:
            cluster_0.append(docs[i])
        else:
            cluster_1.append(docs[i])
            
    print(f"Cluster 0: {cluster_0}")
    print(f"Cluster 1: {cluster_1}")
    
    # We expect perfect separation of Backend vs Frontend topics
    # solely based on the transferred vectors.
    print("\nIf the clusters are thematically consistent, Isotrieve worked!")

if __name__ == "__main__":
    main()
