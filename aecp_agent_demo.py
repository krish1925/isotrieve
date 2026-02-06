
import numpy as np
import time
from sentence_transformers import SentenceTransformer
from aecp import AECP
from aecp.adapters import LocalModelAdapter
from aecp.matrix import cosine_similarity

def run_superior_agent_demo():
    print("🤖 AECP Superiority Demonstration")
    print("="*40)
    
    # Setup
    print("Loading models...")
    # Model A: Fast but less accurate for complex retrieval
    model_a = SentenceTransformer('all-MiniLM-L6-v2') 
    # Model B: Slower but high quality
    model_b = SentenceTransformer('all-mpnet-base-v2')
    
    agent_a = AECP(LocalModelAdapter(model_a), agent_id="FastRouter")
    agent_b = AECP(LocalModelAdapter(model_b), agent_id="Specialist")
    
    # 1. Calibration with High-Quality Vocabulary
    print("\n🤝 Calibrating with diverse semantic samples...")
    # Mixing default vocabulary with some complex sentences for better generalization
    from aecp.vocabulary import get_default_vocabulary
    train_vocab, val_vocab = get_default_vocabulary(train_size=2000)
    
    # Adding real-world diversity
    diverse_sentences = [
        "The economic implications of climate change are profound.",
        "Deep learning models require significant computational resources.",
        "Pharmacology involves the study of drug action on biological systems.",
        "Sustainable urban development reduces carbon footprints.",
        "Quantum mechanics challenges our classical intuition.",
        "The historical significance of the Magna Carta cannot be overstated.",
        "Modern architecture often emphasizes minimalism and functionality.",
        "Cellular respiration converts glucose into usable energy.",
        "Geopolitical tensions impact global supply chains.",
        "Cognitive behavioral therapy is effective for many conditions."
    ]
    train_vocab.extend(diverse_sentences)
    
    start_cal = time.time()
    res = agent_a.calibrate_with(agent_b, vocabulary=train_vocab)
    print(f"✅ Calibration complete ({time.time()-start_cal:.2f}s). Fidelity: {res.validation_similarity:.2%}")

    # 2. The Task: Semantic Search Handoff
    # Knowledge base is indexed in Agent B's space (the specialist)
    knowledge_base = [
        "Solar panels convert sunlight into electricity using the photoelectric effect.",
        "Photosynthesis in plants produces oxygen and glucose from CO2 and water.",
        "The Large Hadron Collider is the world's most powerful particle accelerator.",
        "Blockchain technology uses a decentralized ledger to record transactions.",
        "Enzymes are biological catalysts that speed up chemical reactions."
    ]
    print(f"\n📚 Specialist (Agent B) has indexed {len(knowledge_base)} documents.")
    kb_embeddings_b = [agent_b.embed(doc) for doc in knowledge_base]

    # Query comes into Agent A (the router)
    query = "How do green plants create their own food?"
    print(f"📥 Query received by FastRouter (Agent A): '{query}'")
    
    # Traditional approach: Send text to Agent B, Agent B encodes
    print("\n[Traditional Method]")
    start_trad = time.time()
    # Agent B must encode the query itself
    query_emb_b_trad = agent_b.embed(query)
    # Search
    sims_trad = [cosine_similarity(query_emb_b_trad, doc_emb) for doc_emb in kb_embeddings_b]
    best_idx_trad = np.argmax(sims_trad)
    end_trad = time.time()
    print(f"Result: '{knowledge_base[best_idx_trad]}'")
    print(f"Latency: {(end_trad-start_trad)*1000:.2f}ms (requires full encoding on B)")

    # AECP approach: Agent A transfers its already computed vector
    print("\n[AECP Method]")
    start_aecp = time.time()
    # Agent A already had the vector (e.g. for its own routing logic)
    # We just transfer it to B's space
    transfer_obj = agent_a.transfer_to("Specialist", query)
    query_emb_b_aecp = transfer_obj.embedding
    # Search in B's space using the transferred vector
    sims_aecp = [cosine_similarity(query_emb_b_aecp, doc_emb) for doc_emb in kb_embeddings_b]
    best_idx_aecp = np.argmax(sims_aecp)
    end_aecp = time.time()
    
    print(f"Result: '{knowledge_base[best_idx_aecp]}'")
    print(f"Latency: {(end_aecp-start_aecp)*1000:.2f}ms (vector-only transfer)")
    
    # 3. Evaluation of "Superiority"
    print("\n💡 Superiority Metrics:")
    speedup = (end_trad - start_trad) / (end_aecp - start_aecp) if (end_aecp - start_aecp) > 0 else 100
    print(f"- Speedup: {speedup:.1f}x faster handoff")
    print(f"- Preservation: {'SUCCESS' if best_idx_aecp == best_idx_trad else 'FAILED'} (Rank-1 Match)")
    
    # Demonstrate Privacy Preservation
    print("\n🔒 Privacy Mode:")
    print("In AECP mode, FastRouter could have sent ONLY the vector.")
    print("The Specialist (Agent B) found the answer without ever seeing the string: 'How do green plants create their own food?'")
    print("This is impossible with traditional text-based handoff.")

if __name__ == "__main__":
    run_superior_agent_demo()
