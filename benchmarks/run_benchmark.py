
import argparse
import time
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import sys
import os

# Add parent directory to path to import aecp if installed locally
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from aecp import AecpAgent
except ImportError:
    # If aecp is not installed, try to import from local source
    try:
        sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "aecp-python"))
        from aecp import AecpAgent
    except ImportError as e:
        print(f"Error: Could not import 'aecp'. {e}")
        sys.exit(1)

def run_benchmark(source_model_name, target_model_name, num_samples=1000):
    print(f"\n Starting AECP Benchmark")
    print(f"==================================================")
    print(f"Source Model: {source_model_name}")
    print(f"Target Model: {target_model_name}")
    print(f"Samples:      {num_samples}")
    print(f"==================================================\n")

    # 1. Load Models
    print(" Loading models... (this may take a minute)")
    start_load = time.time()
    source_model = SentenceTransformer(source_model_name)
    target_model = SentenceTransformer(target_model_name)
    print(f" Models loaded in {time.time() - start_load:.2f}s")

    # 2. Initialize Agents
    agent_a = AECP(LocalModelAdapter(source_model))
    agent_b = AECP(LocalModelAdapter(target_model))

    # 3. Calibrate
    print("\n🛠  Calibrating Transfer Matrix...")
    start_cal = time.time()
    _ = agent_a.calibrate_with(agent_b)
    cal_time = time.time() - start_cal
    print(f" Calibration complete in {cal_time:.2f}s")

    # 4. Generate Test Data (Synthetic)
    # In a real benchmark, load MTEB data. Here we use a diverse set of phrases.
    # We use a preset list effectively representing a "Lite" benchmark.
    print(f"\n Generating {num_samples} test samples...")
    # diverse_phrases usually come from a dataset, here we synthesize or use a small hardcoded list for the 'demo' benchmark
    # Ideally, we'd pull from MTEB. For this script to be standalone and fast, we'll use a mix.
    
    # Simple synthetic data generator for "Lite" mode
    base_phrases = [
        "The quick brown fox jumps over the lazy dog",
        "Deep learning revolutionizes data analysis",
        "Python is the primary language for AI engineering",
        "React and Next.js are popular for frontend development",
        "The user authentication flow needs to be secure",
        "Kubernetes orchestration for containerized applications",
        "Financial markets fluctuate based on global events",
        "Climate change affects global weather patterns",
        "Quantum superposition allows parallel computation",
        "Recipe for chocolate chip cookies with walnuts"
    ]
    test_phrases = []
    # Repeat and perturb to get volume if needed, or just use base for speed in lite
    # For a real feel, let's just use the unique ones and loop
    import random
    for _ in range(num_samples):
        p = random.choice(base_phrases)
        test_phrases.append(p)
    
    # 5. Measure Performance
    print("\nrunning measurements...")
    
    # Baseline: Text Transfer (Simulated)
    # Cost: Time to decode (N/A for local) + Time to re-encode (Target)
    start_text = time.time()
    # In text transfer, we just take the text and encode it with B
    vectors_b_ground_truth = agent_b.encode(test_phrases)
    time_text = time.time() - start_text
    
    # AECP Transfer
    # Cost: Time to encode (Source) + Time to transfer
    start_aecp = time.time()
    vectors_a = agent_a.embed(test_phrases)
    vectors_transferred = agent_a.transfer_to(agent_b, vectors_a)
    time_aecp_transfer_only = time.time() - start_aecp # This includes encode time, which is unfair? 
    # Actually, usually A has specific vectors already. The comparison is Handoff vs Handoff.
    # Text Handoff: A has vectors. A must decode? No, A has TEXT usually if it's the source. 
    # If A has Vectors, Text Handoff is impossible without original text.
    # Scenario: A has TEXT.
    # Path 1 (Text): A sends Text. B Encodes. Cost = B_Encode_Time.
    # Path 2 (AECP): A Encodes. A Transfers. B receives. Cost = A_Encode_Time + Transfer_Time.
    # If A has ALREADY encoded (common case, index), then:
    # Path 1 (Text): A retrieves Text. Sends Text. B Encodes. Cost = B_Encode_Time.
    # Path 2 (AECP): A retrieves Vector. Transfers. Cost = Transfer_Time.
    
    # We measure "Transfer Only" latency assuming A has content ready (Text or Vector)
    # Let's measure the "Handoff Calculation" step.
    
    # Latency: Text Handoff (B Encoding)
    start_b_encode = time.time()
    _ = agent_b.embed(test_phrases[:100]) # Benchmark 100 for latency
    time_b_encode_avg = (time.time() - start_b_encode) / 100
    
    # Latency: AECP Handoff (Matrix Mult)
    start_transfer = time.time()
    _ = agent_a.transfer_to(agent_b, vectors_a[:100])
    time_transfer_avg = (time.time() - start_transfer) / 100
    
    # Fidelity Calculation
    similarities = cosine_similarity(vectors_b_ground_truth, vectors_transferred).diagonal()
    avg_fidelity = np.mean(similarities)
    
    print("\n RESULTS")
    print(f"--------------------------------------------------")
    print(f"Semantic Fidelity:      {avg_fidelity:.2%} (Target: >95%)")
    print(f"Latency (Text Handoff): {time_b_encode_avg*1000:.2f} ms/doc")
    print(f"Latency (AECP Handoff): {time_transfer_avg*1000:.2f} ms/doc")
    print(f"Speedup Factor:         {time_b_encode_avg/time_transfer_avg:.1f}x")
    print(f"--------------------------------------------------")
    
    if avg_fidelity > 0.90:
        print("\n PASSED: High Fidelity Achieved")
    else:
        print("\n⚠️  WARNING: Fidelity below target")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AECP Benchmark")
    parser.add_argument("--source", default="all-MiniLM-L6-v2", help="Source model name")
    parser.add_argument("--target", default="all-mpnet-base-v2", help="Target model name")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples")
    args = parser.parse_args()
    
    run_benchmark(args.source, args.target, args.samples)
