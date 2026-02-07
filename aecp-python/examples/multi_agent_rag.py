
import numpy as np
import time
from rich.console import Console
from rich.panel import Panel
from rich.layout import Layout
from aecp import AECPAgent 
from aecp.adapters import LocalModelAdapter 
from sentence_transformers import SentenceTransformer

# We use the integration we just built!
from aecp_langchain import AECPEmbeddings

console = Console()

def run_multi_agent_rag():
    console.print(Panel.fit("[bold blue]AECP Multi-Agent RAG Demo[/bold blue]\n[dim]Demonstrating seamless vector transfer between Writer (MiniLM) and Knowledge Base (MPNet)[/dim]"))

    # ---------------------------------------------------------
    # 1. Setup Agents (The "Hard" Part made Easy)
    # ---------------------------------------------------------
    with console.status("[bold green]Initializing Agents...[/bold green]"):
        # Agent A: "The Writer" - Lightweight, fast model (384d)
        # In reality, this could be a browser-agent or edge device
        writer_model = SentenceTransformer('all-MiniLM-L6-v2')
        writer_agent = AECPAgent(
            agent_id="writer_agent",
            embedder=LocalModelAdapter(writer_model),
            system_prompt="You are a helpful writer."
        )

        # Agent B: "The Knowledge Base" - Heavy, powerful model (768d)
        # This agent holds the vector database
        kb_model = SentenceTransformer('all-mpnet-base-v2') 
        kb_agent = AECPAgent(
            agent_id="kb_agent", 
            embedder=LocalModelAdapter(kb_model),
            system_prompt="You are a knowledge retrieval system."
        )
    
    console.print(f"[green]✔ Agents Online[/green]")
    console.print(f"  • Writer: {writer_agent.agent_id} (Model: MiniLM, Dim: 384)")
    console.print(f"  • KB:     {kb_agent.agent_id}     (Model: MPNet,  Dim: 768)")

    # ---------------------------------------------------------
    # 2. Calibration (One-time handshake)
    # ---------------------------------------------------------
    # In a real app, this happens once and is cached.
    # We force calibration here to show it working.
    with console.status("[bold yellow]Calibrating Agents (Computing Transfer Matrix)...[/bold yellow]"):
        # We use a synthetic vocabulary for speed in this demo
        # Using unique items to ensure numerical stability (avoiding rank deficiency)
        base_vocab = ["machine learning", "artificial intelligence", "neural networks", "database", "sql", "query", "optimization", "performance", "latency", "vector"]
        vocab = []
        for word in base_vocab:
            for i in range(5):
                vocab.append(f"{word} variation {i}")
        
        # Add some random synthetic tokens to reach sufficient rank
        for i in range(100):
            vocab.append(f"synthetic token {i}")

        result = writer_agent.calibrate_with(kb_agent, vocabulary=vocab, verbose=False)
        
    console.print(f"[green]✔ Calibration Complete[/green] (Quality: {result.validation_similarity:.2f})")
    console.print(f"  • Transfer Matrix computed: 384 -> 768 dimensions")

    # ---------------------------------------------------------
    # 3. The LangChain Integration (The Magic)
    # ---------------------------------------------------------
    console.print("\n[bold white]scenario:[/bold white] Writer Agent wants to search KB Agent's database.")
    console.print("[dim]Writer sends a query. It is AUTO-TRANSLATED to KB's vector space.[/dim]\n")

    # Create the "Transfer Embeddings" - this looks like a standard LangChain embeddings class!
    # But internally it handles the dimension upgrade (384->768) and semantic alignment.
    transfer_embeddings = AECPEmbeddings(
        agent=writer_agent,
        target_agent=kb_agent
    )

    query = "How to make SQL faster?"
    console.print(f"[bold cyan]Query:[/bold cyan] {query}")

    # Execution
    start = time.time()
    
    # This single call does:
    # 1. Encodes with MiniLM (384d)
    # 2. Multiplies by Transfer Matrix (384x768)
    # 3. Returns a 768d vector compatible with MPNet
    vector = transfer_embeddings.embed_query(query)
    
    duration = (time.time() - start) * 1000

    # ---------------------------------------------------------
    # 4. Verify Results
    # ---------------------------------------------------------
    vector_np = np.array(vector)
    console.print(f"\n[bold green]✔ Transfer Successful[/bold green]")
    console.print(f"  • Output Shape: {vector_np.shape} (Expected: (768,))")
    console.print(f"  • Latency:      {duration:.2f}ms")
    
    # Verify semantic quality
    # We compare the transferred vector against the "Ground Truth" (what MPNet would have produced)
    ground_truth = kb_model.encode(query)
    
    # Cosine Similarity
    sim = np.dot(vector_np, ground_truth) / (np.linalg.norm(vector_np) * np.linalg.norm(ground_truth))
    
    console.print(f"  • Semantic Fidelity: [bold yellow]{sim:.4f}[/bold yellow] (1.0 = Perfect)")

    if sim > 0.8:
        console.print("\n[bold blue]Result:[/bold blue] High-fidelity transfer! The KB Agent understood the query perfectly without seeing the text.")
    else:
        console.print("\n[bold red]Result:[/bold red] Transfer quality low (expected for tiny demo vocab). Increase vocab size for >0.95.")

if __name__ == "__main__":
    run_multi_agent_rag()
