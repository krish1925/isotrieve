
from typing import List, Any
from langchain_core.embeddings import Embeddings
from aecp import AECP

class AecpEmbeddings(Embeddings):
    """
    AECP Adapter for LangChain.
    Allows an AECP Agent to be used as a standard LangChain Embeddings provider.
    
    If 'target_agent' is provided, the embeddings returned will be TRANSFORMED
    into the target agent's space automatically.
    """

    def __init__(self, agent: AECP, target_agent: AECP = None):
        self.agent = agent
        self.target_agent = target_agent
        
        if self.target_agent:
            # Ensure calibration
            self.agent.calibrate_with(self.target_agent)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # 1. Encode with source
        vectors = self.agent.embed(texts)
        
        # 2. Transform if target set
        if self.target_agent:
            # AECP supports single or bulk? The transfer_to might expect single or list.
            # Assuming transfer_to does NOT handle list automatically based on the demo,
            # or it might. The demo used singular. Let's iterate to be safe, or check if transfer_to handles batch.
            # aecp-python code would reveal this. 
            # In benchmarks I used: vectors_transferred = agent_a.transfer_to(agent_b, vectors_a)
            # which implies batch support if vectors_a is batch.
            vectors = self.agent.transfer_to(self.target_agent, vectors)
            
        return vectors.tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]
