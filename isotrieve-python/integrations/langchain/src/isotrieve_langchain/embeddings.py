from typing import List, Optional, Any
from langchain_core.embeddings import Embeddings
from isotrieve import IsotrieveAgent

class IsotrieveEmbeddings(Embeddings):
    """
    LangChain integration for Isotrieve (Agent Embedding Communication Protocol).
    
    This wrapper allows Isotrieve agents to be used seamlessly within LangChain chains,
    vector stores, and retrievers. It handles the "Vector-First" communication
    by using the Isotrieve agent's transfer capabilities.
    
    Example:
        from isotrieve_langchain import IsotrieveEmbeddings
        from isotrieve import IsotrieveAgent
        
        agent = IsotrieveAgent(model="all-MiniLM-L6-v2")
        embeddings = IsotrieveEmbeddings(agent=agent)
        
        # Use in VectorStore
        vectorstore = Chroma.from_texts(["hullo", "world"], embedding=embeddings)
    """
    
    def __init__(self, agent: IsotrieveAgent, target_agent: Optional[IsotrieveAgent] = None):
        """
        Initialize the Isotrieve Embeddings wrapper.
        
        Args:
            agent: The local source agent.
            target_agent: Optional target agent. If provided, `embed_documents` 
                          will return transferred vectors for this target.
        """
        self.agent = agent
        self.target_agent = target_agent

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        # Get raw embeddings from source agent
        # Use embed_batch for efficiency
        embeddings = self.agent.embed_batch(texts)
        
        # If target is set, transfer them
        if self.target_agent:
            transferred = []
            for vec in embeddings:
                # transfer_to returns SemanticTransfer object
                # agent.transfer_to(target, vector)
                transfer_obj = self.agent.transfer_to(self.target_agent, vec)
                transferred.append(transfer_obj.embedding)
            return [t.tolist() if hasattr(t, 'tolist') else t for t in transferred]
            
        return [e.tolist() if hasattr(e, 'tolist') else e for e in embeddings]

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        # Use embed for single text
        vec = self.agent.embed(text)
        
        if self.target_agent:
            transfer_obj = self.agent.transfer_to(self.target_agent, vec)
            vec = transfer_obj.embedding
            
        return vec.tolist() if hasattr(vec, 'tolist') else vec
