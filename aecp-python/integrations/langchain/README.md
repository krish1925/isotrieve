# AECP LangChain Integration

This package provides [LangChain](https://github.com/langchain-ai/langchain) integration for the Agent Embedding Communication Protocol (AECP).

It allows you to use AECP agents as standard LangChain `Embeddings` models, enabling seamless integration with VectorStores (Chroma, FAISS, Pinecone) and Retrievers while gaining the benefits of AECP's Vector-First communication.

## Installation

```bash
pip install aecp-langchain
```

## Usage

### Basic Embedding
Use an AECP agent like any other embedding model.

```python
from aecp import AecpAgent
from aecp_langchain import AECPEmbeddings

# Initialize the agent
agent = AecpAgent(model="all-MiniLM-L6-v2")

# Wrap in LangChain interface
embeddings = AECPEmbeddings(agent=agent)

# Use in LangChain
vector = embeddings.embed_query("Hello world")
```

### Vector Transfer (The Power of AECP)
Configure the embeddings to automatically transfer vectors to a target agent's space.

```python
# Source Agent (MiniLM)
source_agent = AecpAgent(model="all-MiniLM-L6-v2")

# Target Agent (MPNet) - effectively the "reader" of these vectors
target_agent = AecpAgent(model="all-mpnet-base-v2")

# Create transferring embeddings
transfer_embeddings = AECPEmbeddings(
    agent=source_agent,
    target_agent=target_agent
)

# When you embed documents, they are encoded by Source, 
# then transformed to match Target's space!
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=transfer_embeddings  # <-- Magic happens here
)
```
