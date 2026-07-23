# Isotrieve LangChain Integration

This package provides [LangChain](https://github.com/langchain-ai/langchain) integration for the Agent Embedding Communication Protocol (Isotrieve).

It allows you to use Isotrieve agents as standard LangChain `Embeddings` models, enabling seamless integration with VectorStores (Chroma, FAISS, Pinecone) and Retrievers while gaining the benefits of Isotrieve's Vector-First communication.

## Installation

```bash
pip install isotrieve-langchain
```

## Usage

### Basic Embedding
Use an Isotrieve agent like any other embedding model.

```python
from isotrieve import IsotrieveAgent
from isotrieve_langchain import IsotrieveEmbeddings

# Initialize the agent
agent = IsotrieveAgent(model="all-MiniLM-L6-v2")

# Wrap in LangChain interface
embeddings = IsotrieveEmbeddings(agent=agent)

# Use in LangChain
vector = embeddings.embed_query("Hello world")
```

### Vector Transfer (The Power of Isotrieve)
Configure the embeddings to automatically transfer vectors to a target agent's space.

```python
# Source Agent (MiniLM)
source_agent = IsotrieveAgent(model="all-MiniLM-L6-v2")

# Target Agent (MPNet) - effectively the "reader" of these vectors
target_agent = IsotrieveAgent(model="all-mpnet-base-v2")

# Create transferring embeddings
transfer_embeddings = IsotrieveEmbeddings(
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
