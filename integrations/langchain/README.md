# Isotrieve LangChain Integration

This adapter allows you to use Isotrieve Agents directly within LangChain pipelines.

## Usage

```python
from langchain_community.vectorstores import Chroma
from isotrieve import IsotrieveAgent
from sentence_transformers import SentenceTransformer
from integrations.langchain.isotrieve_langchain import IsotrieveEmbeddings

# 1. Setup Agents
source_model = SentenceTransformer('all-MiniLM-L6-v2')
target_model = SentenceTransformer('all-mpnet-base-v2') 

agent_source = IsotrieveAgent(model=source_model)
agent_dest = IsotrieveAgent(model=target_model)

# 2. Create the Isotrieve Embedding Provider
# This provider uses 'agent_source' to embed, but AUTOMATICALLY translates
# the vectors to 'agent_dest's space before returning them.
embeddings = IsotrieveEmbeddings(agent=agent_source, target_agent=agent_dest)

# 3. Use in LangChain
# The vectors stored in Chroma will be in 'agent_dest' format,
# even though they originated from 'agent_source'.
vectorstore = Chroma.from_texts(
    ["Hello world", "Isotrieve is cool"],
    embedding=embeddings
)

print("Vectors stored in Agent Destination space!")
```
