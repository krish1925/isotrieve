# AECP LangChain Integration

This adapter allows you to use AECP Agents directly within LangChain pipelines.

## Usage

```python
from langchain_community.vectorstores import Chroma
from aecp import AecpAgent
from sentence_transformers import SentenceTransformer
from integrations.langchain.aecp_langchain import AecpEmbeddings

# 1. Setup Agents
source_model = SentenceTransformer('all-MiniLM-L6-v2')
target_model = SentenceTransformer('all-mpnet-base-v2') 

agent_source = AecpAgent(model=source_model)
agent_dest = AecpAgent(model=target_model)

# 2. Create the AECP Embedding Provider
# This provider uses 'agent_source' to embed, but AUTOMATICALLY translates
# the vectors to 'agent_dest's space before returning them.
embeddings = AecpEmbeddings(agent=agent_source, target_agent=agent_dest)

# 3. Use in LangChain
# The vectors stored in Chroma will be in 'agent_dest' format,
# even though they originated from 'agent_source'.
vectorstore = Chroma.from_texts(
    ["Hello world", "AECP is cool"],
    embedding=embeddings
)

print("Vectors stored in Agent Destination space!")
```
