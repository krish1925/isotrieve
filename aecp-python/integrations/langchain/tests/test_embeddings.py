
import pytest
import numpy as np
from unittest.mock import MagicMock
from aecp import AECPAgent
from aecp_langchain.embeddings import AECPEmbeddings

class TestAECPEmbeddings:
    
    def test_initialization(self):
        mock_agent = MagicMock(spec=AECPAgent)
        embeddings = AECPEmbeddings(agent=mock_agent)
        assert embeddings.agent == mock_agent
        assert embeddings.target_agent is None

    def test_embed_query_local(self):
        """Test embedding without transfer (local only)."""
        mock_agent = MagicMock(spec=AECPAgent)
        # Setup mock to return a numpy array
        mock_vec = np.array([0.1, 0.2, 0.3])
        mock_agent.embed.return_value = mock_vec
        
        embeddings = AECPEmbeddings(agent=mock_agent)
        result = embeddings.embed_query("test query")
        
        mock_agent.embed.assert_called_with("test query")
        assert result == [0.1, 0.2, 0.3]
        
    def test_embed_query_transfer(self):
        """Test embedding with transfer."""
        mock_agent = MagicMock(spec=AECPAgent)
        mock_target = MagicMock(spec=AECPAgent)
        
        # Source vector
        vec_a = np.array([1.0, 1.0])
        # Transferred vector (wrapped in SemanticTransfer)
        vec_b = np.array([0.5, 0.5])
        mock_transfer_obj = MagicMock()
        mock_transfer_obj.embedding = vec_b
        
        mock_agent.embed.return_value = vec_a
        mock_agent.transfer_to.return_value = mock_transfer_obj
        
        embeddings = AECPEmbeddings(agent=mock_agent, target_agent=mock_target)
        result = embeddings.embed_query("transfer me")
        
        mock_agent.embed.assert_called_with("transfer me")
        mock_agent.transfer_to.assert_called_with(mock_target, vec_a)
        assert result == [0.5, 0.5]
        
    def test_embed_documents_batch(self):
        """Test embedding a list of documents."""
        mock_agent = MagicMock(spec=AECPAgent)
        
        doc_vecs = [np.array([1.0]), np.array([2.0])]
        mock_agent.embed_batch.return_value = doc_vecs
        
        embeddings = AECPEmbeddings(agent=mock_agent)
        result = embeddings.embed_documents(["doc1", "doc2"])
        
        mock_agent.embed_batch.assert_called_with(["doc1", "doc2"])
        assert len(result) == 2
        assert result[0] == [1.0]
        assert result[1] == [2.0]
