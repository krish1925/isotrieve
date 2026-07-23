"""
Example: Creating a Custom Adapter

Shows how to create your own embedding provider adapter
and use it with Isotrieve.
"""

from typing import List
from isotrieve import Isotrieve, register_adapter, adapter
from isotrieve.adapters.base import BaseAdapter


# Method 1: Regular class + manual registration
class MyCustomAdapter(BaseAdapter):
    """
    Custom adapter that generates simple embeddings.
    
    In a real implementation, you would:
    - Connect to your embedding API
    - Handle authentication
    - Process responses
    """
    
    def __init__(self, api_key: str, model: str = "my-model-v1", **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key
        self.model = model
        self._dimensions = 128  # Your model's dimensions
    
    def _embed_impl(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Replace this with your actual API call.
        """
        # Example: Simple hash-based embedding (replace with real API call)
        import hashlib
        hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
        
        # Generate deterministic values
        embedding = []
        for i in range(self._dimensions):
            val = ((hash_val + i) % 1000) / 1000.0
            embedding.append(val)
        
        return embedding
    
    def _embed_batch_impl(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        return [self._embed_impl(text) for text in texts]
    
    def get_dimensions(self) -> int:
        """Get embedding dimensions."""
        return self._dimensions
    
    def get_model_id(self) -> str:
        """Get model identifier."""
        return f"custom:{self.model}"


# Method 2: Using decorator
@adapter("my-other-provider")
class AnotherCustomAdapter(BaseAdapter):
    """
    Another custom adapter using the decorator pattern.
    Automatically registers when the module is imported.
    """
    
    def __init__(self, api_endpoint: str, **kwargs):
        super().__init__(**kwargs)
        self.api_endpoint = api_endpoint
        self._dimensions = 256
    
    def _embed_impl(self, text: str) -> List[float]:
        # Your implementation here
        return [0.1] * self._dimensions
    
    def _embed_batch_impl(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_impl(text) for text in texts]
    
    def get_dimensions(self) -> int:
        return self._dimensions
    
    def get_model_id(self) -> str:
        return f"another:{self.api_endpoint}"


def example_manual_registration():
    """Example: Manual registration and usage."""
    print("=" * 60)
    print("Example 1: Manual Registration")
    print("=" * 60)
    
    # Register the adapter
    register_adapter("my-custom-provider", MyCustomAdapter)
    
    # Create agent using the custom adapter
    from isotrieve import get_adapter
    
    MyAdapter = get_adapter("my-custom-provider")
    adapter = MyAdapter(api_key="fake-key-123")
    
    agent = Isotrieve(adapter, agent_id="custom_agent")
    
    # Use it
    embedding = agent.embed("Hello from my custom adapter!")
    print(f"Generated embedding with {len(embedding)} dimensions")
    print(f"First 5 values: {embedding[:5]}")


def example_decorator_usage():
    """Example: Using decorator-registered adapter."""
    print("\n" + "=" * 60)
    print("Example 2: Decorator Registration")
    print("=" * 60)
    
    # Already registered via decorator
    from isotrieve import create_adapter
    
    adapter = create_adapter("my-other-provider", api_endpoint="https://api.example.com")
    agent = Isotrieve(adapter, agent_id="another_agent")
    
    embedding = agent.embed("Hello from decorator adapter!")
    print(f"Generated embedding with {len(embedding)} dimensions")


def example_calibration_with_custom():
    """Example: Calibrating custom adapters."""
    print("\n" + "=" * 60)
    print("Example 3: Calibration with Custom Adapters")
    print("=" * 60)
    
    # Register adapter (with override=True since it may already be registered)
    register_adapter("my-custom-provider", MyCustomAdapter, override=True)
    
    from isotrieve import create_adapter
    from isotrieve.adapters import MockAdapter
    
    # Create two agents with different adapters
    custom_adapter = create_adapter("my-custom-provider", api_key="key1")
    mock_adapter = MockAdapter(dimensions=384)
    
    agent1 = Isotrieve(custom_adapter, agent_id="custom_agent")
    agent2 = Isotrieve(mock_adapter, agent_id="mock_agent")
    
    # Calibrate
    print("Calibrating agents...")
    result = agent1.calibrate_with(agent2, verbose=False)
    
    if result.success:
        print(f"✓ Calibration successful!")
        print(f"  Validation similarity: {result.validation_similarity:.4f}")
        
        # Transfer
        transfer = agent1.transfer_to(agent2.agent_id, "Test message")
        print(f"✓ Transfer successful!")
        print(f"  Transferred embedding shape: {transfer.embedding.shape}")
    else:
        print(f"✗ Calibration failed: {result.error_message}")


def example_list_adapters():
    """Example: List all registered adapters."""
    print("\n" + "=" * 60)
    print("Example 4: List All Registered Adapters")
    print("=" * 60)
    
    from isotrieve import list_adapters
    
    adapters = list_adapters()
    print(f"Found {len(adapters)} registered adapters:")
    for name in sorted(adapters.keys()):
        adapter_class = adapters[name]
        print(f"  - {name}: {adapter_class.__name__}")


if __name__ == "__main__":
    # Run all examples
    example_manual_registration()
    example_decorator_usage()
    example_list_adapters()
    example_calibration_with_custom()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
