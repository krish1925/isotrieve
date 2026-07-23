# Isotrieve Protocol Specification v1.0

## Overview

The Agent Embedding Communication Protocol (Isotrieve) enables semantic communication between agents using different embedding models through learned linear transfer matrices.

## Protocol Design Principles

1. **Simplicity**: Linear transformations, no neural networks
2. **Efficiency**: Millisecond-level transfer latency
3. **Quality**: 2x better than text serialization
4. **Generalization**: Validated on unseen data

## Mathematical Foundation

### Transfer Matrix Computation

Given two embedding spaces A and B with vocabularies V:

1. Encode vocabulary: `E_A = embed_A(V)`, `E_B = embed_B(V)`
2. Solve least squares: `W_AB = argmin ||E_A @ W - E_B||²`
3. Compute reverse: `W_BA = argmin ||E_B @ W - E_A||²`

### Transfer Operation

To transfer embedding `e_A` from space A to space B:

```
e_B = e_A @ W_AB
```

### Round-Trip Quality

Quality is measured by round-trip cosine similarity:

```
e_A' = (e_A @ W_AB) @ W_BA
quality = cosine_similarity(e_A, e_A')
```

## Protocol Phases

### 1. Initialization

Agents initialize with:
- Embedding provider
- Agent ID (unique identifier)
- Quality thresholds
- Protocol version

### 2. Capability Exchange

Agents exchange:
- Model name and dimensions
- Protocol version
- Quality requirements
- Batch size limits

### 3. Calibration

**Input:**
- Training vocabulary (N items)
- Validation vocabulary (M items)
- Quality threshold (τ)

**Process:**
1. Both agents encode training vocabulary
2. Compute transfer matrices W_AB and W_BA
3. Validate on held-out validation set
4. Check quality ≥ τ

**Output:**
- Transfer matrices
- Quality metrics
- Validity period (default: 7 days)

### 4. Semantic Transfer

**Input:**
- Source embedding e_A
- Target agent ID

**Process:**
1. Retrieve transfer matrix W_AB
2. Transform: e_B = e_A @ W_AB
3. Package with metadata

**Output:**
- Transferred embedding
- Expected quality
- Transfer ID

### 5. Quality Monitoring

Continuous monitoring of:
- Transfer quality
- Matrix expiration
- Recalibration triggers

## Quality Guarantees

### Calibration Quality

- **Training similarity**: > 0.95 (forward transfer)
- **Validation similarity**: > 0.80 (round-trip on held-out)
- **Worst-case similarity**: > 0.65 (minimum across validation)

### Transfer Quality

Expected quality based on validation similarity:
- Excellent (> 0.90): Near-perfect preservation
- Good (0.80-0.90): Production-ready
- Acceptable (0.75-0.80): Usable with monitoring
- Poor (< 0.75): Recalibration required

## Recalibration Triggers

Recalibration is required when:

1. **Time-based**: Matrix validity expired (default: 7 days)
2. **Quality-based**: Observed quality < threshold
3. **Model-based**: Embedding model updated

## Vocabulary Requirements

### Size Recommendations

- **Minimum**: 200 items (quick testing)
- **Standard**: 1000 items (general use)
- **Production**: 5000+ items (high quality)

### Diversity Requirements

Vocabulary should include:
- Single words (nouns, verbs, adjectives)
- Common phrases (2-3 words)
- Domain-specific terms
- Complete sentences

### Example Vocabulary Structure

```typescript
[
  // Single words
  'information', 'knowledge', 'learning',
  
  // Phrases
  'machine learning', 'neural network',
  
  // Sentences
  'The model achieves high accuracy.',
  'We optimize the algorithm for performance.'
]
```

## Performance Characteristics

### Calibration

- **Time**: O(N × D²) where N = vocabulary size, D = dimensions
- **Memory**: O(N × D) for embeddings
- **Typical**: 2-5 seconds for 1000 items

### Transfer

- **Time**: O(D²) matrix multiplication
- **Memory**: O(D²) for transfer matrix
- **Typical**: < 1ms per embedding

## Implementation Requirements

### Required Operations

1. **Matrix Operations**:
   - Least squares solver
   - Matrix multiplication
   - Vector normalization

2. **Similarity Metrics**:
   - Cosine similarity
   - L2 norm

3. **Storage**:
   - Transfer matrix caching
   - Quality metrics logging

### Optional Optimizations

1. **Compression**:
   - Matrix quantization (8-bit, 4-bit)
   - Sparse matrix representation

2. **Batching**:
   - Batch embedding generation
   - Batch transfer operations

3. **Caching**:
   - Vocabulary embedding cache
   - Transfer matrix persistence

## Security Considerations

### Data Privacy

- Vocabulary should not contain sensitive information
- Transfer matrices reveal model relationships
- Embeddings may leak semantic information

### Integrity

- Transfer IDs prevent replay attacks
- Timestamps enable freshness checks
- Quality metrics detect tampering

### Authentication

- Agents should authenticate before calibration
- API keys should be stored securely
- Transfer matrices should be signed

## Compatibility

### Version Compatibility

- Protocol version must match exactly
- Backward compatibility not guaranteed
- Version negotiation in handshake

### Model Compatibility

Compatible with any embedding model that:
- Produces fixed-dimension vectors
- Supports batch encoding
- Has consistent output

Tested with:
- OpenAI (1536D, 3072D)
- Voyage (1024D, 1536D)
- Cohere (384D, 1024D)
- HuggingFace (384D, 768D)

## Limitations

### Known Limitations

1. **Linear assumption**: May not capture all semantic nuances
2. **Dimension mismatch**: Large differences (e.g., 384D ↔ 3072D) reduce quality
3. **Domain shift**: Quality degrades on out-of-domain content
4. **Model families**: Best results within same model family

### Not Supported

- Non-linear transformations
- Multi-hop transfers (A → B → C)
- Dynamic vocabulary updates
- Real-time adaptation

## Future Extensions

### Planned Features

1. **Adaptive calibration**: Incremental vocabulary updates
2. **Multi-hop transfer**: Composed transfer matrices
3. **Neural transfer**: Non-linear transformation functions
4. **Domain adaptation**: Domain-specific calibration

### Research Directions

1. **Optimal vocabulary selection**: Active learning for calibration
2. **Quality prediction**: Predict transfer quality before execution
3. **Compression**: Efficient matrix representation
4. **Robustness**: Adversarial transfer testing

## References

- Original research: [ENHANCED_SUMMARY.md](../../ENHANCED_SUMMARY.md)
- Implementation: [protocol.py](../../protocol.py)
- Validation: 97% fidelity on 300k vocabulary

## Version History

- **v1.0** (2026-02): Initial release
  - Linear transfer matrices
  - Quality monitoring
  - Multiple provider support

## License

MIT License - See LICENSE file for details
