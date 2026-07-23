# RFC-001: Agent Embedding Communication Protocol (Isotrieve)

| Field | Value |
| :--- | :--- |
| **RFC ID** | 001 |
| **Title** | Agent Embedding Communication Protocol |
| **Status** | Draft |
| **Type** | Standards Track |
| **Created** | 2024-02-06 |

## 1. Abstract

The Agent Embedding Communication Protocol (Isotrieve) defines a standard method for autonomous agents to exchange semantic information via vector embeddings, regardless of the underlying embedding models they use. It specifies the handshake process for learning a linear transfer matrix between two embedding spaces and the wire format for transmitting transformed vectors.

## 2. Terminology

*   **Source Agent ($A$)**: The agent originating the semantic content.
*   **Target Agent ($B$)**: The agent receiving the semantic content.
*   **Source Space ($\mathbb{R}^{d_A}$)**: Experimental vector space of Agent A.
*   **Target Space ($\mathbb{R}^{d_B}$)**: Experimental vector space of Agent B.
*   **Transfer Matrix ($W_{A \to B}$)**: A matrix $W \in \mathbb{R}^{d_A \times d_B}$ that maps vectors from A to B.
*   **Calibration Anchors**: A standardized set of diverse text strings used to align the two spaces.

## 3. The Protocol

### 3.1 Handshake & Calibration

Before exchanging data, agents may optionally perform a handshake to calibrate their transfer matrix.

**Phase 1: Anchor Exchange**
Agents must agree on a set of calibration anchors $\mathcal{V}$.
*   **Standard**: The protocol defines a SHA-256 hash of the "Standard Isotrieve Vocabulary v1" (30,000 diverse English phrases).
*   **Custom**: Agents may negotiate a custom vocabulary subset.

**Phase 2: Matrix Computation**
1.  Agent A encodes $\mathcal{V} \to \mathbf{X} \in \mathbb{R}^{N \times d_A}$.
2.  Agent B encodes $\mathcal{V} \to \mathbf{Y} \in \mathbb{R}^{N \times d_B}$.
3.  The system solves for $W_{A \to B}$ such that $||\mathbf{X}W - \mathbf{Y}||^2_F$ is minimized, typically via Ridge Regression with L2 regularization ($\alpha=1.0$).

### 3.2 Wire Format

Isotrieve messages are JSON objects.

#### 3.2.1 Transfer Message
When Agent A sends a vector to Agent B:

```json
{
  "isotrieve_version": "1.0",
  "type": "transfer",
  "payload": {
    "vector": [0.123, -0.456, ...],  // The TRANSFORMED vector (in B's space)
    "original_model": "voyage-code-2",
    "target_model": "text-embedding-3-small",
    "fidelity_score": 0.972           // Estimated cosine similarity retention
  }
}
```

> **Note**: The vector sent IS already transformed. The Receiver (Agent B) usually ingests it directly.

#### 3.2.2 Error Codes

| Code | Name | Description |
| :--- | :--- | :--- |
| `E100` | `VERSION_MISMATCH` | Protocol version not supported. |
| `E200` | `DIMENSION_MISMATCH` | Vector dimensions do not match expected target model. |
| `E300` | `UNCALIBRATED` | No transfer matrix exists for this model pair. |

## 4. Transfer Logic

The transformation is linear:
$$ \mathbf{v}_B = \mathbf{v}_A \cdot W_{A \to B} $$

Where:
*   $\mathbf{v}_A$ is the row vector $(1 \times d_A)$.
*   $W_{A \to B}$ is the transfer matrix $(d_A \times d_B)$.

## 5. Security Considerations

*   **Privacy**: Isotrieve allows sharing semantic meaning without sharing raw text. However, inversion attacks on embeddings are possible. Isotrieve does not claim to differ privacy-wise from sharing raw embeddings.
*   **adversarial Attacks**: A malicious Agent A could send vectors that "mean" something harmless in text but decode to a jailbreak vector in Agent B's space. Receiver validation is recommended.

## 6. Reference Implementation

The official reference implementation is maintained in the `isotrieve` Python package.
