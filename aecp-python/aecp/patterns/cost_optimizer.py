"""
Cost Optimizer Pattern

Minimize embedding costs by using cheaper models for most work
and expensive models only when high precision is needed.

Includes real-time cost tracking and dashboard integration.
"""

import time
import logging
from typing import List, Optional, Dict, Any, Literal
import numpy as np

from ..types import EmbeddingProvider
from ..protocol import AECP, CalibrationResult
from ..errors import AECPError, CalibrationError

logger = logging.getLogger("aecp.patterns.cost_optimizer")


PrecisionLevel = Literal["low", "medium", "high"]


class CostOptimizer:
    """
    Optimize embedding costs by intelligently routing between models.

    Uses a cheap model for most operations and an expensive model
    only when high precision is required. AECP enables seamless
    transfer between the two embedding spaces.

    Cost savings can be 70-90% compared to always using expensive models.

    Includes real-time cost tracking with:
    - Per-operation cost estimation
    - Cumulative savings tracking
    - Time savings estimation
    - Dashboard integration

    Example:
        >>> from aecp.patterns import CostOptimizer
        >>> from aecp.adapters import OpenAIAdapter, VoyageAdapter
        >>>
        >>> optimizer = CostOptimizer(
        ...     cheap_adapter=OpenAIAdapter(model="text-embedding-3-small"),
        ...     expensive_adapter=VoyageAdapter(model="voyage-large-2"),
        ... )
        >>>
        >>> # Calibrate once
        >>> optimizer.calibrate()
        >>>
        >>> # Use cheap model (default)
        >>> embedding = optimizer.embed("common query")
        >>>
        >>> # Use expensive model when needed
        >>> precise_embedding = optimizer.embed("critical query", precision="high")
        >>>
        >>> # View real-time cost dashboard
        >>> optimizer.cost_dashboard()
        >>>
        >>> # Get cost statistics
        >>> print(optimizer.get_stats())
    """

    def __init__(
        self,
        cheap_adapter: EmbeddingProvider,
        expensive_adapter: EmbeddingProvider,
        cheap_cost_per_token: float = 0.00002,  # $0.02/1M tokens
        expensive_cost_per_token: float = 0.00012,  # $0.12/1M tokens
        avg_tokens_per_text: int = 10,
        auto_precision: bool = False,
        quality_threshold: float = 0.80,
    ):
        """
        Initialize cost optimizer.

        Args:
            cheap_adapter: Low-cost embedding adapter
            expensive_adapter: High-quality embedding adapter
            cheap_cost_per_token: Cost per token for cheap model
            expensive_cost_per_token: Cost per token for expensive model
            avg_tokens_per_text: Average tokens per text (for cost estimation)
            auto_precision: Automatically choose precision based on content
            quality_threshold: Quality threshold for calibration
        """
        if cheap_adapter is None or expensive_adapter is None:
            raise ValueError("Both adapters must be provided")

        self.cheap_agent = AECP(cheap_adapter, agent_id="cheap_agent")
        self.expensive_agent = AECP(expensive_adapter, agent_id="expensive_agent")

        self.cheap_cost_per_token = cheap_cost_per_token
        self.expensive_cost_per_token = expensive_cost_per_token
        self.avg_tokens_per_text = avg_tokens_per_text
        self.auto_precision = auto_precision
        self.quality_threshold = quality_threshold

        self._calibrated = False
        self._calibration_quality = 0.0
        self._start_time = time.time()

        self._stats = {
            "cheap_calls": 0,
            "expensive_calls": 0,
            "transfers": 0,
            "estimated_cost": 0.0,
            "estimated_savings": 0.0,
            "total_time_ms": 0.0,
            "cheap_time_ms": 0.0,
            "expensive_time_ms": 0.0,
            "errors": 0,
            "fallbacks": 0,
        }

        # Per-operation history for real-time tracking
        self._operation_history: List[Dict[str, Any]] = []
        self._max_history = 10000

    def calibrate(
        self,
        vocabulary: Optional[List[str]] = None,
        verbose: bool = True,
    ) -> CalibrationResult:
        """
        Calibrate transfer matrices between cheap and expensive models.

        Args:
            vocabulary: Custom calibration vocabulary
            verbose: Whether to print progress

        Returns:
            CalibrationResult with quality metrics
        """
        try:
            result = self.cheap_agent.calibrate_with(
                self.expensive_agent,
                vocabulary=vocabulary,
                verbose=verbose,
                quality_threshold=self.quality_threshold,
            )

            if result.success:
                self._calibrated = True
                self._calibration_quality = result.validation_similarity
                if verbose:
                    print(f"\n Cost optimizer ready! "
                          f"Quality: {result.validation_similarity:.1%}")
                    print(f"   Cheap model:     ${self.cheap_cost_per_token * 1_000_000:.2f}/1M tokens")
                    print(f"   Expensive model: ${self.expensive_cost_per_token * 1_000_000:.2f}/1M tokens")
                    savings_ratio = (1 - self.cheap_cost_per_token / self.expensive_cost_per_token) * 100
                    print(f"   Potential savings: {savings_ratio:.0f}% per operation")
            else:
                if verbose:
                    print(f"\n⚠️  Calibration failed: {result.error_message}")
                    print("   Cost optimizer will use expensive model as fallback.")

            return result
        except Exception as e:
            logger.error(f"Cost optimizer calibration failed: {e}")
            return CalibrationResult(
                success=False,
                error_message=str(e),
            )

    def embed(
        self,
        text: str,
        precision: PrecisionLevel = "low",
        target_space: Literal["cheap", "expensive"] = "cheap",
    ) -> np.ndarray:
        """
        Generate embedding with specified precision level.

        Args:
            text: Text to embed
            precision: Quality level ("low", "medium", "high")
            target_space: Which embedding space to return

        Returns:
            Embedding vector in the target space

        Raises:
            CalibrationError: If not calibrated and transfer is needed
        """
        start_time = time.time()
        operation = {
            "text_preview": text[:50] + "..." if len(text) > 50 else text,
            "precision": precision,
            "target_space": target_space,
            "timestamp": time.time(),
        }

        try:
            if precision == "high":
                # Use expensive model directly
                embedding = self.expensive_agent.embed(text)
                self._stats["expensive_calls"] += 1
                cost = self._update_cost(expensive=True)
                operation["model"] = "expensive"
                operation["cost"] = cost

                elapsed = (time.time() - start_time) * 1000
                self._stats["expensive_time_ms"] += elapsed

                if target_space == "cheap" and self._calibrated:
                    transfer = self.expensive_agent.transfer_embedding_to(
                        self.cheap_agent.agent_id, embedding
                    )
                    self._stats["transfers"] += 1
                    operation["transferred"] = True
                    self._record_operation(operation, elapsed)
                    return transfer.embedding

                self._record_operation(operation, elapsed)
                return embedding

            elif precision == "medium":
                if not self._calibrated:
                    raise CalibrationError(
                        "Calibration required for medium precision. "
                        "Call calibrate() first.",
                        recoverable=True,
                    )

                embedding = self.cheap_agent.embed(text)
                self._stats["cheap_calls"] += 1
                cost = self._update_cost(expensive=False)
                operation["model"] = "cheap"
                operation["cost"] = cost

                elapsed = (time.time() - start_time) * 1000
                self._stats["cheap_time_ms"] += elapsed

                if target_space == "expensive":
                    transfer = self.cheap_agent.transfer_embedding_to(
                        self.expensive_agent.agent_id, embedding
                    )
                    self._stats["transfers"] += 1
                    operation["transferred"] = True
                    self._record_operation(operation, elapsed)
                    return transfer.embedding

                self._record_operation(operation, elapsed)
                return embedding

            else:  # low precision
                embedding = self.cheap_agent.embed(text)
                self._stats["cheap_calls"] += 1
                cost = self._update_cost(expensive=False)
                operation["model"] = "cheap"
                operation["cost"] = cost

                elapsed = (time.time() - start_time) * 1000
                self._stats["cheap_time_ms"] += elapsed

                if target_space == "expensive" and self._calibrated:
                    transfer = self.cheap_agent.transfer_embedding_to(
                        self.expensive_agent.agent_id, embedding
                    )
                    self._stats["transfers"] += 1
                    operation["transferred"] = True
                    self._record_operation(operation, elapsed)
                    return transfer.embedding

                self._record_operation(operation, elapsed)
                return embedding

        except CalibrationError:
            raise
        except Exception as e:
            self._stats["errors"] += 1
            elapsed = (time.time() - start_time) * 1000

            # Fallback: try expensive model
            logger.warning(f"Cost optimizer error ({e}), falling back to expensive model")
            try:
                embedding = self.expensive_agent.embed(text)
                self._stats["expensive_calls"] += 1
                self._stats["fallbacks"] += 1
                self._update_cost(expensive=True)
                operation["model"] = "expensive_fallback"
                self._record_operation(operation, elapsed)
                return embedding
            except Exception as e2:
                logger.error(f"Fallback also failed: {e2}")
                raise AECPError(
                    f"Both cheap and expensive models failed: {e}, {e2}",
                    recoverable=False,
                )

    def embed_batch(
        self,
        texts: List[str],
        precision: PrecisionLevel = "low",
        target_space: Literal["cheap", "expensive"] = "cheap",
    ) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: Texts to embed
            precision: Quality level
            target_space: Which embedding space to return

        Returns:
            List of embedding vectors
        """
        return [
            self.embed(text, precision=precision, target_space=target_space)
            for text in texts
        ]

    def _update_cost(self, expensive: bool) -> float:
        """Update cost statistics. Returns the cost of this operation."""
        if expensive:
            cost = self.expensive_cost_per_token * self.avg_tokens_per_text
            savings = 0
        else:
            cost = self.cheap_cost_per_token * self.avg_tokens_per_text
            savings = (
                self.expensive_cost_per_token - self.cheap_cost_per_token
            ) * self.avg_tokens_per_text

        self._stats["estimated_cost"] += cost
        self._stats["estimated_savings"] += savings
        return cost

    def _record_operation(self, operation: Dict, elapsed_ms: float) -> None:
        """Record an operation for history tracking."""
        operation["elapsed_ms"] = elapsed_ms
        self._stats["total_time_ms"] += elapsed_ms

        if len(self._operation_history) >= self._max_history:
            self._operation_history.pop(0)
        self._operation_history.append(operation)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get usage and cost statistics.

        Returns:
            Dictionary with comprehensive cost and performance metrics
        """
        total_calls = self._stats["cheap_calls"] + self._stats["expensive_calls"]

        if total_calls > 0:
            all_expensive_cost = (
                total_calls * self.expensive_cost_per_token * self.avg_tokens_per_text
            )
            savings_pct = (
                self._stats["estimated_savings"] / all_expensive_cost * 100
                if all_expensive_cost > 0 else 0
            )
        else:
            all_expensive_cost = 0
            savings_pct = 0

        # Time analysis
        time_if_all_expensive = self._stats.get("expensive_time_ms", 0)
        avg_expensive_time = (
            time_if_all_expensive / max(1, self._stats["expensive_calls"])
        )
        estimated_time_all_expensive = total_calls * avg_expensive_time if avg_expensive_time > 0 else 0
        time_saved = max(0, estimated_time_all_expensive - self._stats["total_time_ms"])

        return {
            **self._stats,
            "total_calls": total_calls,
            "savings_percentage": round(savings_pct, 1),
            "all_expensive_would_cost": round(all_expensive_cost, 8),
            "calibrated": self._calibrated,
            "calibration_quality": self._calibration_quality,
            "session_duration_s": time.time() - self._start_time,
            "avg_time_per_op_ms": (
                self._stats["total_time_ms"] / total_calls if total_calls > 0 else 0
            ),
            "estimated_time_saved_ms": time_saved,
        }

    def cost_dashboard(self) -> str:
        """
        Print a real-time cost dashboard.

        Returns:
            Formatted dashboard string
        """
        stats = self.get_stats()
        elapsed = time.time() - self._start_time

        lines = []
        lines.append("")
        lines.append("╔══════════════════════════════════════════════════╗")
        lines.append("║           Cost Optimizer Dashboard            ║")
        lines.append("╠══════════════════════════════════════════════════╣")
        lines.append(f"║  Session: {elapsed:.0f}s | Calibrated: {'' if self._calibrated else '':>16s}  ║")
        lines.append("╠══════════════════════════════════════════════════╣")

        lines.append("║  OPERATIONS                                      ║")
        lines.append(f"║    Cheap model calls:    {stats['cheap_calls']:>20,d}  ║")
        lines.append(f"║    Expensive model calls:{stats['expensive_calls']:>20,d}  ║")
        lines.append(f"║    AECP transfers:       {stats['transfers']:>20,d}  ║")
        lines.append(f"║    Errors/Fallbacks:     {stats['errors']}/{stats['fallbacks']:>16s}  ║")
        lines.append("╠══════════════════════════════════════════════════╣")

        lines.append("║  COST ANALYSIS                                   ║")
        lines.append(f"║    Actual cost:          ${stats['estimated_cost']:>18.6f}  ║")
        lines.append(f"║    All-expensive cost:   ${stats['all_expensive_would_cost']:>18.6f}  ║")
        lines.append(f"║    Savings:              ${stats['estimated_savings']:>18.6f}  ║")
        lines.append(f"║    Savings %:            {stats['savings_percentage']:>19.1f}%  ║")
        lines.append("╠══════════════════════════════════════════════════╣")

        lines.append("║  PERFORMANCE                                     ║")
        lines.append(f"║    Total time:           {stats['total_time_ms']:>17.0f}ms  ║")
        lines.append(f"║    Avg per operation:    {stats['avg_time_per_op_ms']:>17.1f}ms  ║")
        lines.append(f"║    Est. time saved:      {stats['estimated_time_saved_ms']:>17.0f}ms  ║")

        lines.append("╚══════════════════════════════════════════════════╝")

        output = "\n".join(lines)
        print(output)
        return output

    def reset_stats(self) -> None:
        """Reset usage statistics."""
        self._stats = {
            "cheap_calls": 0,
            "expensive_calls": 0,
            "transfers": 0,
            "estimated_cost": 0.0,
            "estimated_savings": 0.0,
            "total_time_ms": 0.0,
            "cheap_time_ms": 0.0,
            "expensive_time_ms": 0.0,
            "errors": 0,
            "fallbacks": 0,
        }
        self._operation_history.clear()
        self._start_time = time.time()
