"""
AECP Debug Monitor & Dashboard

Real-time monitoring of AECP operations including:
- Transfer quality metrics
- Cost tracking and savings estimation
- Time savings vs text-based communication
- Error rates and circuit breaker status
- Live operation logging

Usage:
    >>> from aecp.debug import DebugMonitor, enable_debug
    >>> 
    >>> # Enable global debug mode
    >>> monitor = enable_debug(level="detailed")
    >>> 
    >>> # Or attach to specific agents
    >>> from aecp import AECP
    >>> agent = AECP(adapter)
    >>> monitor = DebugMonitor.attach(agent)
    >>> 
    >>> # After operations, view dashboard
    >>> monitor.dashboard()
    >>> monitor.cost_report()
    >>> monitor.quality_report()
"""

import time
import threading
import logging
import json
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

logger = logging.getLogger("aecp.debug")


class DebugLevel(Enum):
    """Debug verbosity levels."""
    OFF = 0
    MINIMAL = 1       # Only errors and warnings
    STANDARD = 2      # Key operations + metrics
    DETAILED = 3      # All operations + timing
    VERBOSE = 4       # Everything including internal state


class EventType(Enum):
    """Types of monitored events."""
    HANDSHAKE = "handshake"
    CALIBRATION_START = "calibration_start"
    CALIBRATION_END = "calibration_end"
    TRANSFER = "transfer"
    RECEIVE = "receive"
    EMBED = "embed"
    EMBED_BATCH = "embed_batch"
    NEGOTIATION = "negotiation"
    FALLBACK = "fallback"
    ERROR = "error"
    WARNING = "warning"
    CIRCUIT_BREAK = "circuit_break"
    RETRY = "retry"
    RECOVERY = "recovery"


@dataclass
class DebugEvent:
    """A single monitored event."""
    event_type: EventType
    timestamp: float
    agent_id: str
    partner_id: Optional[str] = None
    duration_ms: Optional[float] = None
    quality_score: Optional[float] = None
    cost_estimate: Optional[float] = None
    tokens_used: Optional[int] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def time_str(self) -> str:
        """Format timestamp as readable string."""
        dt = datetime.fromtimestamp(self.timestamp)
        return dt.strftime("%H:%M:%S.%f")[:-3]


@dataclass
class CostConfig:
    """Cost configuration for different providers."""
    # Cost per 1M tokens
    provider_costs: Dict[str, float] = field(default_factory=lambda: {
        "openai:text-embedding-3-small": 0.02,
        "openai:text-embedding-3-large": 0.13,
        "openai:text-embedding-ada-002": 0.10,
        "voyage:voyage-2": 0.10,
        "voyage:voyage-large-2": 0.12,
        "voyage:voyage-code-2": 0.12,
        "cohere:embed-english-v3.0": 0.10,
        "cohere:embed-multilingual-v3.0": 0.10,
        "huggingface:*": 0.0,  # Local, no API cost
        "mock:*": 0.0,
    })
    # Average tokens per text (rough estimate)
    avg_tokens_per_text: int = 10
    # Text-based communication cost estimate (for comparison)
    text_comm_cost_per_msg: float = 0.003  # ~$3/1000 messages via LLM

    def get_cost_per_token(self, model_id: str) -> float:
        """Get cost per token for a model."""
        # Exact match
        if model_id in self.provider_costs:
            return self.provider_costs[model_id] / 1_000_000

        # Wildcard match (provider:*)
        provider = model_id.split(":")[0] if ":" in model_id else model_id
        wildcard = f"{provider}:*"
        if wildcard in self.provider_costs:
            return self.provider_costs[wildcard] / 1_000_000

        # Default: assume moderate cost
        return 0.10 / 1_000_000


class DebugMonitor:
    """
    Real-time monitoring and debugging for AECP operations.

    Tracks all operations, quality metrics, costs, and timing to provide
    a comprehensive view of AECP performance.

    Thread-safe for use in multi-agent environments.
    """

    _global_instance: Optional['DebugMonitor'] = None
    _lock = threading.Lock()

    def __init__(
        self,
        level: DebugLevel = DebugLevel.STANDARD,
        cost_config: Optional[CostConfig] = None,
        max_events: int = 10000,
        log_to_console: bool = True,
        log_to_file: Optional[str] = None,
    ):
        """
        Initialize debug monitor.

        Args:
            level: Debug verbosity level
            cost_config: Cost configuration for providers
            max_events: Maximum events to retain in memory
            log_to_console: Whether to print events to console
            log_to_file: Optional file path for event logging
        """
        self.level = level
        self.cost_config = cost_config or CostConfig()
        self.max_events = max_events
        self.log_to_console = log_to_console
        self.log_to_file = log_to_file

        # Event storage (thread-safe deque)
        self._events: Deque[DebugEvent] = deque(maxlen=max_events)
        self._event_lock = threading.Lock()

        # Aggregate statistics
        self._stats = {
            "total_events": 0,
            "total_embeds": 0,
            "total_transfers": 0,
            "total_calibrations": 0,
            "total_negotiations": 0,
            "total_fallbacks": 0,
            "total_errors": 0,
            "total_retries": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "total_time_ms": 0.0,
            "estimated_text_cost": 0.0,
        }
        self._stats_lock = threading.Lock()

        # Per-agent statistics
        self._agent_stats: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "embeds": 0,
                "transfers_sent": 0,
                "transfers_received": 0,
                "errors": 0,
                "total_time_ms": 0.0,
                "total_cost": 0.0,
                "avg_quality": 0.0,
                "quality_samples": 0,
            }
        )

        # Per-pair quality tracking
        self._pair_quality: Dict[str, List[float]] = defaultdict(list)

        # Timing tracker
        self._active_timers: Dict[str, float] = {}

        # File logger
        self._file_handler = None
        if log_to_file:
            self._file_handler = open(log_to_file, "a")

        # Attached agents (weak references would be ideal but keep simple)
        self._attached_agents: List[str] = []

        # Session start time
        self._session_start = time.time()

    @classmethod
    def get_global(cls) -> Optional['DebugMonitor']:
        """Get the global debug monitor instance."""
        return cls._global_instance

    @classmethod
    def set_global(cls, monitor: 'DebugMonitor') -> None:
        """Set the global debug monitor instance."""
        with cls._lock:
            cls._global_instance = monitor

    @classmethod
    def attach(cls, agent: Any) -> 'DebugMonitor':
        """
        Attach a debug monitor to an AECP agent.

        Creates a new monitor if no global one exists.

        Args:
            agent: AECP agent to monitor

        Returns:
            The debug monitor instance
        """
        monitor = cls._global_instance or cls(level=DebugLevel.STANDARD)
        if cls._global_instance is None:
            cls.set_global(monitor)

        agent_id = getattr(agent, "agent_id", str(id(agent)))
        monitor._attached_agents.append(agent_id)
        monitor.log_event(DebugEvent(
            event_type=EventType.HANDSHAKE,
            timestamp=time.time(),
            agent_id=agent_id,
            metadata={"action": "monitor_attached"},
        ))
        return monitor

    def log_event(self, event: DebugEvent) -> None:
        """
        Log a debug event.

        Thread-safe event recording with optional console/file output.

        Args:
            event: The event to log
        """
        with self._event_lock:
            self._events.append(event)

        with self._stats_lock:
            self._stats["total_events"] += 1

            if event.event_type == EventType.EMBED:
                self._stats["total_embeds"] += 1
            elif event.event_type == EventType.EMBED_BATCH:
                self._stats["total_embeds"] += event.metadata.get("batch_size", 1)
            elif event.event_type == EventType.TRANSFER:
                self._stats["total_transfers"] += 1
            elif event.event_type == EventType.CALIBRATION_END:
                self._stats["total_calibrations"] += 1
            elif event.event_type == EventType.NEGOTIATION:
                self._stats["total_negotiations"] += 1
            elif event.event_type == EventType.FALLBACK:
                self._stats["total_fallbacks"] += 1
            elif event.event_type == EventType.ERROR:
                self._stats["total_errors"] += 1
            elif event.event_type == EventType.RETRY:
                self._stats["total_retries"] += 1

            if event.duration_ms:
                self._stats["total_time_ms"] += event.duration_ms

            if event.tokens_used:
                self._stats["total_tokens"] += event.tokens_used

            if event.cost_estimate:
                self._stats["total_cost"] += event.cost_estimate

            # Track per-agent stats
            agent_stats = self._agent_stats[event.agent_id]
            if event.event_type in (EventType.EMBED, EventType.EMBED_BATCH):
                agent_stats["embeds"] += event.metadata.get("batch_size", 1)
            elif event.event_type == EventType.TRANSFER:
                agent_stats["transfers_sent"] += 1
                if event.partner_id:
                    self._agent_stats[event.partner_id]["transfers_received"] += 1
            elif event.event_type == EventType.ERROR:
                agent_stats["errors"] += 1

            if event.duration_ms:
                agent_stats["total_time_ms"] += event.duration_ms
            if event.cost_estimate:
                agent_stats["total_cost"] += event.cost_estimate
            if event.quality_score is not None:
                n = agent_stats["quality_samples"]
                avg = agent_stats["avg_quality"]
                agent_stats["avg_quality"] = (avg * n + event.quality_score) / (n + 1)
                agent_stats["quality_samples"] += 1

                # Track pair quality
                if event.partner_id:
                    pair_key = f"{event.agent_id}<->{event.partner_id}"
                    self._pair_quality[pair_key].append(event.quality_score)

        # Console output
        if self.log_to_console and self.level.value >= DebugLevel.STANDARD.value:
            self._print_event(event)

        # File output
        if self._file_handler:
            self._write_event_to_file(event)

    def start_timer(self, operation_id: str) -> None:
        """Start a timer for an operation."""
        self._active_timers[operation_id] = time.time()

    def stop_timer(self, operation_id: str) -> float:
        """Stop a timer and return elapsed milliseconds."""
        start = self._active_timers.pop(operation_id, None)
        if start is None:
            return 0.0
        return (time.time() - start) * 1000

    def estimate_cost(
        self,
        model_id: str,
        num_texts: int = 1,
        avg_tokens: Optional[int] = None,
    ) -> float:
        """
        Estimate cost for an embedding operation.

        Args:
            model_id: Model identifier
            num_texts: Number of texts being embedded
            avg_tokens: Average tokens per text (uses config default if None)

        Returns:
            Estimated cost in USD
        """
        tokens = avg_tokens or self.cost_config.avg_tokens_per_text
        total_tokens = num_texts * tokens
        cost_per_token = self.cost_config.get_cost_per_token(model_id)
        return total_tokens * cost_per_token

    def estimate_text_comm_cost(self, num_messages: int = 1) -> float:
        """
        Estimate cost of text-based communication (for comparison).

        Args:
            num_messages: Number of messages

        Returns:
            Estimated cost in USD
        """
        return num_messages * self.cost_config.text_comm_cost_per_msg

    # ── Dashboard & Reports ──────────────────────────────────────────

    def dashboard(self, compact: bool = False) -> str:
        """
        Generate a real-time dashboard view.

        Args:
            compact: If True, show compact single-line summary

        Returns:
            Formatted dashboard string (also printed to console)
        """
        elapsed = time.time() - self._session_start
        elapsed_str = str(timedelta(seconds=int(elapsed)))

        if compact:
            s = self._stats
            line = (
                f"[AECP Monitor] "
                f"⏱ {elapsed_str} | "
                f" {s['total_transfers']} transfers | "
                f" ${s['total_cost']:.6f} spent | "
                f" {s['total_errors']} errors | "
                f" {s['total_fallbacks']} fallbacks"
            )
            print(line)
            return line

        lines = []
        lines.append("")
        lines.append("╔══════════════════════════════════════════════════════════════╗")
        lines.append("║              AECP Debug Monitor Dashboard                   ║")
        lines.append("╠══════════════════════════════════════════════════════════════╣")
        lines.append(f"║  Session Duration: {elapsed_str:>38s}  ║")
        lines.append(f"║  Debug Level:      {self.level.name:>38s}  ║")
        lines.append("╠══════════════════════════════════════════════════════════════╣")

        # Operations summary
        s = self._stats
        lines.append("║  OPERATIONS                                                  ║")
        lines.append(f"║    Total Embeddings:     {s['total_embeds']:>32,d}  ║")
        lines.append(f"║    Total Transfers:      {s['total_transfers']:>32,d}  ║")
        lines.append(f"║    Total Calibrations:   {s['total_calibrations']:>32,d}  ║")
        lines.append(f"║    Total Negotiations:   {s['total_negotiations']:>32,d}  ║")
        lines.append(f"║    Fallbacks to Text:    {s['total_fallbacks']:>32,d}  ║")
        lines.append("╠══════════════════════════════════════════════════════════════╣")

        # Cost summary
        lines.append("║  COST ANALYSIS                                               ║")
        lines.append(f"║    AECP Cost:            ${s['total_cost']:>30.6f}  ║")

        # Calculate what text-based would have cost
        text_cost = self.estimate_text_comm_cost(
            s["total_transfers"] + s["total_embeds"]
        )
        self._stats["estimated_text_cost"] = text_cost
        savings = max(0, text_cost - s["total_cost"])
        savings_pct = (savings / text_cost * 100) if text_cost > 0 else 0

        lines.append(f"║    Text Comm Would Cost: ${text_cost:>30.6f}  ║")
        lines.append(f"║    Estimated Savings:    ${savings:>30.6f}  ║")
        lines.append(f"║    Savings Percentage:   {savings_pct:>31.1f}%  ║")
        lines.append("╠══════════════════════════════════════════════════════════════╣")

        # Performance
        lines.append("║  PERFORMANCE                                                 ║")
        lines.append(f"║    Total Time:           {s['total_time_ms']:>29.1f}ms  ║")
        total_ops = s["total_embeds"] + s["total_transfers"]
        avg_time = s["total_time_ms"] / total_ops if total_ops > 0 else 0
        lines.append(f"║    Avg Time/Operation:   {avg_time:>29.1f}ms  ║")
        lines.append(f"║    Total Tokens:         {s['total_tokens']:>32,d}  ║")
        lines.append("╠══════════════════════════════════════════════════════════════╣")

        # Reliability
        lines.append("║  RELIABILITY                                                 ║")
        lines.append(f"║    Errors:               {s['total_errors']:>32,d}  ║")
        lines.append(f"║    Retries:              {s['total_retries']:>32,d}  ║")
        total_events = s["total_events"]
        error_rate = (s["total_errors"] / total_events * 100) if total_events > 0 else 0
        lines.append(f"║    Error Rate:           {error_rate:>31.2f}%  ║")
        lines.append("╠══════════════════════════════════════════════════════════════╣")

        # Per-agent summary
        if self._agent_stats:
            lines.append("║  AGENTS                                                      ║")
            for agent_id, stats in self._agent_stats.items():
                quality_str = (
                    f"{stats['avg_quality']:.4f}"
                    if stats["quality_samples"] > 0
                    else "N/A"
                )
                lines.append(
                    f"║    {agent_id[:20]:<20s} "
                    f"emb:{stats['embeds']:>5d} "
                    f"tx:{stats['transfers_sent']:>4d} "
                    f"q:{quality_str:>7s} "
                    f"err:{stats['errors']:>3d}  ║"
                )
            lines.append("╠══════════════════════════════════════════════════════════════╣")

        # Quality per pair
        if self._pair_quality:
            lines.append("║  PAIR QUALITY                                                ║")
            for pair, scores in self._pair_quality.items():
                if scores:
                    import numpy as np
                    avg = float(np.mean(scores))
                    worst = float(np.min(scores))
                    lines.append(
                        f"║    {pair[:25]:<25s} "
                        f"avg:{avg:.4f} "
                        f"worst:{worst:.4f} "
                        f"n:{len(scores):>4d}  ║"
                    )

        lines.append("╚══════════════════════════════════════════════════════════════╝")

        output = "\n".join(lines)
        print(output)
        return output

    def cost_report(self) -> str:
        """
        Generate a detailed cost report.

        Returns:
            Formatted cost report string
        """
        lines = []
        lines.append("\n AECP Cost Report")
        lines.append("=" * 50)

        s = self._stats
        total_ops = s["total_embeds"] + s["total_transfers"]

        lines.append(f"  Total operations:        {total_ops:,}")
        lines.append(f"  Total tokens used:       {s['total_tokens']:,}")
        lines.append(f"  Total AECP cost:         ${s['total_cost']:.6f}")

        # Text-based comparison
        text_cost = self.estimate_text_comm_cost(total_ops)
        lines.append(f"  Text-based would cost:   ${text_cost:.6f}")

        savings = max(0, text_cost - s["total_cost"])
        lines.append(f"  Estimated savings:       ${savings:.6f}")

        if text_cost > 0:
            pct = savings / text_cost * 100
            lines.append(f"  Savings percentage:      {pct:.1f}%")

        # Time savings estimate
        # Assume text-based communication adds ~500ms per message for LLM processing
        time_text_ms = total_ops * 500
        time_aecp_ms = s["total_time_ms"]
        time_saved_ms = max(0, time_text_ms - time_aecp_ms)

        lines.append(f"\n  ⏱ Time Analysis:")
        lines.append(f"  AECP time:               {time_aecp_ms:.0f}ms")
        lines.append(f"  Text-based estimate:     {time_text_ms:.0f}ms")
        lines.append(f"  Time saved:              {time_saved_ms:.0f}ms ({time_saved_ms/1000:.1f}s)")

        # Per-agent breakdown
        if self._agent_stats:
            lines.append(f"\n  Per-Agent Breakdown:")
            for agent_id, stats in self._agent_stats.items():
                lines.append(f"    {agent_id}:")
                lines.append(f"      Cost:     ${stats['total_cost']:.6f}")
                lines.append(f"      Time:     {stats['total_time_ms']:.0f}ms")
                lines.append(f"      Embeds:   {stats['embeds']}")

        output = "\n".join(lines)
        print(output)
        return output

    def quality_report(self) -> str:
        """
        Generate a quality metrics report.

        Returns:
            Formatted quality report string
        """
        lines = []
        lines.append("\n AECP Quality Report")
        lines.append("=" * 50)

        if not self._pair_quality:
            lines.append("  No quality data available yet.")
            lines.append("  Run calibration and transfers to collect data.")
        else:
            import numpy as np

            for pair, scores in self._pair_quality.items():
                if scores:
                    arr = np.array(scores)
                    lines.append(f"\n  {pair}:")
                    lines.append(f"    Samples:    {len(scores)}")
                    lines.append(f"    Mean:       {float(np.mean(arr)):.4f}")
                    lines.append(f"    Median:     {float(np.median(arr)):.4f}")
                    lines.append(f"    Std Dev:    {float(np.std(arr)):.4f}")
                    lines.append(f"    Min:        {float(np.min(arr)):.4f}")
                    lines.append(f"    Max:        {float(np.max(arr)):.4f}")
                    lines.append(f"    P5:         {float(np.percentile(arr, 5)):.4f}")
                    lines.append(f"    P95:        {float(np.percentile(arr, 95)):.4f}")

                    # Quality assessment
                    mean_q = float(np.mean(arr))
                    if mean_q >= 0.95:
                        assessment = " Excellent"
                    elif mean_q >= 0.90:
                        assessment = " Very Good"
                    elif mean_q >= 0.80:
                        assessment = " Good"
                    elif mean_q >= 0.70:
                        assessment = " Acceptable"
                    else:
                        assessment = " Poor - Consider recalibration"
                    lines.append(f"    Assessment: {assessment}")

        output = "\n".join(lines)
        print(output)
        return output

    def recent_events(self, n: int = 20) -> List[DebugEvent]:
        """
        Get the most recent events.

        Args:
            n: Number of events to return

        Returns:
            List of recent events
        """
        with self._event_lock:
            events = list(self._events)
        return events[-n:]

    def get_stats(self) -> Dict[str, Any]:
        """
        Get raw statistics dictionary.

        Returns:
            Dictionary with all tracked statistics
        """
        with self._stats_lock:
            return {
                **self._stats.copy(),
                "session_duration_s": time.time() - self._session_start,
                "agent_stats": dict(self._agent_stats),
                "pair_quality": {
                    k: list(v) for k, v in self._pair_quality.items()
                },
            }

    def export_json(self, filepath: Optional[str] = None) -> str:
        """
        Export all debug data as JSON.

        Args:
            filepath: Optional file path to write JSON to

        Returns:
            JSON string of debug data
        """
        data = {
            "session_start": datetime.fromtimestamp(self._session_start).isoformat(),
            "session_duration_s": time.time() - self._session_start,
            "stats": self._stats.copy(),
            "agent_stats": {k: dict(v) for k, v in self._agent_stats.items()},
            "pair_quality": {
                k: v for k, v in self._pair_quality.items()
            },
            "recent_events": [
                {
                    "type": e.event_type.value,
                    "time": e.time_str,
                    "agent": e.agent_id,
                    "partner": e.partner_id,
                    "duration_ms": e.duration_ms,
                    "quality": e.quality_score,
                    "cost": e.cost_estimate,
                    "error": e.error,
                }
                for e in self.recent_events(100)
            ],
        }

        json_str = json.dumps(data, indent=2, default=str)

        if filepath:
            with open(filepath, "w") as f:
                f.write(json_str)

        return json_str

    def reset(self) -> None:
        """Reset all monitoring data."""
        with self._event_lock:
            self._events.clear()
        with self._stats_lock:
            for key in self._stats:
                if isinstance(self._stats[key], (int, float)):
                    self._stats[key] = 0 if isinstance(self._stats[key], int) else 0.0
            self._agent_stats.clear()
            self._pair_quality.clear()
        self._session_start = time.time()

    def close(self) -> None:
        """Close the monitor and clean up resources."""
        if self._file_handler:
            self._file_handler.close()
            self._file_handler = None

    # ── Internal Methods ─────────────────────────────────────────────

    def _print_event(self, event: DebugEvent) -> None:
        """Print an event to console with formatting."""
        icons = {
            EventType.HANDSHAKE: "",
            EventType.CALIBRATION_START: "",
            EventType.CALIBRATION_END: "",
            EventType.TRANSFER: "",
            EventType.RECEIVE: "",
            EventType.EMBED: "",
            EventType.EMBED_BATCH: "",
            EventType.NEGOTIATION: "",
            EventType.FALLBACK: "",
            EventType.ERROR: "",
            EventType.WARNING: "⚠️",
            EventType.CIRCUIT_BREAK: "",
            EventType.RETRY: "",
            EventType.RECOVERY: "",
        }

        icon = icons.get(event.event_type, "")
        parts = [f"[{event.time_str}]", icon, event.event_type.value.upper()]

        if event.agent_id:
            parts.append(f"agent={event.agent_id}")
        if event.partner_id:
            parts.append(f"partner={event.partner_id}")
        if event.duration_ms is not None:
            parts.append(f"time={event.duration_ms:.1f}ms")
        if event.quality_score is not None:
            parts.append(f"quality={event.quality_score:.4f}")
        if event.cost_estimate is not None:
            parts.append(f"cost=${event.cost_estimate:.6f}")
        if event.error:
            parts.append(f"error={event.error}")

        # Only print detailed events in detailed/verbose mode
        if event.event_type in (EventType.EMBED, EventType.EMBED_BATCH):
            if self.level.value < DebugLevel.DETAILED.value:
                return

        print(" ".join(parts))

    def _write_event_to_file(self, event: DebugEvent) -> None:
        """Write an event to the log file."""
        if not self._file_handler:
            return
        try:
            entry = {
                "type": event.event_type.value,
                "time": event.time_str,
                "agent": event.agent_id,
                "partner": event.partner_id,
                "duration_ms": event.duration_ms,
                "quality": event.quality_score,
                "cost": event.cost_estimate,
                "error": event.error,
                "metadata": event.metadata,
            }
            self._file_handler.write(json.dumps(entry, default=str) + "\n")
            self._file_handler.flush()
        except Exception:
            pass  # Don't let logging failures affect operation


# ── Convenience Functions ────────────────────────────────────────────

def enable_debug(
    level: Union[str, DebugLevel] = "standard",
    cost_config: Optional[CostConfig] = None,
    log_to_console: bool = True,
    log_to_file: Optional[str] = None,
) -> DebugMonitor:
    """
    Enable global AECP debug monitoring.

    Args:
        level: Debug level ("off", "minimal", "standard", "detailed", "verbose")
        cost_config: Optional cost configuration
        log_to_console: Whether to print to console
        log_to_file: Optional log file path

    Returns:
        The global DebugMonitor instance

    Example:
        >>> from aecp.debug import enable_debug
        >>> monitor = enable_debug("detailed")
        >>> # ... run AECP operations ...
        >>> monitor.dashboard()
        >>> monitor.cost_report()
    """
    if isinstance(level, str):
        level_map = {
            "off": DebugLevel.OFF,
            "minimal": DebugLevel.MINIMAL,
            "standard": DebugLevel.STANDARD,
            "detailed": DebugLevel.DETAILED,
            "verbose": DebugLevel.VERBOSE,
        }
        level = level_map.get(level.lower(), DebugLevel.STANDARD)

    monitor = DebugMonitor(
        level=level,
        cost_config=cost_config,
        log_to_console=log_to_console,
        log_to_file=log_to_file,
    )
    DebugMonitor.set_global(monitor)
    return monitor


def disable_debug() -> None:
    """Disable global debug monitoring."""
    monitor = DebugMonitor.get_global()
    if monitor:
        monitor.close()
    DebugMonitor.set_global(None)


def get_monitor() -> Optional[DebugMonitor]:
    """Get the current global debug monitor."""
    return DebugMonitor.get_global()


__all__ = [
    "DebugMonitor",
    "DebugLevel",
    "DebugEvent",
    "EventType",
    "CostConfig",
    "enable_debug",
    "disable_debug",
    "get_monitor",
]
