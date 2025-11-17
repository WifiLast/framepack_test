"""
Profile-Guided Optimization (PGO) utilities for FramePack.

This module provides comprehensive profiling tools to identify performance bottlenecks
and guide optimization decisions with real data.
"""

import os
import time
import json
import torch
import functools
from pathlib import Path
from contextlib import contextmanager
from typing import Optional, Dict, List, Any, Callable
from collections import defaultdict
import threading


class TimingStats:
    """Collect and aggregate timing statistics."""

    def __init__(self):
        self.timings: Dict[str, List[float]] = defaultdict(list)
        self.counts: Dict[str, int] = defaultdict(int)
        self.lock = threading.Lock()

    def record(self, name: str, duration: float):
        """Record a timing measurement."""
        with self.lock:
            self.timings[name].append(duration)
            self.counts[name] += 1

    def get_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a named operation."""
        if name not in self.timings or not self.timings[name]:
            return {}

        times = self.timings[name]
        return {
            'count': len(times),
            'total': sum(times),
            'mean': sum(times) / len(times),
            'min': min(times),
            'max': max(times),
            'last': times[-1],
        }

    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all operations."""
        return {name: self.get_stats(name) for name in self.timings.keys()}

    def clear(self):
        """Clear all statistics."""
        with self.lock:
            self.timings.clear()
            self.counts.clear()

    def print_summary(self, top_n: int = 20):
        """Print a summary of the slowest operations."""
        all_stats = self.get_all_stats()

        # Sort by total time
        sorted_stats = sorted(
            all_stats.items(),
            key=lambda x: x[1].get('total', 0),
            reverse=True
        )

        print("\n" + "="*80)
        print("PROFILING SUMMARY - Top Operations by Total Time")
        print("="*80)
        print(f"{'Operation':<50} {'Count':>8} {'Total':>10} {'Mean':>10} {'Min':>10} {'Max':>10}")
        print("-"*80)

        for name, stats in sorted_stats[:top_n]:
            if not stats:
                continue
            print(
                f"{name[:48]:<50} "
                f"{stats['count']:>8d} "
                f"{stats['total']:>10.3f}s "
                f"{stats['mean']:>10.3f}s "
                f"{stats['min']:>10.3f}s "
                f"{stats['max']:>10.3f}s"
            )

        print("="*80 + "\n")


# Global timing stats instance
_global_stats = TimingStats()


def get_global_stats() -> TimingStats:
    """Get the global timing statistics instance."""
    return _global_stats


@contextmanager
def profile_section(name: str, enabled: bool = True):
    """
    Context manager for profiling a code section.

    Usage:
        with profile_section("encode_text"):
            result = encode_text(prompt)
    """
    if not enabled:
        yield
        return

    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        _global_stats.record(name, duration)


def profile_function(name: Optional[str] = None, enabled: bool = True):
    """
    Decorator for profiling functions.

    Usage:
        @profile_function("my_function")
        def my_function(x):
            return x * 2
    """
    def decorator(func: Callable) -> Callable:
        func_name = name or f"{func.__module__}.{func.__name__}"

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not enabled:
                return func(*args, **kwargs)

            with profile_section(func_name, enabled=enabled):
                return func(*args, **kwargs)

        return wrapper
    return decorator


class PyTorchProfiler:
    """
    Wrapper for PyTorch profiler with easy export to Chrome trace format.
    """

    def __init__(
        self,
        output_dir: str = "./profiling_results",
        enabled: bool = True,
        record_shapes: bool = True,
        profile_memory: bool = True,
        with_stack: bool = False,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.enabled = enabled
        self.record_shapes = record_shapes
        self.profile_memory = profile_memory
        self.with_stack = with_stack
        self.profiler = None

    def start(self, trace_name: str = "trace"):
        """Start profiling."""
        if not self.enabled:
            return

        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        self.profiler = torch.profiler.profile(
            activities=activities,
            record_shapes=self.record_shapes,
            profile_memory=self.profile_memory,
            with_stack=self.with_stack,
        )
        self.profiler.__enter__()
        self.trace_name = trace_name

    def stop(self) -> Optional[str]:
        """Stop profiling and export trace."""
        if not self.enabled or self.profiler is None:
            return None

        self.profiler.__exit__(None, None, None)

        # Export to Chrome trace format
        trace_path = self.output_dir / f"{self.trace_name}.json"
        self.profiler.export_chrome_trace(str(trace_path))

        print(f"\n{'='*80}")
        print(f"PyTorch Profiler trace exported to: {trace_path}")
        print(f"Open in Chrome: chrome://tracing")
        print(f"{'='*80}\n")

        # Print key stats
        print(self.profiler.key_averages().table(
            sort_by="cuda_time_total" if torch.cuda.is_available() else "cpu_time_total",
            row_limit=20
        ))

        self.profiler = None
        return str(trace_path)

    @contextmanager
    def profile(self, trace_name: str = "trace"):
        """Context manager for profiling a block."""
        self.start(trace_name)
        try:
            yield self
        finally:
            self.stop()


class MemoryTracker:
    """Track GPU memory usage over time."""

    def __init__(self, enabled: bool = True, device: str = "cuda"):
        self.enabled = enabled and torch.cuda.is_available()
        self.device = device
        self.snapshots: List[Dict[str, Any]] = []

    def snapshot(self, label: str):
        """Take a memory snapshot."""
        if not self.enabled:
            return

        snapshot = {
            'label': label,
            'timestamp': time.time(),
            'allocated_mb': torch.cuda.memory_allocated(self.device) / 1024**2,
            'reserved_mb': torch.cuda.memory_reserved(self.device) / 1024**2,
            'max_allocated_mb': torch.cuda.max_memory_allocated(self.device) / 1024**2,
        }
        self.snapshots.append(snapshot)

    def print_summary(self):
        """Print memory usage summary."""
        if not self.enabled or not self.snapshots:
            return

        print("\n" + "="*80)
        print("MEMORY USAGE SUMMARY")
        print("="*80)
        print(f"{'Label':<40} {'Allocated':>12} {'Reserved':>12} {'Max Alloc':>12}")
        print("-"*80)

        for snapshot in self.snapshots:
            print(
                f"{snapshot['label']:<40} "
                f"{snapshot['allocated_mb']:>11.1f}M "
                f"{snapshot['reserved_mb']:>11.1f}M "
                f"{snapshot['max_allocated_mb']:>11.1f}M"
            )

        print("="*80 + "\n")

    def export_json(self, output_path: str):
        """Export snapshots to JSON."""
        with open(output_path, 'w') as f:
            json.dump(self.snapshots, f, indent=2)
        print(f"Memory snapshots exported to: {output_path}")

    def clear(self):
        """Clear snapshots and reset max memory."""
        self.snapshots.clear()
        if self.enabled:
            torch.cuda.reset_peak_memory_stats(self.device)


class IterationProfiler:
    """Profile individual iterations in a loop (e.g., diffusion steps)."""

    def __init__(self, name: str = "iteration", enabled: bool = True):
        self.name = name
        self.enabled = enabled
        self.iteration_times: List[float] = []
        self.current_start: Optional[float] = None

    def start_iteration(self):
        """Mark the start of an iteration."""
        if self.enabled:
            self.current_start = time.time()

    def end_iteration(self):
        """Mark the end of an iteration."""
        if self.enabled and self.current_start is not None:
            duration = time.time() - self.current_start
            self.iteration_times.append(duration)
            self.current_start = None

    def get_stats(self) -> Dict[str, float]:
        """Get iteration statistics."""
        if not self.iteration_times:
            return {}

        times = self.iteration_times
        return {
            'count': len(times),
            'total': sum(times),
            'mean': sum(times) / len(times),
            'min': min(times),
            'max': max(times),
            'median': sorted(times)[len(times) // 2],
        }

    def print_summary(self):
        """Print iteration summary."""
        stats = self.get_stats()
        if not stats:
            return

        print(f"\n{'='*80}")
        print(f"ITERATION PROFILING: {self.name}")
        print(f"{'='*80}")
        print(f"Total iterations: {stats['count']}")
        print(f"Total time:       {stats['total']:.3f}s")
        print(f"Mean time/iter:   {stats['mean']:.3f}s ({1/stats['mean']:.2f} it/s)")
        print(f"Min time:         {stats['min']:.3f}s")
        print(f"Max time:         {stats['max']:.3f}s")
        print(f"Median time:      {stats['median']:.3f}s")
        print(f"{'='*80}\n")

    @contextmanager
    def iteration(self):
        """Context manager for a single iteration."""
        self.start_iteration()
        try:
            yield
        finally:
            self.end_iteration()


def export_profiling_report(
    output_dir: str,
    timing_stats: Optional[TimingStats] = None,
    memory_tracker: Optional[MemoryTracker] = None,
    iteration_profiler: Optional[IterationProfiler] = None,
):
    """Export a comprehensive profiling report."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    report = {
        'timestamp': time.time(),
        'timing_stats': {},
        'memory_snapshots': [],
        'iteration_stats': {},
    }

    # Add timing stats
    if timing_stats:
        report['timing_stats'] = timing_stats.get_all_stats()

    # Add memory snapshots
    if memory_tracker:
        report['memory_snapshots'] = memory_tracker.snapshots

    # Add iteration stats
    if iteration_profiler:
        report['iteration_stats'] = iteration_profiler.get_stats()

    # Export to JSON
    report_path = output_path / "profiling_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Profiling report exported to: {report_path}")
    print(f"{'='*80}\n")

    return str(report_path)
