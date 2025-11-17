#!/usr/bin/env python3
"""
Standalone profiling script for FramePack.

This script demonstrates how to use the profiling tools to identify bottlenecks
in your FramePack workflow.

Usage:
    python profile_demo.py

The script will:
1. Profile a simple inference run
2. Generate timing statistics
3. Track memory usage
4. Export results to ./profiling_results/
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from diffusers_helper.profiling import (
    profile_section,
    get_global_stats,
    PyTorchProfiler,
    MemoryTracker,
    export_profiling_report,
)


def main():
    print("="*80)
    print("FramePack Profiling Demo")
    print("="*80)
    print()
    print("This demo shows how to use profiling tools to identify bottlenecks.")
    print("In a real scenario, you would:")
    print("  1. Run your actual workload with --enable-profiling")
    print("  2. Analyze the results")
    print("  3. Optimize the slowest operations")
    print("  4. Re-profile to confirm improvements")
    print()
    print("="*80)
    print()

    # Initialize profilers
    pytorch_profiler = PyTorchProfiler(
        output_dir="./profiling_results",
        enabled=True,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    )
    memory_tracker = MemoryTracker(enabled=True)

    # Start PyTorch profiler
    pytorch_profiler.start("demo_trace")
    memory_tracker.snapshot("demo_start")

    # Simulate some operations with profiling
    print("Simulating operations...")
    print()

    with profile_section("operation_1_fast"):
        import time
        time.sleep(0.1)
        print("  ✓ Fast operation completed (0.1s)")

    memory_tracker.snapshot("after_op1")

    with profile_section("operation_2_slow"):
        time.sleep(0.5)
        print("  ✓ Slow operation completed (0.5s)")

    memory_tracker.snapshot("after_op2")

    # Simulate multiple iterations
    with profile_section("operation_3_repeated"):
        for i in range(10):
            time.sleep(0.05)
        print("  ✓ Repeated operation completed (10x 0.05s)")

    memory_tracker.snapshot("after_op3")

    # Stop PyTorch profiler
    trace_path = pytorch_profiler.stop()

    # Print summaries
    print()
    get_global_stats().print_summary(top_n=10)
    memory_tracker.print_summary()

    # Export comprehensive report
    report_path = export_profiling_report(
        output_dir="./profiling_results",
        timing_stats=get_global_stats(),
        memory_tracker=memory_tracker,
    )

    print()
    print("="*80)
    print("Profiling Complete!")
    print("="*80)
    print()
    print("Results saved to:")
    print(f"  - Chrome trace: {trace_path}")
    print(f"  - JSON report:  {report_path}")
    print()
    print("To view the Chrome trace:")
    print("  1. Open Chrome browser")
    print("  2. Go to: chrome://tracing")
    print(f"  3. Load: {trace_path}")
    print()
    print("="*80)


if __name__ == "__main__":
    main()
