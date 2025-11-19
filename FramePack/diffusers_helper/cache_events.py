from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def _format_meta(meta: Optional[Dict[str, Any]]) -> str:
    if not meta:
        return "-"
    parts = []
    for key, value in meta.items():
        if value is None:
            continue
        if isinstance(value, float):
            parts.append(f"{key}: {value:.4f}")
        else:
            parts.append(f"{key}: {value}")
    return ", ".join(parts) if parts else "-"


@dataclass
class CacheEvent:
    cache_type: str
    time_seconds: float
    time_frame: float
    step_index: int
    chunk_id: int
    meta: Dict[str, Any] = field(default_factory=dict)


class CacheEventRecorder:
    """
    Tracks cache hits during generation and maps them back to video timeline.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._timeline: List[CacheEvent] = []
        self._current_chunk: Optional[Dict[str, Any]] = None
        self._current_step: int = 0
        self._chunk_counter: int = 0

    def start_chunk(self, *, start_frame: float, steps: int, label: Optional[str] = None):
        """
        Begin recording events for a chunk that maps to a section of the video.
        """
        self._current_chunk = {
            "chunk_id": self._chunk_counter,
            "start_frame": float(start_frame),
            "end_frame": float(start_frame),
            "steps": max(1, int(steps)),
            "label": label or f"chunk_{self._chunk_counter}",
            "events": [],
        }
        self._current_step = 0
        self._chunk_counter += 1

    def mark_step(self, step_index: int):
        """
        Update the current diffusion step within the active chunk.
        """
        self._current_step = max(0, int(step_index))

    def record_event(self, cache_type: str, meta: Optional[Dict[str, Any]] = None):
        """
        Record a cache hit for the active chunk.
        """
        if self._current_chunk is None:
            return
        steps = max(1, self._current_chunk["steps"])
        fraction = min(1.0, max(0.0, self._current_step / steps))
        event = {
            "cache_type": cache_type,
            "meta": meta or {},
            "fraction": fraction,
            "step": self._current_step,
        }
        self._current_chunk["events"].append(event)

    def finalize_chunk(self, *, end_frame: float):
        """
        Close out the current chunk and map recorded events to absolute timeline positions.
        """
        if self._current_chunk is None:
            return
        chunk = self._current_chunk
        chunk["end_frame"] = max(float(end_frame), chunk["start_frame"])
        start_time = chunk["start_frame"] / 30.0
        end_time = chunk["end_frame"] / 30.0
        duration = max(end_time - start_time, 1e-6)
        for event in chunk["events"]:
            event_time = start_time + duration * event["fraction"]
            event_frame = chunk["start_frame"] + (chunk["end_frame"] - chunk["start_frame"]) * event["fraction"]
            self._timeline.append(
                CacheEvent(
                    cache_type=event["cache_type"],
                    time_seconds=event_time,
                    time_frame=event_frame,
                    step_index=event["step"],
                    chunk_id=chunk["chunk_id"],
                    meta=event["meta"],
                )
            )
        self._current_chunk = None
        self._current_step = 0

    def cancel_chunk(self):
        """
        Discard the active chunk without recording events.
        """
        self._current_chunk = None
        self._current_step = 0

    def has_events(self) -> bool:
        return bool(self._timeline)

    def events(self) -> List[CacheEvent]:
        return list(self._timeline)

    def to_markdown(self) -> str:
        """
        Render recorded events as a markdown table for the UI.
        """
        if not self._timeline:
            return "No cache hits recorded yet."
        lines = [
            "| Time (s) | Frame | Cache | Details |",
            "| --- | --- | --- | --- |",
        ]
        for event in self._timeline:
            details = _format_meta(event.meta)
            lines.append(f"| {event.time_seconds:.2f} | {event.time_frame:.0f} | {event.cache_type} | {details} |")
        return "\n".join(lines)
