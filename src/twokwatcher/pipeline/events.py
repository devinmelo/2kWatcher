"""A minimal synchronous event bus.

This exists so that later stages — highlight clipping, the live dashboard, the
tracker — can attach without the frame loop needing to know they exist. The
loop publishes what it observes; subscribers decide what to do about it.

Synchronous on purpose. A subscriber that blocks is a bug worth noticing
immediately, not one to paper over with a queue.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Event:
    kind: str
    frame_index: int
    video_ts: float
    data: dict[str, Any] = field(default_factory=dict)


Subscriber = Callable[[Event], None]


class EventBus:
    def __init__(self) -> None:
        self._subscribers: dict[str, list[Subscriber]] = defaultdict(list)

    def subscribe(self, kind: str, fn: Subscriber) -> None:
        """Subscribe to one event kind, or to '*' for all of them."""
        self._subscribers[kind].append(fn)

    def publish(self, event: Event) -> None:
        for fn in (*self._subscribers.get(event.kind, ()),
                   *self._subscribers.get("*", ())):
            fn(event)
