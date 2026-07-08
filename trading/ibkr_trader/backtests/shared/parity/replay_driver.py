from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Generic, Sequence, TypeVar

StateT = TypeVar("StateT")
BarInputT = TypeVar("BarInputT")
OrderUpdateT = TypeVar("OrderUpdateT")
FillT = TypeVar("FillT")
ActionT = TypeVar("ActionT")
EventT = TypeVar("EventT")


@dataclass(slots=True)
class ReplayStep(Generic[BarInputT, OrderUpdateT, FillT]):
    bar_input: BarInputT | None = None
    order_updates: list[OrderUpdateT] = field(default_factory=list)
    fills: list[FillT] = field(default_factory=list)


@dataclass(slots=True)
class ReplayResult(Generic[StateT, ActionT, EventT]):
    state: StateT
    actions: list[ActionT]
    events: list[EventT]


def run_replay(
    initial_state: StateT,
    *,
    steps: Sequence[ReplayStep[BarInputT, OrderUpdateT, FillT]],
    on_bar: Callable[[StateT, BarInputT], tuple[StateT, list[ActionT], list[EventT]]],
    on_order_update: Callable[[StateT, OrderUpdateT], tuple[StateT, list[ActionT], list[EventT]]],
    on_fill: Callable[[StateT, FillT], tuple[StateT, list[ActionT], list[EventT]]],
) -> ReplayResult[StateT, ActionT, EventT]:
    state = initial_state
    action_stream: list[ActionT] = []
    event_stream: list[EventT] = []

    for step in steps:
        if step.bar_input is not None:
            state, actions, events = on_bar(state, step.bar_input)
            action_stream.extend(actions)
            event_stream.extend(events)

        for update in step.order_updates:
            state, actions, events = on_order_update(state, update)
            action_stream.extend(actions)
            event_stream.extend(events)

        for fill in step.fills:
            state, actions, events = on_fill(state, fill)
            action_stream.extend(actions)
            event_stream.extend(events)

    return ReplayResult(state=state, actions=action_stream, events=event_stream)
