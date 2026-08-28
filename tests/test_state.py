from twokwatcher.state import GameState, StateMachine


def _feed(machine, state, count, start=0):
    result = None
    for i in range(count):
        result = machine.update(state, timestamp=float(start + i),
                                frame_index=start + i)
    return result


def test_transition_requires_debounce():
    machine = StateMachine(min_frames=3)
    assert _feed(machine, GameState.LIVE, 2) is None
    assert machine.state is GameState.UNKNOWN

    transition = machine.update(GameState.LIVE, timestamp=3.0, frame_index=3)
    assert transition is not None
    assert transition.current is GameState.LIVE
    assert machine.state is GameState.LIVE


def test_noise_does_not_commit():
    machine = StateMachine(min_frames=3)
    _feed(machine, GameState.LIVE, 3)

    # Two stray MENU frames between LIVE frames must not move us.
    machine.update(GameState.MENU, timestamp=10.0, frame_index=10)
    machine.update(GameState.LIVE, timestamp=11.0, frame_index=11)
    machine.update(GameState.MENU, timestamp=12.0, frame_index=12)
    assert machine.state is GameState.LIVE


def test_is_active_gates_downstream_work():
    machine = StateMachine(min_frames=1)
    machine.update(GameState.MENU, timestamp=0.0, frame_index=0)
    assert not machine.is_active

    machine.update(GameState.LIVE, timestamp=1.0, frame_index=1)
    assert machine.is_active

    machine.update(GameState.REPLAY, timestamp=2.0, frame_index=2)
    assert not machine.is_active


def test_history_records_every_commit():
    machine = StateMachine(min_frames=1)
    for state in (GameState.MENU, GameState.LIVE, GameState.POST_GAME):
        machine.update(state, timestamp=0.0, frame_index=0)
    assert [t.current for t in machine.history] == [
        GameState.MENU, GameState.LIVE, GameState.POST_GAME
    ]
