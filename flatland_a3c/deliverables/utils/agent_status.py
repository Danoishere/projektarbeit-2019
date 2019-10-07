from enum import IntEnum

class RailAgentStatus(IntEnum):
    READY_TO_DEPART = 0  # not in grid yet (position is None) -> prediction as if it were at initial position
    ACTIVE = 1  # in grid (position is not None), not done -> prediction is remaining path
    DONE = 2  # in grid (position is not None), but done -> prediction is stay at target forever
    DONE_REMOVED = 3  # removed from grid (position is None) -> prediction is None
