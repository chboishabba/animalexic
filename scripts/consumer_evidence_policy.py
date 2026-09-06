"""Consumer-relative evidence gates shared with Drosophila x-pollination.

Observation sufficiency is purpose-relative.  A structure/function validation
receipt cannot silently authorize effector, behavioural, or semantic claims.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Consumer(str, Enum):
    STRUCTURE_FUNCTION = "structure_function"
    BODY_STATE = "body_state"
    BEHAVIOUR = "behaviour"
    COMMUNICATION = "communication"


@dataclass(frozen=True)
class EvidenceGate:
    consumer: Consumer
    required_channels: frozenset[str]
    require_independent_replication: bool = False

    def satisfied(
        self,
        channels: frozenset[str],
        provenance_adequate: bool,
        independent_replication: bool,
    ) -> bool:
        if not provenance_adequate:
            return False
        if not self.required_channels.issubset(channels):
            return False
        if self.require_independent_replication and not independent_replication:
            return False
        return True


STRUCTURE_FUNCTION_GATE = EvidenceGate(
    Consumer.STRUCTURE_FUNCTION,
    frozenset({"connectome", "functional_imaging", "registration"}),
)

BODY_STATE_GATE = EvidenceGate(
    Consumer.BODY_STATE,
    frozenset({"pose_or_kinematics"}),
)

BEHAVIOUR_GATE = EvidenceGate(
    Consumer.BEHAVIOUR,
    frozenset({"pose_or_kinematics", "context"}),
)

COMMUNICATION_GATE = EvidenceGate(
    Consumer.COMMUNICATION,
    frozenset({"behaviour", "interaction", "context"}),
    require_independent_replication=True,
)


def cross_consumer_transfer_requires_receipt(source: Consumer, target: Consumer) -> bool:
    return source is not target
