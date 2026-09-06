"""Governed neural/behaviour observation adapter for Animalexic.

This is a runtime-facing contract for connectome/functional-imaging experiments.
It preserves Animalexic's substrate -> candidate -> promoted/abstain/reject
architecture and does not import Drosophila-specific semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Generic, Mapping, TypeVar


class Decision(str, Enum):
    CANDIDATE = "candidate"
    PROMOTED = "promoted"
    ABSTAIN = "abstain"
    REJECT = "reject"


class NeuralObservationModality(str, Enum):
    STRUCTURAL_CONNECTOME = "structural_connectome"
    OPTICAL_CALCIUM = "optical_calcium"
    OPTICAL_VOLTAGE = "optical_voltage"
    ELECTROPHYSIOLOGY = "electrophysiology"
    BODY_KINEMATICS = "body_kinematics"
    CONTACT_FORCE = "contact_force"


@dataclass(frozen=True)
class SourceReceipt:
    author_or_consortium: str
    title: str
    stable_identifier: str

    def __post_init__(self) -> None:
        if not self.author_or_consortium.strip():
            raise ValueError("source author/consortium is required")
        if not self.title.strip():
            raise ValueError("source title is required")
        if not self.stable_identifier.strip():
            raise ValueError("DOI or another stable identifier is required")


@dataclass(frozen=True)
class ProvenanceNode:
    identifier: str
    upstream: tuple[str, ...] = ()


def upstream_closure(nodes: Mapping[str, ProvenanceNode], start: str) -> frozenset[str]:
    seen: set[str] = set()
    todo = [start]
    while todo:
        current = todo.pop()
        if current in seen:
            continue
        seen.add(current)
        node = nodes.get(current)
        if node is not None:
            todo.extend(node.upstream)
    return frozenset(seen)


def independent(nodes: Mapping[str, ProvenanceNode], left: str, right: str) -> bool:
    return upstream_closure(nodes, left).isdisjoint(upstream_closure(nodes, right))


ObservationT = TypeVar("ObservationT")
CandidateT = TypeVar("CandidateT")
PromotedT = TypeVar("PromotedT")
ReceiptT = TypeVar("ReceiptT")
ResidualT = TypeVar("ResidualT")


@dataclass(frozen=True)
class GovernedNeuralObservation(Generic[ObservationT, CandidateT, PromotedT, ReceiptT, ResidualT]):
    observation: ObservationT
    candidate: CandidateT
    decision: Decision
    receipt: ReceiptT | None
    residual: ResidualT
    residual_admissible: Callable[[ResidualT], bool]
    receipt_admissible: Callable[[ObservationT, CandidateT, ReceiptT], bool]
    materialise: Callable[[CandidateT], PromotedT]

    def promote(self) -> PromotedT:
        if self.decision is not Decision.PROMOTED:
            raise ValueError("only promoted candidates can mutate canonical state")
        if self.receipt is None:
            raise ValueError("promotion requires a receipt")
        if not self.residual_admissible(self.residual):
            raise ValueError("residual is outside the promotion margin")
        if not self.receipt_admissible(self.observation, self.candidate, self.receipt):
            raise ValueError("receipt is not admissible")
        return self.materialise(self.candidate)


@dataclass(frozen=True)
class BehaviourMotif:
    identifier: str
    interval_start: float
    interval_end: float
    semantic_label: str | None = None

    @property
    def has_promoted_semantics(self) -> bool:
        return self.semantic_label is not None


def recurrence_does_not_imply_meaning(motif: BehaviourMotif) -> bool:
    return not motif.has_promoted_semantics
