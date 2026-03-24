"""
Execution plan types for chained DeFi actions.

An :class:`ExecutionPlan` is an ordered sequence of :class:`Action` objects.
Each action represents one atomic step – either a single-chain token swap or a
cross-chain bridge transfer.  Plans are produced by the :mod:`pydefi.planner`
module and can be serialised to / deserialised from JSON.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class ActionType(str, Enum):
    """Supported action categories."""

    SWAP = "swap"
    BRIDGE = "bridge"


@dataclass
class SwapAction:
    """Swap tokens on a single chain via a DEX or aggregator.

    Attributes:
        chain_id: EVM chain ID where the swap takes place.
        token_in: Symbol or address of the input token.
        token_out: Symbol or address of the output token.
        amount_in: Human-readable input amount (e.g. ``"1.5"``).
        protocol: Preferred protocol/aggregator name, or ``"auto"`` to let the
            router choose.
        estimated_amount_out: Best-effort estimate of the output amount
            (filled in by the planner; ``None`` when unknown).
    """

    chain_id: int
    token_in: str
    token_out: str
    amount_in: str
    protocol: str = "auto"
    estimated_amount_out: str | None = None

    @property
    def action_type(self) -> ActionType:
        return ActionType.SWAP

    def describe(self) -> str:
        out = f"Swap {self.amount_in} {self.token_in} → {self.token_out}"
        if self.estimated_amount_out:
            out += f" (≈ {self.estimated_amount_out} {self.token_out})"
        out += f" on chain {self.chain_id} via {self.protocol}"
        return out

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["action_type"] = self.action_type.value
        return d


@dataclass
class BridgeAction:
    """Bridge tokens from one chain to another.

    Attributes:
        src_chain_id: Source EVM chain ID.
        dst_chain_id: Destination EVM chain ID.
        token: Symbol or address of the token being bridged.
        amount_in: Human-readable amount to send.
        protocol: Preferred bridge protocol name, or ``"auto"``.
        estimated_amount_out: Best-effort estimate of the received amount after
            bridge fees (``None`` when unknown).
        estimated_time_seconds: Estimated bridge completion time in seconds.
    """

    src_chain_id: int
    dst_chain_id: int
    token: str
    amount_in: str
    protocol: str = "auto"
    estimated_amount_out: str | None = None
    estimated_time_seconds: int | None = None

    @property
    def action_type(self) -> ActionType:
        return ActionType.BRIDGE

    def describe(self) -> str:
        out = (
            f"Bridge {self.amount_in} {self.token} from chain {self.src_chain_id} "
            f"to chain {self.dst_chain_id}"
        )
        if self.estimated_amount_out:
            out += f" (≈ {self.estimated_amount_out} {self.token} received)"
        if self.estimated_time_seconds is not None:
            out += f" — ETA {self.estimated_time_seconds}s"
        out += f" via {self.protocol}"
        return out

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["action_type"] = self.action_type.value
        return d


Action = SwapAction | BridgeAction


@dataclass
class ExecutionPlan:
    """An ordered list of DeFi actions that together fulfil a user's intent.

    Attributes:
        actions: Ordered steps to execute.
        description: Human-readable summary of the overall intent.
        metadata: Optional free-form metadata (e.g. timestamps, IDs).
    """

    actions: list[Action] = field(default_factory=list)
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def add(self, action: Action) -> "ExecutionPlan":
        """Append an action and return *self* for chaining."""
        self.actions.append(action)
        return self

    def describe(self) -> str:
        """Return a multi-line human-readable description of the plan."""
        lines: list[str] = []
        if self.description:
            lines.append(self.description)
            lines.append("")
        for i, action in enumerate(self.actions, 1):
            lines.append(f"  Step {i}: {action.describe()}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "metadata": self.metadata,
            "actions": [a.to_dict() for a in self.actions],
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialise the plan to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExecutionPlan":
        """Deserialise a plan previously produced by :meth:`to_dict`."""
        actions: list[Action] = []
        for raw in data.get("actions", []):
            atype = raw.get("action_type")
            raw_copy = {k: v for k, v in raw.items() if k != "action_type"}
            if atype == ActionType.SWAP.value:
                actions.append(SwapAction(**raw_copy))
            elif atype == ActionType.BRIDGE.value:
                actions.append(BridgeAction(**raw_copy))
            else:
                raise ValueError(f"Unknown action_type: {atype!r}")
        return cls(
            actions=actions,
            description=data.get("description", ""),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_json(cls, text: str) -> "ExecutionPlan":
        """Deserialise a plan from a JSON string."""
        return cls.from_dict(json.loads(text))
