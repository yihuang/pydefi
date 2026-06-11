"""RouteDAG — fluent builder for split/merge swap routes represented as a DAG.

The builder lives here in pathfinder because it is the intermediate
representation between high-level routing algorithms and low-level DeFiVM
bytecode generation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from pydefi._math import MAX_BPS
from pydefi.types import Address, BasePool, RouteAction, RouteBridge, RouteSplit, RouteSplitLeg, RouteSwap, Token


@dataclass
class _RouteSplitLegBuilder:
    fraction_bps: int
    actions: list[RouteAction] = field(default_factory=list)
    current_token: Token | None = None


@dataclass
class _RouteSplitBuilder:
    origin_token: Token
    legs: list[_RouteSplitLegBuilder] = field(default_factory=list)
    active_leg: _RouteSplitLegBuilder | None = None

    def start_leg(self, fraction_bps: int) -> None:
        leg = _RouteSplitLegBuilder(fraction_bps=fraction_bps, current_token=self.origin_token)
        self.legs.append(leg)
        self.active_leg = leg


@dataclass
class RouteDAG:
    """Fluent builder for split/merge swap routes represented as a DAG."""

    token_in: Token | None = None
    actions: list[RouteAction] = field(default_factory=list)
    _current_token: Token | None = None
    _split_stack: list[_RouteSplitBuilder] = field(default_factory=list)

    def from_token(self, token: Token) -> "RouteDAG":
        if self.token_in is not None:
            raise ValueError("RouteDAG.from_token() can only be called once")
        self.token_in = token
        self._current_token = token
        return self

    def swap(self, token_out: Token, pool: BasePool) -> "RouteDAG":
        if self.token_in is None:
            raise ValueError("RouteDAG.from_token() must be called before swap()")
        actions = self._current_actions()
        self._reject_action_after_bridge(actions, "swap")
        actions.append(RouteSwap(token_out=token_out, pool=pool))
        self._set_current_token(token_out)
        return self

    def bridge(
        self,
        token_out: Token,
        *,
        denom: Address | None = None,
        transfer_addr: Address,
        source_client: str,
        receiver: str,
        dest_port: str = "transfer",
        memo: str = "",
        timeout_seconds: int | None = None,
        timeout_timestamp: int | None = None,
    ) -> "RouteDAG":
        """Append an Eureka ICS-20 bridge edge to the current branch.

        Must be the last action on its branch — the program can't continue
        operating on tokens that have been sent to an escrow on another chain.
        Subsequent ``.swap()`` / ``.bridge()`` calls will raise.

        ``denom`` defaults to the running token's address (the upstream
        action's output), which is the source-chain ERC-20 to escrow.
        """
        if self.token_in is None:
            raise ValueError("RouteDAG.from_token() must be called before bridge()")
        actions = self._current_actions()
        self._reject_action_after_bridge(actions, "bridge")
        if denom is None:
            current_token = self._branch_current_token()
            if current_token is None:
                raise ValueError("RouteDAG.bridge: cannot infer denom; supply it explicitly")
            denom = current_token.address
        actions.append(
            RouteBridge(
                denom=denom,
                token_out=token_out,
                transfer_addr=transfer_addr,
                source_client=source_client,
                receiver=receiver,
                dest_port=dest_port,
                memo=memo,
                timeout_seconds=timeout_seconds,
                timeout_timestamp=timeout_timestamp,
            )
        )
        self._set_current_token(token_out)
        return self

    def _branch_current_token(self) -> Token | None:
        """Return the token currently held on the active branch."""
        if self._split_stack:
            leg = self._split_stack[-1].active_leg
            return leg.current_token if leg is not None else None
        return self._current_token

    @staticmethod
    def _reject_action_after_bridge(actions: list[RouteAction], new_action: str) -> None:
        if actions and isinstance(actions[-1], RouteBridge):
            raise ValueError(
                f"RouteDAG.{new_action}() cannot follow .bridge() — bridge must be the last action on its branch"
            )

    def split(self) -> "RouteDAG":
        if self.token_in is None:
            raise ValueError("RouteDAG.from_token() must be called before split()")
        self._reject_action_after_bridge(self._current_actions(), "split")
        if not self._split_stack:
            origin_token = self._current_token
        else:
            parent = self._split_stack[-1]
            if parent.active_leg is None or parent.active_leg.current_token is None:
                raise ValueError("leg() must be called before nested split()")
            origin_token = parent.active_leg.current_token

        if origin_token is None:
            raise ValueError("RouteDAG.from_token() must be called before split()")
        self._split_stack.append(_RouteSplitBuilder(origin_token=origin_token))
        return self

    def leg(self, fraction_bps: int) -> "RouteDAG":
        if not self._split_stack:
            raise ValueError("RouteDAG.leg() must be called inside split()")
        if not (0 < fraction_bps <= MAX_BPS):
            raise ValueError(f"leg fraction_bps must be in (0, {MAX_BPS}], got {fraction_bps}")
        self._split_stack[-1].start_leg(fraction_bps)
        return self

    def merge(self) -> "RouteDAG":
        if not self._split_stack:
            raise ValueError("RouteDAG.merge() called without an active split")

        split_ctx = self._split_stack.pop()
        total_bps = sum(leg.fraction_bps for leg in split_ctx.legs)
        if total_bps != MAX_BPS:
            raise ValueError(f"sum of split leg fraction_bps must be {MAX_BPS}, got {total_bps}")

        if any(not leg.actions for leg in split_ctx.legs):
            raise ValueError("each split leg must contain at least one swap() before merge()")

        end_tokens = {leg.current_token for leg in split_ctx.legs}
        if len(end_tokens) != 1:
            raise ValueError("all split legs must end at the same token before merge()")

        merged_token = next(iter(end_tokens))
        split_action = RouteSplit(
            legs=tuple(
                RouteSplitLeg(fraction_bps=leg.fraction_bps, actions=tuple(leg.actions)) for leg in split_ctx.legs
            ),
            token_out=merged_token,
        )

        if self._split_stack:
            parent = self._split_stack[-1]
            if parent.active_leg is None:
                raise ValueError("internal RouteDAG error: missing parent split leg")
            parent.active_leg.actions.append(split_action)
            parent.active_leg.current_token = merged_token
        else:
            self.actions.append(split_action)
            self._current_token = merged_token
        return self

    def to_dict(self) -> dict[str, Any]:
        if self._split_stack:
            raise ValueError("RouteDAG has unmerged split legs")
        if self.token_in is None:
            raise ValueError("RouteDAG.from_token() must be called before serialization")
        return {"token_in": self.token_in, "actions": _freeze_actions(self.actions)}

    def _current_actions(self) -> list[RouteAction]:
        if not self._split_stack:
            return self.actions
        split_ctx = self._split_stack[-1]
        if split_ctx.active_leg is None:
            raise ValueError("leg() must be called before swap() inside split()")
        return split_ctx.active_leg.actions

    def _set_current_token(self, token: Token) -> None:
        if not self._split_stack:
            self._current_token = token
            return
        split_ctx = self._split_stack[-1]
        if split_ctx.active_leg is None:
            raise ValueError("leg() must be called before swap() inside split()")
        split_ctx.active_leg.current_token = token


def _freeze_actions(actions: Sequence[RouteAction]) -> tuple[RouteAction, ...]:
    frozen: list[RouteAction] = []
    for action in actions:
        if isinstance(action, (RouteSwap, RouteBridge)):
            frozen.append(action)
            continue
        frozen.append(
            RouteSplit(
                legs=tuple(
                    RouteSplitLeg(
                        fraction_bps=leg.fraction_bps,
                        actions=_freeze_actions(leg.actions),
                    )
                    for leg in action.legs
                ),
                token_out=action.token_out,
            )
        )
    return tuple(frozen)
