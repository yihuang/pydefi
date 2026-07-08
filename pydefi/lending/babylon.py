"""Babylon Trustless Bitcoin Vaults (TBV) — Aave V4 integration client.

``AaveAdapter`` is the entry point for borrowing against **native BTC** on Aave
V4: a peg-in mints transfer-restricted ``vaultBTC`` that backs the depositor's
position on the ``BabylonCoreSpoke``. The adapter mirrors the Spoke's lending
entry points (same ``(reserveId, amount, onBehalfOf)`` shape) plus the
vault-aware ops.

This client is the **EVM side** — position reads + lending tx builders. Opening
a position (``activateVault``) needs a signet peg-in proven off-chain (see
:mod:`pydefi.lending.tbv_signet`); position reads revert for an address that has
never opened one.

Docs: https://github.com/babylonlabs-io/babylonlabs.github.io/tree/main/docs/trustless-bitcoin-vault
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from web3 import AsyncWeb3

from pydefi._utils import to_tx
from pydefi.abi.lending import AAVE_ADAPTER
from pydefi.deployments import get_address
from pydefi.lending.aave_v4 import V4UserAccountData, parse_health_factor
from pydefi.types import Address, ChainId, TokenAmount

#: The adapter's four lending entry points that share the ``(reserveId, amount,
#: onBehalfOf)`` calldata shape (mirroring the Spoke). Keyed by op name.
_LENDING_OP_FNS = {
    "supply": AAVE_ADAPTER.fns.supply,
    "withdraw": AAVE_ADAPTER.fns.withdraw,
    "borrow": AAVE_ADAPTER.fns.borrow,
    "repay": AAVE_ADAPTER.fns.repay,
}


@dataclass
class MarketPosition:
    """A depositor's native-BTC lending position behind the adapter.

    Attributes:
        vault_ids: The BTC vaults backing the position.
        total_collateral_btc: Their summed collateral, in sats.
        proxy: The per-user position proxy holding the Spoke position; the zero
            address until the first vault is activated.
    """

    vault_ids: list[bytes]
    total_collateral_btc: int
    proxy: Address


class BabylonAaveAdapter:
    """Babylon TBV ``AaveAdapter`` — borrow against native BTC on Aave V4.

    Args:
        w3: :class:`~web3.AsyncWeb3` for the target chain.
        chain_id: EVM chain ID (Babylon TBV is on Sepolia).
        adapter_address: Address of the ``AaveAdapter`` contract.
    """

    protocol_name: str = "BabylonAaveAdapter"

    def __init__(self, w3: AsyncWeb3, chain_id: int, adapter_address: Address) -> None:
        self.w3 = w3
        self.chain_id = chain_id
        self.adapter_address = adapter_address

    @classmethod
    def from_chain(cls, w3: AsyncWeb3, chain_id: int = ChainId.SEPOLIA) -> BabylonAaveAdapter:
        """Construct from the deployment registry (``BABYLON_AAVE_ADAPTER``)."""
        return cls(w3, chain_id, get_address("BABYLON_AAVE_ADAPTER", chain_id))

    # ------------------------------------------------------------------
    # Reads — wiring
    # ------------------------------------------------------------------

    async def vault_btc(self) -> Address:
        """The ``vaultBTC`` collateral token the adapter mints/manages."""
        return Address(await AAVE_ADAPTER.fns.VAULT_BTC().call(self.w3, to=self.adapter_address))

    async def core_spoke(self) -> Address:
        """The ``BabylonCoreSpoke`` (Aave V4 Spoke) the adapter borrows on."""
        return Address(await AAVE_ADAPTER.fns.BTC_VAULT_CORE_SPOKE().call(self.w3, to=self.adapter_address))

    async def registry(self) -> Address:
        """The ``BTCVaultRegistry`` tracking BTC vault lifecycle."""
        return Address(await AAVE_ADAPTER.fns.BTC_VAULT_REGISTRY().call(self.w3, to=self.adapter_address))

    # ------------------------------------------------------------------
    # Reads — position (revert for an address with no position)
    # ------------------------------------------------------------------

    async def get_position(self, user: Address) -> MarketPosition:
        """Return *user*'s native-BTC position (vaults, collateral, proxy)."""
        raw = await AAVE_ADAPTER.fns.getPosition(user).call(self.w3, to=self.adapter_address)
        return MarketPosition(
            vault_ids=list(raw.vaultIds),
            total_collateral_btc=raw.totalCollateralBTC,
            proxy=Address(raw.proxyContract),
        )

    async def get_user_account_data(self, user: Address) -> V4UserAccountData:
        """Return *user*'s aggregate Aave position (health factor, values)."""
        raw = await AAVE_ADAPTER.fns.getUserAccountData(user).call(self.w3, to=self.adapter_address)
        return V4UserAccountData(
            risk_premium=raw.riskPremium,
            avg_collateral_factor=raw.avgCollateralFactor,
            health_factor=parse_health_factor(raw.healthFactor),
            total_collateral_value=raw.totalCollateralValue,
            total_debt_value_ray=raw.totalDebtValueRay,
            active_collateral_count=raw.activeCollateralCount,
            borrow_count=raw.borrowCount,
        )

    async def get_reserve_total_debt(self, reserve_id: int) -> int:
        """Total outstanding debt for reserve *reserve_id*."""
        return await AAVE_ADAPTER.fns.getReserveTotalDebt(reserve_id).call(self.w3, to=self.adapter_address)

    async def get_user_total_debt(self, reserve_id: int, user: Address) -> int:
        """*user*'s debt (drawn + premium) in reserve *reserve_id*."""
        return await AAVE_ADAPTER.fns.getUserTotalDebt(reserve_id, user).call(self.w3, to=self.adapter_address)

    # ------------------------------------------------------------------
    # Writes — return tx dicts {to, data, value}
    # ------------------------------------------------------------------

    def build_op_tx(self, op: str, reserve_id: int, amount: TokenAmount, on_behalf_of: Address) -> dict[str, Any]:
        """Build a ``(reserveId, amount, onBehalfOf)`` lending tx for *on_behalf_of*.

        *op* is ``supply`` / ``withdraw`` / ``borrow`` / ``repay`` — they share
        the adapter's calldata shape (the Spoke's entry points).
        """
        try:
            fn = _LENDING_OP_FNS[op]
        except KeyError:
            raise ValueError(f"unknown lending op {op!r}; expected one of {sorted(_LENDING_OP_FNS)}") from None
        return to_tx(self.adapter_address, fn(reserve_id, amount.amount, on_behalf_of).data)

    def build_set_collateral_tx(
        self, reserve_id: int, use_as_collateral: bool, on_behalf_of: Address
    ) -> dict[str, Any]:
        """Enable/disable reserve *reserve_id* as collateral for *on_behalf_of*."""
        call = AAVE_ADAPTER.fns.setUsingAsCollateral(reserve_id, use_as_collateral, on_behalf_of).data
        return to_tx(self.adapter_address, call)

    def build_withdraw_collaterals_tx(self, vault_ids: list[bytes]) -> dict[str, Any]:
        """Redeem the BTC collateral *vault_ids* (closing them out)."""
        return to_tx(self.adapter_address, AAVE_ADAPTER.fns.withdrawCollaterals(vault_ids).data)

    def flow(self, on_behalf_of: Address) -> BabylonFlow:
        """Start a :class:`BabylonFlow` that composes operations for *on_behalf_of*."""
        return BabylonFlow(self, on_behalf_of)


# ---------------------------------------------------------------------------
# BabylonFlow — fluent composition of native-BTC lending operations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FlowStep:
    """One step in a :class:`BabylonFlow` — a labelled ``{to, data, value}`` tx."""

    label: str
    tx: dict[str, Any]


class BabylonFlow:
    """Compose ordered native-BTC lending txs for one *on_behalf_of* depositor,
    fluently — e.g. ``flow.lock_btc_to_tbv(activation).op("borrow", USDC, amt)``
    — and yield them via :meth:`build`.

    Steps are built from the adapter; the ERC-20 approvals they need (before
    supply/repay) are the caller's responsibility. ``lock_btc_to_tbv`` only
    *sequences* an activation tx built off-chain by the signet peg-in flow — its
    ``BTCVault`` payload can't be produced on the EVM side.
    """

    def __init__(self, adapter: BabylonAaveAdapter, on_behalf_of: Address) -> None:
        self.adapter = adapter
        self.on_behalf_of = on_behalf_of
        self._steps: list[FlowStep] = []

    # ---- core EVM operations ----------------------------------------------

    def op(self, op: str, reserve_id: int, amount: TokenAmount) -> BabylonFlow:
        """Add a lending op (``supply``/``withdraw``/``borrow``/``repay``) on reserve *reserve_id*."""
        return self._add(f"{op}:{reserve_id}", self.adapter.build_op_tx(op, reserve_id, amount, self.on_behalf_of))

    def set_collateral(self, reserve_id: int, use_as_collateral: bool) -> BabylonFlow:
        """Enable/disable reserve *reserve_id* as collateral."""
        tx = self.adapter.build_set_collateral_tx(reserve_id, use_as_collateral, self.on_behalf_of)
        return self._add(f"setCollateral:{reserve_id}={use_as_collateral}", tx)

    def unlock_btc(self, vault_ids: list[bytes]) -> BabylonFlow:
        """Redeem the BTC collateral *vault_ids*."""
        return self._add("withdrawCollaterals", self.adapter.build_withdraw_collaterals_tx(vault_ids))

    def lock_btc_to_tbv(self, activation_tx: dict[str, Any]) -> BabylonFlow:
        """Sequence an *activation_tx* (``activateVault``) built off-chain by the
        Bitcoin/signet peg-in flow, turning a pegged BTC vault into collateral."""
        return self._add("activateVault", activation_tx)

    # ---- output ------------------------------------------------------------

    def steps(self) -> list[FlowStep]:
        """The composed steps, in order."""
        return list(self._steps)

    def build(self) -> list[dict[str, Any]]:
        """The ordered ``{to, data, value}`` transactions to submit."""
        return [step.tx for step in self._steps]

    def _add(self, label: str, tx: dict[str, Any]) -> BabylonFlow:
        self._steps.append(FlowStep(label=label, tx=tx))
        return self


__all__ = ["BabylonAaveAdapter", "BabylonFlow", "FlowStep", "MarketPosition"]
