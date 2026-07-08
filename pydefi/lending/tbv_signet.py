"""Babylon TBV — Bitcoin (signet) peg-in side of native-BTC lending.

Borrowing against native BTC means first turning signet BTC into ``vaultBTC``:
pick a vault provider and fetch its params, build + broadcast a signet peg-in,
run the depositor presign flow, then ``activateVault`` to mint ``vaultBTC`` and
borrow.

The peg-in build (a Taproot WOTS/HTLC/hashlock vault) and the depositor presign
flow (CWT-auth-gated VP RPCs, see :data:`AUTH_GATED_METHODS`) are **not**
reimplemented here — drive them from Node with ``@babylonlabs-io/babylon-ts-sdk``
(``PeginManager`` / ``runDepositorPresignFlow``).

What lives here: the unauthenticated VP reads (:func:`vault_providers`,
:func:`get_pegin_status`, :func:`vp_rpc`), signet UTXO/broadcast
(:func:`fetch_signet_utxos`, :func:`broadcast_pegin`), and the EVM activation
(:func:`build_activate_vault_tx`) — whose output feeds
:meth:`pydefi.lending.BabylonFlow.lock_btc_to_tbv`.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import astuple, dataclass
from enum import IntEnum
from typing import Any

from pydefi._utils import to_tx
from pydefi.abi.lending import AAVE_ADAPTER
from pydefi.types import Address


class BTCVaultStatus(IntEnum):
    """``IBTCVaultRegistry.BTCVaultStatus`` (on-chain enum order)."""

    NONE = 0
    PENDING = 1
    VERIFIED = 2
    ACTIVE = 3
    REDEEMING = 4
    REDEEMED = 5
    SLASHED = 6


@dataclass(frozen=True)
class BTCVault:
    """EVM-side carrier for a pegged-in BTC vault (``IBTCVaultRegistry.BTCVault``).

    Field order matches the ABI tuple exactly — :meth:`to_abi_tuple` relies on
    it. Every value originates from the off-chain signet peg-in / VP flow.
    """

    depositor: Address
    depositor_btc_pubkey: bytes  # bytes32
    depositor_signed_pegin_tx: bytes
    amount: int
    vault_provider: Address
    status: int
    application_entry_point: Address
    universal_challengers_version: int
    app_vault_keepers_version: int
    offchain_params_version: int
    created_at: int
    verified_at: int
    depositor_wots_pk_hash: bytes  # bytes32
    hashlock: bytes  # bytes32
    htlc_vout: int
    depositor_pop_signature: bytes
    pre_pegin_tx_hash: bytes  # bytes32

    def to_abi_tuple(self) -> tuple[Any, ...]:
        """The ordered tuple for ABI-encoding ``BTCVaultStruct``."""
        return astuple(self)


def build_activate_vault_tx(
    adapter_address: Address,
    vault_id: bytes,
    vault: BTCVault,
    activation_metadata: bytes = b"",
) -> dict[str, Any]:
    """Build the ``activateVault`` tx that mints ``vaultBTC`` from a pegged-in
    BTC *vault* and supplies it as collateral.

    *vault_id* is the 32-byte registry id; *vault* and *activation_metadata*
    come from the off-chain peg-in flow. The result plugs straight into
    :meth:`pydefi.lending.BabylonFlow.lock_btc_to_tbv`.
    """
    call = AAVE_ADAPTER.fns.activateVault(vault_id, vault.to_abi_tuple(), activation_metadata).data
    return to_tx(adapter_address, call)


# ---------------------------------------------------------------------------
# Bitcoin / signet side — stubs (need a signet BTC library + the VP proxy API)
# ---------------------------------------------------------------------------

_VP_PROXY_API = "https://vault-provider-proxy-api.testnet.babylonlabs.io"
#: Public signet mempool (esplora API) — not jurisdiction-gated, unlike
#: Babylon's mempool.signet.babylonlabs.io.
_SIGNET_MEMPOOL = "https://mempool.space/signet"
#: A non-default User-Agent — Cloudflare 403s the bare ``Python-urllib`` UA on
#: these hosts (curl/browser UAs pass).
_UA = "pydefi-tbv/0.1"

#: VP RPC methods that require a CWT ``Authorization: Bearer`` (minted by the
#: VP's ``auth_createDepositorToken``, keyed by peginTxid + auth-anchor + the
#: on-chain server pubkey). The deposit flow that drives them lives in
#: ``@babylonlabs-io/babylon-ts-sdk`` (``runDepositorPresignFlow`` /
#: ``submitWotsPublicKey``) — that is where the auth handshake + presign signing
#: belong; this module only does the unauthenticated reads + transport.
VP_TOKEN_ISSUE_METHOD = "auth_createDepositorToken"
AUTH_GATED_METHODS: frozenset[str] = frozenset(
    {
        "vaultProvider_submitDepositorWotsKey",
        "vaultProvider_submitDepositorPresignatures",
        "vaultProvider_requestDepositorPresignTransactions",
    }
)
#: Gated by a *gRPC-subject* CWT (``auth_createDepositorTokenGrpc``).
GRPC_AUTH_GATED_METHODS: frozenset[str] = frozenset({"vaultProvider_requestDepositorClaimerArtifacts"})
#: Babylon's TS SDK — drives the peg-in build (wraps the btc_vault Rust→WASM)
#: plus the VP auth + depositor presign flow. Reuse it from Node; don't
#: reimplement. https://github.com/babylonlabs-io/babylon-toolkit
_BTC_VAULT_SDK = "@babylonlabs-io/babylon-ts-sdk"


def _vp_addr(vault_provider: Address | str) -> str:
    """Normalise a VP address to the ``0x``-hex the proxy URL requires."""
    addr = vault_provider if isinstance(vault_provider, str) else "0x" + bytes(vault_provider).hex()
    if not re.fullmatch(r"0x[0-9a-fA-F]{40}", addr):
        raise ValueError(f"vault_provider must be a 20-byte EVM address, got {addr!r}")
    return addr


def vp_health(*, vp_proxy_url: str = _VP_PROXY_API, timeout: float = 15.0) -> Any:
    """GET ``{vp_proxy_url}/vp-health`` — vault-provider directory + stats.

    Returns a list of ``{address, disabled, successRate, totalRequests, ...}``,
    so it doubles as VP discovery (see :func:`vault_providers`). It is a
    *sliding-window activity* view, not the on-chain registry: an idle VP drops
    out, so ``[]`` means "nothing served recently", not "no VPs registered".
    ``disabled`` is a separate *administrative* block (not a stat) — Babylon's
    own flow hides disabled VPs from new-deposit selection, unlike runtime-
    unhealthy ones (kept but deprioritised); filter them with ``enabled_only``.
    """
    url = f"{vp_proxy_url.rstrip('/')}/vp-health"
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 — fixed https host
        return json.loads(resp.read().decode())


def vault_providers(
    *, vp_proxy_url: str = _VP_PROXY_API, enabled_only: bool = False, timeout: float = 15.0
) -> list[dict[str, Any]]:
    """List vault providers (address + health stats) from :func:`vp_health` —
    the addresses to use in :func:`vp_rpc` / :func:`get_pegin_status`.

    With *enabled_only*, drop ``disabled`` (administratively-blocked) VPs — the
    filter to use when picking a VP for a new peg-in.
    """
    data = vp_health(vp_proxy_url=vp_proxy_url, timeout=timeout)
    providers = data if isinstance(data, list) else data.get("providers", [])
    if enabled_only:
        providers = [p for p in providers if not p.get("disabled")]
    return providers


def get_pegin_status(
    vault_provider: Address | str,
    *,
    pegin_txid: str | None = None,
    vault_id: str | None = None,
    vp_proxy_url: str = _VP_PROXY_API,
    auth_token: str | None = None,
) -> Any:
    """``vaultProvider_getPeginStatus`` — status of a peg-in at *vault_provider*.

    Provide ``pegin_txid`` or ``vault_id`` (at least one). Wraps :func:`vp_rpc`
    with the ``GetPeginStatusParams`` struct the VP expects.
    """
    if not pegin_txid and not vault_id:
        raise ValueError("get_pegin_status: provide pegin_txid or vault_id")
    param: dict[str, str] = {}
    if pegin_txid:
        param["pegin_txid"] = pegin_txid
    if vault_id:
        param["vault_id"] = vault_id
    return vp_rpc(
        vault_provider, "vaultProvider_getPeginStatus", [param], vp_proxy_url=vp_proxy_url, auth_token=auth_token
    )


def vp_rpc(
    vault_provider: Address | str,
    method: str,
    params: Any = None,
    *,
    vp_proxy_url: str = _VP_PROXY_API,
    auth_token: str | None = None,
    request_id: int = 1,
    timeout: float = 30.0,
) -> Any:
    """Call a JSON-RPC 2.0 *method* on a vault provider via the proxy
    (``POST {vp_proxy_url}/rpc/{vault_provider}``) and return its ``result``.

    *auth_token* is sent as ``Authorization: Bearer`` when given (the proxy
    requires it for authenticated methods; obtain it from the VP auth handshake).
    Raises :class:`RuntimeError` on an HTTP or JSON-RPC error.

    Note: the proxy is geo-restricted (HTTP 451 from some regions) — run from a
    permitted network.
    """
    if auth_token is None and (method in AUTH_GATED_METHODS or method in GRPC_AUTH_GATED_METHODS):
        raise RuntimeError(
            f"{method!r} is auth-gated — it needs a CWT bearer from {VP_TOKEN_ISSUE_METHOD!r} "
            "(keyed by peginTxid + auth-anchor + the on-chain server pubkey). Drive the deposit "
            "flow via @babylonlabs-io/babylon-ts-sdk (runDepositorPresignFlow / submitWotsPublicKey) "
            "and pass its token as auth_token, rather than calling this raw."
        )
    url = f"{vp_proxy_url.rstrip('/')}/rpc/{_vp_addr(vault_provider)}"
    headers = {"Content-Type": "application/json", "User-Agent": _UA}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
    payload = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params if params is not None else []}
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 — fixed https host
            body = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read()[:200].decode(errors="replace")
        raise RuntimeError(f"VP proxy HTTP {exc.code} for {method!r}: {detail!r} ({url})") from exc
    if isinstance(body, dict) and body.get("error"):
        raise RuntimeError(f"VP proxy JSON-RPC error for {method!r}: {body['error']}")
    return body.get("result") if isinstance(body, dict) else body


def build_signed_pegin_tx(params: dict[str, Any], *, amount_sats: int, signer: Any) -> bytes:
    """Build + sign the signet peg-in tx, returning the bytes for
    :attr:`BTCVault.depositor_signed_pegin_tx`.

    Not a single step (and not a plain send): the vault is a Taproot WOTS/HTLC/
    hashlock tree, and the depositor presign flow is VP-auth-gated. Drive it from
    Node with ``@babylonlabs-io/babylon-ts-sdk`` — ``PeginManager`` builds the
    pegin (via the ``btc_vault`` WASM) and ``runDepositorPresignFlow`` /
    ``submitWotsPublicKey`` handle the VP auth + presigning — then :func:`broadcast_pegin`
    the result and continue with :func:`build_activate_vault_tx`.
    """
    raise NotImplementedError(
        f"drive the pegin + depositor presign flow via {_BTC_VAULT_SDK} (PeginManager / "
        "runDepositorPresignFlow); do not hand-roll the vault Taproot/WOTS script"
    )


def fetch_signet_utxos(
    address: str, *, mempool_url: str = _SIGNET_MEMPOOL, timeout: float = 20.0
) -> list[dict[str, Any]]:
    """Return *address*'s spendable signet UTXOs (esplora ``/api/address/{a}/utxo``)
    — the inputs you fund the pre-pegin with. Each item has ``txid``/``vout``/
    ``value`` (sats)/``status``.
    """
    url = f"{mempool_url.rstrip('/')}/api/address/{address}/utxo"
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 — fixed https host
        return json.loads(resp.read().decode())


def broadcast_pegin(signed_tx: bytes | str, *, mempool_url: str = _SIGNET_MEMPOOL, timeout: float = 30.0) -> str:
    """Broadcast a signed raw tx to a signet mempool (esplora ``POST /api/tx``)
    and return its txid.

    *signed_tx* is the raw transaction as bytes or hex. Used for both the funded
    pre-pegin and the pegin. The default mempool (``mempool.space/signet``) is
    public; broadcast first, wait for confirmations, then activate the vault.
    """
    raw_hex = signed_tx.hex() if isinstance(signed_tx, (bytes, bytearray)) else str(signed_tx).removeprefix("0x")
    url = f"{mempool_url.rstrip('/')}/api/tx"
    req = urllib.request.Request(
        url, data=raw_hex.encode(), method="POST", headers={"Content-Type": "text/plain", "User-Agent": _UA}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 — fixed https host
            return resp.read().decode().strip()
    except urllib.error.HTTPError as exc:
        detail = exc.read()[:300].decode(errors="replace")
        raise RuntimeError(f"signet broadcast failed (HTTP {exc.code}): {detail!r} ({url})") from exc


__all__ = [
    "BTCVault",
    "BTCVaultStatus",
    "build_activate_vault_tx",
    "vp_health",
    "vault_providers",
    "vp_rpc",
    "get_pegin_status",
    "AUTH_GATED_METHODS",
    "GRPC_AUTH_GATED_METHODS",
    "VP_TOKEN_ISSUE_METHOD",
    "build_signed_pegin_tx",
    "fetch_signet_utxos",
    "broadcast_pegin",
]
