"""Chainlink CCIP bridge integration.

CCIP transfers tokens and an arbitrary ``data`` payload in one ``ccipSend``
call.  Tokens are 1:1; the per-message fee is paid separately in native gas
or an ERC-20 fee token.  See :class:`CCIPComposer` for the compose path.

Docs: https://docs.chain.link/ccip
"""

from __future__ import annotations

from typing import Any

from eth_abi import encode as abi_encode
from web3 import AsyncWeb3, Web3

from pydefi._utils import address_to_bytes32
from pydefi.abi.bridge import CCIP_ROUTER, CCIPEVM2AnyMessage, CCIPEVMTokenAmount
from pydefi.bridge.base import BaseBridge
from pydefi.exceptions import BridgeError
from pydefi.types import ZERO_ADDRESS, Address, BridgeQuote, Token, TokenAmount

# bytes4(keccak256("CCIP EVMExtraArgsV2")).
EVM_EXTRA_ARGS_V2_TAG: bytes = bytes.fromhex("181dcf10")


# CCIP identifies destination chains by a 64-bit selector, distinct from the
# EVM chain ID.  This map covers every chain pydefi can target as a
# destination; the source-chain Router map below is a subset, since pydefi
# only initiates from chains where a Router is also deployed.
# https://docs.chain.link/ccip/directory/mainnet
_CCIP_CHAIN_SELECTOR: dict[int, int] = {
    1: 5009297550715157269,  # Ethereum
    10: 3734403246176062136,  # Optimism
    56: 11344663589394136015,  # BNB Chain
    137: 4051577828743386545,  # Polygon
    8453: 15971525489660198786,  # Base
    42161: 4949039107694359620,  # Arbitrum
    43114: 6433500567565415381,  # Avalanche
    59144: 4627098889531055414,  # Linea
    81457: 4411394078118774322,  # Blast
    324: 1562403441176082196,  # zkSync Era
    534352: 13264668187771770619,  # Scroll
    130: 1923510734129558909,  # Unichain
    480: 2049429975587534727,  # World Chain
}

# Canonical mainnet Router deployments.
_CCIP_ROUTER: dict[int, str] = {
    1: "0x80226fc0Ee2b096224EeAc085Bb9a8cba1146f7D",  # Ethereum
    10: "0x3206695CaE29952f4b0c22a169725a865bc8Ce0f",  # Optimism
    56: "0x34B03Cb9086d7D758AC55af71584F81A598759FE",  # BNB Chain
    137: "0x849c5ED5a80F5B408Dd4969b78c2C8fdf0565Bfe",  # Polygon
    8453: "0x881e3A65B4d4a04dD529061dd0071cf975F58bCD",  # Base
    42161: "0x141fa059441E0ca23ce184B6A78bafD2A517DdE8",  # Arbitrum
    43114: "0xF4c7E640EdA248ef95972845a62bdC74237805dB",  # Avalanche
    59144: "0xb5e8Cc83af0E1ce05A9c5b2655d50e9CF85753f4",  # Linea
    324: "0xF12CA89aaaeC02E0F2f0F86620586D08D7C9bf85",  # zkSync Era
    534352: "0x6b9a4fb612b9d3D5fE69bbF93bD8a13767E96b62",  # Scroll
}


def _ccip_chain_selector(evm_chain_id: int) -> int:
    selector = _CCIP_CHAIN_SELECTOR.get(evm_chain_id)
    if selector is None:
        raise BridgeError(f"CCIP: unsupported chain {evm_chain_id} (no chain selector known).")
    return selector


# Outbound transaction gas budgets.  CCIP `ccipSend` itself is ~250k baseline;
# we bump both to absorb large message payloads (especially DeFiVM bytecode in
# the compose path) and post-EIP-3860 / EOF overhead.
_DEFAULT_SEND_GAS = 500_000
_DEFAULT_COMPOSE_GAS = 600_000


class CCIP(BaseBridge):
    """Chainlink CCIP bridge — token-only or compose ``ccipSend`` via the Router.

    Tokens are bridged 1:1; the per-message fee is paid in native gas
    (``fee_token=None``) or in an ERC-20 fee token such as LINK.  Lane
    support and token allowlisting are enforced by the Router — unsupported
    tokens / lanes surface as :class:`~pydefi.exceptions.BridgeError` on the
    first :meth:`get_quote` (mirrors :class:`~pydefi.bridge.LayerZeroOFT`).

    Args:
        w3: AsyncWeb3 for the source chain (drives :meth:`Router.getFee`).
        src_chain_id, dst_chain_id: EVM chain IDs.
        router_address: Override the source Router address; defaults to the
            canonical mainnet deployment for ``src_chain_id``.
        fee_token: ERC-20 fee token address, or ``None`` for native gas.  When
            set, the caller approves this token to the Router for the fee.
        fee_token_decimals: Precision of the fee token (default 18 — correct
            for native gas and LINK; pass 6 for USDC-as-fee on lanes that
            accept it).  Flows into :attr:`BridgeQuote.bridge_fee` so
            :class:`~pydefi.bridge.BridgeRouter` can convert the fee.
        fee_token_symbol: Override the synthetic symbol on the fee Token.
    """

    def __init__(
        self,
        w3: AsyncWeb3,
        src_chain_id: int,
        dst_chain_id: int,
        router_address: str | None = None,
        fee_token: Address | None = None,
        fee_token_decimals: int = 18,
        fee_token_symbol: str | None = None,
    ) -> None:
        super().__init__(src_chain_id, dst_chain_id)
        self.w3 = w3

        self.router_address = router_address or _CCIP_ROUTER.get(src_chain_id, "")
        if not self.router_address:
            raise BridgeError(
                f"CCIP: no Router address known for chain {src_chain_id}. Pass router_address explicitly."
            )

        self.fee_token: Address = fee_token if fee_token is not None else ZERO_ADDRESS
        # bool subclasses int — reject explicitly so True/False don't sneak in.
        if not isinstance(fee_token_decimals, int) or isinstance(fee_token_decimals, bool):
            raise BridgeError(f"CCIP: fee_token_decimals must be an int, got {type(fee_token_decimals).__name__}")
        if not 0 <= fee_token_decimals <= 36:
            raise BridgeError(f"CCIP: fee_token_decimals must be in [0, 36], got {fee_token_decimals}")
        self.fee_token_decimals: int = fee_token_decimals
        self.fee_token_symbol: str | None = fee_token_symbol

        # Resolve eagerly so unsupported destinations fail at construction.
        self.dst_chain_selector = _ccip_chain_selector(dst_chain_id)

    @property
    def protocol_name(self) -> str:
        return "CCIP"

    def _build_message(
        self,
        recipient: Address,
        token_in: Token,
        amount_in: TokenAmount,
        data: bytes,
        gas_limit: int,
        allow_out_of_order_execution: bool,
    ) -> CCIPEVM2AnyMessage:
        # ``receiver`` is abi.encode(address) for EVM destinations.
        # ``extraArgs`` v2 layout: 0x181dcf10 ++ abi.encode(uint256, bool).
        extra_args = EVM_EXTRA_ARGS_V2_TAG + abi_encode(
            ["uint256", "bool"], [gas_limit, bool(allow_out_of_order_execution)]
        )
        return CCIPEVM2AnyMessage(
            receiver=bytes(address_to_bytes32(recipient)),
            data=data,
            tokenAmounts=[
                CCIPEVMTokenAmount(
                    token=Web3.to_checksum_address(bytes(token_in.address)),
                    amount=amount_in.amount,
                )
            ],
            feeToken=Web3.to_checksum_address(bytes(self.fee_token)),
            extraArgs=extra_args,
        )

    async def quote_fee(
        self,
        token_in: Token,
        amount_in: TokenAmount,
        recipient: Address,
        data: bytes = b"",
        gas_limit: int = 0,
        allow_out_of_order_execution: bool = False,
    ) -> int:
        """Call ``Router.getFee`` and return the message fee in :attr:`fee_token`.

        Reverts on unsupported lanes or non-allowlisted tokens.
        """
        message = self._build_message(
            recipient=recipient,
            token_in=token_in,
            amount_in=amount_in,
            data=data,
            gas_limit=gas_limit,
            allow_out_of_order_execution=allow_out_of_order_execution,
        )
        try:
            fee = await CCIP_ROUTER.fns.getFee(self.dst_chain_selector, message).call(self.w3, to=self.router_address)
        except Exception as exc:
            raise BridgeError(f"CCIP: getFee failed: {exc}") from exc
        return int(fee)

    # -----------------------------------------------------------------------
    # BaseBridge interface
    # -----------------------------------------------------------------------

    async def get_quote(
        self,
        token_in: Token,
        token_out: Token,
        amount_in: TokenAmount,
        recipient: Address | None = None,
        gas_limit: int = 0,
        allow_out_of_order_execution: bool = False,
        **kwargs: Any,
    ) -> BridgeQuote:
        """Quote a CCIP transfer; bridge_fee is denominated in :attr:`fee_token`.

        ``recipient`` is only used to derive ``getFee`` calldata (the fee is
        recipient-independent); zero address is used when ``None``.
        """
        _recipient = recipient if recipient is not None else ZERO_ADDRESS
        fee = await self.quote_fee(
            token_in=token_in,
            amount_in=amount_in,
            recipient=_recipient,
            gas_limit=gas_limit,
            allow_out_of_order_execution=allow_out_of_order_execution,
        )

        fee_token_symbol = self.fee_token_symbol or ("NATIVE" if self.fee_token == ZERO_ADDRESS else "FEE")
        fee_token = Token(
            chain_id=self.src_chain_id,
            address=self.fee_token,
            symbol=fee_token_symbol,
            decimals=self.fee_token_decimals,
        )

        return BridgeQuote(
            token_in=token_in,
            token_out=token_out,
            amount_in=amount_in,
            amount_out=TokenAmount(token=token_out, amount=amount_in.amount),
            bridge_fee=TokenAmount(token=fee_token, amount=fee),
            # CCIP's public finality target is ~10 minutes on most lanes.
            estimated_time_seconds=600,
            protocol=self.protocol_name,
        )

    def _encode_send_tx(self, message: CCIPEVM2AnyMessage, fee: int, *, gas_budget: int) -> dict[str, Any]:
        """Encode a ``ccipSend`` tx dict for *message* with *fee* already known."""
        call_data = CCIP_ROUTER.fns.ccipSend(self.dst_chain_selector, message).data
        return {
            "to": self.router_address,
            "data": "0x" + call_data.hex(),
            "value": str(fee if self.fee_token == ZERO_ADDRESS else 0),
            "gas": str(gas_budget),
        }

    async def build_bridge_tx(
        self,
        token_in: Token,
        token_out: Token,
        amount_in: TokenAmount,
        recipient: Address,
        slippage_bps: int = 0,
        data: bytes = b"",
        gas_limit: int = 0,
        allow_out_of_order_execution: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build a ``ccipSend`` transaction.

        Native fee → ``value = fee``.  ERC-20 fee → ``value = 0`` and the
        caller must have approved the fee token to the Router.  In both cases
        the caller must approve ``token_in`` for ``amount_in`` first.

        ``slippage_bps`` is ignored (CCIP is 1:1).  Pass DeFiVM bytecode as
        ``data`` and raise ``gas_limit`` (~200 000) for compose flows.
        """
        del token_out, slippage_bps  # unused — CCIP is 1:1 and has no slippage
        message = self._build_message(
            recipient=recipient,
            token_in=token_in,
            amount_in=amount_in,
            data=data,
            gas_limit=gas_limit,
            allow_out_of_order_execution=allow_out_of_order_execution,
        )
        fee = await self.quote_fee(
            token_in=token_in,
            amount_in=amount_in,
            recipient=recipient,
            data=data,
            gas_limit=gas_limit,
            allow_out_of_order_execution=allow_out_of_order_execution,
        )
        return self._encode_send_tx(message, fee, gas_budget=_DEFAULT_SEND_GAS)

    async def build_bridge_compose_tx(
        self,
        token_in: Token,
        amount_in: TokenAmount,
        composer_address: Address,
        program: bytes,
        gas_limit: int = 800_000,
        allow_out_of_order_execution: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build a ``ccipSend`` transaction that triggers a DeFiVM compose.

        Sets the message ``receiver`` to *composer_address* and ``data`` to
        the raw DeFiVM bytecode.  CCIP delivers the bridged tokens to the
        composer and invokes ``ccipReceive``; the composer prepends
        ``(amountReceived, sourceChainSelector)`` and forwards execution to
        DeFiVM.  Same approval rules as :meth:`build_bridge_tx`.

        ``gas_limit`` defaults to 800 000 — enough for a typical
        transfer + DEX swap; raise for fan-out programs.
        """
        if not program:
            raise BridgeError("CCIP: program must not be empty for compose transactions.")
        message = self._build_message(
            recipient=composer_address,
            token_in=token_in,
            amount_in=amount_in,
            data=program,
            gas_limit=gas_limit,
            allow_out_of_order_execution=allow_out_of_order_execution,
        )
        fee = await self.quote_fee(
            token_in=token_in,
            amount_in=amount_in,
            recipient=composer_address,
            data=program,
            gas_limit=gas_limit,
            allow_out_of_order_execution=allow_out_of_order_execution,
        )
        return self._encode_send_tx(message, fee, gas_budget=_DEFAULT_COMPOSE_GAS)
