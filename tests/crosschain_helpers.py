"""Shared constants and factories for the ``pydefi.crosschain`` unit tests.

Keeps the offline test modules free of copy-pasted tokens, addresses, dummy
pools, and bridge/market factories.
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

from hexbytes import HexBytes

from pydefi.bridge.ccip import CCIP
from pydefi.bridge.cctp import CCTP
from pydefi.crosschain.compose import SupplyTemplate
from pydefi.pathfinder.dag import RouteDAG
from pydefi.pathfinder.graph import BasePool
from pydefi.types import ZERO_ADDRESS, Address, ChainId, SwapProtocol, Token, TokenAmount
from pydefi.vm.yield_deposit import AMOUNT_SENTINEL
from pydefi.yields.router import YieldMarket

# Placeholder addresses (the bridge contract doubles as CCTP messenger / CCIP router).
BRIDGE = Address("0x" + "11" * 20)
COMPOSER = Address("0x" + "C0" * 20)
SPENDER = Address("0x" + "CE" * 20)
USER = Address("0x" + "AA" * 20)
VM = Address("0x" + "44" * 20)
LINK = Address("0x" + "1a" * 20)

USDC_ETH = Token(chain_id=ChainId.ETHEREUM, address=Address("0x" + "22" * 20), symbol="USDC", decimals=6)
USDC_BASE = Token(chain_id=ChainId.BASE, address=Address("0x" + "33" * 20), symbol="USDC", decimals=6)
WETH_ETH = Token(chain_id=ChainId.ETHEREUM, address=Address("0x" + "55" * 20), symbol="WETH", decimals=18)

#: A supply-calldata template carrying the runtime-amount sentinel exactly once.
SENTINEL_TEMPLATE = b"\xde\xad\xbe\xef" + b"\x00" * 32 + AMOUNT_SENTINEL.to_bytes(32, "big")


class DummyV2Pool(BasePool):
    """Minimal V2 pool for offline RouteDAG lowering (always zero-for-one)."""

    protocol = SwapProtocol.UNISWAP_V2
    pool_address = HexBytes("0x" + "66" * 20)
    fee_bps = 30

    def __init__(self, token_in: Token, token_out: Token) -> None:
        self.token_in = token_in
        self.token_out = token_out

    def zero_for_one(self, _token_out: HexBytes) -> bool:
        return True


def swap_dag(token_in: Token, token_out: Token) -> RouteDAG:
    """A single V2 hop ``token_in -> token_out``."""
    return RouteDAG().from_token(token_in).swap(token_out, DummyV2Pool(token_in, token_out))


def market(protocol: str = "aave_v3", token: Token = USDC_BASE) -> YieldMarket:
    return YieldMarket(
        protocol=protocol,
        chain_id=token.chain_id,
        token=token,
        supply_apy=Decimal("0.05"),
        utilization=Decimal("0.5"),
        available_liquidity=TokenAmount(token=token, amount=10**12),
        market_id=f"{protocol}:{token.chain_id}:{token.symbol}",
    )


def supply_template(template: bytes = SENTINEL_TEMPLATE) -> SupplyTemplate:
    return SupplyTemplate(template=template, spender=SPENDER, token=USDC_BASE.address)


def cctp() -> CCTP:
    """CCTP lane Ethereum -> Base; the mock w3 is never touched offline."""
    return CCTP(
        w3=MagicMock(),
        src_chain_id=ChainId.ETHEREUM,
        dst_chain_id=ChainId.BASE,
        token_messenger_address=BRIDGE,
        src_usdc_address=USDC_ETH.address,
    )


def ccip(fee_token: Address = ZERO_ADDRESS) -> CCIP:
    """CCIP lane Ethereum -> Base; native fee unless *fee_token* is given."""
    return CCIP(
        w3=MagicMock(),
        src_chain_id=ChainId.ETHEREUM,
        dst_chain_id=ChainId.BASE,
        router_address=BRIDGE.to_0x_hex(),
        fee_token=fee_token,
    )
