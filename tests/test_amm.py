"""Tests for pydefi.amm (no live node required)."""

from dataclasses import replace
from decimal import Decimal

import pytest
from eth_abi import decode as abi_decode
from web3.exceptions import ProviderConnectionError

from pydefi._math import apply_slippage
from pydefi.amm import v4_hooks
from pydefi.amm.uniswap_v2 import UniswapV2
from pydefi.amm.uniswap_v3 import UniswapV3
from pydefi.amm.uniswap_v4 import EXECUTION_GAS, UniswapV4
from pydefi.amm.universal_router import (
    ADDRESS_THIS,
    CONTRACT_BALANCE,
    MSG_SENDER,
    OPEN_DELTA,
    UNIVERSAL_ROUTER_ADDRESSES,
    RouterCommand,
    UniversalRouter,
    V2Hop,
    V3Hop,
    V4Action,
    V4Hop,
)
from pydefi.exceptions import HookedPoolError, InsufficientLiquidityError, PoolFeeTooHighError
from pydefi.pathfinder.graph import V4PoolEdge
from pydefi.pathfinder.v3_tick_math import TickLadder
from pydefi.pool_data.base import PoolData
from pydefi.types import Address, TokenAmount
from pydefi.vm.swap import SwapTransaction
from tests.addrs import (
    DAI,
    ETH_WHALE,
    UNISWAP_V2_ROUTER,
    UNISWAP_V3_QUOTER,
    UNISWAP_V3_ROUTER,
    UNIVERSAL_ROUTER,
    USDC,
    WETH,
    ZERO_ADDR,
)

# ---------------------------------------------------------------------------
# Shared constants and helpers
# ---------------------------------------------------------------------------

ROUTER_V2: Address = UNISWAP_V2_ROUTER
ROUTER_V3: Address = UNISWAP_V3_ROUTER
QUOTER_V3: Address = UNISWAP_V3_QUOTER
UNIVERSAL_ROUTER_ADDR = UNIVERSAL_ROUTER
RECIPIENT = ETH_WHALE

# keccak("execute(bytes,bytes[],uint256)")[:4] / keccak("execute(bytes,bytes[])")[:4]
DEADLINE_SELECTOR = bytes.fromhex("3593564c")
NO_DEADLINE_SELECTOR = bytes.fromhex("24856bc3")

WETH_1 = TokenAmount.from_human(WETH, "1")
USDC_2K = TokenAmount(token=USDC, amount=2_000 * 10**6)
DAI_2K = TokenAmount(token=DAI, amount=2_000 * 10**18)
ETH_AMOUNT = 10**17

V2_WETH_USDC = V2Hop(token_in=WETH, token_out=USDC)
V2_USDC_DAI = V2Hop(token_in=USDC, token_out=DAI)
V3_WETH_USDC = V3Hop(token_in=WETH, token_out=USDC, fee=500)
V3_USDC_DAI = V3Hop(token_in=USDC, token_out=DAI, fee=100)
V4_WETH_USDC = V4Hop(token_in=WETH, token_out=USDC, fee=500, tick_spacing=10)
V4_USDC_DAI = V4Hop(token_in=USDC, token_out=DAI, fee=100, tick_spacing=1)


def decode_execute(tx: SwapTransaction) -> tuple[bytes, list[bytes]]:
    """Decode UniversalRouter ``execute`` calldata into ``(commands, inputs)``."""
    selector, payload = tx.data[:4], tx.data[4:]
    if selector == DEADLINE_SELECTOR:
        commands, inputs, _deadline = abi_decode(["bytes", "bytes[]", "uint256"], payload)
    else:
        assert selector == NO_DEADLINE_SELECTOR
        commands, inputs = abi_decode(["bytes", "bytes[]"], payload)
    return commands, list(inputs)


@pytest.fixture
def router() -> UniversalRouter:
    return UniversalRouter(UNIVERSAL_ROUTER_ADDR)


# ---------------------------------------------------------------------------
# UniswapV2 pure math (no network calls)
# ---------------------------------------------------------------------------


class TestUniswapV2Math:
    def test_get_amount_out_basic(self):
        # 1 ETH in, 1000 ETH reserve, 2_000_000 USDC reserve → ~1997 USDC
        out = UniswapV2.get_amount_out(
            amount_in=10**18,
            reserve_in=1_000 * 10**18,
            reserve_out=2_000_000 * 10**6,
        )
        # rough check: 1 ETH at $2000 minus 0.3% fee
        assert 1_990 * 10**6 < out < 2_000 * 10**6

    def test_get_amount_out_zero_reserve_raises(self):
        with pytest.raises(InsufficientLiquidityError):
            UniswapV2.get_amount_out(10**18, 0, 10**18)

    def test_get_amount_in_basic(self):
        # To buy 1000 USDC from pool with 1000 ETH / 2_000_000 USDC
        amount_in = UniswapV2.get_amount_in(
            amount_out=1_000 * 10**6,
            reserve_in=1_000 * 10**18,
            reserve_out=2_000_000 * 10**6,
        )
        # Should require slightly more than 0.5 ETH (fee overhead)
        assert 0.5 * 10**18 < amount_in < 0.505 * 10**18

    def test_get_amount_in_insufficient_reserve_raises(self):
        with pytest.raises(InsufficientLiquidityError):
            UniswapV2.get_amount_in(
                amount_out=2_000_000 * 10**6,  # more than available
                reserve_in=1_000 * 10**18,
                reserve_out=1_999_999 * 10**6,
            )

    def test_get_amount_out_custom_fee(self):
        # Lower fee should give higher output
        out_standard = UniswapV2.get_amount_out(10**18, 10**21, 10**21, fee_bps=30)
        out_low_fee = UniswapV2.get_amount_out(10**18, 10**21, 10**21, fee_bps=5)
        assert out_low_fee > out_standard

    def test_spot_price(self):
        # 1000 WETH, 2_000_000 USDC → spot = 2000 USDC/WETH
        price = UniswapV2.spot_price(
            reserve_in=1_000 * 10**18,
            reserve_out=2_000_000 * 10**6,
            decimals_in=18,
            decimals_out=6,
        )
        assert price == Decimal("2000")

    def test_spot_price_zero_reserve(self):
        assert UniswapV2.spot_price(0, 10**18) == Decimal(0)

    def test_roundtrip_amount(self):
        """get_amount_in(get_amount_out(x)) ≈ x — within 1 wei due to integer division."""
        reserve_in = 1_000 * 10**18
        reserve_out = 2_000_000 * 10**6
        amount_in = 10**18

        amount_out = UniswapV2.get_amount_out(amount_in, reserve_in, reserve_out)
        amount_in_back = UniswapV2.get_amount_in(amount_out, reserve_in, reserve_out)
        # Integer division may produce a value slightly below amount_in (floor rounding).
        # The difference must be within a negligible fraction of the input (< 0.01%).
        assert abs(amount_in_back - amount_in) < amount_in // 10_000

    def test_apply_slippage(self):

        result = apply_slippage(1_000_000, 50)
        assert result == 995_000

    def test_apply_slippage_zero(self):
        result = apply_slippage(1_000_000, 0)
        assert result == 1_000_000


# ---------------------------------------------------------------------------
# UniswapV3 math (no network calls)
# ---------------------------------------------------------------------------


class TestUniswapV3Math:
    def test_sqrt_price_to_price_equal_decimals(self):
        # At 1:1 ratio with equal decimals: sqrtPrice = 2^96
        price = UniswapV3.sqrt_price_to_price(2**96, 18, 18)
        assert abs(price - Decimal(1)) < Decimal("0.001")

    def test_sqrt_price_to_price_usdc_eth(self):
        # sqrtPriceX96 for ETH/USDC at ~$2000; price is token0 (USDC) per
        # token1 (ETH), approximately 1/2000
        sqrt_price_x96 = 1_771_595_571_142_957_116_569_145_374
        assert UniswapV3.sqrt_price_to_price(sqrt_price_x96, 6, 18) > Decimal(0)

    def test_encode_path_two_tokens(self):
        path = UniswapV3._encode_path([WETH, USDC], [3000])
        # 20 bytes (address) + 3 bytes (fee) + 20 bytes (address) = 43 bytes
        assert len(path) == 43

    def test_encode_path_three_tokens(self):
        path = UniswapV3._encode_path([WETH, DAI, USDC], [3000, 100])
        # 20 + 3 + 20 + 3 + 20 = 66 bytes
        assert len(path) == 66

    def test_encode_path_fee_mismatch_raises(self):
        with pytest.raises(ValueError):
            UniswapV3._encode_path([WETH, USDC], [3000, 500])  # 2 tokens, 2 fees — invalid

    def test_encode_path_contains_token_addresses(self):
        path = UniswapV3._encode_path([WETH, USDC], [3000])
        assert bytes(WETH.address) in path
        assert bytes(USDC.address) in path


# ---------------------------------------------------------------------------
# AMM client instances (no live calls)
# ---------------------------------------------------------------------------


class TestUniswapV2Instance:
    def test_protocol_name_default(self):
        assert UniswapV2(w3=None, router_address=ROUTER_V2).protocol_name == "UniswapV2"

    def test_protocol_name_custom(self):
        sushi = UniswapV2(w3=None, router_address=ROUTER_V2, protocol_name="SushiSwap")
        assert sushi.protocol_name == "SushiSwap"

    def test_router_address_stored(self):
        assert UniswapV2(w3=None, router_address=ROUTER_V2).router_address == ROUTER_V2

    def test_get_pair_contract(self):
        uniswap = UniswapV2(w3=None, router_address=ROUTER_V2)
        assert uniswap.get_pair_contract(Address("0x" + "AB" * 20)) is not None


class TestUniswapV3Instance:
    def test_protocol_name(self):
        v3 = UniswapV3(w3=None, router_address=ROUTER_V3, quoter_address=QUOTER_V3)
        assert v3.protocol_name == "UniswapV3"

    def test_default_fee(self):
        v3 = UniswapV3(w3=None, router_address=ROUTER_V3, quoter_address=QUOTER_V3)
        assert v3.default_fee == 3000

    def test_custom_default_fee(self):
        v3 = UniswapV3(w3=None, router_address=ROUTER_V3, quoter_address=QUOTER_V3, default_fee=500)
        assert v3.default_fee == 500


class TestAttachTickLadders:
    """Enriching candidate pools with exact tick data before pricing them."""

    @staticmethod
    def _pool(addr_byte: str, **kwargs):
        defaults = dict(
            pool_address="0x" + addr_byte * 20,
            protocol="UniswapV3",
            chain_id=1,
            token0=WETH,
            token1=USDC,
            fee_bps=5,
            sqrt_price_x96=2**96,
            liquidity=10**22,
        )
        return PoolData(**{**defaults, **kwargs})

    @staticmethod
    def _v3(monkeypatch, fetch):
        v3 = UniswapV3(w3=None, router_address=ROUTER_V3, quoter_address=QUOTER_V3)
        monkeypatch.setattr(v3, "fetch_tick_ladder", fetch)
        return v3

    async def test_fills_v3_pools_and_skips_pools_without_price_state(self, monkeypatch):
        seen: list[str] = []

        async def fetch(address, **kwargs):
            seen.append(bytes(address).hex())
            return TickLadder([(0, 2**96, 10**18)])

        v3 = self._v3(monkeypatch, fetch)
        v3_pool = self._pool("aa")
        v2_pool = self._pool("bb", protocol="UniswapV2", sqrt_price_x96=0, liquidity=0)

        returned = await v3.attach_tick_ladders([v3_pool, v2_pool])

        assert seen == ["aa" * 20], "only the pool with V3 price state is worth a round trip"
        assert len(v3_pool.tick_ladder) == 1
        assert v2_pool.tick_ladder is None
        assert returned == [v3_pool, v2_pool], "every pool comes back, enriched or not"

    async def test_a_failed_fetch_leaves_that_pool_on_the_estimate(self, monkeypatch):
        """One unreachable pool must not cost the others their ladders."""

        async def fetch(address, **kwargs):
            if bytes(address)[0] == 0xAA:
                raise ValueError("no ticks for you")
            return TickLadder([(0, 2**96, 10**18)])

        v3 = self._v3(monkeypatch, fetch)
        failing, ok = self._pool("aa"), self._pool("cc")

        await v3.attach_tick_ladders([failing, ok])

        assert failing.tick_ladder is None, "falls back to the single-tick estimate"
        assert ok.tick_ladder is not None

    async def test_ladder_reaches_the_edge_that_prices_the_swap(self, monkeypatch):
        """The point of the whole chain: fetch -> PoolData -> V3PoolEdge.amount_out."""

        async def fetch(address, **kwargs):
            return TickLadder([(-60, 2**95, -(10**18)), (60, 2**97, 10**18)])

        v3 = self._v3(monkeypatch, fetch)
        pool = self._pool("aa")
        await v3.attach_tick_ladders([pool])

        edges = pool.to_pool_edges()
        assert all(e.tick_ladder is pool.tick_ladder for e in edges)
        assert edges[0].amount_out(10**18) > 0


# ---------------------------------------------------------------------------
# Universal Router constants and enums
# ---------------------------------------------------------------------------


def test_router_command_values():
    assert RouterCommand.V3_SWAP_EXACT_IN == 0x00
    assert RouterCommand.V3_SWAP_EXACT_OUT == 0x01
    assert RouterCommand.V2_SWAP_EXACT_IN == 0x08
    assert RouterCommand.V2_SWAP_EXACT_OUT == 0x09
    assert RouterCommand.WRAP_ETH == 0x0B
    assert RouterCommand.UNWRAP_WETH == 0x0C
    assert RouterCommand.V4_SWAP == 0x10
    assert RouterCommand.V4_INITIALIZE_POOL == 0x13
    assert RouterCommand.EXECUTE_SUB_PLAN == 0x21
    assert RouterCommand.ACROSS_V4_DEPOSIT_V3 == 0x40
    assert RouterCommand.ALLOW_REVERT_FLAG == 0x80
    assert RouterCommand.V3_SWAP_EXACT_OUT | RouterCommand.ALLOW_REVERT_FLAG == 0x81


def test_v4_action_values():
    assert V4Action.SWAP_EXACT_IN_SINGLE == 0x06
    assert V4Action.SWAP_EXACT_IN == 0x07
    assert V4Action.SWAP_EXACT_OUT_SINGLE == 0x08
    assert V4Action.SWAP_EXACT_OUT == 0x09
    assert V4Action.SETTLE_ALL == 0x0C
    assert V4Action.TAKE_ALL == 0x0F


def test_contract_balance_sentinels():
    # v4-periphery ActionConstants: one CONTRACT_BALANCE sentinel (1 << 255)
    # shared by V2/V3 swap amounts and V4 SETTLE amounts; OPEN_DELTA (0) is
    # the V4 swap-amount sentinel.
    assert CONTRACT_BALANCE == 1 << 255
    assert OPEN_DELTA == 0


class TestUniversalRouterConstants:
    def test_known_addresses(self):
        # UniversalRouterV2 (supports Uniswap V4)
        assert UNIVERSAL_ROUTER_ADDRESSES[1] == UNIVERSAL_ROUTER
        assert 42161 in UNIVERSAL_ROUTER_ADDRESSES

    def test_sentinels(self):
        assert MSG_SENDER == Address("0x0000000000000000000000000000000000000001")
        assert ADDRESS_THIS == Address("0x0000000000000000000000000000000000000002")

    def test_class_known_addresses(self, router):
        assert router.KNOWN_ADDRESSES[1] == UNIVERSAL_ROUTER_ADDRESSES[1]

    def test_router_address_stored(self, router):
        assert router.router_address == UNIVERSAL_ROUTER_ADDR


class TestHopDataclasses:
    def test_fields(self):
        assert V2_WETH_USDC.token_in is WETH
        assert V2_WETH_USDC.token_out is USDC
        assert V3_WETH_USDC.fee == 500

    def test_v4_hop_defaults_and_custom_hooks(self):
        assert V4_WETH_USDC.hooks == ZERO_ADDR
        assert V4_WETH_USDC.hook_data == b""
        hooks = Address("0x1234567890abcdef1234567890abcdef12345678")
        hop = V4Hop(token_in=WETH, token_out=USDC, fee=500, tick_spacing=10, hooks=hooks)
        assert hop.hooks == hooks


# ---------------------------------------------------------------------------
# Command-input encoders (no network calls)
# ---------------------------------------------------------------------------

# (encoder thunk, expected encoded length; fixed-size = N 32-byte ABI words)
FIXED_LENGTH_ENCODERS = [
    pytest.param(lambda: UniversalRouter.encode_wrap_eth(ADDRESS_THIS, 10**18), 64, id="wrap-eth"),
    pytest.param(lambda: UniversalRouter.encode_unwrap_weth(RECIPIENT, 0), 64, id="unwrap-weth"),
    pytest.param(lambda: UniversalRouter.encode_sweep(WETH.address, RECIPIENT, 0), 96, id="sweep"),
    pytest.param(lambda: UniversalRouter.encode_v4_settle_all_params(WETH.address, 10**18), 64, id="v4-settle-all"),
    pytest.param(lambda: UniversalRouter.encode_v4_take_all_params(USDC.address, 0), 64, id="v4-take-all"),
    pytest.param(lambda: UniversalRouter.encode_v4_settle_params(WETH.address, 10**18, False), 96, id="v4-settle"),
    pytest.param(lambda: UniversalRouter.encode_v4_settle_params(WETH.address, 10**18, True), 96, id="v4-settle-user"),
    pytest.param(lambda: UniversalRouter.encode_v4_take_params(USDC.address, RECIPIENT, 0), 96, id="v4-take"),
]

V3_PATH = UniswapV3._encode_path([WETH, USDC], [3000])

VARIABLE_LENGTH_ENCODERS = [
    pytest.param(
        lambda: UniversalRouter.encode_v3_swap_exact_in(RECIPIENT, 10**18, 1_800 * 10**6, V3_PATH),
        id="v3-exact-in",
    ),
    pytest.param(
        lambda: UniversalRouter.encode_v3_swap_exact_out(RECIPIENT, 10**18, 2_200 * 10**6, V3_PATH),
        id="v3-exact-out",
    ),
    pytest.param(
        lambda: UniversalRouter.encode_v2_swap_exact_in(RECIPIENT, 10**18, 1_800 * 10**6, [WETH.address, USDC.address]),
        id="v2-exact-in",
    ),
    pytest.param(
        lambda: UniversalRouter.encode_v2_swap_exact_out(
            RECIPIENT, 1_800 * 10**6, 10**18, [WETH.address, USDC.address]
        ),
        id="v2-exact-out",
    ),
]


class TestCommandEncoders:
    @pytest.mark.parametrize("encode,expected_len", FIXED_LENGTH_ENCODERS)
    def test_fixed_length(self, encode, expected_len):
        encoded = encode()
        assert isinstance(encoded, bytes)
        assert len(encoded) == expected_len

    @pytest.mark.parametrize("encode", VARIABLE_LENGTH_ENCODERS)
    def test_nonempty_bytes(self, encode):
        encoded = encode()
        assert isinstance(encoded, bytes)
        assert len(encoded) > 0
        assert len(encoded) % 32 == 0

    def test_v3_payer_is_user_changes_encoding(self):
        enc_user = UniversalRouter.encode_v3_swap_exact_in(RECIPIENT, 10**18, 0, V3_PATH, payer_is_user=True)
        enc_router = UniversalRouter.encode_v3_swap_exact_in(RECIPIENT, 10**18, 0, V3_PATH, payer_is_user=False)
        assert enc_user != enc_router

    def test_v4_take_params_open_delta_amount(self):
        # amount=0 means OPEN_DELTA (take all available credit): last ABI word is zero
        encoded = UniversalRouter.encode_v4_take_params(USDC.address, RECIPIENT, 0)
        assert encoded[-32:] == b"\x00" * 32

    def test_encode_v4_swap_actions_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length"):
            UniversalRouter.encode_v4_swap_actions([V4Action.SWAP_EXACT_IN_SINGLE], [])

    def test_sort_currencies(self):
        # USDC (0xA0...) < WETH (0xC0...) so USDC is currency0, whatever the input order
        for pair in [(WETH.address, USDC.address), (USDC.address, WETH.address)]:
            c0, c1 = UniversalRouter._sort_v4_currencies(*pair)
            assert c0 == USDC.address
            assert c1 == WETH.address


# ---------------------------------------------------------------------------
# V4 hook-permission classification
# ---------------------------------------------------------------------------

HookFlag = v4_hooks.HookFlag


def _hook_addr(flags: int) -> Address:
    """Synthetic hook address with the given permission bits."""
    return Address(int(flags).to_bytes(20, "big"))


def _hooked_edge(lp_fee_pips: int = 500) -> V4PoolEdge:
    """Edge for a delta-hook pool at price 1.0 with plenty of liquidity."""
    return V4PoolEdge(
        token_in=WETH,
        token_out=USDC,
        pool_address=UNIVERSAL_ROUTER_ADDR,
        protocol="UniswapV4",
        fee_bps=lp_fee_pips // 100,
        sqrt_price_x96=2**96,
        liquidity=10**24,
        is_token0_in=True,
        tick_spacing=10,
        hooks=_hook_addr(HookFlag.AFTER_SWAP_RETURNS_DELTA),
        lp_fee_pips=lp_fee_pips,
        key_fee_pips=lp_fee_pips,
        hook_affects_pricing=True,
    )


def _v4(**kwargs) -> UniswapV4:
    """Offline V4 client: every on-chain call is stubbed or patched by the test."""
    return UniswapV4(
        w3=None,
        pool_manager_address=UNIVERSAL_ROUTER_ADDR,
        state_view_address=UNIVERSAL_ROUTER_ADDR,
        quoter_address=UNIVERSAL_ROUTER_ADDR,
        **kwargs,
    )


class TestV4Hooks:
    def test_flags_read_low_14_address_bits(self):
        assert v4_hooks.hook_flags(ZERO_ADDR) == 0
        addr = Address(bytes.fromhex("ff" * 18 + "0088"))  # high bits are identity, not permissions
        assert v4_hooks.hook_flags(addr) == HookFlag.BEFORE_SWAP | HookFlag.BEFORE_SWAP_RETURNS_DELTA

    def test_swap_hook_detection(self):
        assert v4_hooks.has_swap_hook(_hook_addr(HookFlag.BEFORE_SWAP))
        assert v4_hooks.has_swap_hook(_hook_addr(HookFlag.AFTER_SWAP))
        assert not v4_hooks.has_swap_hook(_hook_addr(HookFlag.BEFORE_ADD_LIQUIDITY))

    @pytest.mark.parametrize(
        "flags,dynamic,expected",
        [
            pytest.param(0, False, False, id="no-hook"),
            pytest.param(HookFlag.BEFORE_SWAP_RETURNS_DELTA, False, True, id="before-swap-delta"),
            pytest.param(HookFlag.AFTER_SWAP_RETURNS_DELTA, False, True, id="after-swap-delta"),
            # beforeSwap alone can veto but not re-price — unless the pool fee
            # is dynamic, where it may override lpFee per swap
            pytest.param(HookFlag.BEFORE_SWAP, False, False, id="veto-only-static"),
            pytest.param(HookFlag.BEFORE_SWAP, True, True, id="fee-override-dynamic"),
            # liquidity/initialize permissions never touch swap amounts
            pytest.param(
                HookFlag.BEFORE_ADD_LIQUIDITY
                | HookFlag.AFTER_REMOVE_LIQUIDITY_RETURNS_DELTA
                | HookFlag.BEFORE_INITIALIZE,
                True,
                False,
                id="liquidity-hooks",
            ),
        ],
    )
    def test_affects_swap_pricing(self, flags, dynamic, expected):
        assert v4_hooks.affects_swap_pricing(_hook_addr(flags), is_dynamic_fee=dynamic) is expected

    def test_edge_carries_hook_classification(self):
        edge = _hooked_edge()
        assert edge.hook_affects_pricing
        # local math still produces a (hook-blind) estimate for ranking
        assert edge.amount_out(10**15) > 0


# ---------------------------------------------------------------------------
# V4 protocol fee (slot0.protocolFee, stacked on top of the LP fee)
# ---------------------------------------------------------------------------


class _StubStateViewCall:
    def __init__(self, result):
        self._result = result

    async def call(self, _w3, to=None):
        return self._result


class _StubStateView:
    """Stands in for the StateView contract, returning canned slot0/liquidity."""

    def __init__(self, slot0, liquidity):
        self.fns = self
        self._slot0, self._liquidity = slot0, liquidity

    def getSlot0(self, _pool_id):  # noqa: N802 - mirrors the on-chain ABI name
        return _StubStateViewCall(self._slot0)

    def getLiquidity(self, _pool_id):  # noqa: N802 - mirrors the on-chain ABI name
        return _StubStateViewCall(self._liquidity)


@pytest.fixture
def state_view(monkeypatch):
    """Install a StateView stub at price 1.0; *protocol_fee* is the packed slot0 field."""

    def _install(lp_fee_pips: int, protocol_fee: int = 0):
        monkeypatch.setattr(
            "pydefi.amm.uniswap_v4.UNISWAP_V4_STATE_VIEW",
            _StubStateView(slot0=(2**96, 0, protocol_fee, lp_fee_pips), liquidity=10**24),
        )

    return _install


class TestV4ProtocolFee:
    """slot0.protocolFee stacks on lpFee; omitting it under-prices every V4 swap."""

    def _edge(self, protocol_fee_pips: int, lp_fee_pips: int = 500) -> V4PoolEdge:
        """Hookless twin of the standard test edge, carrying a protocol fee."""
        return replace(
            _hooked_edge(lp_fee_pips),
            hooks=ZERO_ADDR,
            hook_affects_pricing=False,
            protocol_fee_pips=protocol_fee_pips,
        )

    def test_composes_rather_than_sums(self):
        # pf + lp - pf*lp/1e6: on a 0.3% pool that lands 3 pips under the
        # naive 4000 sum (at 500 pips the cross term floors to 0 and they tie)
        assert self._edge(protocol_fee_pips=1000, lp_fee_pips=3000).effective_fee_pips() == 3997
        assert self._edge(protocol_fee_pips=0).effective_fee_pips() == 500

    def test_mainnet_125_pip_fee_reproduces_625(self):
        # the exact configuration that broke the live tests
        assert self._edge(protocol_fee_pips=125).effective_fee_pips() == 625

    def test_protocol_fee_reduces_output(self):
        charged = self._edge(protocol_fee_pips=125).amount_out(10**16)
        assert charged < self._edge(protocol_fee_pips=0).amount_out(10**16)

    def test_calibrated_edge_does_not_double_charge(self):
        # calibration already measured the total take, protocol fee included
        edge = self._edge(protocol_fee_pips=125)
        edge.lp_fee_pips, edge.hook_fee_calibrated = 625, True
        assert edge.effective_fee_pips() == 625

    def test_unset_fee_falls_back_to_fee_bps(self):
        edge = self._edge(protocol_fee_pips=125)
        edge.lp_fee_pips = 0  # edge built without slot0 data
        assert edge.effective_fee_pips() == 625

    @pytest.mark.parametrize(
        "token_in,token_out,expected",
        [
            # USDC (0xA0b8…) sorts below WETH (0xC02a…), so USDC is currency0:
            # USDC→WETH is zeroForOne (low 12 bits), WETH→USDC is oneForZero (high).
            pytest.param(USDC, WETH, 100, id="zero-for-one"),
            pytest.param(WETH, USDC, 300, id="one-for-zero"),
        ],
    )
    async def test_get_pool_edge_picks_direction(self, state_view, token_in, token_out, expected):
        state_view(lp_fee_pips=500, protocol_fee=(300 << 12) | 100)  # oneForZero=300, zeroForOne=100

        edge = await _v4().get_pool_edge(token_in, token_out)

        assert edge.protocol_fee_pips == expected
        assert edge.lp_fee_pips == 500  # LP fee stays bare, for PoolKey rebuild


# ---------------------------------------------------------------------------
# V4 routing gates: hooked pools are opt-in, fees are capped
# ---------------------------------------------------------------------------

HOOK = _hook_addr(HookFlag.BEFORE_SWAP)


class TestHookPolicy:
    """A hook can re-price, veto or re-enter a swap, so hooked pools are opt-in."""

    @pytest.mark.parametrize(
        "route",
        [
            pytest.param(lambda v4: v4.get_pool_edge(USDC, WETH, hooks=HOOK), id="pool-edge"),
            pytest.param(lambda v4: v4.build_swap_route(WETH_1, USDC, hooks=HOOK), id="swap-route"),
        ],
    )
    async def test_hooked_pool_is_refused_before_any_rpc(self, route):
        # no StateView / Quoter stub installed: the refusal must come first
        with pytest.raises(HookedPoolError, match="allow_hooks=True"):
            await route(_v4())

    def test_default_hooks_are_refused_too(self):
        with pytest.raises(HookedPoolError):
            _v4(default_hooks=HOOK)

    async def test_allow_hooks_admits_it(self, state_view):
        state_view(lp_fee_pips=500)
        v4 = _v4(default_hooks=HOOK, allow_hooks=True)

        assert (await v4.get_pool_edge(USDC, WETH)).hooks == HOOK

    async def test_hookless_pool_is_unaffected(self, state_view):
        state_view(lp_fee_pips=500)

        assert (await _v4().get_pool_edge(USDC, WETH)).hooks == ZERO_ADDR


class TestFeeCap:
    """A pool charging past every real fee tier is a hook skimming, not a tier."""

    async def test_top_fee_tier_stays_routable(self, state_view):
        # Uniswap's own 1% tier plus the mainnet protocol fee composes to 10 124
        state_view(lp_fee_pips=10_000, protocol_fee=125)

        edge = await _v4().get_pool_edge(USDC, WETH)

        assert edge.effective_fee_pips() == 10_124

    async def test_pool_over_the_cap_is_rejected(self, state_view):
        state_view(lp_fee_pips=128_000)  # 12.8%, the toxic-pool take

        with pytest.raises(PoolFeeTooHighError, match="128000 pips"):
            await _v4().get_pool_edge(USDC, WETH)

    async def test_cap_is_configurable(self, state_view):
        state_view(lp_fee_pips=30_000)  # 3%

        assert await _v4(max_fee_pips=30_000).get_pool_edge(USDC, WETH)
        with pytest.raises(PoolFeeTooHighError):
            await _v4(max_fee_pips=29_999).get_pool_edge(USDC, WETH)

    async def test_cap_sees_the_calibrated_hook_take(self, state_view, monkeypatch):
        # a cheap stored fee says nothing: the hook's cut only shows up once
        # calibration has measured it, so the cap has to be applied after
        state_view(lp_fee_pips=500)
        v4 = _v4(allow_hooks=True)

        async def fake_calibrate(edge, **_kwargs):
            edge.lp_fee_pips, edge.hook_fee_calibrated = 128_000, True

        monkeypatch.setattr(v4, "calibrate_hook_fee", fake_calibrate)

        hooks = _hook_addr(HookFlag.BEFORE_SWAP_RETURNS_DELTA)
        with pytest.raises(PoolFeeTooHighError):
            await v4.get_pool_edge(USDC, WETH, hooks=hooks, calibrate_hooks=True)


# ---------------------------------------------------------------------------
# V4 quote environment (gas / sender a hook can read)
# ---------------------------------------------------------------------------


class _StubQuoterCall:
    def __init__(self, recorder: list[dict], result):
        self._recorder, self._result = recorder, result

    async def call(self, _w3, to=None, **tx):
        self._recorder.append(tx)
        return self._result


class _StubQuoter:
    """Stands in for the V4 Quoter, recording the eth_call fields it was sent."""

    def __init__(self, result=(1_000, 0)):
        self.fns = self
        self.calls: list[dict] = []
        self._result = result

    def quoteExactInputSingle(self, _params):  # noqa: N802 - mirrors the on-chain ABI name
        return _StubQuoterCall(self.calls, self._result)

    def quoteExactOutputSingle(self, _params):  # noqa: N802 - mirrors the on-chain ABI name
        return _StubQuoterCall(self.calls, self._result)


class TestV4QuoteEnvironment:
    """gasleft() and msg.sender are readable from a hook, so a quote that leaves
    them at eth_call defaults is a quote the hook can tell apart from the trade."""

    @pytest.fixture
    def quoter(self, monkeypatch) -> _StubQuoter:
        quoter = _StubQuoter()
        monkeypatch.setattr("pydefi.amm.uniswap_v4.UNISWAP_V4_QUOTER", quoter)
        return quoter

    async def test_default_quote_leaves_the_environment_unset(self, quoter):
        await _v4().quote_exact_input_single(WETH_1, USDC)

        assert quoter.calls == [{}]  # node gas cap, zero-address sender

    async def test_gas_and_sender_reach_the_call(self, quoter):
        await _v4().quote_exact_input_single(WETH_1, USDC, gas=EXECUTION_GAS, sender=RECIPIENT)

        assert quoter.calls == [{"gas": EXECUTION_GAS, "from": RECIPIENT}]

    async def test_every_hop_quotes_in_the_same_environment(self, quoter):
        await _v4().get_amounts_out(WETH_1, [WETH, USDC, DAI], gas=EXECUTION_GAS, sender=RECIPIENT)

        assert quoter.calls == [{"gas": EXECUTION_GAS, "from": RECIPIENT}] * 2

    async def test_exact_output_quote_takes_the_environment(self, quoter):
        await _v4().get_amounts_in(USDC_2K, [WETH, USDC], gas=EXECUTION_GAS, sender=RECIPIENT)

        assert quoter.calls == [{"gas": EXECUTION_GAS, "from": RECIPIENT}]


# ---------------------------------------------------------------------------
# V4 hook-fee calibration (faked quoter, no network)
# ---------------------------------------------------------------------------


def _nonlinear_fee(amount: int) -> int:
    """Size-dependent fake-quoter fee: jumps between the 1x probe (1e20 at the
    standard test edge) and the 4x probe, so calibration must reject it."""
    return 700 if amount <= 10**20 else 1_500


def _fake_quoter(v4: UniswapV4, edge: V4PoolEdge, fee_pips_for_amount, *, execution_fee_pips=None):
    """Patch v4.quote_exact_input_single to apply fee_pips_for_amount(amount) to the raw curve.

    Records every ``fee`` kwarg passed by the caller in ``v4.quoted_key_fees``.
    *execution_fee_pips* is charged instead whenever the caller pins ``gas``,
    modelling a hook that branches on ``gasleft()``; an exception is raised there.
    """
    raw_curve = replace(edge, lp_fee_pips=0, fee_bps=0, hook_fee_calibrated=False)
    v4.quoted_key_fees = []

    async def fake(amount_in, token_out, **kwargs):
        v4.quoted_key_fees.append(kwargs.get("fee"))
        fee = fee_pips_for_amount(amount_in.amount)
        if kwargs.get("gas") is not None and execution_fee_pips is not None:
            if isinstance(execution_fee_pips, Exception):
                raise execution_fee_pips
            fee = execution_fee_pips
        out = raw_curve.amount_out(amount_in.amount) * (1_000_000 - fee)
        return TokenAmount(token=token_out, amount=out // 1_000_000)

    v4.quote_exact_input_single = fake
    return v4


class TestHookFeeCalibration:
    @pytest.mark.parametrize("lp_fee,effective", [(500, 700), (0, 10_000)], ids=["hook-take", "zero-fee-key"])
    async def test_linear_hook_fee_folded_into_edge(self, lp_fee, effective):
        edge = _hooked_edge(lp_fee_pips=lp_fee)
        v4 = _fake_quoter(_v4(), edge, lambda _amount: effective)

        result = await v4.calibrate_hook_fee(edge)

        assert result.linear
        assert abs(result.implied_fee_pips - effective) <= 1
        assert edge.hook_fee_calibrated
        assert abs(edge.lp_fee_pips - effective) <= 1
        # repeat call re-quotes the same pool key, never the mutated lp_fee_pips
        assert (await v4.calibrate_hook_fee(edge)).linear
        # per calibration: gas probe (cap + execution legs) + the 4x probe
        assert v4.quoted_key_fees == [lp_fee] * 6

    async def test_nonlinear_hook_stays_estimate_only(self):
        edge = _hooked_edge(lp_fee_pips=500)
        v4 = _fake_quoter(_v4(), edge, _nonlinear_fee)

        result = await v4.calibrate_hook_fee(edge)

        assert not result.linear
        assert result.deviation_pips > 20
        assert not edge.hook_fee_calibrated
        assert edge.lp_fee_pips == 500  # pricing fee unchanged

    async def test_handbuilt_edge_backfills_key_fee(self):
        # without key_fee_pips the fallback uses lp_fee_pips once, then pins it
        # so recalibration keys the same pool after lp_fee_pips is mutated
        edge = _hooked_edge(lp_fee_pips=500)
        edge.key_fee_pips = 0
        v4 = _fake_quoter(_v4(), edge, lambda _amount: 700)

        await v4.calibrate_hook_fee(edge)
        assert edge.key_fee_pips == 500
        await v4.calibrate_hook_fee(edge)
        assert v4.quoted_key_fees == [500] * 6

    async def test_nonlinear_recalibration_revokes_stale_trust(self):
        edge = _hooked_edge(lp_fee_pips=500)
        v4 = _fake_quoter(_v4(), edge, lambda _amount: 700)
        await v4.calibrate_hook_fee(edge)
        assert edge.hook_fee_calibrated

        _fake_quoter(v4, edge, _nonlinear_fee)  # hook behaviour changes
        assert not (await v4.calibrate_hook_fee(edge)).linear
        assert not edge.hook_fee_calibrated

    async def test_calibrated_zero_fee_is_not_treated_as_unset(self):
        edge = _hooked_edge(lp_fee_pips=500)
        v4 = _fake_quoter(_v4(), edge, lambda _amount: 0)  # hook refunds the lpFee

        assert (await v4.calibrate_hook_fee(edge)).linear
        assert edge.lp_fee_pips == 0
        # pricing must use the calibrated 0, not fall back to fee_bps
        assert edge.amount_out(10**18) == replace(edge, lp_fee_pips=0, fee_bps=0).amount_out(10**18)

    def test_probe_amount_scales_with_liquidity(self):
        edge = _hooked_edge()
        # price 1.0 (sqrtP = 2^96) → probe ≈ L * 1e-4 for either direction
        assert UniswapV4._probe_amount(edge) == edge.liquidity // 10_000
        edge.is_token0_in = False
        assert UniswapV4._probe_amount(edge) == edge.liquidity // 10_000


class TestGasDependentHook:
    """A hook reading gasleft() quotes free to an eth_call and taxes the trade."""

    async def test_honest_pool_prices_the_same_either_way(self):
        edge = _hooked_edge(lp_fee_pips=500)
        v4 = _fake_quoter(_v4(), edge, lambda _amount: 700)

        probe = await v4.probe_gas_dependence(edge)

        assert not probe.divergent
        assert probe.deviation_bps == 0
        assert probe.execution_gas_amount_out == probe.quote_gas_amount_out

    async def test_gas_dependent_take_is_divergent(self):
        edge = _hooked_edge(lp_fee_pips=500)
        # 0% to the gas-capped quote, 12.8% to the trade (the BSC WBNB/USDC pattern)
        v4 = _fake_quoter(_v4(), edge, lambda _amount: 0, execution_fee_pips=128_000)

        probe = await v4.probe_gas_dependence(edge)

        assert probe.divergent
        assert probe.deviation_bps == pytest.approx(1_280, abs=1)

    async def test_execution_leg_revert_fails_closed(self):
        edge = _hooked_edge(lp_fee_pips=500)
        out_of_gas = InsufficientLiquidityError("V4 quoteExactInputSingle reverted: out of gas")
        v4 = _fake_quoter(_v4(), edge, lambda _amount: 0, execution_fee_pips=out_of_gas)

        probe = await v4.probe_gas_dependence(edge)

        # unquotable at execution gas = unpriceable, not "quoted at the cap price"
        assert probe.divergent
        assert probe.execution_gas_amount_out is None
        assert probe.deviation_bps is None

    async def test_transport_error_on_the_execution_leg_propagates(self):
        # only the node's verdict on the swap counts as a failed leg
        edge = _hooked_edge(lp_fee_pips=500)
        v4 = _fake_quoter(_v4(), edge, lambda _amount: 0, execution_fee_pips=ProviderConnectionError("rpc down"))

        with pytest.raises(ProviderConnectionError):
            await v4.probe_gas_dependence(edge)

    async def test_calibration_refuses_a_gas_dependent_zero(self):
        edge = _hooked_edge(lp_fee_pips=500)
        v4 = _fake_quoter(_v4(), edge, lambda _amount: 0, execution_fee_pips=128_000)

        result = await v4.calibrate_hook_fee(edge)

        assert result.linear  # the fake 0% take is perfectly proportional...
        assert not result.trusted  # ...and still must not be believed
        assert edge.hook_gas_dependent
        assert not edge.hook_fee_calibrated
        assert edge.lp_fee_pips == 500  # pricing fee untouched

    async def test_uncalibrated_edge_keeps_charging_the_protocol_fee(self):
        # the amplification: a calibrated 0 would zero the protocol fee too,
        # pricing the poisoned pool below every honest one in the graph
        edge = replace(_hooked_edge(lp_fee_pips=500), protocol_fee_pips=125)
        v4 = _fake_quoter(_v4(), edge, lambda _amount: 0, execution_fee_pips=128_000)

        await v4.calibrate_hook_fee(edge)

        assert edge.effective_fee_pips() == 625

    async def test_gas_dependence_revokes_an_earlier_calibration(self):
        edge = _hooked_edge(lp_fee_pips=500)
        v4 = _fake_quoter(_v4(), edge, lambda _amount: 700)
        await v4.calibrate_hook_fee(edge)
        assert edge.hook_fee_calibrated

        _fake_quoter(v4, edge, lambda _amount: 0, execution_fee_pips=128_000)
        assert not (await v4.calibrate_hook_fee(edge)).trusted
        assert not edge.hook_fee_calibrated

    async def test_probe_can_be_skipped(self):
        edge = _hooked_edge(lp_fee_pips=500)
        v4 = _fake_quoter(_v4(), edge, lambda _amount: 700)

        result = await v4.calibrate_hook_fee(edge, gas_probe=False)

        assert result.gas_probe is None
        assert result.trusted and edge.hook_fee_calibrated
        assert v4.quoted_key_fees == [500] * 2  # 1x and 4x only


# ---------------------------------------------------------------------------
# V4 swap-params encoders — decode round-trips against the deployed structs
# ---------------------------------------------------------------------------

# Universal Router >= 2.1.1 V4 structs (with minHopPriceX36); these decode
# checks pin the exact ABI layouts (see encode_v4_exact_in_single_params).
EXACT_SINGLE_ABI = "((address,address,uint24,int24,address),bool,uint128,uint128,uint256,bytes)"
EXACT_MULTI_ABI = "(address,(address,uint24,int24,address,bytes)[],uint256[],uint128,uint128)"


class TestV4SwapParamsEncoders:
    def test_exact_in_single_decode_round_trip(self):
        c0, c1 = UniversalRouter._sort_v4_currencies(WETH.address, USDC.address)
        encoded = UniversalRouter.encode_v4_exact_in_single_params(
            currency0=c0,
            currency1=c1,
            fee=500,
            tick_spacing=10,
            hooks=ZERO_ADDR,
            zero_for_one=False,
            amount_in=10**18,
            amount_out_minimum=1_800 * 10**6,
            hook_data=b"\x01\x02",
        )
        pool_key, zero_for_one, amount_in, amount_out_min, min_hop_price, hook_data = abi_decode(
            [EXACT_SINGLE_ABI], encoded
        )[0]
        assert pool_key == (c0.to_0x_hex().lower(), c1.to_0x_hex().lower(), 500, 10, ZERO_ADDR.to_0x_hex())
        assert zero_for_one is False
        assert amount_in == 10**18
        assert amount_out_min == 1_800 * 10**6
        assert min_hop_price == 0  # no per-hop price limit
        assert hook_data == b"\x01\x02"

    def test_exact_out_single_decode_round_trip(self):
        c0, c1 = UniversalRouter._sort_v4_currencies(WETH.address, USDC.address)
        encoded = UniversalRouter.encode_v4_exact_out_single_params(
            currency0=c0,
            currency1=c1,
            fee=500,
            tick_spacing=10,
            hooks=ZERO_ADDR,
            zero_for_one=False,
            amount_out=2_000 * 10**6,
            amount_in_maximum=2 * 10**18,
            hook_data=b"\x01\x02",
        )
        pool_key, zero_for_one, amount_out, amount_in_max, min_hop_price, hook_data = abi_decode(
            [EXACT_SINGLE_ABI], encoded
        )[0]
        assert pool_key == (c0.to_0x_hex().lower(), c1.to_0x_hex().lower(), 500, 10, ZERO_ADDR.to_0x_hex())
        assert zero_for_one is False
        assert amount_out == 2_000 * 10**6
        assert amount_in_max == 2 * 10**18
        assert min_hop_price == 0  # no per-hop price limit
        assert hook_data == b"\x01\x02"

    def test_exact_in_params_decode_round_trip(self):
        # WETH → USDC → DAI: exact-in PathKeys carry each hop's *output* token.
        path = [
            (USDC.address, 500, 10, ZERO_ADDR, b""),
            (DAI.address, 100, 1, ZERO_ADDR, b""),
        ]
        encoded = UniversalRouter.encode_v4_exact_in_params(
            currency_in=WETH.address,
            path=path,
            amount_in=10**18,
            amount_out_minimum=0,
        )
        currency_in, decoded_path, min_hop_prices, amount_in, amount_out_min = abi_decode([EXACT_MULTI_ABI], encoded)[0]
        assert currency_in == WETH.address.to_0x_hex().lower()
        assert [p[0] for p in decoded_path] == [
            USDC.address.to_0x_hex().lower(),
            DAI.address.to_0x_hex().lower(),
        ]
        assert min_hop_prices == ()  # empty = no per-hop price limits
        assert amount_in == 10**18
        assert amount_out_min == 0

    def test_exact_out_params_decode_round_trip(self):
        # WETH → USDC → DAI: exact-out PathKeys carry each hop's *input* token.
        path = [
            (WETH.address, 500, 10, ZERO_ADDR, b""),
            (USDC.address, 100, 1, ZERO_ADDR, b""),
        ]
        encoded = UniversalRouter.encode_v4_exact_out_params(
            currency_out=DAI.address,
            path=path,
            amount_out=2_000 * 10**18,
            amount_in_maximum=2 * 10**18,
        )
        currency_out, decoded_path, min_hop_prices, amount_out, amount_in_max = abi_decode([EXACT_MULTI_ABI], encoded)[
            0
        ]
        assert currency_out == DAI.address.to_0x_hex().lower()
        assert [p[0] for p in decoded_path] == [
            WETH.address.to_0x_hex().lower(),
            USDC.address.to_0x_hex().lower(),
        ]
        assert min_hop_prices == ()  # empty = no per-hop price limits
        assert amount_out == 2_000 * 10**18
        assert amount_in_max == 2 * 10**18


# ---------------------------------------------------------------------------
# build_execute_calldata
# ---------------------------------------------------------------------------


class TestBuildExecuteCalldata:
    def test_selector_depends_on_deadline(self):
        input_data = UniversalRouter.encode_v3_swap_exact_in(RECIPIENT, 10**18, 0, V3_PATH)
        with_deadline = UniversalRouter.build_execute_calldata(
            [RouterCommand.V3_SWAP_EXACT_IN], [input_data], deadline=1_700_000_000
        )
        without = UniversalRouter.build_execute_calldata([RouterCommand.V3_SWAP_EXACT_IN], [input_data])
        assert with_deadline[:4] == DEADLINE_SELECTOR
        assert without[:4] == NO_DEADLINE_SELECTOR

    def test_mismatched_commands_inputs_raises(self):
        with pytest.raises(ValueError, match="length"):
            UniversalRouter.build_execute_calldata(
                [RouterCommand.V3_SWAP_EXACT_IN, RouterCommand.WRAP_ETH],
                [b"only_one_input"],
            )


# ---------------------------------------------------------------------------
# High-level transaction builders
#
# Every case is checked for: SwapTransaction shape, router address, tx.value,
# the exact router command bytes, V4 action sequences where applicable, and
# both execute() selectors (with/without deadline).
# ---------------------------------------------------------------------------

# (build(router, **kw) -> tx, exact commands bytes, action/data fragments, tx.value)
BUILDER_CASES = [
    pytest.param(
        lambda r, **kw: r.build_v3_exact_in_transaction(
            amount_in=WETH_1, token_out=USDC, recipient=RECIPIENT, amount_out_minimum=0, fee=3000, **kw
        ),
        bytes([RouterCommand.V3_SWAP_EXACT_IN]),
        [],
        0,
        id="v3-exact-in",
    ),
    pytest.param(
        lambda r, **kw: r.build_v3_multihop_exact_in_transaction(
            amount_in=WETH_1, path=[WETH, USDC, DAI], fees=[500, 100], recipient=RECIPIENT, amount_out_minimum=0, **kw
        ),
        bytes([RouterCommand.V3_SWAP_EXACT_IN]),
        [],
        0,
        id="v3-multihop-exact-in",
    ),
    pytest.param(
        lambda r, **kw: r.build_v3_exact_out_transaction(
            amount_out=USDC_2K, token_in=WETH, recipient=RECIPIENT, amount_in_maximum=2 * 10**18, fee=3000, **kw
        ),
        bytes([RouterCommand.V3_SWAP_EXACT_OUT]),
        [],
        0,
        id="v3-exact-out",
    ),
    pytest.param(
        lambda r, **kw: r.build_v2_exact_in_transaction(
            amount_in=WETH_1, path=[WETH, USDC], recipient=RECIPIENT, amount_out_minimum=0, **kw
        ),
        bytes([RouterCommand.V2_SWAP_EXACT_IN]),
        [],
        0,
        id="v2-exact-in",
    ),
    pytest.param(
        lambda r, **kw: r.build_v2_exact_out_transaction(
            amount_out=USDC_2K, path=[WETH, USDC], recipient=RECIPIENT, amount_in_maximum=2 * 10**18, **kw
        ),
        bytes([RouterCommand.V2_SWAP_EXACT_OUT]),
        [],
        0,
        id="v2-exact-out",
    ),
    pytest.param(
        lambda r, **kw: r.build_wrap_and_v3_swap_transaction(
            eth_amount=ETH_AMOUNT,
            weth_token=WETH,
            token_out=USDC,
            recipient=RECIPIENT,
            amount_out_minimum=0,
            fee=3000,
            **kw,
        ),
        bytes([RouterCommand.WRAP_ETH, RouterCommand.V3_SWAP_EXACT_IN]),
        [],
        ETH_AMOUNT,
        id="wrap-and-v3",
    ),
    pytest.param(
        lambda r, **kw: r.build_v4_exact_in_single_transaction(
            amount_in=WETH_1, token_out=USDC, fee=500, tick_spacing=10, recipient=RECIPIENT, amount_out_minimum=0, **kw
        ),
        bytes([RouterCommand.V4_SWAP]),
        [bytes([V4Action.SWAP_EXACT_IN_SINGLE, V4Action.SETTLE_ALL, V4Action.TAKE])],
        0,
        id="v4-exact-in-single",
    ),
    pytest.param(
        lambda r, **kw: r.build_v4_exact_out_single_transaction(
            token_in=WETH,
            amount_out=USDC_2K,
            fee=500,
            tick_spacing=10,
            recipient=RECIPIENT,
            amount_in_maximum=2 * 10**18,
            **kw,
        ),
        bytes([RouterCommand.V4_SWAP]),
        [bytes([V4Action.SWAP_EXACT_OUT_SINGLE, V4Action.SETTLE_ALL, V4Action.TAKE])],
        0,
        id="v4-exact-out-single",
    ),
    pytest.param(
        lambda r, **kw: r.build_wrap_and_v4_swap_transaction(
            eth_amount=ETH_AMOUNT,
            weth_token=WETH,
            token_out=USDC,
            fee=500,
            tick_spacing=10,
            recipient=RECIPIENT,
            amount_out_minimum=0,
            **kw,
        ),
        bytes([RouterCommand.WRAP_ETH, RouterCommand.V4_SWAP]),
        [bytes([V4Action.SWAP_EXACT_IN_SINGLE, V4Action.SETTLE, V4Action.TAKE])],
        ETH_AMOUNT,
        id="wrap-and-v4",
    ),
    pytest.param(
        lambda r, **kw: r.build_v4_multihop_exact_in_transaction(
            amount_in=WETH_1, hops=[V4_WETH_USDC, V4_USDC_DAI], recipient=RECIPIENT, amount_out_minimum=0, **kw
        ),
        bytes([RouterCommand.V4_SWAP]),
        [bytes([V4Action.SWAP_EXACT_IN, V4Action.SETTLE_ALL, V4Action.TAKE])],
        0,
        id="v4-multihop-exact-in",
    ),
    pytest.param(
        lambda r, **kw: r.build_v4_multihop_exact_in_transaction(
            amount_in=WETH_1, hops=[V4_WETH_USDC], recipient=RECIPIENT, amount_out_minimum=0, payer_is_user=False, **kw
        ),
        bytes([RouterCommand.V4_SWAP]),
        [bytes([V4Action.SWAP_EXACT_IN, V4Action.SETTLE, V4Action.TAKE])],
        0,
        id="v4-multihop-exact-in-router-pays",
    ),
    pytest.param(
        lambda r, **kw: r.build_v4_multihop_exact_out_transaction(
            amount_out=DAI_2K, hops=[V4_WETH_USDC, V4_USDC_DAI], recipient=RECIPIENT, amount_in_maximum=2 * 10**18, **kw
        ),
        bytes([RouterCommand.V4_SWAP]),
        [bytes([V4Action.SWAP_EXACT_OUT, V4Action.SETTLE_ALL, V4Action.TAKE])],
        0,
        id="v4-multihop-exact-out",
    ),
    pytest.param(
        lambda r, **kw: r.build_v4_multihop_exact_out_transaction(
            amount_out=USDC_2K,
            hops=[V4_WETH_USDC],
            recipient=RECIPIENT,
            amount_in_maximum=2 * 10**18,
            payer_is_user=False,
            **kw,
        ),
        bytes([RouterCommand.V4_SWAP]),
        [bytes([V4Action.SWAP_EXACT_OUT, V4Action.SETTLE, V4Action.TAKE])],
        0,
        id="v4-multihop-exact-out-router-pays",
    ),
    pytest.param(
        lambda r, **kw: r.build_wrap_and_v4_multihop_swap_transaction(
            eth_amount=ETH_AMOUNT, weth_token=WETH, hops=[V4_WETH_USDC], recipient=RECIPIENT, amount_out_minimum=0, **kw
        ),
        bytes([RouterCommand.WRAP_ETH, RouterCommand.V4_SWAP]),
        [bytes([V4Action.SWAP_EXACT_IN, V4Action.SETTLE, V4Action.TAKE])],
        ETH_AMOUNT,
        id="wrap-and-v4-multihop",
    ),
    pytest.param(
        lambda r, **kw: r.build_wrap_and_multihop_exact_in_transaction(
            eth_amount=ETH_AMOUNT, weth_token=WETH, hops=[V3_WETH_USDC], recipient=RECIPIENT, amount_out_minimum=0, **kw
        ),
        bytes([RouterCommand.WRAP_ETH, RouterCommand.V3_SWAP_EXACT_IN]),
        [],
        ETH_AMOUNT,
        id="wrap-and-generic-multihop",
    ),
    pytest.param(
        lambda r, **kw: r.build_wrap_and_multihop_exact_in_transaction(
            eth_amount=ETH_AMOUNT,
            weth_token=WETH,
            hops=[V3_WETH_USDC, V4Hop(token_in=USDC, token_out=WETH, fee=500, tick_spacing=10)],
            recipient=RECIPIENT,
            amount_out_minimum=0,
            **kw,
        ),
        bytes([RouterCommand.WRAP_ETH, RouterCommand.V3_SWAP_EXACT_IN, RouterCommand.V4_SWAP]),
        [],
        ETH_AMOUNT,
        id="wrap-and-generic-multihop-cross-type",
    ),
]


class TestTransactionBuilders:
    @pytest.mark.parametrize("build,commands,fragments,value", BUILDER_CASES)
    def test_builds_expected_transaction(self, router, build, commands, fragments, value):
        tx = build(router)
        assert isinstance(tx, SwapTransaction)
        assert tx.to == UNIVERSAL_ROUTER_ADDR
        assert tx.value == value
        assert tx.data[:4] == NO_DEADLINE_SELECTOR
        decoded_commands, inputs = decode_execute(tx)
        assert decoded_commands == commands
        assert len(inputs) == len(commands)
        for fragment in fragments:
            assert fragment in tx.data

    @pytest.mark.parametrize("build,commands,fragments,value", BUILDER_CASES)
    def test_deadline_selector(self, router, build, commands, fragments, value):
        tx = build(router, deadline=1_700_000_000)
        assert tx.data[:4] == DEADLINE_SELECTOR
        decoded_commands, _inputs = decode_execute(tx)
        assert decoded_commands == commands

    def test_v4_exact_in_single_with_hooks(self, router):
        custom_hooks = Address("0x1234567890abcdef1234567890abcdef12345678")
        tx = router.build_v4_exact_in_single_transaction(
            amount_in=WETH_1,
            token_out=USDC,
            fee=500,
            tick_spacing=10,
            recipient=RECIPIENT,
            amount_out_minimum=0,
            hooks=custom_hooks,
            hook_data=b"\x01\x02\x03",
        )
        assert isinstance(tx, SwapTransaction)
        assert custom_hooks in tx.data
        assert b"\x01\x02\x03" in tx.data


# ---------------------------------------------------------------------------
# Generic multi-hop builder — segment merging and cross-pool-type paths
# ---------------------------------------------------------------------------

# (hops, exact commands bytes): consecutive same-type hops merge into one
# command; type changes start a new command, in path order.
GENERIC_MULTIHOP_CASES = [
    pytest.param([V2_WETH_USDC], bytes([RouterCommand.V2_SWAP_EXACT_IN]), id="v2"),
    pytest.param([V3_WETH_USDC], bytes([RouterCommand.V3_SWAP_EXACT_IN]), id="v3"),
    pytest.param([V4_WETH_USDC], bytes([RouterCommand.V4_SWAP]), id="v4"),
    pytest.param([V2_WETH_USDC, V2_USDC_DAI], bytes([RouterCommand.V2_SWAP_EXACT_IN]), id="v2-v2-merged"),
    pytest.param([V3_WETH_USDC, V3_USDC_DAI], bytes([RouterCommand.V3_SWAP_EXACT_IN]), id="v3-v3-merged"),
    pytest.param([V4_WETH_USDC, V4_USDC_DAI], bytes([RouterCommand.V4_SWAP]), id="v4-v4-merged"),
    pytest.param(
        [V2_WETH_USDC, V3_USDC_DAI],
        bytes([RouterCommand.V2_SWAP_EXACT_IN, RouterCommand.V3_SWAP_EXACT_IN]),
        id="v2-then-v3",
    ),
    pytest.param(
        [V3_WETH_USDC, V4_USDC_DAI],
        bytes([RouterCommand.V3_SWAP_EXACT_IN, RouterCommand.V4_SWAP]),
        id="v3-then-v4",
    ),
    pytest.param(
        [V2_WETH_USDC, V4_USDC_DAI],
        bytes([RouterCommand.V2_SWAP_EXACT_IN, RouterCommand.V4_SWAP]),
        id="v2-then-v4",
    ),
    pytest.param(
        [V4_WETH_USDC, V3_USDC_DAI],
        bytes([RouterCommand.V4_SWAP, RouterCommand.V3_SWAP_EXACT_IN]),
        id="v4-then-v3",
    ),
    pytest.param(
        [V2_WETH_USDC, V3_USDC_DAI, V4Hop(token_in=DAI, token_out=WETH, fee=500, tick_spacing=10)],
        bytes([RouterCommand.V2_SWAP_EXACT_IN, RouterCommand.V3_SWAP_EXACT_IN, RouterCommand.V4_SWAP]),
        id="v2-v3-v4",
    ),
]


class TestGenericMultihopBuilder:
    @pytest.mark.parametrize("hops,commands", GENERIC_MULTIHOP_CASES)
    def test_segments_and_commands(self, router, hops, commands):
        tx = router.build_multihop_exact_in_transaction(
            amount_in=WETH_1,
            hops=hops,
            recipient=RECIPIENT,
            amount_out_minimum=0,
        )
        assert isinstance(tx, SwapTransaction)
        assert tx.to == UNIVERSAL_ROUTER_ADDR
        assert tx.value == 0
        decoded_commands, inputs = decode_execute(tx)
        assert decoded_commands == commands
        assert len(inputs) == len(commands)

    def test_deadline_selector(self, router):
        tx = router.build_multihop_exact_in_transaction(
            amount_in=WETH_1,
            hops=[V3_WETH_USDC],
            recipient=RECIPIENT,
            amount_out_minimum=0,
            deadline=1_700_000_000,
        )
        assert tx.data[:4] == DEADLINE_SELECTOR


V2_SWAP_ABI = ["address", "uint256", "uint256", "address[]", "bool", "uint256[]"]
V3_SWAP_ABI = ["address", "uint256", "uint256", "bytes", "bool", "uint256[]"]


def _segment_amount_out_min(command: int, input_data: bytes) -> int:
    """Pull the amountOutMinimum a segment was encoded with, whatever its type."""
    if command == RouterCommand.V2_SWAP_EXACT_IN:
        return abi_decode(V2_SWAP_ABI, input_data)[2]
    if command == RouterCommand.V3_SWAP_EXACT_IN:
        return abi_decode(V3_SWAP_ABI, input_data)[2]
    assert command == RouterCommand.V4_SWAP
    actions, params = abi_decode(["bytes", "bytes[]"], input_data)
    swap_params = params[list(actions).index(V4Action.SWAP_EXACT_IN)]
    return abi_decode([EXACT_MULTI_ABI], swap_params)[0][4]


class TestPerHopMinimums:
    """An intermediate hop encoded with min 0 lets a mid-route pool skim what
    the route's final bound absorbs."""

    HOPS = [V2_WETH_USDC, V4_USDC_DAI, V3Hop(token_in=DAI, token_out=WETH, fee=500)]
    FINAL_MIN = 9 * 10**17

    def _minimums(self, router, **kwargs) -> list[int]:
        tx = router.build_multihop_exact_in_transaction(
            amount_in=WETH_1,
            hops=self.HOPS,
            recipient=RECIPIENT,
            amount_out_minimum=self.FINAL_MIN,
            **kwargs,
        )
        commands, inputs = decode_execute(tx)
        return [_segment_amount_out_min(c, i) for c, i in zip(commands, inputs)]

    def test_intermediate_segments_are_unbounded_by_default(self, router):
        assert self._minimums(router) == [0, 0, self.FINAL_MIN]

    def test_each_segment_takes_its_hop_minimum(self, router):
        hop_minimums = [1_900 * 10**6, 1_800 * 10**18, self.FINAL_MIN]

        assert self._minimums(router, hop_amount_out_minimums=hop_minimums) == hop_minimums

    def test_merged_segment_takes_the_hop_it_ends_on(self, router):
        # two V2 hops merge into one command, which lands on the second hop's token
        tx = router.build_multihop_exact_in_transaction(
            amount_in=WETH_1,
            hops=[V2_WETH_USDC, V2_USDC_DAI, V3Hop(token_in=DAI, token_out=WETH, fee=500)],
            recipient=RECIPIENT,
            amount_out_minimum=self.FINAL_MIN,
            hop_amount_out_minimums=[1_900 * 10**6, 1_800 * 10**18, self.FINAL_MIN],
        )
        commands, inputs = decode_execute(tx)

        assert len(commands) == 2
        assert _segment_amount_out_min(commands[0], inputs[0]) == 1_800 * 10**18

    def test_wrap_variant_takes_them_too(self, router):
        tx = router.build_wrap_and_multihop_exact_in_transaction(
            eth_amount=ETH_AMOUNT,
            weth_token=WETH,
            hops=[V2_WETH_USDC, V3_USDC_DAI],
            recipient=RECIPIENT,
            amount_out_minimum=self.FINAL_MIN,
            hop_amount_out_minimums=[1_900 * 10**6, self.FINAL_MIN],
        )
        commands, inputs = decode_execute(tx)

        # commands[0] is WRAP_ETH; the V2 segment follows
        assert _segment_amount_out_min(commands[1], inputs[1]) == 1_900 * 10**6

    def test_length_mismatch_raises(self, router):
        with pytest.raises(ValueError, match="must equal hops"):
            self._minimums(router, hop_amount_out_minimums=[0, 0])


# (builder invoked with hops=[]) — every multi-hop builder rejects empty hops
EMPTY_HOPS_CASES = [
    pytest.param(
        lambda r: r.build_multihop_exact_in_transaction(
            amount_in=WETH_1, hops=[], recipient=RECIPIENT, amount_out_minimum=0
        ),
        id="generic-multihop",
    ),
    pytest.param(
        lambda r: r.build_v4_multihop_exact_in_transaction(
            amount_in=WETH_1, hops=[], recipient=RECIPIENT, amount_out_minimum=0
        ),
        id="v4-multihop-exact-in",
    ),
    pytest.param(
        lambda r: r.build_v4_multihop_exact_out_transaction(
            amount_out=USDC_2K, hops=[], recipient=RECIPIENT, amount_in_maximum=2 * 10**18
        ),
        id="v4-multihop-exact-out",
    ),
    pytest.param(
        lambda r: r.build_wrap_and_v4_multihop_swap_transaction(
            eth_amount=ETH_AMOUNT, weth_token=WETH, hops=[], recipient=RECIPIENT, amount_out_minimum=0
        ),
        id="wrap-and-v4-multihop",
    ),
    pytest.param(
        lambda r: r.build_wrap_and_multihop_exact_in_transaction(
            eth_amount=ETH_AMOUNT, weth_token=WETH, hops=[], recipient=RECIPIENT, amount_out_minimum=0
        ),
        id="wrap-and-generic-multihop",
    ),
]


@pytest.mark.parametrize("build", EMPTY_HOPS_CASES)
def test_multihop_builders_raise_on_empty_hops(router, build):
    with pytest.raises(ValueError, match="empty"):
        build(router)
