"""Tests for pydefi.amm (no live node required)."""

from dataclasses import replace
from decimal import Decimal

import pytest
from eth_abi import decode as abi_decode

from pydefi._math import apply_slippage
from pydefi.amm import v4_hooks
from pydefi.amm.uniswap_v2 import UniswapV2
from pydefi.amm.uniswap_v3 import UniswapV3
from pydefi.amm.uniswap_v4 import UniswapV4
from pydefi.amm.universal_router import (
    ADDRESS_THIS,
    CONTRACT_BALANCE_V3,
    CONTRACT_BALANCE_V4,
    MSG_SENDER,
    UNIVERSAL_ROUTER_ADDRESSES,
    RouterCommand,
    UniversalRouter,
    V2Hop,
    V3Hop,
    V4Action,
    V4Hop,
)
from pydefi.exceptions import InsufficientLiquidityError
from pydefi.pathfinder.graph import V4PoolEdge
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
    assert CONTRACT_BALANCE_V3 == (1 << 256) - 1
    assert CONTRACT_BALANCE_V4 == (1 << 128) - 1


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


def _nonlinear_fee(amount: int) -> int:
    """Size-dependent fake-quoter fee: jumps between the 1x probe (1e20 at the
    standard test edge) and the 4x probe, so calibration must reject it."""
    return 700 if amount <= 10**20 else 1_500


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
# V4 hook-fee calibration (faked quoter, no network)
# ---------------------------------------------------------------------------


def _fake_quoter(v4: UniswapV4, edge: V4PoolEdge, fee_pips_for_amount):
    """Patch v4.quote_exact_input_single to apply fee_pips_for_amount(amount) to the raw curve.

    Records every ``fee`` kwarg passed by the caller in ``v4.quoted_key_fees``.
    """
    raw_curve = replace(edge, lp_fee_pips=0, fee_bps=0, hook_fee_calibrated=False)
    v4.quoted_key_fees = []

    async def fake(amount_in, token_out, **kwargs):
        v4.quoted_key_fees.append(kwargs.get("fee"))
        out = raw_curve.amount_out(amount_in.amount) * (1_000_000 - fee_pips_for_amount(amount_in.amount))
        return TokenAmount(token=token_out, amount=out // 1_000_000)

    v4.quote_exact_input_single = fake
    return v4


class TestHookFeeCalibration:
    def _v4(self) -> UniswapV4:
        return UniswapV4(
            w3=None,
            pool_manager_address=UNIVERSAL_ROUTER_ADDR,
            state_view_address=UNIVERSAL_ROUTER_ADDR,
            quoter_address=UNIVERSAL_ROUTER_ADDR,
        )

    @pytest.mark.parametrize("lp_fee,effective", [(500, 700), (0, 10_000)], ids=["hook-take", "zero-fee-key"])
    async def test_linear_hook_fee_folded_into_edge(self, lp_fee, effective):
        edge = _hooked_edge(lp_fee_pips=lp_fee)
        v4 = _fake_quoter(self._v4(), edge, lambda _amount: effective)

        result = await v4.calibrate_hook_fee(edge)

        assert result.linear
        assert abs(result.implied_fee_pips - effective) <= 1
        assert edge.hook_fee_calibrated
        assert abs(edge.lp_fee_pips - effective) <= 1
        # repeat call re-quotes the same pool key, never the mutated lp_fee_pips
        assert (await v4.calibrate_hook_fee(edge)).linear
        assert v4.quoted_key_fees == [lp_fee] * 4

    async def test_nonlinear_hook_stays_estimate_only(self):
        edge = _hooked_edge(lp_fee_pips=500)
        v4 = _fake_quoter(self._v4(), edge, _nonlinear_fee)

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
        v4 = _fake_quoter(self._v4(), edge, lambda _amount: 700)

        await v4.calibrate_hook_fee(edge)
        assert edge.key_fee_pips == 500
        await v4.calibrate_hook_fee(edge)
        assert v4.quoted_key_fees == [500] * 4

    async def test_nonlinear_recalibration_revokes_stale_trust(self):
        edge = _hooked_edge(lp_fee_pips=500)
        v4 = _fake_quoter(self._v4(), edge, lambda _amount: 700)
        await v4.calibrate_hook_fee(edge)
        assert edge.hook_fee_calibrated

        _fake_quoter(v4, edge, _nonlinear_fee)  # hook behaviour changes
        assert not (await v4.calibrate_hook_fee(edge)).linear
        assert not edge.hook_fee_calibrated

    async def test_calibrated_zero_fee_is_not_treated_as_unset(self):
        edge = _hooked_edge(lp_fee_pips=500)
        v4 = _fake_quoter(self._v4(), edge, lambda _amount: 0)  # hook refunds the lpFee

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
