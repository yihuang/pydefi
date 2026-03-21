"""
Uniswap Universal Router integration.

The Universal Router (https://docs.uniswap.org/contracts/universal-router)
consolidates Uniswap V2, V3, and V4 swapping into a single contract using a
command-based encoding scheme.  The current on-chain version (UniversalRouterV2)
adds full support for Uniswap V4 pools.

Execute interface::

    // With deadline
    function execute(bytes calldata commands, bytes[] calldata inputs, uint256 deadline)
        external payable

    // Without deadline
    function execute(bytes calldata commands, bytes[] calldata inputs) external payable

Each byte in ``commands`` identifies one sub-command.  The corresponding
element in ``inputs`` is ABI-encoded parameters for that sub-command.

V4 swaps use a nested action encoding: the single ``V4_SWAP`` command takes
``abi.encode(bytes actions, bytes[] params)`` as its input, where ``actions``
is a sequence of :class:`V4Action` bytes and ``params`` holds the ABI-encoded
arguments for each action.
"""

from __future__ import annotations

from enum import IntEnum

from eth_abi import encode as abi_encode

from pydefi.amm.uniswap_v3 import UniswapV3
from pydefi.types import SwapTransaction, Token, TokenAmount

# ---------------------------------------------------------------------------
# Well-known Universal Router V2 deployment addresses
#
# Source: https://github.com/Uniswap/universal-router/tree/main/deploy-addresses
# These are the latest UniversalRouterV2 addresses which support Uniswap V4.
# ---------------------------------------------------------------------------

#: Mapping from chain ID to the canonical UniversalRouterV2 deployment address.
UNIVERSAL_ROUTER_ADDRESSES: dict[int, str] = {
    1: "0x66a9893cC07D91D95644AEDD05D03f95e1dBA8Af",        # Ethereum mainnet
    10: "0x851116D9223fabED8E56C0E6b8Ad0c31d98B3507",       # Optimism
    56: "0x1906c1d672b88cD1B9aC7593301cA990F94Eae07",        # BNB Chain
    130: "0xEf740bf23aCaE26f6492B10de645D6B98dC8Eaf3",       # Unichain
    137: "0x1095692A6237d83C6a72F3F5eFEdb9A670C49223",       # Polygon
    480: "0x8ac7bEE993bb44dAb564Ea4bc9EA67Bf9Eb5e743",       # WorldChain
    8453: "0x6fF5693b99212Da76ad316178A184AB56D299b43",      # Base
    42161: "0xA51afAFe0263b40EdaEf0Df8781eA9aa03E381a3",    # Arbitrum One
    43114: "0x4Dae2f939ACf50408e13d58534Ff8c2776d45265",    # Avalanche C-Chain (V1_2)
    81457: "0xeAbBcB3E8E415306207ef514f660A3F820025BE3",     # Blast
    7777777: "0x3315ef7cA28dB74aBADC6c44570efDF06b04B020",  # Zora
    11155111: "0x3A9D48AB9751398BbFa63ad67599Bb04e4BdF98b", # Sepolia (testnet)
}

# ---------------------------------------------------------------------------
# Special recipient sentinel addresses
# ---------------------------------------------------------------------------

#: Use as ``recipient`` to send output tokens to the transaction sender.
MSG_SENDER: str = "0x0000000000000000000000000000000000000001"

#: Use as ``recipient`` to keep output tokens inside the router
#: (useful as an intermediate step in multi-command transactions).
ADDRESS_THIS: str = "0x0000000000000000000000000000000000000002"

# ---------------------------------------------------------------------------
# Function selectors
# ---------------------------------------------------------------------------

# keccak256("execute(bytes,bytes[],uint256)")[:4] == 0x3593564c
_SELECTOR_EXECUTE_DEADLINE: bytes = bytes.fromhex("3593564c")

# keccak256("execute(bytes,bytes[])")[:4] == 0x24856bc3
_SELECTOR_EXECUTE: bytes = bytes.fromhex("24856bc3")


# ---------------------------------------------------------------------------
# Command definitions
# ---------------------------------------------------------------------------


class RouterCommand(IntEnum):
    """Uniswap Universal Router command bytes.

    Each command occupies one byte in the ``commands`` argument of
    ``execute()``.  The high bit (``ALLOW_REVERT_FLAG``) may be OR-ed
    with any command byte to allow that sub-call to revert without
    reverting the entire transaction.

    Reference:
        https://github.com/Uniswap/universal-router/blob/main/contracts/libraries/Commands.sol
    """

    # 0x00–0x07: first nested-if block
    V3_SWAP_EXACT_IN = 0x00
    V3_SWAP_EXACT_OUT = 0x01
    PERMIT2_TRANSFER_FROM = 0x02
    PERMIT2_PERMIT_BATCH = 0x03
    SWEEP = 0x04
    TRANSFER = 0x05
    PAY_PORTION = 0x06
    # 0x07 reserved

    # 0x08–0x0f: second nested-if block
    V2_SWAP_EXACT_IN = 0x08
    V2_SWAP_EXACT_OUT = 0x09
    PERMIT2_PERMIT = 0x0A
    WRAP_ETH = 0x0B
    UNWRAP_WETH = 0x0C
    PERMIT2_TRANSFER_FROM_BATCH = 0x0D
    BALANCE_CHECK_ERC20 = 0x0E
    # 0x0f reserved

    # 0x10–0x20: third nested-if block (V4 + position management)
    V4_SWAP = 0x10
    V3_POSITION_MANAGER_PERMIT = 0x11
    V3_POSITION_MANAGER_CALL = 0x12
    V4_INITIALIZE_POOL = 0x13
    V4_POSITION_MANAGER_CALL = 0x14
    # 0x15–0x20 reserved

    # 0x21: sub-plan execution
    EXECUTE_SUB_PLAN = 0x21

    # Third-party integration commands (>= 0x40)
    ACROSS_V4_DEPOSIT_V3 = 0x40

    #: OR this flag with any command byte to allow the sub-call to revert
    #: without reverting the whole transaction.
    ALLOW_REVERT_FLAG = 0x80


# ---------------------------------------------------------------------------
# V4 action definitions
# ---------------------------------------------------------------------------


class V4Action(IntEnum):
    """Action codes used inside a ``V4_SWAP`` command payload.

    The ``V4_SWAP`` command input is ``abi.encode(bytes actions, bytes[] params)``.
    Each byte in ``actions`` is a :class:`V4Action` code; the corresponding
    element in ``params`` is the ABI-encoded parameters for that action.

    A typical exact-input-single swap requires three actions in sequence:
    ``SWAP_EXACT_IN_SINGLE``, ``SETTLE_ALL``, ``TAKE_ALL``.

    Reference:
        https://github.com/Uniswap/v4-periphery/blob/main/src/libraries/Actions.sol
    """

    # Swap actions
    SWAP_EXACT_IN_SINGLE = 0x06
    SWAP_EXACT_IN = 0x07
    SWAP_EXACT_OUT_SINGLE = 0x08
    SWAP_EXACT_OUT = 0x09

    # Settlement / payment actions
    SETTLE = 0x0B
    SETTLE_ALL = 0x0C
    SETTLE_PAIR = 0x0D
    TAKE = 0x0E
    TAKE_ALL = 0x0F
    TAKE_PORTION = 0x10
    TAKE_PAIR = 0x11

    # Utility actions
    CLOSE_CURRENCY = 0x12
    CLEAR_OR_TAKE = 0x13
    SWEEP = 0x14
    WRAP = 0x15
    UNWRAP = 0x16


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class UniversalRouter:
    """Uniswap Universal Router (V2) transaction builder.

    Supports encoding calldata for Uniswap V2, V3, and V4 swaps via the
    command-based Universal Router (UniversalRouterV2) which introduced
    full Uniswap V4 pool support.

    Provides static helpers to ABI-encode the input bytes for each command
    type, as well as high-level convenience methods that build complete
    :class:`~pydefi.types.SwapTransaction` objects ready for submission.

    Args:
        router_address: Address of the Universal Router contract.

    Example::

        from pydefi.amm.universal_router import UniversalRouter
        from pydefi.types import Token, TokenAmount, ChainId

        WETH = Token(ChainId.ETHEREUM, "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2", "WETH")
        USDC = Token(ChainId.ETHEREUM, "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48", "USDC", 6)

        router = UniversalRouter(UniversalRouter.KNOWN_ADDRESSES[1])
        # V3 single-hop
        tx = router.build_v3_exact_in_transaction(
            amount_in=TokenAmount.from_human(WETH, "1"),
            token_out=USDC,
            recipient="0xYourAddress",
            amount_out_minimum=1900_000_000,
            fee=500,
            deadline=1_700_000_000,
        )
        # V4 single-hop
        tx = router.build_v4_exact_in_single_transaction(
            amount_in=TokenAmount.from_human(WETH, "1"),
            token_out=USDC,
            fee=500,
            tick_spacing=10,
            recipient="0xYourAddress",
            amount_out_minimum=1900_000_000,
        )
        # tx.to, tx.data, tx.value are ready to use
    """

    #: Canonical Universal Router addresses keyed by chain ID.
    KNOWN_ADDRESSES: dict[int, str] = UNIVERSAL_ROUTER_ADDRESSES

    def __init__(self, router_address: str) -> None:
        self.router_address = router_address

    # ------------------------------------------------------------------
    # Low-level: per-command input encoders (static, no network calls)
    # ------------------------------------------------------------------

    @staticmethod
    def encode_v3_swap_exact_in(
        recipient: str,
        amount_in: int,
        amount_out_minimum: int,
        path: bytes,
        payer_is_user: bool = True,
    ) -> bytes:
        """Encode the ABI input bytes for a ``V3_SWAP_EXACT_IN`` command.

        Args:
            recipient: Address that receives the output tokens.
                Use :data:`MSG_SENDER` or :data:`ADDRESS_THIS` for special routing.
            amount_in: Exact amount of the input token to swap (raw units).
            amount_out_minimum: Minimum acceptable output amount (raw units).
            path: V3 encoded path bytes (see :meth:`~pydefi.amm.UniswapV3._encode_path`).
            payer_is_user: If ``True`` (default), tokens are pulled from
                ``msg.sender`` via Permit2; if ``False``, the router uses
                tokens already held in the contract.

        Returns:
            ABI-encoded bytes ready to be used as an element in ``inputs``.
        """
        return abi_encode(
            ["address", "uint256", "uint256", "bytes", "bool"],
            [recipient, amount_in, amount_out_minimum, path, payer_is_user],
        )

    @staticmethod
    def encode_v3_swap_exact_out(
        recipient: str,
        amount_out: int,
        amount_in_maximum: int,
        path: bytes,
        payer_is_user: bool = True,
    ) -> bytes:
        """Encode the ABI input bytes for a ``V3_SWAP_EXACT_OUT`` command.

        Args:
            recipient: Address that receives the output tokens.
            amount_out: Exact amount of the output token to receive (raw units).
            amount_in_maximum: Maximum input amount the caller is willing to
                spend (raw units).
            path: V3 encoded path bytes in **reverse** order
                (token_out → … → token_in).
            payer_is_user: See :meth:`encode_v3_swap_exact_in`.

        Returns:
            ABI-encoded bytes.
        """
        return abi_encode(
            ["address", "uint256", "uint256", "bytes", "bool"],
            [recipient, amount_out, amount_in_maximum, path, payer_is_user],
        )

    @staticmethod
    def encode_v2_swap_exact_in(
        recipient: str,
        amount_in: int,
        amount_out_minimum: int,
        path: list[str],
        payer_is_user: bool = True,
    ) -> bytes:
        """Encode the ABI input bytes for a ``V2_SWAP_EXACT_IN`` command.

        Args:
            recipient: Address that receives the output tokens.
            amount_in: Exact input amount (raw units).
            amount_out_minimum: Minimum acceptable output amount (raw units).
            path: Ordered list of token addresses (token_in → … → token_out).
            payer_is_user: See :meth:`encode_v3_swap_exact_in`.

        Returns:
            ABI-encoded bytes.
        """
        return abi_encode(
            ["address", "uint256", "uint256", "address[]", "bool"],
            [recipient, amount_in, amount_out_minimum, path, payer_is_user],
        )

    @staticmethod
    def encode_v2_swap_exact_out(
        recipient: str,
        amount_out: int,
        amount_in_maximum: int,
        path: list[str],
        payer_is_user: bool = True,
    ) -> bytes:
        """Encode the ABI input bytes for a ``V2_SWAP_EXACT_OUT`` command.

        Args:
            recipient: Address that receives the output tokens.
            amount_out: Exact output amount (raw units).
            amount_in_maximum: Maximum input amount (raw units).
            path: Ordered list of token addresses (token_in → … → token_out).
            payer_is_user: See :meth:`encode_v3_swap_exact_in`.

        Returns:
            ABI-encoded bytes.
        """
        return abi_encode(
            ["address", "uint256", "uint256", "address[]", "bool"],
            [recipient, amount_out, amount_in_maximum, path, payer_is_user],
        )

    @staticmethod
    def encode_wrap_eth(recipient: str, amount_min: int) -> bytes:
        """Encode the ABI input bytes for a ``WRAP_ETH`` command.

        Args:
            recipient: Address that receives the WETH (often :data:`ADDRESS_THIS`).
            amount_min: Minimum amount of ETH to wrap (raw units).
                Pass ``0`` to wrap the entire ETH balance held by the router.

        Returns:
            ABI-encoded bytes.
        """
        return abi_encode(["address", "uint256"], [recipient, amount_min])

    @staticmethod
    def encode_unwrap_weth(recipient: str, amount_min: int) -> bytes:
        """Encode the ABI input bytes for an ``UNWRAP_WETH`` command.

        Args:
            recipient: Address that receives the native ETH.
            amount_min: Minimum acceptable ETH amount (raw units).
                Pass ``0`` to unwrap the entire WETH balance held by the router.

        Returns:
            ABI-encoded bytes.
        """
        return abi_encode(["address", "uint256"], [recipient, amount_min])

    @staticmethod
    def encode_sweep(token: str, recipient: str, amount_min: int) -> bytes:
        """Encode the ABI input bytes for a ``SWEEP`` command.

        Sweeps the entire ERC-20 (or native ETH) balance held by the router
        to *recipient*, ensuring at least *amount_min* is transferred.

        Args:
            token: Token address to sweep, or the native-currency sentinel
                ``"0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE"``.
            recipient: Address that receives the swept tokens.
            amount_min: Minimum acceptable amount (raw units).

        Returns:
            ABI-encoded bytes.
        """
        return abi_encode(["address", "address", "uint256"], [token, recipient, amount_min])

    # ------------------------------------------------------------------
    # Mid-level: calldata builder
    # ------------------------------------------------------------------

    @staticmethod
    def build_execute_calldata(
        commands: list[RouterCommand | int],
        inputs: list[bytes],
        deadline: int | None = None,
    ) -> bytes:
        """Build the complete ``execute`` calldata from commands and inputs.

        Args:
            commands: Ordered sequence of command bytes.  Each element may be
                a :class:`RouterCommand` value or a plain :class:`int`.
            inputs: ABI-encoded input for each command.
                Must have the same length as *commands*.
            deadline: Unix timestamp after which the transaction reverts.
                When provided, the ``execute(bytes,bytes[],uint256)`` variant
                (selector ``0x3593564c``) is used; otherwise the no-deadline
                variant ``execute(bytes,bytes[])`` (selector ``0x24856bc3``)
                is used.

        Returns:
            Full calldata bytes including the 4-byte function selector.

        Raises:
            ValueError: If *commands* and *inputs* have different lengths.
        """
        if len(commands) != len(inputs):
            raise ValueError(
                f"commands length ({len(commands)}) must equal inputs length ({len(inputs)})"
            )

        commands_bytes = bytes([int(c) for c in commands])

        if deadline is not None:
            selector = _SELECTOR_EXECUTE_DEADLINE
            encoded = abi_encode(
                ["bytes", "bytes[]", "uint256"],
                [commands_bytes, list(inputs), deadline],
            )
        else:
            selector = _SELECTOR_EXECUTE
            encoded = abi_encode(
                ["bytes", "bytes[]"],
                [commands_bytes, list(inputs)],
            )

        return selector + encoded

    # ------------------------------------------------------------------
    # High-level: transaction builders
    # ------------------------------------------------------------------

    def build_v3_exact_in_transaction(
        self,
        amount_in: TokenAmount,
        token_out: Token,
        recipient: str,
        amount_out_minimum: int,
        fee: int = 3000,
        deadline: int | None = None,
        payer_is_user: bool = True,
    ) -> SwapTransaction:
        """Build a single-hop V3 exact-input swap transaction.

        Args:
            amount_in: Exact input token and amount.
            token_out: Desired output token.
            recipient: Address that receives the output tokens.
            amount_out_minimum: Minimum acceptable output amount (raw units).
            fee: V3 pool fee tier in hundredths of a basis point
                (e.g. ``3000`` = 0.3 %).  Defaults to ``3000``.
            deadline: Unix timestamp after which the transaction reverts.
                If ``None``, the no-deadline ``execute`` variant is used.
            payer_is_user: See :meth:`encode_v3_swap_exact_in`.

        Returns:
            A :class:`~pydefi.types.SwapTransaction` with ``to``, ``data``,
            and ``value`` set.
        """
        path = UniswapV3._encode_path([amount_in.token, token_out], [fee])
        input_data = self.encode_v3_swap_exact_in(
            recipient, amount_in.amount, amount_out_minimum, path, payer_is_user
        )
        calldata = self.build_execute_calldata(
            [RouterCommand.V3_SWAP_EXACT_IN], [input_data], deadline
        )
        return SwapTransaction(to=self.router_address, data=calldata)

    def build_v3_multihop_exact_in_transaction(
        self,
        amount_in: TokenAmount,
        path: list[Token],
        fees: list[int],
        recipient: str,
        amount_out_minimum: int,
        deadline: int | None = None,
        payer_is_user: bool = True,
    ) -> SwapTransaction:
        """Build a multi-hop V3 exact-input swap transaction.

        Args:
            amount_in: Exact input token and amount.
            path: Ordered list of tokens (token_in → … → token_out).
                Must have at least two elements.
            fees: Fee tier for each hop (length must equal ``len(path) - 1``).
            recipient: Address that receives the output tokens.
            amount_out_minimum: Minimum acceptable output amount (raw units).
            deadline: Unix timestamp after which the transaction reverts.
            payer_is_user: See :meth:`encode_v3_swap_exact_in`.

        Returns:
            A :class:`~pydefi.types.SwapTransaction`.
        """
        encoded_path = UniswapV3._encode_path(path, fees)
        input_data = self.encode_v3_swap_exact_in(
            recipient, amount_in.amount, amount_out_minimum, encoded_path, payer_is_user
        )
        calldata = self.build_execute_calldata(
            [RouterCommand.V3_SWAP_EXACT_IN], [input_data], deadline
        )
        return SwapTransaction(to=self.router_address, data=calldata)

    def build_v3_exact_out_transaction(
        self,
        amount_out: TokenAmount,
        token_in: Token,
        recipient: str,
        amount_in_maximum: int,
        fee: int = 3000,
        deadline: int | None = None,
        payer_is_user: bool = True,
    ) -> SwapTransaction:
        """Build a single-hop V3 exact-output swap transaction.

        The Universal Router encodes V3 exact-output paths in **reverse**
        order (token_out → token_in), which this method handles automatically.

        Args:
            amount_out: Desired output token and exact amount.
            token_in: Token to spend.
            recipient: Address that receives the output tokens.
            amount_in_maximum: Maximum input amount the caller will spend
                (raw units).
            fee: V3 pool fee tier.  Defaults to ``3000``.
            deadline: Unix timestamp after which the transaction reverts.
            payer_is_user: See :meth:`encode_v3_swap_exact_in`.

        Returns:
            A :class:`~pydefi.types.SwapTransaction`.
        """
        # Exact-output paths are reversed: output token first, input token last
        path = UniswapV3._encode_path([amount_out.token, token_in], [fee])
        input_data = self.encode_v3_swap_exact_out(
            recipient, amount_out.amount, amount_in_maximum, path, payer_is_user
        )
        calldata = self.build_execute_calldata(
            [RouterCommand.V3_SWAP_EXACT_OUT], [input_data], deadline
        )
        return SwapTransaction(to=self.router_address, data=calldata)

    def build_v2_exact_in_transaction(
        self,
        amount_in: TokenAmount,
        path: list[Token],
        recipient: str,
        amount_out_minimum: int,
        deadline: int | None = None,
        payer_is_user: bool = True,
    ) -> SwapTransaction:
        """Build a V2 exact-input swap transaction.

        Args:
            amount_in: Exact input token and amount.
            path: Ordered list of tokens (token_in → … → token_out).
                Must have at least two elements.
            recipient: Address that receives the output tokens.
            amount_out_minimum: Minimum acceptable output amount (raw units).
            deadline: Unix timestamp after which the transaction reverts.
            payer_is_user: See :meth:`encode_v3_swap_exact_in`.

        Returns:
            A :class:`~pydefi.types.SwapTransaction`.
        """
        addresses = [t.address for t in path]
        input_data = self.encode_v2_swap_exact_in(
            recipient, amount_in.amount, amount_out_minimum, addresses, payer_is_user
        )
        calldata = self.build_execute_calldata(
            [RouterCommand.V2_SWAP_EXACT_IN], [input_data], deadline
        )
        return SwapTransaction(to=self.router_address, data=calldata)

    def build_v2_exact_out_transaction(
        self,
        amount_out: TokenAmount,
        path: list[Token],
        recipient: str,
        amount_in_maximum: int,
        deadline: int | None = None,
        payer_is_user: bool = True,
    ) -> SwapTransaction:
        """Build a V2 exact-output swap transaction.

        Args:
            amount_out: Desired output token and exact amount.
            path: Ordered list of tokens (token_in → … → token_out).
            recipient: Address that receives the output tokens.
            amount_in_maximum: Maximum input amount (raw units).
            deadline: Unix timestamp after which the transaction reverts.
            payer_is_user: See :meth:`encode_v3_swap_exact_in`.

        Returns:
            A :class:`~pydefi.types.SwapTransaction`.
        """
        addresses = [t.address for t in path]
        input_data = self.encode_v2_swap_exact_out(
            recipient, amount_out.amount, amount_in_maximum, addresses, payer_is_user
        )
        calldata = self.build_execute_calldata(
            [RouterCommand.V2_SWAP_EXACT_OUT], [input_data], deadline
        )
        return SwapTransaction(to=self.router_address, data=calldata)

    def build_wrap_and_v3_swap_transaction(
        self,
        eth_amount: int,
        weth_token: Token,
        token_out: Token,
        recipient: str,
        amount_out_minimum: int,
        fee: int = 3000,
        deadline: int | None = None,
    ) -> SwapTransaction:
        """Build a two-command transaction: WRAP_ETH then V3_SWAP_EXACT_IN.

        Useful when the user wants to swap native ETH → ERC-20 via a WETH
        pool.  The ETH is wrapped inside the router and then immediately
        swapped via V3.

        Args:
            eth_amount: Amount of native ETH to wrap and swap (in wei).
            weth_token: The WETH token on the target chain.  Must be the
                canonical WETH contract for that network (e.g.
                ``0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2`` on Ethereum
                mainnet).
            token_out: Desired output ERC-20 token.
            recipient: Address that receives the output tokens.
            amount_out_minimum: Minimum acceptable output amount (raw units).
            fee: V3 pool fee tier.  Defaults to ``3000``.
            deadline: Unix timestamp after which the transaction reverts.

        Returns:
            A :class:`~pydefi.types.SwapTransaction` with ``value`` set to
            *eth_amount* so the caller knows how much ETH to attach.
        """
        wrap_input = self.encode_wrap_eth(ADDRESS_THIS, eth_amount)
        path = UniswapV3._encode_path([weth_token, token_out], [fee])
        swap_input = self.encode_v3_swap_exact_in(
            recipient, eth_amount, amount_out_minimum, path, payer_is_user=False
        )
        calldata = self.build_execute_calldata(
            [RouterCommand.WRAP_ETH, RouterCommand.V3_SWAP_EXACT_IN],
            [wrap_input, swap_input],
            deadline,
        )
        return SwapTransaction(to=self.router_address, data=calldata, value=eth_amount)

    # ------------------------------------------------------------------
    # V4 helpers (static, no network calls)
    # ------------------------------------------------------------------

    @staticmethod
    def _sort_v4_currencies(addr_a: str, addr_b: str) -> tuple[str, str]:
        """Return *(currency0, currency1)* with the lower address first.

        Uniswap V4 ``PoolKey`` requires ``currency0 < currency1`` by address
        value.  The native currency (ETH) is represented as ``address(0)``
        and is always ``currency0``.

        Args:
            addr_a: First token address (checksummed or lowercase hex).
            addr_b: Second token address.

        Returns:
            A ``(currency0, currency1)`` tuple sorted in ascending address order.
        """
        return (addr_a, addr_b) if addr_a.lower() < addr_b.lower() else (addr_b, addr_a)

    @staticmethod
    def encode_v4_swap_actions(
        actions: list[V4Action | int],
        params: list[bytes],
    ) -> bytes:
        """Encode the full ``V4_SWAP`` command input.

        The V4_SWAP command takes ``abi.encode(bytes actions, bytes[] params)``
        as its input.  Each byte in ``actions`` is a :class:`V4Action` code
        and the corresponding element of ``params`` contains the
        ABI-encoded arguments for that action.

        Args:
            actions: Ordered sequence of V4 action codes.
            params: ABI-encoded parameter bytes for each action.
                Must have the same length as *actions*.

        Returns:
            ABI-encoded bytes suitable for use as the input to
            :meth:`build_execute_calldata` with ``RouterCommand.V4_SWAP``.

        Raises:
            ValueError: If *actions* and *params* have different lengths.
        """
        if len(actions) != len(params):
            raise ValueError(
                f"actions length ({len(actions)}) must equal params length ({len(params)})"
            )
        actions_bytes = bytes([int(a) for a in actions])
        return abi_encode(["bytes", "bytes[]"], [actions_bytes, list(params)])

    @staticmethod
    def encode_v4_exact_in_single_params(
        currency0: str,
        currency1: str,
        fee: int,
        tick_spacing: int,
        hooks: str,
        zero_for_one: bool,
        amount_in: int,
        amount_out_minimum: int,
        hook_data: bytes = b"",
    ) -> bytes:
        """Encode ABI params for a ``SWAP_EXACT_IN_SINGLE`` V4 action.

        The V4 ``ExactInputSingleParams`` struct is::

            struct ExactInputSingleParams {
                PoolKey poolKey;          // (currency0, currency1, fee, tickSpacing, hooks)
                bool zeroForOne;
                uint128 amountIn;
                uint128 amountOutMinimum;
                uint256 minHopPriceX36;   // 0 = no price limit
                bytes hookData;
            }

        Args:
            currency0: Lower-address token of the pool (as returned by
                :meth:`_sort_v4_currencies`).
            currency1: Higher-address token of the pool.
            fee: Pool fee in hundredths of a basis point (e.g. ``500`` = 0.05%).
            tick_spacing: Tick spacing of the pool (e.g. ``10`` for 0.05% tier).
            hooks: Address of the hooks contract, or ``address(0)`` for no hooks.
            zero_for_one: ``True`` to swap currency0 → currency1;
                ``False`` for the reverse direction.
            amount_in: Exact input amount (raw units).
            amount_out_minimum: Minimum acceptable output amount (raw units).
            hook_data: Optional arbitrary bytes forwarded to the hooks contract.

        Returns:
            ABI-encoded bytes for this action's ``params`` slot.
        """
        pool_key = (currency0, currency1, fee, tick_spacing, hooks)
        # v4-periphery CalldataDecoder.decodeSwapExactInSingleParams reads the
        # first word as an ABI offset pointer (abi.decode-style), so the params
        # must be encoded as a *struct* (outer tuple) — not as flat fields.
        # abi_encode(["((T1,...),T2,...,Tn)"], [(v1,...,vn)]) produces the
        # outer 0x20 offset that the decoder expects.
        return abi_encode(
            ["((address,address,uint24,int24,address),bool,uint128,uint128,uint256,bytes)"],
            [(pool_key, zero_for_one, amount_in, amount_out_minimum, 0, hook_data)],
        )

    @staticmethod
    def encode_v4_settle_all_params(currency: str, max_amount: int) -> bytes:
        """Encode ABI params for a ``SETTLE_ALL`` V4 action.

        ``SETTLE_ALL`` pays the full owed balance for *currency* from
        ``msgSender()`` via Permit2 (for ERC-20) or native ETH (for
        ``address(0)``).

        Args:
            currency: Token address, or ``address(0)`` for native ETH.
            max_amount: Maximum amount the caller permits to be settled.
                Use ``2**256 - 1`` to allow any amount.

        Returns:
            ABI-encoded bytes.
        """
        return abi_encode(["address", "uint256"], [currency, max_amount])

    @staticmethod
    def encode_v4_settle_params(
        currency: str, amount: int, payer_is_user: bool
    ) -> bytes:
        """Encode ABI params for a ``SETTLE`` V4 action.

        ``SETTLE`` pays *amount* of *currency* from either ``msgSender()``
        (when *payer_is_user* is ``True``, via Permit2) or from the router's
        own balance (when *payer_is_user* is ``False``).

        Pass *payer_is_user* = ``False`` when the router already holds the
        input tokens (e.g. after a preceding ``WRAP_ETH`` command).

        Args:
            currency: Token address, or ``address(0)`` for native ETH.
            amount: Amount to settle.  Use ``0`` (``ActionConstants.OPEN_DELTA``)
                to settle the full outstanding debt automatically.
            payer_is_user: ``True`` → pull from caller via Permit2;
                ``False`` → use the router's own token balance.

        Returns:
            ABI-encoded bytes.
        """
        return abi_encode(["address", "uint256", "bool"], [currency, amount, payer_is_user])

    @staticmethod
    def encode_v4_take_all_params(currency: str, min_amount: int) -> bytes:
        """Encode ABI params for a ``TAKE_ALL`` V4 action.

        ``TAKE_ALL`` transfers the entire positive delta for *currency* to
        ``msgSender()`` (the caller of the ``execute`` function).

        Args:
            currency: Token address, or ``address(0)`` for native ETH.
            min_amount: Minimum amount that must be taken (slippage guard).

        Returns:
            ABI-encoded bytes.
        """
        return abi_encode(["address", "uint256"], [currency, min_amount])

    @staticmethod
    def encode_v4_take_params(
        currency: str, recipient: str, amount: int
    ) -> bytes:
        """Encode ABI params for a ``TAKE`` V4 action.

        ``TAKE`` transfers *amount* of *currency* to an explicit *recipient*.
        Use ``amount = 0`` (``ActionConstants.OPEN_DELTA``) to take the full
        available credit automatically.

        Args:
            currency: Token address, or ``address(0)`` for native ETH.
            recipient: Address that receives the tokens.  Use
                :data:`MSG_SENDER` or :data:`ADDRESS_THIS` sentinels where
                appropriate.
            amount: Exact amount to take, or ``0`` to take all available credit.

        Returns:
            ABI-encoded bytes.
        """
        return abi_encode(["address", "address", "uint256"], [currency, recipient, amount])

    # ------------------------------------------------------------------
    # V4 high-level transaction builders
    # ------------------------------------------------------------------

    def build_v4_exact_in_single_transaction(
        self,
        amount_in: TokenAmount,
        token_out: Token,
        fee: int,
        tick_spacing: int,
        recipient: str,
        amount_out_minimum: int,
        hooks: str = "0x0000000000000000000000000000000000000000",
        hook_data: bytes = b"",
        deadline: int | None = None,
    ) -> SwapTransaction:
        """Build a single-hop Uniswap V4 exact-input swap transaction.

        Encodes three V4 actions: ``SWAP_EXACT_IN_SINGLE``, ``SETTLE_ALL``
        (pulls input tokens from the caller via Permit2), and ``TAKE``
        (sends output tokens to *recipient*), then wraps them in a
        ``V4_SWAP`` Universal Router command.

        The caller must have approved Permit2 for the input token and
        granted the UniversalRouter an allowance via ``permit2.approve()``
        before calling this transaction.  For a flow that avoids Permit2,
        see :meth:`build_wrap_and_v4_swap_transaction`.

        Args:
            amount_in: Exact input token and amount.
            token_out: Desired output token.
            fee: V4 pool fee tier in hundredths of a basis point
                (e.g. ``500`` = 0.05 %, ``3000`` = 0.3 %).
            tick_spacing: Tick spacing that matches the pool's fee tier
                (e.g. ``10`` for 0.05 %, ``60`` for 0.3 %).
            recipient: Address that receives the output tokens.
                Use :data:`MSG_SENDER` to send to the transaction sender.
            amount_out_minimum: Minimum acceptable output amount (raw units).
            hooks: Address of the hooks contract.
                Defaults to ``address(0)`` (no hooks).
            hook_data: Arbitrary bytes forwarded to the hooks contract.
                Defaults to empty.
            deadline: Unix timestamp after which the transaction reverts.
                If ``None``, the no-deadline ``execute`` variant is used.

        Returns:
            A :class:`~pydefi.types.SwapTransaction`.
        """
        token_in = amount_in.token
        addr_in = token_in.address
        addr_out = token_out.address

        currency0, currency1 = self._sort_v4_currencies(addr_in, addr_out)
        zero_for_one = addr_in.lower() == currency0.lower()

        swap_params = self.encode_v4_exact_in_single_params(
            currency0=currency0,
            currency1=currency1,
            fee=fee,
            tick_spacing=tick_spacing,
            hooks=hooks,
            zero_for_one=zero_for_one,
            amount_in=amount_in.amount,
            amount_out_minimum=amount_out_minimum,
            hook_data=hook_data,
        )
        settle_params = self.encode_v4_settle_all_params(addr_in, amount_in.amount)
        # TAKE with amount=0 (OPEN_DELTA) takes all available credit
        take_params = self.encode_v4_take_params(addr_out, recipient, 0)

        v4_input = self.encode_v4_swap_actions(
            [V4Action.SWAP_EXACT_IN_SINGLE, V4Action.SETTLE_ALL, V4Action.TAKE],
            [swap_params, settle_params, take_params],
        )
        calldata = self.build_execute_calldata(
            [RouterCommand.V4_SWAP], [v4_input], deadline
        )
        return SwapTransaction(to=self.router_address, data=calldata)

    def build_wrap_and_v4_swap_transaction(
        self,
        eth_amount: int,
        weth_token: Token,
        token_out: Token,
        fee: int,
        tick_spacing: int,
        recipient: str,
        amount_out_minimum: int,
        hooks: str = "0x0000000000000000000000000000000000000000",
        hook_data: bytes = b"",
        deadline: int | None = None,
    ) -> SwapTransaction:
        """Build a two-command transaction: ``WRAP_ETH`` then ``V4_SWAP``.

        Wraps native ETH into WETH inside the router, then performs a
        single-hop V4 exact-input swap using the router's own WETH balance
        (``SETTLE`` with *payer_is_user* = ``False``).  This avoids the need
        for any Permit2 approval, making it suitable for ETH → ERC-20 swaps.

        Args:
            eth_amount: Amount of native ETH to wrap and swap (in wei).
            weth_token: The WETH token on the target chain.
            token_out: Desired output ERC-20 token.
            fee: V4 pool fee tier in hundredths of a basis point
                (e.g. ``500`` = 0.05 %, ``3000`` = 0.3 %).
            tick_spacing: Tick spacing that matches the pool's fee tier.
            recipient: Address that receives the output tokens.
                Use :data:`MSG_SENDER` to send to the transaction sender.
            amount_out_minimum: Minimum acceptable output amount (raw units).
            hooks: Address of the hooks contract.
                Defaults to ``address(0)`` (no hooks).
            hook_data: Arbitrary bytes forwarded to the hooks contract.
                Defaults to empty.
            deadline: Unix timestamp after which the transaction reverts.

        Returns:
            A :class:`~pydefi.types.SwapTransaction` with ``value`` set to
            *eth_amount* so the caller knows how much ETH to attach.
        """
        addr_in = weth_token.address
        addr_out = token_out.address

        currency0, currency1 = self._sort_v4_currencies(addr_in, addr_out)
        zero_for_one = addr_in.lower() == currency0.lower()

        # Command 1: wrap ETH → WETH inside the router
        wrap_input = self.encode_wrap_eth(ADDRESS_THIS, eth_amount)

        # Command 2: V4_SWAP — three sub-actions:
        #   a) SWAP_EXACT_IN_SINGLE: execute the pool swap
        #   b) SETTLE(payerIsUser=False): router pays WETH from its own balance
        #   c) TAKE(amount=0): take all USDC credit and send to recipient
        swap_params = self.encode_v4_exact_in_single_params(
            currency0=currency0,
            currency1=currency1,
            fee=fee,
            tick_spacing=tick_spacing,
            hooks=hooks,
            zero_for_one=zero_for_one,
            amount_in=eth_amount,
            amount_out_minimum=amount_out_minimum,
            hook_data=hook_data,
        )
        # SETTLE with payerIsUser=False: pay WETH from the router's own balance
        settle_params = self.encode_v4_settle_params(
            addr_in, eth_amount, payer_is_user=False
        )
        # TAKE with amount=0 (ActionConstants.OPEN_DELTA): take all credit
        take_params = self.encode_v4_take_params(addr_out, recipient, 0)

        v4_input = self.encode_v4_swap_actions(
            [V4Action.SWAP_EXACT_IN_SINGLE, V4Action.SETTLE, V4Action.TAKE],
            [swap_params, settle_params, take_params],
        )
        calldata = self.build_execute_calldata(
            [RouterCommand.WRAP_ETH, RouterCommand.V4_SWAP],
            [wrap_input, v4_input],
            deadline,
        )
        return SwapTransaction(to=self.router_address, data=calldata, value=eth_amount)
