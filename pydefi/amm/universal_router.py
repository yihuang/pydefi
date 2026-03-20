"""
Uniswap Universal Router integration.

The Universal Router (https://docs.uniswap.org/contracts/universal-router)
consolidates Uniswap V2 and V3 swapping into a single contract using a
command-based encoding scheme.

Execute interface::

    // With deadline
    function execute(bytes calldata commands, bytes[] calldata inputs, uint256 deadline)
        external payable

    // Without deadline
    function execute(bytes calldata commands, bytes[] calldata inputs) external payable

Each byte in ``commands`` identifies one sub-command.  The corresponding
element in ``inputs`` is ABI-encoded parameters for that sub-command.
"""

from __future__ import annotations

from enum import IntEnum

from eth_abi import encode as abi_encode

from pydefi.amm.uniswap_v3 import UniswapV3
from pydefi.types import SwapTransaction, Token, TokenAmount

# ---------------------------------------------------------------------------
# Well-known Universal Router deployment addresses
# ---------------------------------------------------------------------------

#: Mapping from chain ID to the canonical Universal Router deployment address.
UNIVERSAL_ROUTER_ADDRESSES: dict[int, str] = {
    1: "0x3fC91A3afd70395Cd496C647d5a6CC9D4B2b7FAD",       # Ethereum mainnet
    10: "0xCb1355ff08Ab38bBCE60111F1bb2B784bE25D7e8",      # Optimism
    56: "0x5Dc88340E1c5c6366864Ee415d6034cadd1A9897",      # BNB Chain
    137: "0x3fC91A3afd70395Cd496C647d5a6CC9D4B2b7FAD",     # Polygon
    42161: "0x5E325eDA8064b456f4781070C0738d849c824258",   # Arbitrum One
    43114: "0x82635AF6146972cD895487D368088B603dfA0bd2",   # Avalanche C-Chain
    8453: "0x3fC91A3afd70395Cd496C647d5a6CC9D4B2b7FAD",    # Base
    59144: "0x3fC91A3afd70395Cd496C647d5a6CC9D4B2b7FAD",   # Linea
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

    V3_SWAP_EXACT_IN = 0x00
    V3_SWAP_EXACT_OUT = 0x01
    PERMIT2_TRANSFER_FROM = 0x02
    PERMIT2_PERMIT_BATCH = 0x03
    SWEEP = 0x04
    TRANSFER = 0x05
    PAY_PORTION = 0x06
    # 0x07 reserved
    V2_SWAP_EXACT_IN = 0x08
    V2_SWAP_EXACT_OUT = 0x09
    PERMIT2_PERMIT = 0x0A
    WRAP_ETH = 0x0B
    UNWRAP_WETH = 0x0C
    PERMIT2_TRANSFER_FROM_BATCH = 0x0D
    BALANCE_CHECK_ERC20 = 0x0E

    #: OR this flag with any command byte to allow the sub-call to revert
    #: without reverting the whole transaction.
    ALLOW_REVERT_FLAG = 0x80


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class UniversalRouter:
    """Uniswap Universal Router transaction builder.

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

        router = UniversalRouter("0x3fC91A3afd70395Cd496C647d5a6CC9D4B2b7FAD")
        tx = router.build_v3_exact_in_transaction(
            amount_in=TokenAmount.from_human(WETH, "1"),
            token_out=USDC,
            recipient="0xYourAddress",
            amount_out_minimum=1900_000_000,  # 1900 USDC in raw units
            fee=500,
            deadline=1_700_000_000,
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
