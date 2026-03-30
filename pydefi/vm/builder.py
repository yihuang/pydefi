"""DeFiVM fluent program builder.

:class:`Program` provides a method-chaining interface over the low-level
instruction builders in :mod:`pydefi.vm.program`.  It adds three higher-level
features that are awkward with the raw byte-concatenation approach:

1. **Label-based jumps** — define named positions with :meth:`label` and
   reference them in :meth:`jump` / :meth:`jumpi` without computing byte
   offsets by hand.  Labels are resolved when :meth:`build` is called.

2. **``call_contract`` helper** — wraps the four-item stack protocol required
   by the ``CALL`` opcode into a single method call.

3. **Program composition** — combine independent sub-programs with
   :meth:`extend` / ``+`` / ``+=`` or :meth:`compose`.

4. **Calldata surgery** — :meth:`call_with_patches` embeds runtime values
   (static, from returndata, or from a register) into a calldata template
   before dispatching the external call.

Basic usage::

    from pydefi.vm import Program
    from eth_contract.erc20 import ERC20

    ROUTER  = "0xE592427A0AEce92De3Edee1F18E0157C05861564"
    TOKEN   = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
    AMOUNT  = 10 ** 18

    bytecode = (
        Program()
        # approve router to spend tokens
        .call_contract(TOKEN, ERC20.fns.approve(ROUTER, AMOUNT).data)
        .pop()  # consume CALL success flag
        # swap (pre-built calldata)
        .call_contract(ROUTER, swap_calldata, value=0, gas=0)
        .pop()  # consume CALL success flag
        # check minimum output
        .push_addr(RECIPIENT)
        .push_addr(TOKEN)
        .push_u256(MIN_OUT)
        .assert_ge("slippage: amount_out too low")
        .build()
    )

Label-based conditional example::

    bytecode = (
        Program()
        .push_u256(condition_value)
        .jumpi("skip")          # jump if condition != 0
        .push_bytes(calldata_a)
        .push_u256(0).push_addr(CONTRACT_A).push_u256(0)
        .call()
        .pop()
        .label("skip")
        .build()
    )

Composition example::

    from eth_contract.erc20 import ERC20

    approve = Program().call_contract(TOKEN, ERC20.fns.approve(ROUTER, MAX_U256).data).pop()
    swap    = Program().call_contract(ROUTER, swap_calldata).pop()

    full = approve + swap            # returns a new Program
    # or: approve.extend(swap)       # in-place
    # or: Program.compose([approve, swap])

Calldata surgery example — embed amount from last returndata::

    from pydefi.vm.program import ret_u256, load_reg

    # double_sel(5) → 10; patch that into double_sel(0) template → double_sel(10) → 20
    bytecode = (
        Program()
        .call_contract(ADAPTER, double_calldata)
        .pop()
        .call_with_patches(
            ADAPTER,
            template_calldata,               # double(0) placeholder template
            patches=[
                ("u256", 4, ret_u256(0)),    # offset 4, value from last returndata[0:32]
            ],
        )
        .pop()
        .build()
    )

Calldata surgery with a register source::

    from pydefi.vm.program import load_reg

    # Amount was saved to reg 0 earlier in the program
    bytecode = (
        Program()
        .store_reg(0)                        # save amount from stack top
        .call_with_patches(
            ROUTER,
            swap_template,
            patches=[
                ("u256", 36, load_reg(0)),   # offset 36, value from register 0
            ],
        )
        .pop()
        .build()
    )

Split-swap example — swap token0 → token1, then split the output and route to
two separate destinations using arithmetic and composition::

    from pydefi.vm.program import load_reg

    # Prerequisite: swap01_template produces token1 from token0 (amount in reg 1 from
    # the CCTPComposer / OFTComposer prologue, or a prior STORE_REG).
    #
    # Program structure:
    #   1. swap token0→token1, store amount1 in reg 0
    #   2. share0 = amount1 * NUMERATOR / DENOMINATOR  (60% example)
    #   3. share1 = amount1 - share0
    #   4. swap token1 → token2 using share0
    #   5. swap token1 → token3 using share1

    NUMERATOR   = 60
    DENOMINATOR = 100

    # ── Step 1: swap token0 → token1 ────────────────────────────────────────
    step1 = (
        Program()
        # call swap adapter; retdata[0] = amount1
        .call_with_patches(SWAP01, swap01_template, []).pop()
        .ret_u256(0)          # push amount1
        .store_reg(0)         # reg[0] = amount1
    )

    # ── Step 2-3: compute shares ─────────────────────────────────────────────
    split = (
        Program()
        .load_reg(0)          # [amount1]
        .push_u256(NUMERATOR) # [amount1, 60]
        .mul()                # [amount1 * 60]
        .push_u256(DENOMINATOR)
        .div()                # [share0 = amount1*60//100]
        .store_reg(1)         # reg[1] = share0
        .load_reg(0)          # [amount1]
        .load_reg(1)          # [amount1, share0]
        .sub()                # [share1 = amount1 - share0]
        .store_reg(2)         # reg[2] = share1
    )

    # ── Step 4: swap token1 → token2 (share0 from reg 1) ────────────────────
    step4 = (
        Program()
        .call_with_patches(
            SWAP12, swap12_template,
            patches=[("u256", AMOUNT_OFFSET, load_reg(1))],
        )
        .pop()
    )

    # ── Step 5: swap token1 → token3 (share1 from reg 2) ────────────────────
    step5 = (
        Program()
        .call_with_patches(
            SWAP13, swap13_template,
            patches=[("u256", AMOUNT_OFFSET, load_reg(2))],
        )
        .pop()
    )

    bytecode = Program.compose([step1, split, step4, step5]).build()
"""

from __future__ import annotations

import struct

from pydefi.vm.program import (
    OP_JUMP,
    OP_JUMPI,
    add,
    assert_ge,
    assert_le,
    balance_of,
    call,
    div,
    dup,
    jump,
    jumpi,
    load_reg,
    mod,
    mul,
    patch_addr,
    patch_u256,
    pop,
    push_addr,
    push_bytes,
    push_u256,
    ret_slice,
    ret_u256,
    revert_if,
    self_addr,
    store_reg,
    sub,
    swap,
)

# ---------------------------------------------------------------------------
# Patch source types
# ---------------------------------------------------------------------------

#: A *patch source* is raw DeFiVM opcode bytes that, when executed, push exactly
#: one value onto the stack.  That value is then used to overwrite the calldata
#: field at the specified offset.
#:
#: Any instruction sequence that leaves exactly one item on the stack is valid.
#: Common examples::
#:
#:     from pydefi.vm.program import ret_u256, load_reg, push_u256, push_addr
#:
#:     ret_u256(0)        # uint256 from last call's returndata at offset 0
#:     load_reg(2)        # value from VM register 2
#:     push_u256(1000)    # static uint256 literal (pre-encode if value is known)
#:     push_addr("0x…")   # static address literal
PatchSource = bytes

#: A single patch descriptor: ``(kind, calldata_offset, opcodes)`` where:
#:
#: - *kind*: ``"u256"`` (patch a 32-byte word) or ``"addr"`` (patch a 20-byte address).
#: - *calldata_offset*: byte offset inside the calldata template to overwrite.
#: - *opcodes*: :data:`PatchSource` — raw bytecode that pushes the patch value.
PatchSpec = tuple[str, int, PatchSource]

# ---------------------------------------------------------------------------
# Program builder
# ---------------------------------------------------------------------------


class Program:
    """Fluent DeFiVM bytecode builder with label support.

    All instruction methods return ``self`` so calls can be chained.
    Call :meth:`build` at the end to obtain the final ``bytes`` bytecode.
    """

    def __init__(self) -> None:
        self._buf: bytearray = bytearray()
        self._labels: dict[str, int] = {}
        self._fixups: list[tuple[int, str]] = []  # (u16 offset in _buf, label name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _emit(self, data: bytes) -> "Program":
        self._buf.extend(data)
        return self

    # ------------------------------------------------------------------
    # Label management
    # ------------------------------------------------------------------

    def label(self, name: str) -> "Program":
        """Mark the current program position with *name*.

        Use the same name as the target in :meth:`jump` or :meth:`jumpi`
        to create a labelled branch without computing byte offsets.

        Raises :exc:`ValueError` if the label has already been defined.
        """
        if name in self._labels:
            raise ValueError(f"Program: duplicate label {name!r}")
        self._labels[name] = len(self._buf)
        return self

    # ------------------------------------------------------------------
    # Stack / register instructions
    # ------------------------------------------------------------------

    def push_u256(self, n: int) -> "Program":
        """Emit PUSH_U256."""
        return self._emit(push_u256(n))

    def push_addr(self, a: str) -> "Program":
        """Emit PUSH_ADDR."""
        return self._emit(push_addr(a))

    def push_bytes(self, data: bytes) -> "Program":
        """Emit PUSH_BYTES."""
        return self._emit(push_bytes(data))

    def dup(self) -> "Program":
        """Emit DUP."""
        return self._emit(dup())

    def swap(self) -> "Program":
        """Emit SWAP."""
        return self._emit(swap())

    def pop(self) -> "Program":
        """Emit POP."""
        return self._emit(pop())

    def load_reg(self, i: int) -> "Program":
        """Emit LOAD_REG *i*."""
        return self._emit(load_reg(i))

    def store_reg(self, i: int) -> "Program":
        """Emit STORE_REG *i*."""
        return self._emit(store_reg(i))

    # ------------------------------------------------------------------
    # Control flow instructions
    # ------------------------------------------------------------------

    def jump(self, target: str | int) -> "Program":
        """Emit JUMP.

        *target* may be either a raw byte offset (``int``) or a label name
        (``str``).  Label references are resolved at :meth:`build` time.
        """
        if isinstance(target, int):
            return self._emit(jump(target))
        self._buf.append(OP_JUMP)
        self._fixups.append((len(self._buf), target))
        self._buf.extend(b"\x00\x00")
        return self

    def jumpi(self, target: str | int) -> "Program":
        """Emit JUMPI.

        *target* may be a raw byte offset (``int``) or a label name (``str``).
        JUMPI pops the condition from the top of the stack and jumps if it is
        non-zero.
        """
        if isinstance(target, int):
            return self._emit(jumpi(target))
        self._buf.append(OP_JUMPI)
        self._fixups.append((len(self._buf), target))
        self._buf.extend(b"\x00\x00")
        return self

    def revert_if(self, msg: str) -> "Program":
        """Emit REVERT_IF with message *msg*."""
        return self._emit(revert_if(msg))

    def assert_ge(self, msg: str = "") -> "Program":
        """Emit ASSERT_GE — revert if top-of-stack ``a < b``."""
        return self._emit(assert_ge(msg))

    def assert_le(self, msg: str = "") -> "Program":
        """Emit ASSERT_LE — revert if top-of-stack ``a > b``."""
        return self._emit(assert_le(msg))

    # ------------------------------------------------------------------
    # External / introspection instructions
    # ------------------------------------------------------------------

    def call(self, require_success: bool = True) -> "Program":
        """Emit CALL.

        The caller must have pushed (top to bottom):
        ``gasLimit``, ``to``, ``value``, ``calldataBufIdx``.

        After execution, CALL pushes a single success flag (``1`` on success,
        ``0`` on failure) onto the stack.  Callers that do not rely on an
        automatic revert (for example, when ``require_success=False``) must
        explicitly :meth:`pop` or otherwise consume this flag to avoid stack
        mismanagement.
        """
        return self._emit(call(require_success))

    def balance_of(self) -> "Program":
        """Emit BALANCE_OF — pop ``token``, ``account``; push ERC-20 balance."""
        return self._emit(balance_of())

    def self_addr(self) -> "Program":
        """Emit SELF_ADDR — push the VM contract's own address."""
        return self._emit(self_addr())

    def sub(self) -> "Program":
        """Emit SUB — pop ``a`` (top), ``b``; push ``a - b`` (saturates to 0)."""
        return self._emit(sub())

    def add(self) -> "Program":
        """Emit ADD — pop ``a`` (top), ``b``; push ``a + b`` (wrapping uint256)."""
        return self._emit(add())

    def mul(self) -> "Program":
        """Emit MUL — pop ``a`` (top), ``b``; push ``a * b`` (wrapping uint256)."""
        return self._emit(mul())

    def div(self) -> "Program":
        """Emit DIV — pop ``a`` (top), ``b``; push ``a / b`` (0 if ``b == 0``)."""
        return self._emit(div())

    def mod(self) -> "Program":
        """Emit MOD — pop ``a`` (top), ``b``; push ``a % b`` (0 if ``b == 0``)."""
        return self._emit(mod())

    # ------------------------------------------------------------------
    # ABI / data instructions
    # ------------------------------------------------------------------

    def patch_u256(self, offset: int) -> "Program":
        """Emit PATCH_U256 at *offset*."""
        return self._emit(patch_u256(offset))

    def patch_addr(self, offset: int) -> "Program":
        """Emit PATCH_ADDR at *offset*."""
        return self._emit(patch_addr(offset))

    def ret_u256(self, offset: int) -> "Program":
        """Emit RET_U256 — push uint256 from last returndata at *offset*."""
        return self._emit(ret_u256(offset))

    def ret_slice(self, offset: int, length: int) -> "Program":
        """Emit RET_SLICE — push bytes slice from last returndata."""
        return self._emit(ret_slice(offset, length))

    # ------------------------------------------------------------------
    # High-level helpers
    # ------------------------------------------------------------------

    def call_contract(
        self,
        to: str,
        calldata: bytes,
        *,
        value: int = 0,
        gas: int = 0,
        require_success: bool = True,
    ) -> "Program":
        """Emit a complete external-call sequence for a pre-built calldata buffer.

        This is a convenience wrapper that pushes the four items required by
        the ``CALL`` opcode in the correct stack order::

            push_bytes(calldata)   # calldataBufIdx (bottom)
            push_u256(value)
            push_addr(to)
            push_u256(gas)         # gasLimit (top)
            CALL

        Args:
            to: Target contract address (checksummed or lowercase hex).
            calldata: Pre-encoded ABI calldata (use :mod:`pydefi.vm.abi` helpers).
            value: ETH value to forward with the call (wei), default 0.
            gas: Gas limit for the sub-call (0 = forward all remaining gas).
            require_success: If ``True`` (default), revert if the sub-call fails.

        Returns:
            ``self`` for chaining.
        """
        return (
            self._emit(push_bytes(calldata))
            ._emit(push_u256(value))
            ._emit(push_addr(to))
            ._emit(push_u256(gas))
            ._emit(call(require_success))
        )

    def call_contract_abi(
        self,
        to: str,
        abi_sig: str,
        *args: object,
        value: int = 0,
        gas: int = 0,
        require_success: bool = True,
    ) -> "Program":
        """Emit an external call encoded from a human-readable ABI signature and args.

        This is a higher-level companion to :meth:`call_contract` that builds the
        calldata automatically from a **human-readable ABI function signature** and
        the Python argument values, using
        :class:`eth_contract.contract.ContractFunction` internally.

        The ``function`` keyword in *abi_sig* is optional — both bare
        ``"transfer(address,uint256)"`` and fully qualified
        ``"function transfer(address to, uint256 amount) external"`` forms are
        accepted.  Parameter names are also optional.

        All Solidity primitive types as well as nested tuples and arrays are
        supported (anything that :func:`eth_abi.encode` can handle).

        Args:
            to: Target contract address (hex string with ``0x`` prefix).
            abi_sig: Human-readable function signature, e.g.
                ``"transfer(address,uint256)"`` or
                ``"function exactInputSingle((address,address,uint24,...) params)"``.
            *args: Positional arguments matching the signature's input parameters.
                Addresses must be ``str``; numbers must be ``int``.
                Tuple parameters are passed as Python ``tuple`` (or ``NamedTuple``).
            value: ETH value to forward with the call (wei), default 0.
            gas: Gas limit for the sub-call (0 = forward all remaining gas).
            require_success: If ``True`` (default), revert if the sub-call fails.

        Returns:
            ``self`` for chaining.

        Example::

            # ERC-20 transfer — no need to pre-build calldata
            bytecode = (
                Program()
                .call_contract_abi(TOKEN, "transfer(address,uint256)", RECIPIENT, 10**18)
                .pop()
                .build()
            )

            # Uniswap V3 exactInputSingle with a struct argument
            bytecode = (
                Program()
                .call_contract_abi(
                    ROUTER,
                    "function exactInputSingle("
                    "  (address tokenIn, address tokenOut, uint24 fee,"
                    "   address recipient, uint256 deadline,"
                    "   uint256 amountIn, uint256 amountOutMinimum,"
                    "   uint160 sqrtPriceLimitX96) params"
                    ")",
                    (TOKEN_IN, TOKEN_OUT, 3000, RECIPIENT, deadline, amount_in, 0, 0),
                )
                .pop()
                .build()
            )
        """
        from eth_contract.contract import ContractFunction

        normalised = abi_sig if abi_sig.lstrip().startswith("function ") else "function " + abi_sig
        calldata = bytes(ContractFunction.from_abi(normalised)(*args).data)
        return self.call_contract(to, calldata, value=value, gas=gas, require_success=require_success)

    def call_with_patches(
        self,
        to: str,
        calldata: bytes,
        patches: list[PatchSpec],
        *,
        value: int = 0,
        gas: int = 0,
        require_success: bool = True,
    ) -> "Program":
        """Emit a patched external call — embed runtime values into a calldata template.

        This is the **calldata surgery** helper.  It pushes a mutable copy of
        *calldata* as a buffer, applies each patch from *patches* (each one
        overwrites a field at a specific byte offset using a value produced at
        runtime by arbitrary opcodes), then issues the ``CALL`` opcode.

        Each entry in *patches* is a 3-tuple ``(kind, offset, opcodes)``:

        - *kind* — ``"u256"`` to overwrite a 32-byte word, ``"addr"`` for 20 bytes.
        - *offset* — byte offset in the calldata template to overwrite.
        - *opcodes* — raw DeFiVM bytecode (``bytes``) that, when executed, pushes
          exactly one value onto the stack.  Any instruction sequence that leaves a
          single item on the stack is valid.  For example::

              from pydefi.vm.program import ret_u256, load_reg, push_u256, push_addr

              ret_u256(0)        # uint256 from last call's returndata
              load_reg(2)        # value from VM register 2
              push_u256(1000)    # static uint256 literal
              push_addr("0x…")   # static address literal

        Example::

            from pydefi.vm.program import ret_u256, load_reg

            # Embed the output of a previous call (from returndata) as amountIn
            program = (
                Program()
                .call_contract(QUOTER, quote_calldata)
                .pop()
                .call_with_patches(
                    ROUTER,
                    swap_template,          # swap(0, ...) — amount placeholder at offset 36
                    patches=[
                        ("u256", 36, ret_u256(0)),   # fill amount from last retdata
                    ],
                )
                .pop()
                .build()
            )

        Args:
            to: Target contract address.
            calldata: Mutable calldata template bytes.
            patches: List of ``(kind, offset, opcodes)`` patch descriptors.
            value: ETH value to forward (wei), default 0.
            gas: Sub-call gas limit (0 = forward all remaining gas).
            require_success: Revert if the sub-call fails (default ``True``).

        Returns:
            ``self`` for chaining.
        """
        self._emit(push_bytes(calldata))  # [bufIdx]

        for kind, offset, opcodes in patches:
            if kind not in ("u256", "addr"):
                raise ValueError(f"call_with_patches: unknown patch kind {kind!r}; expected 'u256' or 'addr'")
            if not isinstance(opcodes, (bytes, bytearray)):
                raise TypeError(
                    f"call_with_patches: opcodes must be bytes or bytearray, got {type(opcodes).__name__!r}"
                )

            self._emit(opcodes)  # push the patch value onto the stack

            if kind == "u256":
                self._emit(patch_u256(offset))
            else:  # kind == "addr"
                self._emit(patch_addr(offset))

        # Stack now: [bufIdx] — ready for CALL prologue
        self._emit(push_u256(value))
        self._emit(push_addr(to))
        self._emit(push_u256(gas))
        self._emit(call(require_success))
        return self

    # ------------------------------------------------------------------
    # Composition
    # ------------------------------------------------------------------

    def extend(self, other: "Program") -> "Program":
        """Append *other*'s instructions to this program **in-place**.

        All byte offsets in *other*'s labels and fixup table are adjusted by
        the current length of ``self`` so that label references remain correct
        after merging.

        Raises :exc:`ValueError` if *other* defines a label that already exists
        in ``self``.

        Returns:
            ``self`` for chaining.
        """
        # Pre-validate label collisions before mutating internal state to
        # avoid leaving this Program instance in a partially-updated state.
        for name in other._labels:
            if name in self._labels:
                raise ValueError(f"Program: duplicate label {name!r} during extend")
        offset = len(self._buf)
        self._buf.extend(other._buf)
        for name, pos in other._labels.items():
            self._labels[name] = pos + offset
        for fixup_off, name in other._fixups:
            self._fixups.append((fixup_off + offset, name))
        return self

    def __add__(self, other: "Program") -> "Program":
        """Return a new :class:`Program` that concatenates *self* and *other*.

        Neither ``self`` nor ``other`` is modified.

        Raises :exc:`ValueError` on duplicate label names.
        """
        result = Program()
        result._buf.extend(self._buf)
        result._labels.update(self._labels)
        result._fixups.extend(self._fixups)
        result.extend(other)
        return result

    def __iadd__(self, other: "Program") -> "Program":
        """Extend this program in-place (``self += other``)."""
        return self.extend(other)

    @classmethod
    def compose(cls, programs: list["Program"]) -> "Program":
        """Compose a sequence of programs into a single :class:`Program`.

        Equivalent to reducing the list with ``+``, but more efficient for
        large numbers of sub-programs.

        Example::

            parts = [approve_prog, wrap_prog, swap_prog, unwrap_prog]
            bytecode = Program.compose(parts).build()

        Raises :exc:`ValueError` on duplicate label names across sub-programs.
        """
        result = cls()
        for prog in programs:
            result.extend(prog)
        return result

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> bytes:
        """Resolve label fixups and return the final bytecode.

        Raises :exc:`ValueError` if any label referenced in a jump has not
        been defined, or if a label's target offset does not fit in 16 bits.
        """
        buf = bytearray(self._buf)
        for fixup_offset, name in self._fixups:
            if name not in self._labels:
                raise ValueError(f"Program: undefined label {name!r}")
            target = self._labels[name]
            if not 0 <= target <= 0xFFFF:
                raise ValueError(f"Program: label {name!r} target offset {target} out of range for 16-bit jump")
            struct.pack_into(">H", buf, fixup_offset, target)
        return bytes(buf)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __bytes__(self) -> bytes:
        """Allow ``bytes(program)`` as an alias for ``program.build()``."""
        return self.build()

    def __len__(self) -> int:
        """Return the current (unresolved) byte length of the program."""
        return len(self._buf)

    def __repr__(self) -> str:
        return f"Program(len={len(self._buf)}, labels={list(self._labels)!r})"
