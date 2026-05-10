"""High-level Venom IR program builder.

:class:`ProgramContext` inherits from :class:`~vyper.codegen_venom.context.VenomCodegenContext`
and is self-contained — it creates the :class:`~vyper.venom.context.IRContext` and
:class:`~vyper.venom.builder.VenomBuilder` internally, provides a dummy
:class:`~vyper.semantics.types.module.ModuleT`, and adds:

* :meth:`abi_encode` / :meth:`abi_decode` — type-safe helpers that don't need AST nodes
* :meth:`compile` — runs the full Venom optimisation pipeline and emits EVM bytecode

Usage::

    from pydefi.vm.context import ProgramContext
    from vyper.semantics.types.shortcuts import UINT256_T
    from vyper.semantics.types.primitives import AddressT
    from vyper.venom.basicblock import IRLiteral
    from eth_contract import ContractFunction

    TRANSFER = ContractFunction.from_abi("function transfer(address recipient, uint256 amount)")
    ctx = ProgramContext()
    calldata = ctx.abi_encode(
        [recipient, 10**18],
        TRANSFER.input_types,
        method_id=TRANSFER.selector,
    )
    ctx.builder.stop()
    bytecode = ctx.compile()
"""

from __future__ import annotations

from typing import Any, Sequence

from eth_typing import ABIComponent
from vyper import ast as vy_ast
from vyper.codegen_venom.abi.abi_decoder import abi_decode_to_buf
from vyper.codegen_venom.abi.abi_encoder import abi_encode_to_buf
from vyper.codegen_venom.buffer import Buffer, Ptr
from vyper.codegen_venom.context import VenomCodegenContext
from vyper.codegen_venom.value import VyperValue
from vyper.compiler.settings import VenomOptimizationFlags
from vyper.evm.assembler.core import assembly_to_evm
from vyper.semantics.data_locations import DataLocation
from vyper.semantics.types import BytesT, TupleT, VyperType
from vyper.semantics.types.module import ModuleT
from vyper.venom import VenomCompiler, run_passes_on
from vyper.venom.basicblock import IRLabel, IRLiteral, IRVariable
from vyper.venom.builder import VenomBuilder
from vyper.venom.context import IRContext

from pydefi.vm.abiutils import abi_to_vyper, load_object

# Module-level dummy ModuleT shared by all ProgramContext instances.
# VenomCodegenContext requires a ModuleT, but our operations (abi encode/decode,
# variable management) only use ctx.builder, ctx.new_temporary_value(),
# ctx.unwrap(), and ctx.copy_memory() — none of which touch module_ctx.
_dummy_ast = vy_ast.Module(body=[], name="", doc_string=None, source_id=0)
_dummy_ast.path = ""  # required by ModuleT.__init__
_DUMMY_MODULE_T: ModuleT = ModuleT(_dummy_ast)


class ProgramContext(VenomCodegenContext):
    """High-level Venom IR program builder.

    Inherits all of :class:`VenomCodegenContext` (``new_variable``, ``unwrap``,
    ``store_vyper_value``, ``allocate_buffer``, ``copy_memory``, ``block_scope``,
    etc.) and adds self-contained construction, ABI encode/decode helpers, and
    ``compile()``.
    """

    def __init__(self, ir_ctx: IRContext | None = None, fn_name: str = "main", *, set_entry: bool = True) -> None:
        """Create a ProgramContext for a function within an IRContext.

        Args:
            ir_ctx: Shared IRContext to add the function to.  When ``None``
                (default) a fresh context is created.
            fn_name: Name for the function.  Defaults to ``"main"``.
            set_entry: If ``True`` (default) and *ir_ctx* has no entry
                function yet, this function is set as the entry point.
                Pass ``False`` when building library/utility functions
                that should not be the entry point.
        """
        if ir_ctx is None:
            ir_ctx = IRContext()
        self._ir_ctx = ir_ctx
        fn = ir_ctx.create_function(fn_name)
        if set_entry and ir_ctx.entry_function is None:
            ir_ctx.entry_function = fn
            # VenomCompiler emits functions in dict insertion order. Ensure
            # the entry function is first once at construction time.
            functions = ir_ctx.functions
            if next(iter(functions)) != fn.name:
                entry_fn = functions.pop(fn.name)
                reordered = {fn.name: entry_fn, **functions}
                functions.clear()
                functions.update(reordered)
        builder = VenomBuilder(ir_ctx, fn)
        super().__init__(module_ctx=_DUMMY_MODULE_T, builder=builder)

    # ------------------------------------------------------------------
    # ABI helpers (no AST nodes)
    # ------------------------------------------------------------------

    def abi_encode(
        self,
        args: Sequence[Any],
        types: Sequence[VyperType | ABIComponent],
        *,
        method_id: bytes | None = None,
        ensure_tuple: bool = True,
    ) -> VyperValue:
        if method_id is not None and len(method_id) != 4:
            raise ValueError("method_id must be 4 bytes")
        if len(args) != len(types):
            raise ValueError("args and types must have the same length")

        b = self.builder

        vyper_types = [abi_to_vyper(comp) if not isinstance(comp, VyperType) else comp for comp in types]

        if len(args) == 1 and not ensure_tuple:
            value = args[0]
            vyper_type = vyper_types[0]
        else:
            value = args
            vyper_type = TupleT(tuple(vyper_types))

        vyper_value = load_object(self, value, vyper_type)

        # abi_encode_to_buf expects a memory pointer for src, even for
        # primitives.  If load_object returned a stack value (primitives),
        # store it to a temporary first.
        if vyper_value.is_stack_value:
            tmp = self.new_temporary_value(vyper_type)
            assert isinstance(tmp.operand, IRVariable)
            b.mstore(tmp.operand, vyper_value.operand)
            encode_src = tmp.operand
        else:
            encode_src = vyper_value.operand

        # Allocate output buffer: [length word] [method_id?] [data]
        offset = 4 if method_id is not None else 0
        maxlen = vyper_type.abi_type.size_bound() + offset
        buf_t = BytesT(maxlen)
        buf_val = self.new_temporary_value(buf_t)
        assert isinstance(buf_val.operand, IRVariable)

        if method_id is not None:
            method_id_word = int.from_bytes(method_id.ljust(32, b"\x00"), "big")
            b.mstore(b.add(buf_val.operand, IRLiteral(32)), IRLiteral(method_id_word))
            data_dst = b.add(buf_val.operand, IRLiteral(36))
        else:
            data_dst = b.add(buf_val.operand, IRLiteral(32))

        encoded_len = abi_encode_to_buf(self, data_dst, encode_src, vyper_type)
        if offset > 0:
            encoded_len = b.add(encoded_len, IRLiteral(offset))
        b.mstore(buf_val.operand, encoded_len)

        return buf_val

    def abi_decode(
        self,
        data: IRVariable,
        output_type: VyperType,
        *,
        unwrap_tuple: bool = True,
    ) -> VyperValue:
        """ABI-decode a Bytes buffer and return the decoded value in memory.

        Args:
            data: Memory pointer to the Bytes buffer (length word + ABI data).
            output_type: The VyperType to decode into.
            unwrap_tuple: If True (default), single-element tuples are
                unwrapped to the element type.

        Returns:
            ``VyperValue`` pointing to the decoded value in Vyper memory layout.
        """
        b = self.builder

        # The Bytes buffer: [length_word][ABI data ...]
        data_len = b.mload(data)
        data_ptr = b.add(data, IRLiteral(32))

        # Determine the ABI-level type (may be wrapped in a tuple).
        wrapped_typ = output_type
        # For ABI conformance, external return types are wrapped in tuples.
        # We only wrap if the output_type itself isn't already a tuple and
        # unwrap_tuple is True (meaning the caller wants us to handle the
        # tuple wrapping/unwrapping automatically).
        if unwrap_tuple and not isinstance(output_type, TupleT):
            wrapped_typ = TupleT((output_type,))

        # Validate size bounds.
        abi_min_size = wrapped_typ.abi_type.static_size()
        abi_max_size = wrapped_typ.abi_type.size_bound()
        if abi_min_size == abi_max_size:
            b.assert_(b.eq(data_len, IRLiteral(abi_min_size)))
        else:
            ge_min = b.iszero(b.lt(data_len, IRLiteral(abi_min_size)))
            le_max = b.iszero(b.gt(data_len, IRLiteral(abi_max_size)))
            b.assert_(b.and_(ge_min, le_max))

        # Allocate output buffer and decode.
        output_val = self.new_temporary_value(wrapped_typ)
        assert isinstance(output_val.operand, IRVariable)

        hi = b.add(data_ptr, data_len)
        buf = Buffer(_ptr=data_ptr, size=wrapped_typ.memory_bytes_required, annotation="abi_decode_src")
        ptr = Ptr(operand=data_ptr, location=DataLocation.MEMORY, buf=buf)
        src = VyperValue.from_ptr(ptr, wrapped_typ)
        abi_decode_to_buf(self, output_val.operand, src, hi=hi)

        # Unwrap single-element tuple if requested.
        if unwrap_tuple and isinstance(wrapped_typ, TupleT) and wrapped_typ != output_type:
            return VyperValue.from_ptr(output_val.ptr(), output_type)
        return output_val

    # ------------------------------------------------------------------
    # Compilation
    # ------------------------------------------------------------------

    def compile(
        self,
        flags: VenomOptimizationFlags | None = None,
    ) -> bytes:
        """Compile this context to EVM bytecode.

        Runs the full Venom IR optimisation pipeline and generates EVM
        bytecode via :class:`~vyper.venom.VenomCompiler`.

        Returns:
            Compiled EVM bytecode as :class:`bytes`.
        """
        if flags is None:
            flags = VenomOptimizationFlags()
        run_passes_on(self._ir_ctx, flags)
        compiler = VenomCompiler(self._ir_ctx)
        asm = compiler.generate_evm_assembly()
        bytecode, _ = assembly_to_evm(asm)
        return bytecode

    # ------------------------------------------------------------------
    # Data sections
    # ------------------------------------------------------------------

    def append_data_section(self, name: str) -> None:
        """Append a named data section."""
        self._ir_ctx.append_data_section(IRLabel(name))

    def append_data_item(self, data: IRLabel | bytes) -> None:
        """Append a data item to the most-recently opened data section."""
        self._ir_ctx.append_data_item(data)

    def runtime_buffer(self, size: int) -> IRVariable:
        """Allocate a buffer and return its pointer.

        The returned buffer has undefined contents — use calldatacopy or
        codecopy to fill it at runtime.  The ``alloca`` is tracked by the
        memory allocator like any other static allocation.

        Compared to :meth:`allocate_buffer`, this is the raw ``alloca``
        output without a ``Buffer`` wrapper.
        """
        return self.builder.alloca(size)

    def embed_and_load(self, data: bytes) -> IRVariable:
        """Embed *data* in a data section and copy it into a memory buffer
        at runtime using a volatile ``codecopy``.

        Unlike ``mstore`` with compile-time constants, ``codecopy`` cannot
        be eliminated by the optimiser, making this safe for use as input
        to ``abi_decode`` even when the data is known at build time.
        """
        b = self.builder
        label_name = f"_data_{len(self._ir_ctx.data_segment)}"
        label = IRLabel(label_name)
        self._ir_ctx.append_data_section(label)
        self._ir_ctx.append_data_item(data)
        buf = b.alloca(len(data))
        src = b.offset(IRLiteral(0), label)
        b.codecopy(buf, src, IRLiteral(len(data)))
        return buf

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def ir_ctx(self) -> IRContext:
        """The underlying :class:`~vyper.venom.context.IRContext`."""
        return self._ir_ctx
