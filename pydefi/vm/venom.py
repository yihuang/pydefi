"""Venom IR utilities for pydefi.

This module provides :class:`ModuleBuilder` — a thin :class:`~vyper.venom.builder.VenomBuilder`
subclass that owns one :class:`~vyper.venom.context.IRContext` configured with a *prefix*,
so multiple independently-built modules can be merged into a single context without label
collisions.

Label namespacing, str-overloads (for ``append_data_section``/``append_data_item``/
``offset``/``codecopy``), and collision-safe ``merge`` all live on
:class:`~vyper.venom.context.IRContext` upstream — see
``vyper/venom/context.py``.

Calling convention for internal functions
-----------------------------------------

When building functions callable via ``invoke``, the last ``param()`` is always
the **return PC** (the address to jump back to after the function completes).
Correspondingly, ``ret()`` must receive the return values first and the return PC
**last**:

.. code-block:: python

    fn = mod.ctx.create_function("add")
    mod.set_block(fn.entry)
    x          = mod.param()          # first argument
    y          = mod.param()          # second argument
    return_pc  = mod.param()          # return-PC is always the last param
    result     = mod.add(x, y)
    mod.ret(result, return_pc)        # value(s), then return_pc

    # Caller side:
    rets = main.invoke(mod.ctx.named_label("add"), [IRLiteral(3), IRLiteral(4)], returns=1)

Data sections
-------------

Use :meth:`~vyper.venom.context.IRContext.append_data_section` /
:meth:`~vyper.venom.context.IRContext.append_data_item` to attach constant byte data.
To read it at runtime, combine :meth:`~vyper.venom.builder.VenomBuilder.offset`
with :meth:`~vyper.venom.builder.VenomBuilder.codecopy` — **do not** use
:meth:`~vyper.venom.builder.VenomBuilder.dload` (which adds ``code_end``, the total
bytecode length, instead of the code-section end):

.. code-block:: python

    mod.ctx.append_data_section("table")
    mod.ctx.append_data_item(b"\\x00" * 31 + b"\\x2a")   # literal 42

    src  = mod.offset(IRLiteral(0), "table")
    buf  = mod.alloca(32)
    mod.codecopy(buf, src, IRLiteral(32))
    val  = mod.mload(buf)

Typical usage::

    from pydefi.vm.venom import ModuleBuilder
    from vyper.venom.basicblock import IRLiteral

    mod_a = ModuleBuilder("mod_a")
    mod_a.ctx.append_data_section("table")
    mod_a.ctx.append_data_item(b"\\x01\\x02\\x03\\x04")
    addr = mod_a.offset(IRLiteral(0), "table")
    buf  = mod_a.alloca(32)
    mod_a.codecopy(buf, addr, IRLiteral(4))
    mod_a.stop()

    main = ModuleBuilder("main")
    main.ctx.merge(mod_a.ctx)
    bytecode = main.compile()
"""

from __future__ import annotations

from vyper.compiler.settings import VenomOptimizationFlags
from vyper.evm.assembler.core import assembly_to_evm
from vyper.venom import VenomCompiler, run_passes_on
from vyper.venom.basicblock import IRBasicBlock
from vyper.venom.builder import VenomBuilder
from vyper.venom.context import IRContext
from vyper.venom.function import IRFunction

__all__ = ["ModuleBuilder"]


class ModuleBuilder(VenomBuilder):
    """A :class:`~vyper.venom.builder.VenomBuilder` bound to a prefixed
    :class:`~vyper.venom.context.IRContext`.

    Each builder owns one context (with namespaced labels) and one entry
    function ``<prefix>.main``.  Merge sibling contexts into ``self.ctx``
    via :meth:`IRContext.merge <vyper.venom.context.IRContext.merge>` and
    compile via :meth:`compile`.

    Args:
        prefix: Namespace prefix for every generated label (e.g. ``"mod_a"``).
                When empty, behaviour is identical to plain :class:`VenomBuilder`.
        ctx:    Existing context to use; a fresh ``IRContext(prefix=prefix)`` is
                created when ``None``.
        fn:     Existing entry function; ``ctx.create_function("main")`` is used
                when ``None``.
    """

    def __init__(
        self,
        prefix: str = "",
        ctx: IRContext | None = None,
        fn: IRFunction | None = None,
    ) -> None:
        if ctx is None:
            ctx = IRContext(prefix=prefix)
        if fn is None:
            fn = ctx.create_function("main")
        # Designate the first function as entry so optimization passes
        # (e.g. FunctionInlinerPass) can find it.
        if ctx.entry_function is None:
            ctx.entry_function = fn
        super().__init__(ctx, fn)

    def set_block(self, bb: IRBasicBlock) -> None:
        """Switch the emission target to *bb* and update the active function.

        Overrides :meth:`~vyper.venom.builder.VenomBuilder.set_block` to also
        update :attr:`fn` to ``bb.parent``, so :meth:`create_block` /
        :meth:`append_block` target the same function as the current block.
        """
        self._current_bb = bb
        self.fn = bb.parent

    def compile(
        self,
        flags: VenomOptimizationFlags | None = None,
        *,
        disable_mem_checks: bool = True,
    ) -> bytes:
        """Compile this builder's context to EVM bytecode.

        Runs the full Venom IR optimization pipeline and generates EVM bytecode.

        Args:
            flags: Venom optimization flags. Defaults to standard O2.
            disable_mem_checks: Skip the concrete memory-address check in
                :func:`~vyper.venom.run_passes_on`. Required when using literal
                memory addresses such as ``PUSH0`` for the ``return_`` offset.
        """
        if flags is None:
            flags = VenomOptimizationFlags()
        run_passes_on(self.ctx, flags, disable_mem_checks=disable_mem_checks)
        compiler = VenomCompiler(self.ctx)
        asm = compiler.generate_evm_assembly(no_optimize=True)
        bytecode, _ = assembly_to_evm(asm)
        return bytecode
