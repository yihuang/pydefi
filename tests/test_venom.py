"""Tests for the venom IR ModuleBuilder abstraction.

Verifies:
1. Label namespacing — all generated labels are prefixed with the module name
2. Data section naming — data section labels carry the prefix
3. Code generation and bytecode compilation
4. Two-module merge without label collisions
5. Cross-module function calls (correct param/ret calling convention)
6. Data section access via offset + codecopy
7. Runnable bytecode verified with mini_evm
"""

from __future__ import annotations

import pytest
from vyper.venom.basicblock import IRLabel, IRLiteral

from pydefi.vm.venom import ModuleBuilder
from tests.conftest import mini_evm


# ---------------------------------------------------------------------------
# 1. Label namespacing
# ---------------------------------------------------------------------------


def test_named_label_with_prefix():
    mod = ModuleBuilder("mymod")
    label = mod.named_label("foo")
    assert label.value == "mymod.foo"
    assert label.is_symbol is True


def test_named_label_without_prefix():
    mod = ModuleBuilder("")
    label = mod.named_label("bar")
    assert label.value == "bar"


def test_get_next_label_with_prefix():
    mod = ModuleBuilder("abc")
    l1 = mod.get_next_label()
    l2 = mod.get_next_label("loop")
    assert l1.value.startswith("abc.")
    assert "loop" in l2.value


def test_get_next_label_no_prefix():
    mod = ModuleBuilder("")
    l1 = mod.get_next_label()
    # When no prefix the label comes directly from IRContext counter
    assert l1.value != ""


def test_create_function_namespaced():
    mod = ModuleBuilder("svc")
    fn = mod.create_function("helper")
    assert "svc.helper" in str(fn.name)


def test_entry_function_is_main():
    mod = ModuleBuilder("svc")
    assert mod.ctx.entry_function is not None
    assert "svc.main" in str(mod.ctx.entry_function.name)


# ---------------------------------------------------------------------------
# 2. Data section naming
# ---------------------------------------------------------------------------


def test_data_section_label_prefixed():
    mod = ModuleBuilder("ds")
    mod.append_data_section("table")
    assert len(mod.ctx.data_segment) == 1
    lbl = mod.ctx.data_segment[0].label
    assert "ds.table" in str(lbl)


def test_data_item_appended():
    mod = ModuleBuilder("ds")
    mod.append_data_section("lut")
    mod.append_data_item(b"\xde\xad\xbe\xef")
    items = mod.ctx.data_segment[0].data_items
    assert len(items) == 1
    assert items[0].data == b"\xde\xad\xbe\xef"


# ---------------------------------------------------------------------------
# 3. Basic compilation — no collisions, produces valid bytecode
# ---------------------------------------------------------------------------


def test_simple_compile_no_error():
    mod = ModuleBuilder("simple")
    result = mod.add(IRLiteral(3), IRLiteral(5))
    buf = mod.alloca(32)
    mod.mstore(buf, result)
    mod.return_(buf, IRLiteral(32))

    bytecode = mod.compile()
    assert isinstance(bytecode, bytes)
    assert len(bytecode) > 0


def test_simple_add_bytecode_result():
    """Compile an ADD and verify the EVM returns the correct value."""
    mod = ModuleBuilder("add_test")
    result = mod.add(IRLiteral(7), IRLiteral(11))
    buf = mod.alloca(32)
    mod.mstore(buf, result)
    mod.return_(buf, IRLiteral(32))

    bytecode = mod.compile()
    evm_result = mini_evm(bytecode)
    assert not evm_result.is_error
    assert int.from_bytes(evm_result.output, "big") == 18


def test_simple_mul_bytecode_result():
    mod = ModuleBuilder("mul_test")
    result = mod.mul(IRLiteral(6), IRLiteral(7))
    buf = mod.alloca(32)
    mod.mstore(buf, result)
    mod.return_(buf, IRLiteral(32))

    bytecode = mod.compile()
    evm_result = mini_evm(bytecode)
    assert not evm_result.is_error
    assert int.from_bytes(evm_result.output, "big") == 42


# ---------------------------------------------------------------------------
# 4. Module merging — no label collisions
# ---------------------------------------------------------------------------


def test_merge_two_modules_no_collision():
    """Merging two modules with different prefixes must not raise."""
    mod_a = ModuleBuilder("mod_a")
    mod_a.stop()

    mod_b = ModuleBuilder("mod_b")
    mod_b.stop()

    main = ModuleBuilder("main")
    result = main.add(IRLiteral(2), IRLiteral(3))
    buf = main.alloca(32)
    main.mstore(buf, result)
    main.return_(buf, IRLiteral(32))
    main.merge(mod_a.ctx, mod_b.ctx)

    # All three modules' functions should be present
    fn_names = [str(name) for name in main.ctx.functions]
    assert any("main.main" in n for n in fn_names)
    assert any("mod_a.main" in n for n in fn_names)
    assert any("mod_b.main" in n for n in fn_names)


def test_merge_produces_correct_result():
    """After merging two modules, the combined bytecode runs correctly."""
    mod_a = ModuleBuilder("mod_a")
    mod_a.stop()

    mod_b = ModuleBuilder("mod_b")
    mod_b.stop()

    main = ModuleBuilder("main")
    result = main.mul(IRLiteral(5), IRLiteral(8))
    buf = main.alloca(32)
    main.mstore(buf, result)
    main.return_(buf, IRLiteral(32))
    main.merge(mod_a.ctx, mod_b.ctx)

    bytecode = main.compile()
    evm_result = mini_evm(bytecode)
    assert not evm_result.is_error
    assert int.from_bytes(evm_result.output, "big") == 40


# ---------------------------------------------------------------------------
# 5. Cross-module function calls (correct param / ret calling convention)
#
# Venom IR internal-function calling convention:
#   - Caller: invoke(target_label, [arg0, arg1, ...], returns=N)
#   - Callee: last param() receives the return PC; preceding param()s get args
#   - Callee returns: ret(*values, return_pc)   ← return_pc is always last
# ---------------------------------------------------------------------------


def test_cross_module_invoke():
    """mod_a exposes 'compute(x, y) -> x+y'; main invokes it cross-module."""
    # --- Module A: defines compute ---
    mod_a = ModuleBuilder("mod_a")
    mod_a.stop()  # terminate the default entry; we only use mod_a.compute

    fn_compute = mod_a.create_function("compute")
    mod_a.set_block(fn_compute.entry)
    x = mod_a.param()          # first argument
    y = mod_a.param()          # second argument
    return_pc = mod_a.param()  # return-PC — always last param
    result = mod_a.add(x, y)
    mod_a.ret(result, return_pc)

    # --- Main module: calls mod_a.compute(10, 20) ---
    main = ModuleBuilder("main")
    rets = main.invoke(IRLabel("mod_a.compute"), [IRLiteral(10), IRLiteral(20)], returns=1)
    buf = main.alloca(32)
    main.mstore(buf, rets[0])
    main.return_(buf, IRLiteral(32))
    main.merge(mod_a.ctx)

    bytecode = main.compile()
    evm_result = mini_evm(bytecode)
    assert not evm_result.is_error
    assert int.from_bytes(evm_result.output, "big") == 30


def test_cross_module_invoke_mul():
    """mod_a.multiply(x, y) -> x*y; called from main."""
    mod_a = ModuleBuilder("mod_a")
    mod_a.stop()

    fn_mul = mod_a.create_function("multiply")
    mod_a.set_block(fn_mul.entry)
    a = mod_a.param()
    b = mod_a.param()
    ret_pc = mod_a.param()
    product = mod_a.mul(a, b)
    mod_a.ret(product, ret_pc)

    main = ModuleBuilder("main")
    rets = main.invoke(IRLabel("mod_a.multiply"), [IRLiteral(6), IRLiteral(7)], returns=1)
    buf = main.alloca(32)
    main.mstore(buf, rets[0])
    main.return_(buf, IRLiteral(32))
    main.merge(mod_a.ctx)

    bytecode = main.compile()
    evm_result = mini_evm(bytecode)
    assert not evm_result.is_error
    assert int.from_bytes(evm_result.output, "big") == 42


# ---------------------------------------------------------------------------
# 6. Data section access via offset + codecopy
#
# NOTE: Use offset(literal, label) + codecopy to read data sections.
# Do NOT use dload, which adds code_end (total bytecode size) instead of the
# code-only size, resulting in an out-of-bounds CODECOPY.
# ---------------------------------------------------------------------------


def test_data_section_read():
    """Store a known value in a data section and read it back at runtime."""
    mod = ModuleBuilder("data_test")

    mod.append_data_section("table")
    # Store 42 as a 32-byte big-endian word
    mod.append_data_item(b"\x00" * 31 + b"\x2a")

    # Read 32 bytes from the table via offset + codecopy
    src = mod.offset(IRLiteral(0), mod.named_label("table"))
    buf = mod.alloca(32)
    mod.codecopy(buf, src, IRLiteral(32))
    val = mod.mload(buf)
    out = mod.alloca(32)
    mod.mstore(out, val)
    mod.return_(out, IRLiteral(32))

    bytecode = mod.compile()
    evm_result = mini_evm(bytecode)
    assert not evm_result.is_error
    assert int.from_bytes(evm_result.output, "big") == 42


def test_data_section_second_item():
    """Access the second 32-byte word in a data section."""
    mod = ModuleBuilder("data2")

    mod.append_data_section("lut")
    mod.append_data_item(b"\x00" * 31 + b"\x01")   # word 0 = 1
    mod.append_data_item(b"\x00" * 31 + b"\x63")   # word 1 = 99

    # Read the second word: offset = 32 bytes into the table
    src = mod.offset(IRLiteral(32), mod.named_label("lut"))
    buf = mod.alloca(32)
    mod.codecopy(buf, src, IRLiteral(32))
    val = mod.mload(buf)
    out = mod.alloca(32)
    mod.mstore(out, val)
    mod.return_(out, IRLiteral(32))

    bytecode = mod.compile()
    evm_result = mini_evm(bytecode)
    assert not evm_result.is_error
    assert int.from_bytes(evm_result.output, "big") == 99


def test_two_modules_separate_data_sections():
    """Each module has its own data section; both are accessible after merge."""
    # Module A: has data section with value 11
    mod_a = ModuleBuilder("mod_a")
    mod_a.append_data_section("data")
    mod_a.append_data_item(b"\x00" * 31 + b"\x0b")  # 11
    mod_a.stop()  # mod_a just declares data; main reads it

    # Main: reads from mod_a.data and returns it
    main = ModuleBuilder("main")
    src = main.offset(IRLiteral(0), IRLabel("mod_a.data"))
    buf = main.alloca(32)
    main.codecopy(buf, src, IRLiteral(32))
    val = main.mload(buf)
    out = main.alloca(32)
    main.mstore(out, val)
    main.return_(out, IRLiteral(32))
    main.merge(mod_a.ctx)

    bytecode = main.compile()
    evm_result = mini_evm(bytecode)
    assert not evm_result.is_error
    assert int.from_bytes(evm_result.output, "big") == 11
