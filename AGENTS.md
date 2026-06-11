# AGENTS.md

Guidelines for agents (and humans) contributing to pydefi.

## Project conventions

Detailed rules live in `.github/agents/` — read those first:

| File | Covers |
|------|--------|
| `.github/agents/conversion-conventions.md` | `Address` / `Hash` type aliases, conversions at peripheries, padding helpers |
| `.github/agents/abi-patterns.md` | Human-readable ABI definitions, `ABIStruct`, `ALL_CAPS` module constants |
| `.github/agents/contract-interactions.md` | `eth_contract` usage, calldata-first, `multicall`, reusing existing ABIs |

This file only states principles that aren't already covered there.

## Design principles

1. **Pythonic.** Follow `python -c "import this"`. Flat over nested. Explicit over implicit. Namespaces over class hierarchies.

2. **Imports at module top level.** All ``import`` / ``from`` statements belong at the top of the module — never inside a function, method, or conditional body. The only exceptions are:

   * To workaround circular imports.
   * ``TYPE_CHECKING`` guards for forward references that would otherwise cause circular imports.
   * ``if sys.version_info`` guards for version-dependent stdlib imports.

3. **Modules as namespaces, not classes as abstractions.** Prefer module-level pure functions over class methods. Use classes only when stateful composition is genuinely needed (e.g. `ProgramContext` in `pydefi/vm/`). Avoid abstract base classes that exist only to define a method signature — Python's duck typing and `typing.Protocol` are lighter-weight alternatives.

4. **Human-readable ABI and data model definitions.** ABIs live in `pydefi/abi/` as `ALL_CAPS` `Contract` objects or `ABIStruct` subclasses (see `.github/agents/abi-patterns.md`). Data models (`Token`, `SwapRoute`, `BridgeQuote`, …) are frozen `@dataclass` in `pydefi/types.py`. Keep types.py to pure data — move builder/stateful logic elsewhere.

5. **Thin abstractions on top of primitives.** Build pure functions first. Only add a class wrapper when state or lifecycle management justifies it. A convenience class that delegates to module-level functions is fine.

6. **Pure functions are independently testable.** Math helpers (`apply_slippage`, constant-product formulas, rate conversions, calldata builders) belong at module level as pure functions — not as `@staticmethod` inside a protocol class. They should be importable and testable without instantiating a client.

7. **No duplicate code.** A single `pydefi/_math.py` for shared arithmetic (`apply_slippage`, `slippage_to_fraction`, `slippage_to_percent`), not copy-pasted into every base class.
