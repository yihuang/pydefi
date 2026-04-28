"""Backwards-compat shim for the merged ``Program`` / ``ProgramContext``.

The ``Program`` class was folded into :class:`pydefi.vm.context.ProgramContext`
in commit 07f59c9 — :class:`Program` is now an alias for ``ProgramContext`` and
this module simply re-exports it (and :data:`Value`) so callers using the old
``from pydefi.vm.program import Program`` import still work.

Prefer ``from pydefi.vm import ProgramContext`` (or ``from pydefi.vm.context``)
in new code.
"""

from __future__ import annotations

from pydefi.vm.context import Program, ProgramContext, Value, ValueLike

__all__ = ["Program", "ProgramContext", "Value", "ValueLike"]
