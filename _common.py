"""
scenarios/_common.py
====================
Shared presentation helpers (Palette + print primitives).

Extracted from the previous monolithic demo_runner.py so scenario modules
can import them without pulling in the whole CLI.
"""

from __future__ import annotations


class Palette:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def _wrap(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.enabled else text

    def bold(self, t: str) -> str: return self._wrap("1", t)
    def dim(self, t: str) -> str: return self._wrap("2", t)
    def red(self, t: str) -> str: return self._wrap("31", t)
    def green(self, t: str) -> str: return self._wrap("32", t)
    def yellow(self, t: str) -> str: return self._wrap("33", t)
    def blue(self, t: str) -> str: return self._wrap("34", t)
    def magenta(self, t: str) -> str: return self._wrap("35", t)
    def cyan(self, t: str) -> str: return self._wrap("36", t)


def banner(pal: Palette, title: str) -> None:
    line = "═" * 72
    print()
    print(pal.bold(pal.cyan(line)))
    print(pal.bold(pal.cyan(f"  {title}")))
    print(pal.bold(pal.cyan(line)))


def step(pal: Palette, label: str, detail: str = "") -> None:
    print(f"{pal.blue('▸')} {pal.bold(label)}" + (f"  {pal.dim(detail)}" if detail else ""))


def ok(pal: Palette, text: str) -> None:
    print(f"  {pal.green('✓')} {text}")


def warn(pal: Palette, text: str) -> None:
    print(f"  {pal.yellow('⚠')} {text}")


def err(pal: Palette, text: str) -> None:
    print(f"  {pal.red('✗')} {text}")


def kv(pal: Palette, key: str, value: str) -> None:
    print(f"    {pal.dim(key + ':')} {value}")
