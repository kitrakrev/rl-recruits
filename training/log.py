"""
Shared lightweight logger for the training pipeline.

Usage:
    from training.log import log
    log.info("rollout", "Starting episode 3")
    log.warn("reinforce", "Episode failed", exc=e)
    log.err("rollout", "env.step() crashed", exc=e, fatal=True)
"""
from __future__ import annotations

import sys
import traceback
from datetime import datetime


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


class _Logger:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    RED    = "\033[31m"
    YELLOW = "\033[33m"
    GREEN  = "\033[32m"
    CYAN   = "\033[36m"
    GREY   = "\033[90m"

    def _write(self, level: str, color: str, src: str, msg: str,
               exc: BaseException | None, extra: dict) -> None:
        ts   = _ts()
        tag  = f"[{level}][{src}]"
        line = f"{self.GREY}{ts}{self.RESET} {color}{self.BOLD}{tag}{self.RESET} {msg}"
        print(line, flush=True)
        if extra:
            for k, v in extra.items():
                print(f"  {self.GREY}{k}:{self.RESET} {v}", flush=True)
        if exc is not None:
            tb = traceback.format_exc()
            print(f"  {self.RED}--- traceback ---{self.RESET}", flush=True)
            for tb_line in tb.splitlines():
                print(f"  {self.RED}{tb_line}{self.RESET}", flush=True)

    def info(self, src: str, msg: str, **extra) -> None:
        self._write("INFO", self.CYAN, src, msg, None, extra)

    def ok(self, src: str, msg: str, **extra) -> None:
        self._write(" OK ", self.GREEN, src, msg, None, extra)

    def warn(self, src: str, msg: str, exc: BaseException | None = None, **extra) -> None:
        self._write("WARN", self.YELLOW, src, msg, exc, extra)

    def err(self, src: str, msg: str, exc: BaseException | None = None,
            fatal: bool = False, **extra) -> None:
        self._write("ERR ", self.RED, src, msg, exc, extra)
        if fatal:
            sys.exit(1)

    def step(self, week: int, turn: int, tool: str, status: str,
             cash: float, profit: float, delta: float, reward: float) -> None:
        color = self.GREEN if delta > 0 else (self.RED if delta < 0 else self.GREY)
        print(
            f"  {self.GREY}{_ts()}{self.RESET}  "
            f"W{week:02d}T{turn:02d}  "
            f"{color}{tool:<26}{self.RESET}  "
            f"{status:<7}  "
            f"cash=${cash:>9,.0f}  "
            f"profit=${profit:>9,.0f}  "
            f"Δ={delta:>+8,.0f}  "
            f"rew={reward:>+8.2f}",
            flush=True,
        )

    def parse_fail(self, week: int, turn: int, raw: str, cash: float, profit: float) -> None:
        snippet = raw.strip().replace("\n", " ")[:140]
        print(
            f"  {self.GREY}{_ts()}{self.RESET}  "
            f"W{week:02d}T{turn:02d}  "
            f"{self.YELLOW}[parse-fail]              {self.RESET}  "
            f"FAILED   "
            f"cash=${cash:>9,.0f}  "
            f"profit=${profit:>9,.0f}  "
            f"raw={self.GREY}\"{snippet}\"{self.RESET}",
            flush=True,
        )


log = _Logger()
