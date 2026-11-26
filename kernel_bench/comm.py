"""A launch over a group of processes: the ranks a collective cell needs.

`kernel_bench/ARCHITECTURE.md`, "How a collective is measured", is the
contract; in short:

* The driver process is rank 0 on card 0. `Group.of(world)` spawns ranks
  1..world-1 on the next visible cards the first time a cell of that group
  size comes up and keeps them until a one-card cell comes up
  (`Group.release`) or the run ends. NCCL carries the data; the commands go
  through the group's `TCPStore`, which is not an aten op, so a launch trace
  around a call sees the collective and nothing else.
* A provider's `build` is ONE rank's launch (it asks torch.distributed
  which rank it is). `Group.case` builds it on every rank: rank 0 from the
  driver's inputs, each peer the same provider, found in the registry by its
  `kernel_impl`, from the cell's canonical inputs regenerated from the
  cell's own seed. The inputs each rank holds its own copy of
  (`RefSpec.per_rank`) are turned by `rank_tensor`.
* Every rank makes the same calls the same number of times: the case's
  `run` is one untimed launch on every rank, and `Group.timed` puts every
  rank into the one clock (`timing.time_callable(..., group=)`), whose
  sizing reads the slowest rank (`agree`), whose windows open on a GPU
  meeting point (`align`) and whose samples are the slowest rank's
  (`slowest`).
* Every command is answered by every peer. A peer that died, or a command
  that failed anywhere, marks the group broken: it is shut down and the next
  collective cell starts a fresh one, rather than reusing a communicator
  that may hold a half-finished collective.

Nothing here decides what a cell needs: the cell's `n_comm_ranks` says the
group, `Provider.refusals` says whether the env has the cards.
"""
from __future__ import annotations

import atexit
import dataclasses
import os
import pickle
from contextlib import contextmanager
from datetime import timedelta
from typing import Any, ClassVar, Dict, Iterator, List, Optional

#: How long a rank waits for the others inside one NCCL call before the
#: communicator is torn down. A rank that died mid-call would otherwise hang
#: its peers forever; the supervisor's stall watchdog is the backstop, this is
#: the first answer.
NCCL_TIMEOUT = timedelta(minutes=5)

#: How long a peer waits for the next command. Long: a run measures a model's
#: one-card cells between two collective ones, and an idle peer is fine.
CONTROL_TIMEOUT = timedelta(hours=24)

#: How often rank 0, waiting on a peer's answer, looks whether it is alive.
_POLL = timedelta(seconds=2)


def rank_tensor(t: Any, rank: int, world: int) -> Any:
    """Rank `rank`'s copy of a canonical buffer in a group of `world`: the
    same values, rotated along the flattened buffer. Different data on every
    rank — so a kernel that skipped the exchange does not agree with the
    oracle — and rebuildable by rank 0 from its own copy, which is what lets
    the oracle compute every rank's contribution locally. Rank 0's is the
    canonical buffer itself.

    The rotation is `rank * (k + 1)` elements, `k` an equal shard
    (`n // world`): rank r's r-th shard then starts at element -r of the
    canonical buffer, so no two ranks send the same shard — a rotation by
    whole shards would hand every rank of an all-gather the same one, and a
    launch that copied its own shard everywhere would pass — and no two
    ranks' first slices (what an all-to-all hands rank 0) coincide."""
    n = t.numel()
    if rank == 0 or n <= 1:
        return t
    shift = (rank * (n // world + 1)) % n
    return t.reshape(-1).roll(shift).reshape(t.shape)


#: How long every rank keeps its card busy around forming the communicator
#: (`_busy_card`): NCCL's setup takes a second or two, and the card must not
#: drop its link in the middle of it.
_BUSY_S = 5.0


def _busy_card() -> Any:
    """Keep this card busy, on a side stream, for `_BUSY_S`; returns the
    stream so the caller can wait it out once the communicator is formed.

    An idle card sits in its lowest power state with its PCIe link down to
    Gen1, and NCCL sizes its channels and protocols from the link speed it
    reads while forming a communicator: measured on the 4090 pair, a group
    formed with either card idle read the link at 3 GB/s instead of 24 and
    ran a 4096-element all-reduce at 29.7 us instead of 13.2. A card running
    a kernel holds its link up (Gen4 within 50 ms), and a serving framework
    forms its group with its cards busy loading weights, so that is the
    state to measure in. Every rank waits for the others first (`ready`),
    so no card idles at the rendezvous after being woken."""
    import torch

    from .timing import _hold_cycles
    side = torch.cuda.Stream()
    chunk = _hold_cycles(torch, 0.05)              # one 50 ms spin
    with torch.cuda.stream(side):
        for _ in range(int(_BUSY_S / 0.05)):
            torch.cuda._sleep(chunk)
    return side


def _ready(store: Any, rank: int, world: int) -> None:
    """Every rank of the group has started and is about to form it."""
    store.set(f"ready/{rank}", b"1")
    store.wait([f"ready/{r}" for r in range(world)])


def _said(e: BaseException) -> str:
    return f"{type(e).__name__}: {e}"


def _provider(signature: str, kernel_impl: str) -> Any:
    from .providers import all_providers, discover
    discover()
    return next(p for p in all_providers()
                if p.signature == signature and p.kernel_impl == kernel_impl)


class _Rank:
    """What every rank of a group shares: the meeting points the clock uses
    (`timing.RankGroup`) and the control channel, numbered commands and
    per-rank statuses in the group's store."""

    def __init__(self, rank: int, world: int, store: Any) -> None:
        import torch
        import torch.distributed as dist
        self.rank, self.world, self.store = rank, world, store
        self._dist = dist
        self._one = torch.zeros(1, dtype=torch.float32, device="cuda")
        self._seq = 0

    # --- timing.RankGroup -------------------------------------------------
    def agree(self, value: float) -> float:
        import torch
        t = torch.tensor([float(value)], dtype=torch.float64, device="cuda")
        self._dist.all_reduce(t, op=self._dist.ReduceOp.MAX)
        return float(t.item())

    def align(self) -> None:
        self._dist.all_reduce(self._one)

    def slowest(self, samples: List[float]) -> List[float]:
        import torch
        t = torch.tensor([float(x) for x in samples], dtype=torch.float64,
                         device="cuda")
        self._dist.all_reduce(t, op=self._dist.ReduceOp.MAX)
        return [float(x) for x in t.tolist()]

    # --- the control channel ---------------------------------------------
    def command(self, cmd: Optional[tuple] = None) -> tuple:
        """Rank 0 posts `cmd`; every other rank waits for it."""
        self._seq += 1
        key = f"cmd/{self._seq}"
        if self.rank == 0:
            self.store.set(key, pickle.dumps(cmd))
            return cmd
        return pickle.loads(self.store.get(key))

    def report(self, status: Optional[str]) -> None:
        """A peer's outcome of the last command (None = fine)."""
        self.store.set(f"status/{self._seq}/{self.rank}", pickle.dumps(status))


class Group(_Rank):
    """Rank 0 of a group of `world` processes, one per card; see the module
    docstring. At most one per process (one default process group)."""

    _open: ClassVar[Dict[int, "Group"]] = {}

    @classmethod
    def of(cls, world: int) -> "Group":
        group = cls._open.get(world)
        if group is not None and group.usable():
            return group
        cls.release()
        group = cls._open[world] = cls(world)
        return group

    @classmethod
    def release(cls) -> None:
        """Shut down whatever group is open: its peers leave their cards and
        its NCCL buffers leave card 0."""
        for group in list(cls._open.values()):
            group.shutdown()
        cls._open.clear()

    def __init__(self, world: int) -> None:
        import torch
        import torch.distributed as dist
        import torch.multiprocessing as mp
        if torch.cuda.device_count() < world:
            raise RuntimeError(f"a group of {world} needs {world} visible cards; "
                               f"this process sees {torch.cuda.device_count()}")
        if not dist.is_nccl_available():
            raise RuntimeError("this torch build has no NCCL")
        self._closed = False
        self._broken = False
        store = dist.TCPStore("127.0.0.1", 0, world, is_master=True,
                              timeout=CONTROL_TIMEOUT, wait_for_workers=False)
        ctx = mp.get_context("spawn")
        self._procs = [ctx.Process(target=_peer_main,
                                   args=(r, world, store.port, os.getpid()),
                                   daemon=True)
                       for r in range(1, world)]
        for p in self._procs:
            p.start()
        torch.cuda.set_device(0)
        _ready(store, 0, world)
        busy = _busy_card()
        # `device_id`: the communicator is formed now, while every card is
        # busy, not lazily at the first collective.
        dist.init_process_group("nccl", store=store, rank=0, world_size=world,
                                timeout=NCCL_TIMEOUT,
                                device_id=torch.device("cuda", 0))
        busy.synchronize()
        super().__init__(0, world, store)
        self.local: Optional[Any] = None
        atexit.register(self.shutdown)

    def usable(self) -> bool:
        return (not self._closed and not self._broken
                and all(p.is_alive() for p in self._procs))

    def _answer(self, r: int) -> Optional[str]:
        """Peer `r`'s answer to the last command, waiting while it is alive."""
        key = f"status/{self._seq}/{r}"
        proc = self._procs[r - 1]
        waited = timedelta(0)
        while True:
            try:
                self.store.wait([key], _POLL)
                break
            except Exception:
                waited += _POLL
                if not proc.is_alive():
                    self._broken = True
                    raise RuntimeError(
                        f"rank {r} died (exit {proc.exitcode}) before "
                        f"answering") from None
                if waited > NCCL_TIMEOUT + timedelta(minutes=1):
                    # Longer than any collective may take: whatever it is
                    # stuck in, this group is not coming back.
                    self._broken = True
                    raise RuntimeError(
                        f"rank {r} gave no answer in {waited}") from None
        out = pickle.loads(self.store.get(key))
        self.store.delete_key(key)
        return out

    def _expect_all(self, what: str, mine: Optional[str] = None) -> None:
        """Every peer's answer to the last command; raises when any rank
        (`mine` is rank 0's) failed, and the group is then broken."""
        said = [(0, mine)] + [(r, self._answer(r)) for r in range(1, self.world)]
        self.store.delete_key(f"cmd/{self._seq}")
        bad = [(r, s) for r, s in said if s]
        if bad:
            self._broken = True
            raise RuntimeError(f"{what}: " + "; ".join(
                f"rank {r}: {s}" for r, s in bad))

    def _settle(self, what: str, failed: Optional[BaseException]) -> None:
        """Collect every peer's answer to the last command. When rank 0's own
        part failed, that exception is the one that propagates (its type is
        what the driver classifies by: a timeout, an out-of-memory)."""
        if failed is None:
            self._expect_all(what)
            return
        try:
            self._expect_all(what, _said(failed))
        except RuntimeError:
            pass                           # rank 0's own exception says it
        raise failed

    def case(self, provider: Any, inv: Any, env: Any,
             inputs: Dict[str, Any]) -> Any:
        """The cell as rank 0 measures it: `provider` built on every rank,
        the untimed `run` launching on every rank, `group` this group (which
        is how the driver's clock gets the other ranks in, `timed`). Rank 0
        builds first: a cell it cannot build never reaches the peers."""
        from .providers import BenchCase
        self.local = None
        local = provider.build(inv, env, inputs)
        self.command(("open", provider.signature, provider.kernel_impl, inv))
        self._settle("building the other ranks", None)
        self.local = local

        def run():
            self.command(("run",))
            failed, out = None, None
            try:
                out = local.run()
            except BaseException as e:
                failed = e
            self._settle("launching on the other ranks", failed)
            return out
        return BenchCase(run=run, to_canonical=local.to_canonical, group=self)

    @contextmanager
    def timed(self) -> Iterator[Any]:
        """Every rank enters the clock now; yields rank 0's own launch for
        `time_callable(launch, group=self)`. If rank 0's clock raises, that
        exception is what propagates (a `TooSlowToMeasure` stays one, so the
        driver records a timeout, not a crash) once the peers have answered;
        a peer's own failure is raised when rank 0's clock succeeded."""
        self.command(("time",))
        try:
            yield self.local.run
        except BaseException as e:
            self._settle("timing the other ranks", e)
        self._settle("timing the other ranks", None)

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if all(p.is_alive() for p in self._procs):
                self.command(("exit",))
            self._dist.destroy_process_group()
        except Exception:
            pass
        for p in self._procs:
            p.join(timeout=10)
            if p.is_alive():
                p.kill()


def _die_with_parent() -> None:
    """Linux: SIGKILL this peer when the driver exits, however it exits — a
    peer outliving its rank 0 would sit on a card the next run locks."""
    try:
        import ctypes
        import signal
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG
    except Exception:
        pass


def _peer_main(rank: int, world: int, port: int, parent: int) -> None:
    _die_with_parent()
    if os.getppid() != parent:
        return                                  # the driver is already gone
    import torch
    import torch.distributed as dist

    from .env import detect_env
    from .refs import get_ref, input_generator
    from .timing import time_callable

    torch.cuda.set_device(rank)
    store = dist.TCPStore("127.0.0.1", port, world, is_master=False,
                          timeout=CONTROL_TIMEOUT)
    _ready(store, rank, world)
    busy = _busy_card()
    dist.init_process_group("nccl", store=store, rank=rank, world_size=world,
                            timeout=NCCL_TIMEOUT,
                            device_id=torch.device("cuda", rank))
    busy.synchronize()
    me = _Rank(rank, world, store)
    env = dataclasses.replace(detect_env(), devices=world)
    local: Optional[Any] = None
    while True:
        cmd = me.command()
        kind = cmd[0]
        if kind == "exit":
            break
        status = None
        try:
            if kind == "open":
                local = None
                torch.cuda.empty_cache()
                _, signature, kernel_impl, inv = cmd
                provider = _provider(signature, kernel_impl)
                spec = get_ref(inv.signature)
                inputs = spec.make_inputs(inv, "cuda", input_generator(inv, "cuda"))
                for name in spec.per_rank:
                    inputs[name] = rank_tensor(inputs[name], rank, world)
                inputs = provider.convert_inputs(inv, inputs)
                local = provider.build(inv, env, inputs)
            elif kind == "run":
                local.run()
            elif kind == "time":
                time_callable(local.run, group=me)
        except Exception as e:
            status = _said(e)
        me.report(status)
    dist.destroy_process_group()
