from __future__ import annotations

import argparse
import json
from pathlib import Path
import secrets


def main(argv=None):
    parser = argparse.ArgumentParser(prog="covbench")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="evaluate one frozen candidate on paired synthetic draws")
    run.add_argument("--candidate", type=Path, required=True)
    run.add_argument("--suite", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--root-seed", type=int, default=None, help="omit for a fresh evaluator seed; never passed to candidate")
    run.add_argument("--replications", type=int, default=3)
    run.add_argument("--timing-repeats", type=int, default=3)
    run.add_argument("--mode", choices=("offline", "online", "both"), default=None)
    run.add_argument("--threads", type=int, default=1)
    run.add_argument("--timeout", type=float, default=60.)
    run.add_argument("--memory-mib", type=float, default=1024., help="worker RSS ceiling; zero disables")
    run.add_argument("--batch-size", type=int, default=128, help="online delivery batch; use 1 for scalar arrival latency")
    frontier = commands.add_parser("frontier", help="derive nondominated records within identical workload contracts")
    frontier.add_argument("reports", type=Path, nargs="+")
    frontier.add_argument("--objectives", nargs="+", default=["completed_ns", "covariance_nmse"])
    frontier.add_argument("--output", type=Path)
    grade = commands.add_parser("regrade", help="recalculate losses from saved timed outputs without executing a candidate")
    grade.add_argument("report", type=Path)
    grade.add_argument("--output", type=Path, required=True, help="new report file beside the original")
    args = parser.parse_args(argv)
    if args.command == "run":
        from .runner import evaluate
        modes = None if args.mode is None else (["offline", "online"] if args.mode == "both" else [args.mode])
        report = evaluate(args.candidate, args.suite, args.output,
                          root_seed=args.root_seed if args.root_seed is not None else secrets.randbits(64),
                          replications=args.replications, timing_repeats=args.timing_repeats,
                          modes=modes, threads=args.threads, timeout=args.timeout,
                          memory_mib=args.memory_mib, batch_size=args.batch_size)
        print(json.dumps(dict(results=str(args.output / "results.json"), root_seed=report["root_seed"],
                              records=len(report["records"]), failed=sum(r["status"]=="failed" for r in report["records"]))))
        return 0
    if args.command == "regrade":
        from .runner import regrade
        regrade(args.report, args.output)
        print(args.output)
        return 0
    from .pareto import frontiers
    records = []
    for path in args.reports:
        records.extend(json.loads(path.read_text())["records"])
    result = frontiers(records, args.objectives,
                       comparison_keys=["comparison_key", "scenario_id", "mode", "track", "checkpoint"])
    text = json.dumps(result, indent=2, allow_nan=False)+"\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
