"""Command line (recommended for the full 50-episode run).

  python -m anchorcut.cli run    path/to/pack.zip [--label duke]   parse + full pipeline
  python -m anchorcut.cli resume <run_id>                           continue after a pause / crash
  python -m anchorcut.cli rebuild <run_id>                          re-verify + re-tier, no LLM calls
  python -m anchorcut.cli status <run_id>
  python -m anchorcut.cli parse  path/to/pack.zip                   parse only, print issues
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from . import pipeline
from .parser import load_pack


def _p(stage, done, total, msg):
    print(f"[{stage:8s}] {done:>3}/{total:<3} {msg}", flush=True)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(prog="anchorcut")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("zip"); r.add_argument("--label", default="")
    for name in ("resume", "rebuild", "status"):
        sub.add_parser(name).add_argument("run_id")
    sub.add_parser("parse").add_argument("zip")
    a = ap.parse_args(argv)

    if a.cmd == "parse":
        eps, pack_issues = load_pack(Path(a.zip))
        print(f"{len(eps)} episodes, {sum(len(e.beats) for e in eps)} beats")
        for i in pack_issues + [i for e in eps for i in e.issues]:
            print(f"  {i.file}:{i.line_no} {i.reason} | {i.line}")
        return 0
    if a.cmd == "status":
        print(json.dumps(pipeline.status(a.run_id), indent=2))
        return 0
    if a.cmd == "rebuild":
        promos = pipeline.rebuild(a.run_id)
        print(f"{sum(1 for p in promos if p.post_slot)} scheduled / {len(promos)} candidates")
        return 0
    run_id = pipeline.ingest(Path(a.zip).read_bytes(), a.label or Path(a.zip).stem) if a.cmd == "run" else a.run_id
    print(f"run_id: {run_id}")
    pipeline.run(run_id, progress=_p)
    st = pipeline.status(run_id)
    print(f"state={st.get('state')}  error={st.get('last_error')}")
    print(f"results: {pipeline.run_dir(run_id) / 'results'}")
    if st.get("state") == "paused":
        print(f"resume later with:  python -m anchorcut.cli resume {run_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
