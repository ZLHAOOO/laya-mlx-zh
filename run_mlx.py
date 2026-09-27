"""Run the zh_short_commands benchmark against an MLX checkpoint (laya-mlx-zh).

Mirrors run.py exactly: same cases, same prompts, same temperature/regime and
judging code (imported from run.py itself). Only the scoring engine differs --
``laya_mlx`` (MLX, Apple silicon) instead of the torch ``laya`` package. A
sys.modules shim routes the official code's ``from laya.common import ...``
to the MLX port so the judging path stays byte-identical.

    <laya-venv-python> run_mlx.py --stub            # offline pipeline check
    <laya-venv-python> run_mlx.py --checkpoint <ckpt-dir> --out <dir>

Clamped temperatures only: the MLX checkpoint carries no raw (unclamped)
temperature table, so this runner scores under the same regime as the
committed v1 multilingual results (``unclamped: false``).
"""
import argparse
import json
import os
import sys
import types
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# --- shim: official judging code does `from laya.common import QTYPES/temp_bucket`;
# route it to the MLX port so no torch laya install is needed.
import laya_mlx  # noqa: E402

_shim = types.ModuleType("laya")
_shim.common = laya_mlx.common
sys.modules.setdefault("laya", _shim)
sys.modules.setdefault("laya.common", laya_mlx.common)

import prompts  # noqa: E402
from research.eval.laya_eval import (  # noqa: E402
    ece, macro_f1, softmax_t, summarise, temperature_for,
)
from research.benchmarks.zh_short_commands.run import (  # noqa: E402
    CASES_PATH, choice_records, load_cases, noul_records, sha256_file, stub_logits,
)


def score_cases_mlx(agent, cases):
    """Raw marker logits per case, mirroring laya_eval.score_cases on MLX.

    Same construction as official score_cases: build every sequence, collate,
    run the model once per batch, and slice each row to its own marker count.
    No temperature is applied here (the caller does softmax_t under each regime).
    """
    import numpy as np
    from laya_mlx.agent import collate_items

    items = []
    for state, questions in cases:
        got, _ = agent.prepare(state, questions)
        items.extend(got)
    out = []
    bs = agent.batch_size
    for s in range(0, len(items), bs):
        chunk = items[s:s + bs]
        batch = collate_items(
            chunk,
            agent.tok.pad_token_id,
            pad_to_multiple=getattr(agent, "pad_to_multiple", None),
            max_length=agent.cfg.get("max_len", 512),
        )
        raw = agent.forward(batch)
        logits = np.asarray(raw[0] if isinstance(raw, tuple) else raw)
        for row, it in enumerate(chunk):
            out.append(logits[row, :len(it["markers"])])
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="zh-short-commands-mlx")
    parser.add_argument("--checkpoint",
                        default=os.path.expanduser("~/.pi/agent/tools/laya-zh/ckpt-zh-v4-mlx"),
                        help="local MLX checkpoint directory (laya_mlx.load-able)")
    parser.add_argument("--configs", default=",".join(prompts.CONFIGS),
                        help="comma list of configs to run")
    parser.add_argument("--out", default=None, help="output directory for report.json")
    parser.add_argument("--stub", action="store_true",
                        help="skip the checkpoint and score with a fixed pseudo-random vector")
    args = parser.parse_args(argv)

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    for config in configs:
        if config not in prompts.CONFIGS:
            parser.error("unknown config %r; known: %s" % (config, ", ".join(prompts.CONFIGS)))

    cases = load_cases()
    print("cases: %d  configs: %s" % (len(cases), ", ".join(configs)))

    agent, info = None, {"engine": "stub"}
    if not args.stub:
        import mlx.core as mx
        agent = laya_mlx.load(args.checkpoint)
        info = {
            "engine": "laya_mlx",
            "laya_mlx": getattr(laya_mlx, "__version__", "unknown"),
            "mlx": getattr(mx, "__version__", "unknown"),
            "checkpoint": args.checkpoint,
            "device": str(getattr(agent, "device", None)),
        }

    report, summary, all_cases = {}, {}, []
    for config in configs:
        pairs = prompts.cases_for(config, [c["text"] for c in cases])
        logits = stub_logits(pairs) if args.stub else score_cases_mlx(agent, pairs)
        if config in prompts.NOUL_CONFIGS:
            out = noul_records(config, cases, logits, agent, False)
        else:
            out = choice_records(config, cases, logits, agent, False)
        report[config] = out["report"]
        report[config]["cases"] = out["cases"]
        summary[config] = {"accuracy": out["report"].get("accuracy"),
                           "mean_confidence": out["report"].get("mean_confidence")}
        line = "  %-18s acc=%s" % (config, out["report"].get("accuracy"))
        extras = []
        if out["report"].get("macro_f1") is not None:
            extras.append("macro_f1=%s" % out["report"]["macro_f1"])
        if out["report"].get("ece") is not None:
            extras.append("ece=%s" % out["report"]["ece"])
        if out["report"].get("std_confidence") is not None:
            extras.append("std_conf=%s" % out["report"]["std_confidence"])
        extras.append("mean_conf=%s" % out["report"].get("mean_confidence"))
        print("%s  %s" % (line, " ".join(extras)))
        all_cases.extend(out["cases"])

    document = {
        "config": {
            "benchmark": "zh_short_commands",
            "cases_file": os.path.relpath(CASES_PATH, ROOT).replace(os.sep, "/"),
            "cases_sha256": sha256_file(CASES_PATH),
            "prompts_sha256": sha256_file(os.path.join(HERE, "prompts.py")),
            "configs": configs,
            "runner": "run_mlx.py",
            "unclamped": False,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **info,
            **({} if args.stub else {
                "weight_sha256": sha256_file(os.path.join(args.checkpoint, "model.safetensors")),
                "weight_bytes": os.path.getsize(os.path.join(args.checkpoint, "model.safetensors")),
            }),
        },
        "report": report,
        "summary": summary,
        "cases": all_cases,
    }

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        target = os.path.join(args.out, "report.json")
        with open(target, "w", encoding="utf-8", newline="\n") as f:
            json.dump(document, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print("wrote %s" % target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
