from __future__ import annotations

import argparse
from pathlib import Path

from market_engine import __version__
from market_engine.config import PROJECT_ROOT
from market_engine.evaluation.universe_intelligence import (
    build_universe_intelligence,
    export_universe_intelligence,
)


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _report(args: argparse.Namespace) -> int:
    source = _resolve(args.input)
    result = build_universe_intelligence(
        input_path=source,
        preferred_sheet=args.sheet,
        expected_tickers=args.expected_tickers,
        engine_version=args.engine_version,
        regime_label=args.regime_label,
        top_n=args.top,
    )
    paths = export_universe_intelligence(result, _resolve(args.output_dir), args.label)
    print("\nUNIVERSE INTELLIGENCE")
    print(result.executive_summary.to_string(index=False))
    print("\nCOBERTURA")
    print(result.coverage.to_string(index=False))
    for kind, path in paths.items():
        print(f"{kind}: {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="market-universe")
    parser.add_argument("--input", required=True)
    parser.add_argument("--sheet", default="Mejor_Memoria")
    parser.add_argument("--label", default="UNIVERSE_INTELLIGENCE_REPORT")
    parser.add_argument("--expected-tickers", type=int, default=260)
    parser.add_argument("--engine-version", default=__version__)
    parser.add_argument("--regime-label", default="REGIME_2026")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--output-dir", default="reports/universe")
    parser.set_defaults(handler=_report)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
