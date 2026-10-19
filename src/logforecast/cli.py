"""CLI para tareas batch (ECS/Fargate o local): python -m logforecast.cli <comando>."""
from __future__ import annotations

import argparse
import json
import logging

from logforecast import io, pipeline
from logforecast.config import Settings
from logforecast.quality import check_quality


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(prog="logforecast")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("quality", help="reporte de calidad de datos")
    sub.add_parser("train", help="entrena y persiste modelo + estado del gate")
    args = parser.parse_args(argv)

    settings = Settings.from_env()
    if args.cmd == "quality":
        rep = check_quality(
            io.load_orders(settings), io.load_forecasts(settings),
            io.load_capacity(settings), io.load_plan(settings),
        )
        print(rep.to_string(index=False))
    elif args.cmd == "train":
        print(json.dumps(pipeline.train(settings), indent=2))


if __name__ == "__main__":
    main()
