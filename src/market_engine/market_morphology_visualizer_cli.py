from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from openpyxl.chart import LineChart, Reference
from openpyxl.chart.axis import DateAxis
from openpyxl.styles import Font

from market_engine.cli import _excel_safe
from market_engine.config import PROJECT_ROOT
from market_engine.evaluation.sample_morphology_fidelity import compare_sample_to_reference


def _resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read(path: Path, sheet: str | None) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path, sheet_name=sheet or 0)


def _add_line_chart(
    worksheet,
    *,
    title: str,
    y_title: str,
    data_columns: tuple[int, ...],
    anchor: str,
    max_row: int,
) -> None:
    chart = LineChart()
    chart.title = title
    chart.y_axis.title = y_title
    chart.x_axis = DateAxis(crosses="autoZero")
    chart.x_axis.title = "Fecha"
    chart.height = 10
    chart.width = 22

    dates = Reference(worksheet, min_col=1, min_row=2, max_row=max_row)
    for column in data_columns:
        values = Reference(worksheet, min_col=column, min_row=1, max_row=max_row)
        chart.add_data(values, titles_from_data=True)
    chart.set_categories(dates)
    worksheet.add_chart(chart, anchor)


def export_visual_comparison(
    sample: pd.DataFrame,
    reference: pd.DataFrame,
    output: Path,
    *,
    smoothing_window: int = 5,
    rolling_window: int = 20,
    sample_label: str = "NASDAQ30",
    reference_label: str = "NASDAQ100",
) -> None:
    result = compare_sample_to_reference(
        sample,
        reference,
        smoothing_window=smoothing_window,
        rolling_window=rolling_window,
    )

    aligned = result.aligned_geometry.copy()
    visual = aligned[[
        "date",
        "sample_market_cap_index",
        "reference_market_cap_index",
        "sample_drawdown_pct",
        "reference_drawdown_pct",
        "sample_smoothed_slope",
        "reference_smoothed_slope",
        "sample_acceleration",
        "reference_acceleration",
        "index_error",
        "drawdown_error",
    ]].rename(columns={
        "sample_market_cap_index": f"{sample_label}_index",
        "reference_market_cap_index": f"{reference_label}_index",
        "sample_drawdown_pct": f"{sample_label}_drawdown_pct",
        "reference_drawdown_pct": f"{reference_label}_drawdown_pct",
        "sample_smoothed_slope": f"{sample_label}_slope",
        "reference_smoothed_slope": f"{reference_label}_slope",
        "sample_acceleration": f"{sample_label}_acceleration",
        "reference_acceleration": f"{reference_label}_acceleration",
    })

    output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        _excel_safe(result.summary).to_excel(writer, sheet_name="Resumen", index=False)
        _excel_safe(result.metrics).to_excel(writer, sheet_name="Metricas", index=False)
        _excel_safe(visual).to_excel(writer, sheet_name="Comparacion_Visual", index=False)
        _excel_safe(result.rolling_fidelity).to_excel(writer, sheet_name="Fidelidad_Movil", index=False)

        workbook = writer.book
        worksheet = workbook["Comparacion_Visual"]
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions
        for cell in worksheet[1]:
            cell.font = Font(bold=True)

        max_row = worksheet.max_row
        if max_row >= 3:
            _add_line_chart(
                worksheet,
                title=f"Geometría normalizada: {sample_label} vs {reference_label}",
                y_title="Índice base 100",
                data_columns=(2, 3),
                anchor="M2",
                max_row=max_row,
            )
            _add_line_chart(
                worksheet,
                title="Drawdown comparado",
                y_title="Drawdown (%)",
                data_columns=(4, 5),
                anchor="M22",
                max_row=max_row,
            )
            _add_line_chart(
                worksheet,
                title="Pendiente suavizada comparada",
                y_title="Pendiente logarítmica",
                data_columns=(6, 7),
                anchor="M42",
                max_row=max_row,
            )
            _add_line_chart(
                worksheet,
                title="Aceleración comparada",
                y_title="Aceleración",
                data_columns=(8, 9),
                anchor="M62",
                max_row=max_row,
            )

        summary_sheet = workbook["Resumen"]
        summary_sheet["A4"] = "Interpretación"
        summary_sheet["A4"].font = Font(bold=True)
        summary_sheet["A5"] = (
            "El score mide cuánta geometría conserva la muestra frente a la referencia; "
            "no constituye una señal de inversión."
        )


def _evaluate(args: argparse.Namespace) -> int:
    sample_path = _resolve(args.sample)
    reference_path = _resolve(args.reference)
    if not sample_path.exists():
        raise FileNotFoundError(f"No existe la muestra: {sample_path}")
    if not reference_path.exists():
        raise FileNotFoundError(f"No existe la referencia: {reference_path}")

    output = _resolve(args.output)
    export_visual_comparison(
        _read(sample_path, args.sample_sheet),
        _read(reference_path, args.reference_sheet),
        output,
        smoothing_window=args.smoothing_window,
        rolling_window=args.rolling_window,
        sample_label=args.sample_label,
        reference_label=args.reference_label,
    )
    print(f"Excel visual generado: {output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="market-morphology-visualizer")
    parser.add_argument("--sample", required=True, help="CSV/XLSX con date,ticker,market_cap")
    parser.add_argument("--reference", required=True, help="CSV/XLSX de referencia")
    parser.add_argument("--sample-sheet")
    parser.add_argument("--reference-sheet")
    parser.add_argument("--sample-label", default="NASDAQ30")
    parser.add_argument("--reference-label", default="NASDAQ100")
    parser.add_argument("--smoothing-window", type=int, default=5)
    parser.add_argument("--rolling-window", type=int, default=20)
    parser.add_argument(
        "--output",
        default="reports/market_morphology/nasdaq30_vs_nasdaq100_visual.xlsx",
    )
    parser.set_defaults(handler=_evaluate)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
