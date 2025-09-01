# src/main.py
import sys
import json
import click
import yaml
from pathlib import Path

from parsing.dxf_parser import parse_dxf_files
from vision.ocr import OCRService
from vision.detector import SymbolDetector
from graph.model import KnowledgeGraphBuilder
from qa.rules import QARunner
from reporting.report import Reporter

@click.command()
@click.option("--config", "-c", type=click.Path(exists=True), default="config.yaml")
def run(config):
    cfg = yaml.safe_load(Path(config).read_text())

    # 1) Parse CAD
    files = [str(p) for p in Path(".").glob(cfg["input"]["glob"])]
    if not files:
        click.secho("No input files found.", fg="red", bold=True)
        sys.exit(1)

    parsed = parse_dxf_files(
        files,
        wire_layer_regex=cfg["input"]["wire_layer_regex"],
        text_layer_regex=cfg["input"]["text_layer_regex"],
        sheet_layer_regex=cfg["input"]["sheet_layer_regex"],
    )

    # 2) Vision (optional)
    ocr = OCRService(cfg) if cfg.get("vision", {}).get("ocr", {}).get("engine") else None
    det = SymbolDetector(cfg) if cfg.get("vision", {}).get("yolo", {}).get("enabled") else None

    if ocr or det:
        for doc in parsed:
            img_path = doc.rasterize()  # export page to image (see dxf_parser)
            if ocr:
                doc.ocr_items = ocr.read(img_path)
            if det:
                doc.detections = det.detect(img_path)

    # 3) Build graph
    builder = KnowledgeGraphBuilder(cfg)
    G, provenance = builder.build(parsed)

    # 4) QA
    qa_runner = QARunner(cfg)
    findings = qa_runner.run(G)

    # 5) Reporting
    reporter = Reporter(cfg)
    reporter.emit(findings=findings, graph=G, provenance=provenance)

    click.secho(f"Done. Reports in {cfg['reporting']['out_dir']}", fg="green", bold=True)

if __name__ == "__main__":
    run()
