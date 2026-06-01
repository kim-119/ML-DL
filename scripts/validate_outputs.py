"""GitHub Actions에서 빠르게 실행하는 포트폴리오 구조 검증 스크립트."""

from __future__ import annotations

import csv
import json
import py_compile
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FAILURES: list[str] = []


def report(status: str, message: str) -> None:
    """검증 결과를 CI 로그에서 읽기 쉬운 형식으로 출력한다."""
    print(f"[{status}] {message}")
    if status == "FAIL":
        FAILURES.append(message)


def require(path: str) -> None:
    """저장소 운영에 필수인 파일은 누락 시 실패 처리한다."""
    target = ROOT / path
    report("OK" if target.exists() else "FAIL", f"{path} {'존재' if target.exists() else '누락'}")


def optional(path: str) -> None:
    """포트폴리오 결과 파일은 초기 단계 누락 시 경고만 출력한다."""
    target = ROOT / path
    report("OK" if target.exists() else "WARN", f"{path} {'존재' if target.exists() else '누락'}")


def validate_python() -> None:
    """분석 코드가 있으면 외부 패키지 설치 없이 문법만 검사한다."""
    target = ROOT / "bridge_ai_analysis.py"
    if not target.exists():
        report("WARN", "bridge_ai_analysis.py 누락")
        return
    try:
        py_compile.compile(str(target), doraise=True)
        report("OK", "bridge_ai_analysis.py 문법 검사 통과")
    except py_compile.PyCompileError as exc:
        report("FAIL", f"bridge_ai_analysis.py 문법 오류: {exc}")


def validate_notebook() -> None:
    """Notebook이 있으면 JSON 파싱 가능 여부를 검사한다."""
    target = ROOT / "bridge_ai_analysis.ipynb"
    if not target.exists():
        report("WARN", "bridge_ai_analysis.ipynb 누락")
        return
    try:
        with target.open(encoding="utf-8") as file:
            json.load(file)
        report("OK", "bridge_ai_analysis.ipynb JSON 검사 통과")
    except (OSError, json.JSONDecodeError) as exc:
        report("FAIL", f"bridge_ai_analysis.ipynb JSON 오류: {exc}")


def validate_metrics() -> None:
    """모델 성능표가 있으면 핵심 컬럼과 실제 결과 행을 검사한다."""
    target = ROOT / "outputs" / "model_metrics.csv"
    if not target.exists():
        report("WARN", "outputs/model_metrics.csv 누락")
        return
    try:
        with target.open(encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
        required = {"모델", "Accuracy", "Precision", "Recall", "F1-score"}
        headers = set(rows[0]) if rows else set()
        if not rows or not required.issubset(headers):
            report("WARN", "outputs/model_metrics.csv가 비어 있거나 필수 컬럼이 부족함")
            return
        report("OK", f"outputs/model_metrics.csv 검증 통과: {len(rows)}개 모델")
    except OSError as exc:
        report("WARN", f"outputs/model_metrics.csv 읽기 실패: {exc}")


def validate_figures() -> None:
    """PNG 그래프가 하나 이상 있는지 검사한다."""
    figure_dir = ROOT / "outputs" / "figures"
    figures = list(figure_dir.glob("*.png")) if figure_dir.exists() else []
    report("OK" if figures else "WARN", f"outputs/figures PNG 그래프 {len(figures)}개")


def main() -> int:
    """필수 구조와 선택 결과물을 검증한다."""
    print("ML/DL 포트폴리오 저장소 검증 시작")
    for path in ["README.md", "requirements.txt", ".gitignore", "scripts/validate_outputs.py"]:
        require(path)
    validate_python()
    validate_notebook()
    optional("outputs/portfolio_model_table.md")
    validate_metrics()
    validate_figures()
    if FAILURES:
        print(f"[FAIL] 필수 검증 실패 {len(FAILURES)}건")
        return 1
    print("[OK] 포트폴리오 저장소 검증 완료")
    return 0


if __name__ == "__main__":
    sys.exit(main())

