"""공개 교량 데이터 기반 점검 우선순위 후보 분석 전체 실행 스크립트."""

from __future__ import annotations

import json
import os
import platform
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from sklearn.cluster import KMeans
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        precision_recall_fscore_support,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
except ImportError as exc:
    raise SystemExit("필수 패키지가 없습니다. 먼저 `pip install -r requirements.txt`를 실행하세요.") from exc

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:
    raise SystemExit("PyTorch가 없습니다. 먼저 `pip install -r requirements.txt`를 실행하세요.") from exc


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
NOW = pd.Timestamp.now().normalize()
CURRENT_YEAR = NOW.year
RANDOM_STATE = 42
DISCLAIMER = "본 분석은 공개 데이터 기반의 점검 우선순위 후보 분석이며, 실제 교량 안전진단을 대체하지 않는다."

# 의미별 컬럼 후보를 한곳에서 관리하여 표준명 차이를 흡수한다.
COLUMN_KEYWORDS = {
    "교량명": ["교량명", "시설명", "bridge"],
    "시도명": ["시도", "광역", "province", "sido"],
    "시군구명": ["시군구", "구군", "county", "sigungu"],
    "도로종류": ["도로종류", "도로구분", "road_type"],
    "도로노선명": ["도로노선", "노선명", "route"],
    "교량연장": ["교량연장", "연장", "길이", "length"],
    "교량폭": ["교량폭", "폭", "width"],
    "차로수": ["차로수", "차선수", "lane"],
    "상부구조형식": ["상부구조", "구조형식", "superstructure"],
    "준공연도": ["준공연도", "준공", "완공", "completion", "year"],
    "최종안전점검일자": ["최종안전점검일자", "안전점검일자", "점검일자", "inspection"],
    "최종안전점검결과": ["최종안전점검결과", "안전점검결과", "점검결과", "inspection_result"],
    "내진설계적용여부": ["내진설계적용", "내진설계", "seismic_design"],
    "내진성능확보여부": ["내진성능", "성능확보", "seismic_performance"],
    "위도": ["위도", "latitude", "lat"],
    "경도": ["경도", "longitude", "lon", "lng"],
}


def warn(message: str) -> None:
    """사용자가 놓치지 않도록 경고 문구를 통일한다."""
    print(f"[경고] {message}")


def setup_dirs() -> None:
    """결과 폴더를 미리 생성한다."""
    FIGURES.mkdir(parents=True, exist_ok=True)


def set_seed() -> None:
    """NumPy와 PyTorch 난수 시드를 고정한다."""
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_STATE)


def configure_korean_font() -> None:
    """운영체제에 맞는 한글 폰트를 가능한 범위에서 선택한다."""
    from matplotlib import font_manager

    candidates = ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR", "DejaVu Sans"]
    installed = {font.name for font in font_manager.fontManager.ttflist}
    selected = next((font for font in candidates if font in installed), "DejaVu Sans")
    plt.rcParams["font.family"] = selected
    plt.rcParams["axes.unicode_minus"] = False
    if selected == "DejaVu Sans":
        warn("한글 전용 폰트를 찾지 못했습니다. 일부 그래프의 한글이 깨질 수 있습니다.")


def read_csv_with_fallback(path: Path, nrows: int | None = None) -> tuple[pd.DataFrame, str]:
    """여러 인코딩을 순서대로 시도하여 CSV를 읽는다."""
    failures = []
    for encoding in ["utf-8-sig", "cp949", "euc-kr", "utf-8"]:
        try:
            return pd.read_csv(path, encoding=encoding, nrows=nrows, low_memory=False), encoding
        except Exception as exc:  # CSV 파서 오류도 다음 인코딩으로 넘긴다.
            failures.append(f"{encoding}: {type(exc).__name__}")
    raise RuntimeError(f"CSV 읽기에 실패했습니다: {path}\n시도 결과: {', '.join(failures)}")


def find_source_csv() -> Path:
    """결과 CSV를 제외하고 키워드 우선, 행 수 차선으로 원본 CSV를 고른다."""
    candidates = [p for p in ROOT.rglob("*.csv") if OUTPUTS not in p.parents]
    if not candidates:
        raise FileNotFoundError("현재 폴더와 하위 폴더에서 CSV 파일을 찾지 못했습니다.")
    keywords = ["bridge", "교량", "전국교량", "국토교통부", "standard", "data"]
    preferred = [p for p in candidates if any(k.lower() in p.name.lower() for k in keywords)]
    pool = preferred or candidates
    row_counts = {}
    for path in pool:
        try:
            # 전체 행 수를 세어 가장 큰 기본 데이터 파일을 선택한다.
            frame, _ = read_csv_with_fallback(path)
            row_counts[path] = len(frame)
        except Exception as exc:
            warn(f"CSV 후보를 읽지 못해 제외합니다: {path.name} ({exc})")
    if not row_counts:
        raise RuntimeError("읽을 수 있는 CSV 파일이 없습니다.")
    selected = max(row_counts, key=row_counts.get)
    print(f"선택된 CSV 파일 경로: {selected}")
    return selected


def normalize_name(value: object) -> str:
    """컬럼 비교를 위해 공백과 기호를 제거한다."""
    return re.sub(r"[^0-9a-zA-Z가-힣]", "", str(value)).lower()


def detect_columns(columns: list[str]) -> dict[str, str | None]:
    """키워드 기반으로 실제 컬럼을 의미 컬럼에 연결한다."""
    detected: dict[str, str | None] = {}
    for meaning, keywords in COLUMN_KEYWORDS.items():
        normalized_keywords = [normalize_name(keyword) for keyword in keywords]
        exact = [col for col in columns if normalize_name(col) in normalized_keywords]
        partial = [col for col in columns if any(k in normalize_name(col) for k in normalized_keywords)]
        detected[meaning] = (exact or partial or [None])[0]
        if detected[meaning] is None:
            warn(f"'{meaning}' 의미 컬럼을 찾지 못했습니다. 관련 분석은 기본값 또는 제외 처리합니다.")
    return detected


def print_basic_info(raw: pd.DataFrame, path: Path) -> None:
    """CSV 로드 직후 요청된 데이터 기본 정보를 터미널에 출력한다."""
    print("\n" + "=" * 70)
    print("데이터 기본 정보")
    print(f"- 데이터 파일명: {path.name}")
    print(f"- 데이터 행 수: {len(raw):,}")
    print(f"- 데이터 열 수: {len(raw.columns):,}")
    print(f"- 컬럼 목록: {list(raw.columns)}")
    print(f"- 중복 행 개수: {raw.duplicated().sum():,}")
    print("\n상위 5개 행:")
    print(raw.head().to_string())
    print("\n데이터 타입:")
    print(raw.dtypes.to_string())
    print("\n결측치 개수:")
    print(raw.isna().sum().to_string())
    print("\n결측치 비율(%):")
    print((raw.isna().mean() * 100).round(2).to_string())


def numeric_clean(series: pd.Series) -> pd.Series:
    """단위와 문자 노이즈를 제거하고 숫자로 변환한다."""
    cleaned = series.astype("string").str.replace(",", "", regex=False).str.replace(" ", "", regex=False)
    cleaned = cleaned.str.replace(r"[^0-9eE+\-.]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def fill_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    """수치형 결측을 중앙값으로 채우되 전체 결측이면 기본값을 사용한다."""
    median = series.median()
    return series.fillna(default if pd.isna(median) else median)


def preprocess(raw: pd.DataFrame, columns: dict[str, str | None]) -> tuple[pd.DataFrame, list[str], list[str]]:
    """원본을 보존하면서 숫자, 날짜, 파생변수를 만든다."""
    df = raw.copy()
    derived: list[str] = []
    limits: list[str] = []

    # 의미 컬럼을 통일된 이름으로 복사하여 이후 코드의 분기를 줄인다.
    for meaning, source in columns.items():
        if source is not None and meaning not in df.columns:
            df[meaning] = df[source]

    for col in ["교량연장", "교량폭", "차로수", "준공연도"]:
        if col in df.columns:
            df[col] = numeric_clean(df[col])

    for col in ["교량연장", "교량폭", "차로수"]:
        if col in df.columns:
            df.loc[df[col] <= 0, col] = np.nan

    if "준공연도" in df.columns:
        df.loc[df["준공연도"] > CURRENT_YEAR, "준공연도"] = np.nan
        df["준공연도"] = fill_numeric(df["준공연도"], CURRENT_YEAR - 30)
        df["bridge_age"] = CURRENT_YEAR - df["준공연도"]
        df.loc[(df["bridge_age"] < 0) | (df["bridge_age"] > 150), "bridge_age"] = np.nan
        df["bridge_age"] = fill_numeric(df["bridge_age"], 30)
    else:
        df["bridge_age"] = 30.0
        limits.append("준공연도 컬럼이 없어 bridge_age를 임시값 30으로 생성했습니다.")
    derived.append("bridge_age")

    if "최종안전점검일자" in df.columns:
        df["최종안전점검일자"] = pd.to_datetime(df["최종안전점검일자"], errors="coerce")
        elapsed = (NOW - df["최종안전점검일자"]).dt.days.astype(float)
        elapsed = elapsed.mask(elapsed < 0)
        df["inspection_elapsed_days"] = fill_numeric(elapsed, 365)
    else:
        df["inspection_elapsed_days"] = 365.0
        limits.append("최종안전점검일자 컬럼이 없어 inspection_elapsed_days를 임시값 365로 생성했습니다.")
    derived.append("inspection_elapsed_days")

    for col in ["교량연장", "교량폭", "차로수"]:
        if col in df.columns:
            df[col] = fill_numeric(df[col], 0)

    if {"교량연장", "교량폭"}.issubset(df.columns):
        df["length_width_ratio"] = df["교량연장"] / df["교량폭"]
        df["length_width_ratio"] = fill_numeric(df["length_width_ratio"].replace([np.inf, -np.inf], np.nan), 0)
        derived.append("length_width_ratio")
    else:
        warn("교량연장 또는 교량폭이 없어 length_width_ratio를 생성하지 않습니다.")

    if {"교량폭", "차로수"}.issubset(df.columns):
        df["width_per_lane"] = df["교량폭"] / df["차로수"]
        df["width_per_lane"] = fill_numeric(df["width_per_lane"].replace([np.inf, -np.inf], np.nan), 0)
        derived.append("width_per_lane")
    else:
        warn("교량폭 또는 차로수가 없어 width_per_lane을 생성하지 않습니다.")

    risk_tokens = {"n", "아니오", "미확보", "무", "없음", "부", "x", "결측", "unknown", "미적용"}
    seismic_cols = [col for col in ["내진설계적용여부", "내진성능확보여부"] if col in df.columns]
    if seismic_cols:
        def is_risk(row: pd.Series) -> int:
            values = [str(row[col]).strip().lower() if pd.notna(row[col]) else "unknown" for col in seismic_cols]
            return int(any(any(token in value for token in risk_tokens) for value in values))
        df["seismic_risk_flag"] = df.apply(is_risk, axis=1)
    else:
        df["seismic_risk_flag"] = 1
        limits.append("내진 관련 컬럼이 없어 seismic_risk_flag를 보수적으로 1로 생성했습니다.")
    derived.append("seismic_risk_flag")

    df["old_bridge_flag"] = (df["bridge_age"] >= 30).astype(int)
    df["long_inspection_gap_flag"] = (df["inspection_elapsed_days"] >= 730).astype(int)
    derived.extend(["old_bridge_flag", "long_inspection_gap_flag"])

    # 범주형 결측은 Unknown으로 채워 모델과 결과 CSV에서 의미를 유지한다.
    categorical = ["교량명", "시도명", "시군구명", "도로종류", "도로노선명", "상부구조형식", "최종안전점검결과",
                   "내진설계적용여부", "내진성능확보여부"]
    for col in categorical:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str)
    return df, derived, limits


def add_priority(df: pd.DataFrame) -> pd.DataFrame:
    """공개 데이터 기반 점검 우선순위 후보 점수와 분석용 라벨을 만든다."""
    score = pd.Series(0.0, index=df.index)
    score += np.select([df["bridge_age"] >= 40, df["bridge_age"] >= 30, df["bridge_age"] >= 20], [30, 20, 10], default=0)
    score += np.select([df["inspection_elapsed_days"] >= 1095, df["inspection_elapsed_days"] >= 730,
                        df["inspection_elapsed_days"] >= 365], [25, 15, 5], default=0)
    if "교량연장" in df.columns:
        score += (df["교량연장"] >= df["교량연장"].quantile(0.75)).astype(int) * 10
    if "교량폭" in df.columns:
        score += (df["교량폭"] <= df["교량폭"].quantile(0.25)).astype(int) * 10
    if "width_per_lane" in df.columns:
        score += (df["width_per_lane"] <= df["width_per_lane"].quantile(0.25)).astype(int) * 10
    score += df["seismic_risk_flag"] * 20
    if "최종안전점검결과" in df.columns:
        result = df["최종안전점검결과"].fillna("").astype(str).str.upper()
        score += result.str.contains(r"나쁨|불량|미흡|위험|D|E", regex=True).astype(int) * 30
        score += result.str.contains(r"보통|C", regex=True).astype(int) * 15
    df["priority_score"] = score.astype(float)
    low_q, high_q = score.quantile([0.25, 0.75])
    df["target_priority"] = np.select([score >= high_q, score <= low_q], ["High", "Low"], default="Medium")
    return df


def save_bar(series: pd.Series, title: str, xlabel: str, ylabel: str, filename: str) -> None:
    """빈 데이터도 오류 없이 안내 그래프로 저장한다."""
    plt.figure(figsize=(10, 6))
    if series.empty:
        plt.text(0.5, 0.5, "사용 가능한 데이터 없음", ha="center", va="center")
    else:
        series.plot(kind="bar", color="#4C78A8")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.savefig(FIGURES / filename, bbox_inches="tight", dpi=150)
    plt.close()


def save_hist(series: pd.Series, title: str, xlabel: str, ylabel: str, filename: str) -> None:
    """수치형 분포를 히스토그램으로 저장한다."""
    plt.figure(figsize=(10, 6))
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        plt.text(0.5, 0.5, "사용 가능한 데이터 없음", ha="center", va="center")
    else:
        plt.hist(values, bins=30, color="#4C78A8", edgecolor="white")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(FIGURES / filename, bbox_inches="tight", dpi=150)
    plt.close()


def create_eda(df: pd.DataFrame) -> None:
    """필수 EDA 그래프 여덟 개를 모두 저장한다."""
    save_bar(df.get("시도명", pd.Series(dtype=str)).value_counts().head(10), "지역별 교량 수 Top 10", "시도명", "교량 수", "sido_bridge_count_top10.png")
    save_hist(df.get("준공연도", pd.Series(dtype=float)), "준공연도별 교량 수 분포", "준공연도", "교량 수", "completion_year_distribution.png")
    save_hist(df["bridge_age"], "교량 나이 분포", "교량 나이(년)", "교량 수", "bridge_age_distribution.png")
    save_hist(df.get("교량연장", pd.Series(dtype=float)), "교량연장 분포", "교량연장(m)", "교량 수", "bridge_length_distribution.png")
    save_bar(df.get("상부구조형식", pd.Series(dtype=str)).value_counts().head(10), "상부구조형식별 교량 수 Top 10", "상부구조형식", "교량 수", "superstructure_top10.png")
    save_bar(df.get("최종안전점검결과", pd.Series(dtype=str)).value_counts().head(15), "최종안전점검결과 분포", "점검결과", "교량 수", "inspection_result_distribution.png")
    save_bar(df.get("내진성능확보여부", pd.Series(dtype=str)).value_counts().head(15), "내진성능확보여부 분포", "내진성능확보여부", "교량 수", "seismic_performance_distribution.png")
    save_bar(df["target_priority"].value_counts(), "분석용 점검 우선순위 후보 분포", "target_priority", "교량 수", "priority_distribution.png")


def feature_lists(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """실제 존재하는 모델 입력 컬럼만 반환한다."""
    numeric_candidates = ["bridge_age", "inspection_elapsed_days", "length_width_ratio", "width_per_lane",
                          "seismic_risk_flag", "old_bridge_flag", "long_inspection_gap_flag", "교량연장", "교량폭", "차로수"]
    categorical_candidates = ["시도명", "시군구명", "도로종류", "도로노선명", "상부구조형식", "내진설계적용여부", "내진성능확보여부"]
    return [c for c in numeric_candidates if c in df.columns], [c for c in categorical_candidates if c in df.columns]


def make_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    """공통 전처리기를 생성한다."""
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    category_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                              ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", numeric_pipe, numeric), ("cat", category_pipe, categorical)])


def metrics_row(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    """macro 지표를 공통 형식으로 계산한다."""
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {"모델": name, "Accuracy": accuracy_score(y_true, y_pred), "Precision": precision, "Recall": recall, "F1-score": f1}


def split_data(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """가능하면 계층 분할을 사용하고 실패 시 일반 분할로 전환한다."""
    X, y = df[features], df["target_priority"]
    try:
        return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    except ValueError as exc:
        warn(f"stratify 분할을 사용할 수 없어 일반 분할로 진행합니다: {exc}")
        return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)


class MLPClassifier(nn.Module):
    """요청된 구조의 Tabular MLP."""
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.2),
                                    nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
                                    nn.Linear(64, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def train_mlp(x_train, x_test, y_train, y_test, label_encoder: LabelEncoder) -> tuple[dict[str, object], list[float]]:
    """전처리된 희소 행렬을 받아 MLP를 학습한다."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"MLP 학습 장치: {device}")
    x_train_np = x_train.toarray().astype("float32") if hasattr(x_train, "toarray") else np.asarray(x_train, dtype="float32")
    x_test_np = x_test.toarray().astype("float32") if hasattr(x_test, "toarray") else np.asarray(x_test, dtype="float32")
    y_train_np = label_encoder.transform(y_train).astype("int64")
    y_test_np = label_encoder.transform(y_test).astype("int64")
    loader = DataLoader(TensorDataset(torch.from_numpy(x_train_np), torch.from_numpy(y_train_np)), batch_size=128, shuffle=True)
    model = MLPClassifier(x_train_np.shape[1], len(label_encoder.classes_)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    losses = []
    for _ in range(15):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        losses.append(epoch_loss / len(loader.dataset))
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(x_test_np).to(device)).argmax(dim=1).cpu().numpy()
    return metrics_row("MLP 딥러닝", y_test_np, pred), losses


def save_mlp_loss(losses: list[float]) -> None:
    """MLP 학습 손실 곡선을 저장한다."""
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(losses) + 1), losses, marker="o")
    plt.title("MLP 학습 손실")
    plt.xlabel("Epoch")
    plt.ylabel("CrossEntropy Loss")
    plt.savefig(FIGURES / "mlp_training_loss.png", bbox_inches="tight", dpi=150)
    plt.close()


def train_models(df: pd.DataFrame) -> tuple[pd.DataFrame, Pipeline, list[str], list[str], list[str]]:
    """머신러닝 네 종과 MLP를 학습하여 비교표를 만든다."""
    numeric, categorical = feature_lists(df)
    features = numeric + categorical
    if not features:
        raise RuntimeError("학습에 사용할 수 있는 feature가 없습니다.")
    x_train, x_test, y_train, y_test = split_data(df, features)
    preprocessor = make_preprocessor(numeric, categorical)
    rows: list[dict[str, object]] = []
    unavailable: list[str] = []

    models: list[tuple[str, object]] = [
        ("Logistic Regression", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
        ("RandomForest", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1)),
    ]
    try:
        from xgboost import XGBClassifier
        encoder = LabelEncoder().fit(y_train)
        xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.08, random_state=RANDOM_STATE,
                            eval_metric="mlogloss", n_jobs=-1)
        xgb_pipe = Pipeline([("preprocessor", make_preprocessor(numeric, categorical)), ("model", xgb)])
        xgb_pipe.fit(x_train, encoder.transform(y_train))
        pred = encoder.inverse_transform(xgb_pipe.predict(x_test).astype(int))
        rows.append(metrics_row("XGBoost", y_test.to_numpy(), pred))
    except ImportError:
        unavailable.append("XGBoost")
    except Exception as exc:
        unavailable.append("XGBoost")
        warn(f"XGBoost 학습을 건너뜁니다: {exc}")
    try:
        from lightgbm import LGBMClassifier
        models.append(("LightGBM", LGBMClassifier(n_estimators=200, learning_rate=0.08, random_state=RANDOM_STATE, verbosity=-1)))
    except ImportError:
        unavailable.append("LightGBM")

    rf_pipe = None
    for name, model in models:
        pipe = Pipeline([("preprocessor", make_preprocessor(numeric, categorical)), ("model", model)])
        try:
            pipe.fit(x_train, y_train)
            rows.append(metrics_row(name, y_test.to_numpy(), pipe.predict(x_test)))
            if name == "RandomForest":
                rf_pipe = pipe
                save_confusion(y_test, pipe.predict(x_test))
        except Exception as exc:
            unavailable.append(name)
            warn(f"{name} 학습을 건너뜁니다: {exc}")
    if rf_pipe is None:
        raise RuntimeError("RandomForest 학습에 실패했습니다.")

    # MLP는 RandomForest와 동일한 데이터 분할 및 별도 전처리 결과를 사용한다.
    mlp_preprocessor = make_preprocessor(numeric, categorical)
    xt = mlp_preprocessor.fit_transform(x_train)
    xv = mlp_preprocessor.transform(x_test)
    label_encoder = LabelEncoder().fit(y_train)
    mlp_row, losses = train_mlp(xt, xv, y_train, y_test, label_encoder)
    rows.append(mlp_row)
    save_mlp_loss(losses)
    metrics = pd.DataFrame(rows)
    order = ["Logistic Regression", "RandomForest", "XGBoost", "LightGBM", "MLP 딥러닝"]
    metrics["_order"] = metrics["모델"].map({name: idx for idx, name in enumerate(order)})
    metrics = metrics.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    return metrics, rf_pipe, numeric, categorical, unavailable


def save_confusion(y_true: pd.Series, y_pred: np.ndarray) -> None:
    """RandomForest 혼동행렬을 matplotlib로 저장한다."""
    labels = ["High", "Medium", "Low"]
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, cmap="Blues")
    plt.colorbar()
    plt.xticks(range(len(labels)), labels)
    plt.yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, matrix[i, j], ha="center", va="center")
    plt.title("RandomForest 혼동행렬")
    plt.xlabel("예측 클래스")
    plt.ylabel("실제 클래스")
    plt.savefig(FIGURES / "confusion_matrix_randomforest.png", bbox_inches="tight", dpi=150)
    plt.close()


def save_feature_importance(rf_pipe: Pipeline) -> list[str]:
    """OneHotEncoder 이후 중요도 상위 15개를 복원하고 저장한다."""
    names = rf_pipe.named_steps["preprocessor"].get_feature_names_out()
    values = rf_pipe.named_steps["model"].feature_importances_
    importance = pd.Series(values, index=names).sort_values(ascending=False).head(15).sort_values()
    plt.figure(figsize=(10, 7))
    importance.plot(kind="barh", color="#59A14F")
    plt.title("RandomForest Feature Importance Top 15")
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.savefig(FIGURES / "feature_importance_top15.png", bbox_inches="tight", dpi=150)
    plt.close()
    top = list(reversed(importance.index.tolist()))
    print("\nRandomForest 중요 변수 Top 15:")
    for item in top:
        print(f"- {item}")
    return top


class AutoEncoder(nn.Module):
    """수치형 패턴 복원을 위한 작은 AutoEncoder."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU(),
                                 nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, input_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """AutoEncoder 복원 오차 상위 5%를 이상 교량 후보로 표시한다."""
    candidates = ["bridge_age", "inspection_elapsed_days", "length_width_ratio", "width_per_lane",
                  "seismic_risk_flag", "old_bridge_flag", "long_inspection_gap_flag", "교량연장", "교량폭", "차로수", "priority_score"]
    features = [col for col in candidates if col in df.columns]
    matrix = SimpleImputer(strategy="median").fit_transform(df[features])
    matrix = StandardScaler().fit_transform(matrix).astype("float32")
    tensor = torch.from_numpy(matrix)
    loader = DataLoader(TensorDataset(tensor), batch_size=128, shuffle=True)
    model = AutoEncoder(matrix.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    for _ in range(20):
        for (batch,) in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch), batch)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        reconstruction = model(tensor)
        errors = torch.mean((reconstruction - tensor) ** 2, dim=1).numpy()
    df["anomaly_score"] = errors
    threshold = float(np.quantile(errors, 0.95))
    df["anomaly_flag"] = (df["anomaly_score"] >= threshold).astype(int)
    save_hist(df["anomaly_score"], "AutoEncoder 이상치 점수 분포", "anomaly_score", "교량 수", "anomaly_score_distribution.png")
    return df


def add_clusters(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """세 개의 데이터 특성 기반 그룹을 생성한다."""
    candidates = ["bridge_age", "inspection_elapsed_days", "length_width_ratio", "width_per_lane",
                  "seismic_risk_flag", "old_bridge_flag", "long_inspection_gap_flag", "교량연장", "교량폭", "차로수", "priority_score"]
    features = [col for col in candidates if col in df.columns]
    matrix = SimpleImputer(strategy="median").fit_transform(df[features])
    matrix = StandardScaler().fit_transform(matrix)
    df["cluster"] = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10).fit_predict(matrix)
    summary = df.groupby("cluster")[["bridge_age", "inspection_elapsed_days", "priority_score"]].mean().round(2)
    plt.figure(figsize=(9, 6))
    for cluster, group in df.groupby("cluster"):
        plt.scatter(group["bridge_age"], group["priority_score"], s=10, alpha=0.35, label=f"cluster {cluster}")
    plt.title("데이터 특성 기반 그룹: 교량 나이와 우선순위 점수")
    plt.xlabel("bridge_age")
    plt.ylabel("priority_score")
    plt.legend()
    plt.savefig(FIGURES / "cluster_priority_scatter.png", bbox_inches="tight", dpi=150)
    plt.close()
    return df, summary


def markdown_table(metrics: pd.DataFrame, unavailable: list[str]) -> str:
    """설치 또는 실행 불가 모델도 행을 유지한 Markdown 표를 만든다."""
    order = ["Logistic Regression", "RandomForest", "XGBoost", "LightGBM", "MLP 딥러닝"]
    indexed = metrics.set_index("모델")
    lines = ["| 모델 | Accuracy | Precision | Recall | F1-score |", "|---|---:|---:|---:|---:|"]
    for name in order:
        if name in indexed.index:
            row = indexed.loc[name]
            values = [f"{float(row[col]):.4f}" for col in ["Accuracy", "Precision", "Recall", "F1-score"]]
        else:
            values = ["설치 필요"] * 4
        lines.append(f"| {name} | {' | '.join(values)} |")
    return "\n".join(lines)


def save_model_outputs(metrics: pd.DataFrame, unavailable: list[str]) -> str:
    """모델 CSV, Markdown 표, F1 비교 그래프를 저장한다."""
    metrics.round(6).to_csv(OUTPUTS / "model_metrics.csv", index=False, encoding="utf-8-sig")
    table = markdown_table(metrics, unavailable)
    (OUTPUTS / "portfolio_model_table.md").write_text(table + "\n", encoding="utf-8")
    plt.figure(figsize=(9, 5))
    plt.bar(metrics["모델"], metrics["F1-score"], color="#F28E2B")
    plt.title("모델별 F1-score 비교")
    plt.xlabel("모델")
    plt.ylabel("F1-score")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=25, ha="right")
    plt.savefig(FIGURES / "model_comparison_f1.png", bbox_inches="tight", dpi=150)
    plt.close()
    return table


def save_result_csv(df: pd.DataFrame) -> None:
    """요청 순서대로 존재하는 최종 결과 컬럼만 저장한다."""
    wanted = ["교량명", "시도명", "시군구명", "도로종류", "도로노선명", "교량연장", "교량폭", "차로수",
              "상부구조형식", "준공연도", "최종안전점검일자", "최종안전점검결과", "내진설계적용여부",
              "내진성능확보여부", "bridge_age", "inspection_elapsed_days", "length_width_ratio", "width_per_lane",
              "seismic_risk_flag", "old_bridge_flag", "long_inspection_gap_flag", "priority_score", "target_priority",
              "anomaly_score", "anomaly_flag", "cluster"]
    selected = [col for col in wanted if col in df.columns]
    df[selected].to_csv(OUTPUTS / "bridge_priority_result.csv", index=False, encoding="utf-8-sig")
    anomaly = df[df["anomaly_flag"] == 1].sort_values("anomaly_score", ascending=False).head(20)
    anomaly[selected].to_csv(OUTPUTS / "anomaly_bridge_candidates.csv", index=False, encoding="utf-8-sig")


def dataframe_preview(df: pd.DataFrame, rows: int = 5) -> str:
    """Markdown 미지원 환경에서도 기본 정보가 저장되도록 변환한다."""
    try:
        return df.head(rows).to_markdown(index=False)
    except ImportError:
        return "```\n" + df.head(rows).to_string(index=False) + "\n```"


def save_analysis_summary(raw: pd.DataFrame, path: Path, encoding: str, detected: dict[str, str | None],
                          df: pd.DataFrame, top_features: list[str], cluster_summary: pd.DataFrame) -> None:
    """기본 정보와 분석 해석을 Markdown으로 저장한다."""
    missing = pd.DataFrame({"결측치 개수": raw.isna().sum(), "결측치 비율": (raw.isna().mean() * 100).round(2)})
    lines = [
        "# AI 기반 전국 교량 노후도 및 안전점검 우선순위 분석 요약",
        "",
        f"> {DISCLAIMER}",
        "",
        "## 데이터 기본 정보",
        f"- 데이터 파일명: `{path.name}`",
        f"- 인코딩: `{encoding}`",
        f"- 데이터 행 수: `{len(raw):,}`",
        f"- 데이터 열 수: `{len(raw.columns):,}`",
        f"- 중복 행 개수: `{raw.duplicated().sum():,}`",
        f"- 컬럼 목록: `{', '.join(map(str, raw.columns))}`",
        "",
        "### 상위 5개 행",
        dataframe_preview(raw),
        "",
        "### 데이터 타입",
        "```",
        raw.dtypes.to_string(),
        "```",
        "",
        "### 결측치",
        dataframe_preview(missing.reset_index().rename(columns={"index": "컬럼"}), len(missing)),
        "",
        "## 자동 탐색 컬럼",
        dataframe_preview(pd.DataFrame({"의미": list(detected), "실제 컬럼": [detected[k] or "미탐색" for k in detected]}), len(detected)),
        "",
        "## 모델 해석",
        "RandomForest 모델은 교량 나이, 점검 경과일수, 내진 관련 정보, 교량연장, 교량폭 등에서 생성된 변수를 주요 판단 변수로 사용하였다.",
        f"상위 중요 변수: `{', '.join(top_features)}`",
        "",
        "## 데이터 특성 기반 그룹",
        "아래 cluster는 실제 위험 등급이 아니라 수치형 데이터 특성 기반 그룹이다.",
        dataframe_preview(cluster_summary.reset_index(), len(cluster_summary)),
        "",
        "## 우선순위 라벨 주의사항",
        "`target_priority`는 실제 정밀안전진단 등급이 아니라 분석용 우선순위 라벨이다.",
    ]
    (OUTPUTS / "analysis_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def mock_report(metrics: pd.DataFrame, df: pd.DataFrame, top_features: list[str]) -> str:
    """API 키 없이도 완성되는 자동 리포트를 생성한다."""
    best = metrics.sort_values("F1-score", ascending=False).iloc[0]
    anomaly_count = int(df["anomaly_flag"].sum())
    return f"""# OpenAI API 또는 자동 생성 리포트 예시

> {DISCLAIMER}

## 전체 데이터 분석 요약
총 {len(df):,}개 교량을 대상으로 노후도, 점검 경과일수, 구조 특성, 내진 관련 정보를 정리했다. `target_priority`는 규칙 기반으로 생성한 분석용 우선순위 후보 라벨이다.

## 모델 성능 비교 요약
가장 높은 F1-score를 기록한 모델은 `{best["모델"]}`이며 F1-score는 `{best["F1-score"]:.4f}`이다. 이 수치는 실제 위험 예측력이 아니라 규칙 기반 라벨 재현 성능으로 해석해야 한다.

## 주요 변수 해석
RandomForest 중요 변수 상위 항목은 `{", ".join(top_features[:7])}` 등이다.

## 이상치 후보 해석
AutoEncoder 복원 오차 상위 5% 기준으로 {anomaly_count:,}개 교량을 일반적인 데이터 패턴에서 벗어난 이상 교량 후보로 표시했다. 이 후보는 추가 검토 대상을 좁히기 위한 참고 목록이다.

## 포트폴리오용 프로젝트 설명문
국토교통부 전국교량표준데이터를 활용하여 전처리, EDA, 규칙 기반 점검 우선순위 후보 산정, 머신러닝 및 딥러닝 분류, AutoEncoder 이상치 탐지를 수행했다.
"""


def save_openai_report(metrics: pd.DataFrame, df: pd.DataFrame, top_features: list[str]) -> None:
    """OPENAI_API_KEY가 있으면 API를 시도하고 실패하면 자동 리포트를 저장한다."""
    report = mock_report(metrics, df, top_features)
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            from openai import OpenAI
            prompt = report + "\n위 내용을 포트폴리오용 한국어 리포트로 다듬고 면책 문장을 반드시 유지하세요."
            response = OpenAI(api_key=api_key).responses.create(model="gpt-4.1-mini", input=prompt)
            report = "# OpenAI API 생성 리포트\n\n" + response.output_text + f"\n\n> {DISCLAIMER}\n"
        except Exception as exc:
            warn(f"OpenAI API 호출에 실패하여 자동 리포트를 저장합니다: {exc}")
    else:
        print("OPENAI_API_KEY가 없어 mock report를 생성합니다.")
    (OUTPUTS / "openai_report_examples.md").write_text(report, encoding="utf-8")


def save_readme(table: str, limits: list[str], unavailable: list[str]) -> None:
    """실제 실행 결과 표를 포함한 포트폴리오 README를 저장한다."""
    optional = ", ".join(unavailable) if unavailable else "없음: XGBoost와 LightGBM도 정상 학습됨"
    limitation_lines = "\n".join(f"- {item}" for item in limits) if limits else "- 주요 파생변수는 원본 컬럼을 사용해 생성했다."
    figures = "\n".join(f"- `outputs/figures/{p.name}`" for p in sorted(FIGURES.glob("*.png")))
    content = f"""# AI 기반 전국 교량 노후도 및 안전점검 우선순위 분석

## 1. 프로젝트 개요
국토교통부_전국교량표준데이터를 활용하여 전국 교량의 노후도, 점검 이력, 구조적 특성, 내진 관련 정보를 분석하고, 머신러닝과 딥러닝을 통해 점검 우선순위 후보를 산정한 프로젝트이다.

## 2. 데이터 출처
국토교통부_전국교량표준데이터

## 3. 분석 목적
공개 교량 데이터를 기반으로 노후도와 점검 이력을 정리하고, 공개 데이터 기반 점검 우선순위 후보를 도출한다.

## 4. 사용 기술
- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- PyTorch
- XGBoost 선택
- LightGBM 선택
- OpenAI API 선택

## 5. 폴더 구조
`bridge_ai_analysis.py`를 실행하면 `outputs/` 아래에 결과 CSV, 분석 요약, 모델 성능표, 선택형 자연어 리포트, `figures/` 그래프가 생성된다.

## 6. 실행 방법
```bash
pip install -r requirements.txt
python bridge_ai_analysis.py
```

## 7. 전처리 내용
- 결측치 처리
- 숫자형 변환
- 날짜형 변환
- 파생변수 생성
- 우선순위 점수 생성

## 8. 머신러닝 모델
- Logistic Regression
- RandomForest
- XGBoost
- LightGBM

선택 모델 실행 불가 항목: {optional}

## 9. 딥러닝 모델
- Tabular MLP
- AutoEncoder

## 10. 모델 성능 비교
{table}

## 11. 주요 시각화 결과
{figures}

## 12. OpenAI API 활용
환경변수 `OPENAI_API_KEY`가 있으면 분석 결과를 자연어 리포트로 변환하는 기능을 시도한다. API 키가 없거나 호출이 실패하면 mock report를 생성한다.

## 13. 한계점
- 본 분석은 공개 데이터 기반의 점검 우선순위 후보 분석이며 실제 교량 안전진단을 대체하지 않는다.
- `priority_score`는 분석 목적의 보조 지표이다.
- 실제 구조 안전성 판단에는 전문 진단 데이터, 현장 점검, 구조해석 정보가 필요하다.
- `target_priority`가 규칙 기반으로 생성된 경우, 모델 성능은 실제 위험 예측력이 아니라 규칙 기반 라벨을 얼마나 잘 재현했는지를 의미한다.
{limitation_lines}

## 14. 포트폴리오용 요약문
본 프로젝트는 국토교통부 전국교량표준데이터를 활용하여 전국 교량의 노후도, 점검 이력, 구조적 특성, 내진 관련 정보를 분석하고, 머신러닝과 딥러닝을 통해 점검 우선순위 후보를 산정한 데이터 분석 프로젝트이다. Logistic Regression, RandomForest, XGBoost, LightGBM, Tabular MLP를 활용하여 우선순위 분류 모델을 구축했고, AutoEncoder를 통해 일반적인 교량 패턴에서 벗어나는 이상 교량 후보를 탐지했다. 또한 OpenAI API 또는 자동 리포트 생성 방식을 활용하여 분석 결과를 자연어 리포트로 정리했다.
"""
    (ROOT / "README.md").write_text(content, encoding="utf-8")


def save_notebook() -> None:
    """초보자가 셀 하나로 실행할 수 있는 최소 Notebook을 생성한다."""
    notebook = {
        "cells": [{"cell_type": "markdown", "metadata": {}, "source": ["# AI 기반 전국 교량 노후도 및 안전점검 우선순위 분석\n", "아래 셀은 전체 분석 스크립트를 실행합니다."]},
                  {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["%run bridge_ai_analysis.py\n"]}],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (ROOT / "bridge_ai_analysis.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    """전체 프로젝트를 순서대로 실행한다."""
    warnings.filterwarnings("ignore", category=UserWarning)
    setup_dirs()
    set_seed()
    configure_korean_font()
    csv_path = find_source_csv()
    raw, encoding = read_csv_with_fallback(csv_path)
    print_basic_info(raw, csv_path)
    detected = detect_columns(list(raw.columns))
    df, derived, limits = preprocess(raw, detected)
    df = add_priority(df)
    create_eda(df)
    metrics, rf_pipe, _, _, unavailable = train_models(df)
    top_features = save_feature_importance(rf_pipe)
    df = detect_anomalies(df)
    df, cluster_summary = add_clusters(df)
    save_result_csv(df)
    table = save_model_outputs(metrics, unavailable)
    save_analysis_summary(raw, csv_path, encoding, detected, df, top_features, cluster_summary)
    save_openai_report(metrics, df, top_features)
    save_readme(table, limits, unavailable)
    save_notebook()

    best = metrics.sort_values("F1-score", ascending=False).iloc[0]
    files = sorted(str(p.relative_to(ROOT)) for p in OUTPUTS.rglob("*") if p.is_file())
    summary = ("본 프로젝트는 국토교통부 전국교량표준데이터를 활용하여 노후도, 점검 이력, 구조 특성, "
               "내진 관련 정보를 분석하고 공개 데이터 기반 점검 우선순위 후보를 산정했다. "
               "분류 모델 비교와 AutoEncoder 이상 교량 후보 탐지를 수행했으며 실제 교량 안전진단을 대체하지 않는다.")
    print("\n" + "=" * 70)
    print(f"1. 선택된 CSV 파일명: {csv_path.name}")
    print(f"2. 데이터 행/열 수: {len(raw):,}행 / {len(raw.columns):,}열")
    print(f"3. 생성된 파생변수 목록: {', '.join(derived + ['priority_score', 'target_priority', 'anomaly_score', 'anomaly_flag', 'cluster'])}")
    print("4. target_priority 클래스 분포:")
    print(df["target_priority"].value_counts().to_string())
    print("\n5. 모델 성능 비교표:")
    print(table)
    print(f"\n6. 가장 F1-score가 높은 모델: {best['모델']} ({best['F1-score']:.4f})")
    print(f"7. 이상치 후보 교량 수: {int(df['anomaly_flag'].sum()):,}")
    print("8. 저장된 결과 파일 목록:")
    for file in files:
        print(f"- {file}")
    print(f"9. 포트폴리오에 붙여넣을 요약문:\n{summary}")
    print("\n분석 완료: outputs/portfolio_model_table.md의 표를 Notion 포트폴리오 모델 성능 비교 영역에 붙여넣으면 됩니다.")


if __name__ == "__main__":
    main()
