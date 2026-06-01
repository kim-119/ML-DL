# AI 기반 전국 교량 노후도 및 안전점검 우선순위 분석

## 1. 프로젝트 개요
국토교통부_전국교량표준데이터를 활용하여 전국 교량의 노후도, 점검 이력, 구조적 특성, 내진 관련 정보를 분석하고, 머신러닝과 딥러닝을 통해 점검 우선순위 후보를 산정한 데이터 분석 프로젝트이다.

## 2. 데이터 출처
- 공공데이터포털
- 국토교통부_전국교량표준데이터

## 3. 분석 목적
- 공개 교량 데이터를 기반으로 노후도와 점검 이력을 정리
- 공개 데이터 기반 점검 우선순위 후보 산정
- 머신러닝 모델 성능 비교
- 딥러닝 기반 이상치 후보 탐지
- OpenAI API 또는 자동 리포트 생성 방식으로 분석 결과 요약

## 4. 사용 기술
- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- PyTorch
- XGBoost
- LightGBM
- OpenAI API 선택 사용
- GitHub Actions
- GitHub Pages

`XGBoost`와 `LightGBM`은 선택 패키지이다. 설치되지 않아도 분석 코드가 중단되지 않도록 optional import로 처리한다.

## 5. 폴더 구조
```text
ML-DL/
├── .github/workflows/
│   ├── ci.yml
│   └── pages.yml
├── data/
│   └── README.md
├── docs/
│   └── index.html
├── outputs/
│   ├── figures/
│   ├── model_metrics.csv
│   ├── portfolio_model_table.md
│   ├── bridge_priority_result.csv
│   ├── anomaly_bridge_candidates.csv
│   ├── analysis_summary.md
│   └── openai_report_examples.md
├── scripts/
│   └── validate_outputs.py
├── bridge_ai_analysis.py
├── bridge_ai_analysis.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

## 6. 실행 방법
원본 CSV는 [data/README.md](data/README.md) 안내에 따라 `data/bridge_data.csv`로 배치한다.

```bash
pip install -r requirements.txt
python bridge_ai_analysis.py
```

## 7. 주요 전처리
- 결측치 처리
- 숫자형 변환
- 날짜형 변환
- `bridge_age` 생성
- `inspection_elapsed_days` 생성
- `length_width_ratio` 생성
- `width_per_lane` 생성
- `seismic_risk_flag` 생성
- `priority_score` 생성
- `target_priority` 생성

## 8. 머신러닝 모델
- Logistic Regression
- RandomForest
- XGBoost
- LightGBM

## 9. 딥러닝 모델
- Tabular MLP
- AutoEncoder 이상치 탐지

## 10. 모델 성능 비교
| 모델 | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.7970 | 0.7778 | 0.7805 | 0.7769 |
| RandomForest | 0.9439 | 0.9388 | 0.9273 | 0.9323 |
| XGBoost | 0.9586 | 0.9551 | 0.9478 | 0.9511 |
| LightGBM | 0.9586 | 0.9548 | 0.9478 | 0.9510 |
| MLP 딥러닝 | 0.9138 | 0.9011 | 0.8951 | 0.8978 |

## 11. 주요 결과 파일
- `outputs/model_metrics.csv`
- `outputs/portfolio_model_table.md`
- `outputs/bridge_priority_result.csv`
- `outputs/anomaly_bridge_candidates.csv`
- `outputs/analysis_summary.md`
- `outputs/openai_report_examples.md`
- `outputs/figures/`

## 12. CI/CD
- CI는 GitHub Actions를 통해 저장소 구조, Python 문법, Notebook JSON, 결과 파일 존재 여부를 검증한다.
- CD는 GitHub Pages를 통해 `docs/index.html` 정적 리포트 페이지를 배포한다.
- 모델 학습은 GitHub Actions에서 수행하지 않고 로컬에서 수행한다.

## 13. GitHub Pages 설정
Repository → Settings → Pages → Build and deployment → Source를 `GitHub Actions`로 설정한다.

## 14. 보안 주의사항
아래 항목은 절대 GitHub에 올리지 않는다.
- `.env`
- `OPENAI_API_KEY`
- 공공데이터포털 서비스키
- Notion Token
- GitHub Token

## 15. 한계점
- 본 분석은 공개 데이터 기반의 점검 우선순위 후보 분석이며 실제 교량 안전진단을 대체하지 않는다.
- `priority_score`는 분석 목적의 보조 지표이다.
- 실제 구조 안전성 판단에는 전문 진단 데이터, 현장 점검, 구조해석 정보가 필요하다.
- `target_priority`가 규칙 기반으로 생성된 경우, 모델 성능은 실제 위험 예측력이 아니라 규칙 기반 라벨을 얼마나 잘 재현했는지를 의미한다.

## 16. 포트폴리오용 요약문
본 프로젝트는 국토교통부 전국교량표준데이터를 활용하여 전국 교량의 노후도, 점검 이력, 구조적 특성, 내진 관련 정보를 분석하고, 머신러닝과 딥러닝을 통해 점검 우선순위 후보를 산정한 데이터 분석 프로젝트이다. RandomForest, GradientBoosting, Tabular MLP를 활용하여 우선순위 분류 모델을 구축하였고, AutoEncoder를 통해 일반적인 교량 패턴에서 벗어나는 이상 교량 후보를 탐지하였다. 또한 OpenAI API 또는 자동 리포트 생성 방식을 활용하여 분석 결과를 자연어 리포트로 정리하였다.

