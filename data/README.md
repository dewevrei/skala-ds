# Data

원본 데이터는 용량 때문에 저장소에 포함하지 않습니다.

## 다운로드

[Kaggle — MIT-Stanford Dataset](https://www.kaggle.com/datasets/itshpark/data-driven-prediction-of-battery-cycle) 에서 아래 3개 파일을 받아 `data/raw/` 에 둡니다.

| 파일 | 용도 | 셀 (사용 / 전체) |
|---|---|---|
| `2017-05-12_batchdata_updated_struct_errorcorrect.mat` | Batch 1 — 학습 | 41 / 46 |
| `2018-02-20_batchdata_updated_struct_errorcorrect.mat` | Batch 2 — 테스트 | 39 / 47 |
| `2018-04-12_batchdata_updated_struct_errorcorrect.mat` | Batch 3 — 추가 검증 | 40 / 46 |

`2018-04-03_varcharge` 파일은 사용하지 않습니다.

원 출처 : [data.matr.io — Data-driven prediction of battery cycle life before capacity degradation](https://data.matr.io/1/projects/5c48dd2bc625d700019f3204)

## 전처리

```bash
python -m src.preprocess
```

`data/processed/` 에 분석용 캐시(약 90MB)가 생성됩니다.

| 파일 | 내용 |
|---|---|
| `cells.parquet` | 셀 단위 메타데이터, 충전 정책 파싱, 수명 보정, 제외 여부와 사유 |
| `summary.parquet` | 사이클 단위 요약 (QD, QC, IR, 온도, 충전·방전 시간) |
| `qdlin.npz` | 사이클 0~100 의 전압축 방전 용량 곡선 Qdlin / Tdlin (셀 × 사이클 × 1,000) |
| `features.parquet` | 모델링 데이터셋 (`notebooks/02_feature_engineering.ipynb` 실행 시 생성) |

## 정제 규칙 (원논문 공식 코드 기준)

- Batch 1 : 80% 용량 도달 전 시험이 중단된 5셀 제외 (b1c8, b1c10, b1c12, b1c13, b1c22)
- Batch 1 : 원논문 Batch 2(2017-06-30)로 이어서 측정된 5셀(b1c0~b1c4)은 이어진 사이클 수만큼 수명 보정
- Batch 2 : 수명 레이블이 없는 VarCharge·SLOWCYCLE 8셀 제외
- Batch 3 : 노이즈 채널 6셀 제외 (b3c2, b3c23, b3c32, b3c37, b3c42, b3c43)
