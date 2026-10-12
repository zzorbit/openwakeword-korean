# 한국어 호출어 학습 가이드

이 가이드는 **새 호출어를 생성하고 모델을 학습한 뒤, 평가 결과를 사이트로 확인하는 과정**을 안내합니다. 예시는 ‘보라야’이며, 명령은 저장소 폴더에서 실행합니다.

먼저 [README의 설치 절차](../README.md#설치)에 따라 생성·학습 기능까지 설치하세요. 아래 명령은 Windows PowerShell 기준입니다. 창을 다시 열었다면 다음을 실행합니다.

```powershell
. .\scripts\Use-ProjectStorage.ps1
.venv\Scripts\Activate.ps1
```

## 시작 전에 알아둘 용어

| 용어 | 의미 |
| --- | --- |
| Positive | 인식해야 할 호출어 음성 |
| Negative | 호출어가 아닌 대화·음악·환경음 |
| 참조 음성 | Qwen TTS가 음색을 따라 생성할 때 사용하는 합성 음성 |
| Feature(특징) | 음성을 모델이 학습할 수 있는 숫자 배열로 바꾼 자료 |
| 개발용 평가(Development) | 모델과 감지 설정을 고를 때 사용하는 평가 |
| 최종 평가(Final) | 모델과 설정을 확정한 뒤 별도 자료로 확인하는 평가 |

**호출어는 단독으로 생성**합니다. 뒤에 명령을 붙이지 않으며, 개인의 실제 호출 녹음은 학습·설정 조정·TTS 참조에 사용하지 않습니다.

## 1. 프로젝트 만들기

```powershell
koww init --wakeword 보라야 --id bora --output projects/bora --interactive
koww check
koww prepare-frontend --project projects/bora/project.json
koww storage --project projects/bora/project.json
```

첫 명령에서 **TTS 원본 수 / 원본당 증강 횟수 / 모델당 학습량 / 일반 비호출 시간 / 한국어 비호출 시간 / 환경소음 시간**을 선택합니다. [README의 세 단계](../README.md#호출어-학습-시작) 중 처음에는 **동작 확인용**을 권장합니다. 값 없이 Enter를 누르면 전체 규모가 적용됩니다.

| 단계 | 호출어 원본 / 증강 / 모델당 학습량 | 일반 비호출 / 한국어 비호출 / 환경소음 목표 |
| --- | --- | --- |
| 동작 확인용 | 1,000개 / R1 / 1,000회 | 10 / 1 / 1시간 |
| 중간 규모 | 20,000개 / R3 / 20,000회 | 500 / 50 / 10시간 |
| 전체 규모 | 100,000개 / R5 / 50,000회 | 2,000 / 200 / 30시간 |

세 단계 모두 CNN128 모델 3개와 같은 학습 방법을 사용합니다. 총 학습량은 모델당 학습량의 3배입니다. 작은 두 단계의 수치는 작업량을 줄인 시작 예시이며 성능 검증을 마친 설정은 아닙니다. 동작 확인용의 완료 기준은 **새 호출어를 실제 생성·증강·학습하고, 만든 모델로 감지·평가 결과를 얻는 것**입니다. 설치 확인이나 동봉 모델 실행만으로 이 과정을 대신하지 않습니다.

학습 설정을 비교·선정했어도 **위 데이터량의 최적값을 확인한 것은 아닙니다.** 새 호출어의 자료 규모는 [공식 자료·공개 사례와 데이터량 선정 기준](../plan.md#데이터량의-근거와-선택-원칙)을 검토해 선택합니다.

자료 시간은 준비 목표입니다. 기존 자료를 재사용할 수 있고, 이 단계에서 다운로드하거나 사용 구간을 자동으로 자르지는 않습니다. 정식 학습 전에는 선택한 비호출 시간과 실제 준비한 시간이 맞는지 확인합니다. 환경소음은 규모와 관계없이 다양한 종류를 확보하고 별도 배치 항목으로 세지 않습니다. 전체 규모는 30~50시간부터 시작합니다. 잔향은 서로 다른 공간의 자료를 데이터 준비 단계에서 연결합니다.

설정은 `project.json`, 예상 특징 수·용량·학습량은 `reports/setup_summary.json`에 저장됩니다. 예상 용량에는 호출어 특징 배열만 포함되므로 원본 음원·비호출 자료·평가 오디오 공간도 확보하세요. `init`은 설정만 만들고 학습을 시작하지 않습니다.

컴퓨터의 CPU·GPU 학습 속도를 간단히 비교하려면 다음을 실행합니다.

```powershell
koww benchmark --output projects/bora/reports/cpu_gpu.json
```

이 결과는 짧은 학습 연산의 비교입니다. 다운로드나 전체 음성 생성 시간까지 나타내지는 않습니다. 큰 작업은 대표 자료의 실제 처리 속도와 메모리 사용량도 확인하세요.

## 2. 합성 음색 준비·검수

처음에는 Qwen TTS로 참조 음성을 만듭니다. 기본 음색 후보는 학습용 100개, 개발용 10개, 최종 평가용 10개로 분리되어 있습니다. 후보 개수만으로 실제 음색 다양성이 보장되지는 않으므로 자연스러운 한국어인지, 지나치게 비슷한 음색이 반복되는지 들어봅니다.

**동작 확인용**이라면 아래 학습용 `references` 명령에 `--limit 4`, 개발용 명령에 `--limit 2`를 붙여 소수 음색으로 시험할 수 있습니다. 검수는 같은 방식으로 진행합니다. 이후 다양한 화자에 대한 성능을 확인할 때는 새 프로젝트에서 필요한 음색 계열을 확보하세요.

학습용 참조를 생성하고 검수 화면을 엽니다.

```powershell
koww references --project projects/bora/project.json --split train
koww review --project projects/bora/project.json --queue projects/bora/data/references/train/references.jsonl --kind references --port 8790
```

브라우저에서 [http://127.0.0.1:8790/](http://127.0.0.1:8790/)을 열고 연속 듣기를 시작하세요. 끝까지 들은 항목은 자동 보존됩니다. 문제가 있는 번호만 폐기·재검수로 바꾸고 **검수 완료·저장**을 누릅니다. 결과는 서버에 바로 저장되므로 채팅에 복사할 필요가 없습니다.

저장 후 터미널에서 **Ctrl+C로 검수 서버를 종료**하고 다음 단계로 진행하세요. 서버를 다시 열 때는 같은 `review` 명령을 실행합니다.

개발용 참조도 별도로 생성·검수합니다.

```powershell
koww references --project projects/bora/project.json --split validation
koww review --project projects/bora/project.json --queue projects/bora/data/references/validation/references.jsonl --kind references --port 8790
```

**최종 평가용 참조는 8단계의 모델·설정 확정 후에 생성**합니다. 이미 승인한 합성 참조를 다른 프로젝트에서 가져오는 방법은 아래의 추가 설정을 참고하세요.

## 3. 호출어를 소량 생성해 발음 확인하기

```powershell
koww generate --project projects/bora/project.json --pilot --limit 24
koww review --project projects/bora/project.json --queue projects/bora/data/pilot/train/candidates.jsonl --kind pilot --port 8790
```

파일럿에서는 **호출어가 명확히 들리고, 다른 말이 붙지 않는지** 확인합니다. 문제가 있으면 대량 생성 전에 참조 음색이나 생성 방식을 수정하고 변경한 부분을 다시 확인하세요. 폐기·재검수 항목이 남은 생성 방식을 자동 승인하지 않습니다.

이미 생성한 검수 목록은 한 번에 제공합니다. 새 음색이나 생성 방식이 바뀌었을 때 필요한 대표 검수만 추가합니다.

## 4. 선택한 수량의 호출어 생성하기

파일럿 승인 후 실행합니다.

```powershell
koww generate --project projects/bora/project.json --split train
koww generate --project projects/bora/project.json --split validation
koww status --project projects/bora/project.json
```

학습용은 처음 선택한 수량, 개발용은 기본 약 5,000개를 생성합니다. 무음·깨짐·길이·중복 등을 자동 검사하며 **음원마다 별도로 음성 인식(STT)을 실행하지 않습니다.** 생성 방식에 대한 대표 청취와 대량 생성 결과의 기술 검사는 구분합니다.

**동작 확인용**에서는 개발용 생성 명령을 아래처럼 바꿔 우선 100개만 만드세요. 학습용과 다른 개발용 음색·자료를 사용하는 원칙은 유지합니다.

```powershell
koww generate --project projects/bora/project.json --split validation --limit 100
```

100개 시험은 기본 개발용 목표 5,000개를 충족한 것으로 기록하지 않습니다. 성능을 검증하는 단계로 넘어갈 때는 `--limit` 없이 별도 실행해 목표 수량을 채웁니다.

각 생성 폴더의 `summary.json`에서 `target`(목표 수), `technical_pass`(자동 검사 통과 수), `full_target_reached`(목표 달성 여부)를 확인하세요. 실패한 항목은 정해진 재시도 후 기록에 남습니다.

중단된 생성은 같은 명령을 다시 실행하면 검증된 완료 항목을 재사용합니다. `--limit`는 추가 생성 수가 아니라 그 실행의 총 목표 수입니다.

## 5. 비호출·소음·잔향 자료 연결하기

[데이터 준비 안내](DATASETS.md)에 따라 공개 자료를 내려받습니다. 다운로드만으로 학습용 자료 목록이 만들어지지는 않습니다. 사용할 원본을 선택하고 출처·이용조건·학습/평가 구분·검수 결과를 기록해야 합니다.

[자료 목록 작성법](DATA_FORMAT.md)의 예시를 따라 다음 파일을 준비합니다.

| 프로젝트 파일 | 넣을 자료 |
| --- | --- |
| `data/negative/korean.jsonl` | 학습용 한국어 중심 비호출 음성 |
| `data/materials/noise.jsonl` | 학습용 환경음·음악·겹치는 대화 |
| `data/materials/rir.jsonl` | 학습용 잔향 자료 |
| `data/materials/validation_noise.jsonl` | 개발용 평가의 세 종류 배경음 |
| `data/materials/final_noise.jsonl` | 최종 평가용으로 따로 분리한 배경음 |

공개 일반 비호출 특징 파일(Generic)을 쓴다면 다운로드한 파일을 연결합니다.

```powershell
koww import-generic --project projects/bora/project.json --features shared/openwakeword_features_ACAV100M_2000_hrs_16bit.npy --output projects/bora/features/generic_negative
```

이 파일은 약 17.28GB이며 처음 연결할 때 전체 파일을 검사합니다. 원본 WAV가 없으므로 이 자료에 소음을 추가했다고 기록하지 않습니다.

한국어 원본은 학습할 2초 구간 목록으로 바꿉니다. 음원 자체를 구간마다 복사하지 않습니다.

```powershell
koww audit --project projects/bora/project.json
koww windows --project projects/bora/project.json --pool korean_negative --output projects/bora/data/negative/korean_windows.jsonl
```

완료 후 `project.json`의 `manifests` 안에 있는 `korean_negative` 값을 **`data/negative/korean_windows.jsonl`**로 바꾸세요. 학습을 시작한 뒤에는 이 설정을 변경하지 않습니다.

학습 전에는 **처음 선택한 일반 비호출·한국어 비호출 목표 시간**을 확인합니다. 전체 규모는 각각 2,000시간 / 200시간이며 작은 단계는 1단계 표의 시간을 명시적으로 선택합니다. 호출어 원본 수만 줄여도 자료 목표 시간이 자동으로 줄어들지는 않습니다. 공개 Generic 사전 계산 자료는 작은 단계에서도 같은 전체 파일을 연결하므로 다운로드·첫 검증 비용은 같습니다. 환경소음은 시간보다 종류의 다양성을 함께 확인합니다.

## 6. 특징 만들기·모델 학습하기

```powershell
koww features --project projects/bora/project.json --role korean_negative
koww features --project projects/bora/project.json --role positive
koww train --project projects/bora/project.json --output projects/bora/runs/selected_budget
koww package --project projects/bora/project.json --run projects/bora/runs/selected_budget --output projects/bora/models/candidate_selected
```

`features`는 소음·잔향·음량 변형을 적용하고 학습용 숫자 배열을 만듭니다. 호출어 원본 수 × 증강 횟수와 실제 생성 행 수가 일치해야 합니다. 같은 계열의 증강을 호출어와 원본 파형이 있는 비호출 음성에 적용합니다.

`train`은 선택한 학습량으로 모델을 만들고, `package`는 모델·공통 음성 처리 모델·실행 설정을 한 폴더에 모읍니다. 기본 구성은 독립적으로 학습한 3개 모델의 점수를 평균하는 앙상블입니다.

특징 생성이나 학습이 중단되면 **같은 설정·코드·입력**에서 원래 명령에 `--resume`을 붙여 이어갑니다. 완료한 결과를 덮어쓰려 하지 말고 다른 실험은 새 출력 폴더를 사용하세요.

## 7. 개발용 평가·감지 기준 선택하기

개발용 호출어와 배경음이 준비되면 같은 호출에 같은 소음 구간을 섞어 평가 오디오를 만듭니다.

```powershell
koww prepare-eval --project projects/bora/project.json --output projects/bora/data/eval/paired_v1 --split validation --originals 5000 --noise-sections 5
```

Clean(소음 없음)과 SNR 20·15·10·5·0dB를 만듭니다. SNR이 낮을수록 호출어에 비해 소음이 큽니다. 환경소음·음악·겹치는 한국어 대화 결과를 따로 비교합니다. 처음 흐름을 확인할 때는 `--originals 100`으로 줄일 수 있으나 최종 모델 선택은 계획한 평가 규모로 확인합니다. 여러 조건의 WAV가 생성되므로 [저장 공간](STORAGE_POLICY.md)도 확인하세요.

**동작 확인용**에서는 위 명령을 아래처럼 바꿔 호출 원본 100개와 배경 종류별 소음 구간 1개로 시험합니다.

```powershell
koww prepare-eval --project projects/bora/project.json --output projects/bora/data/eval/paired_v1 --split validation --originals 100 --noise-sections 1
```

동작 확인만 할 때는 아래 **`evaluate` 명령의 결과 저장까지 확인하면 이 단계를 마칩니다.** `tune`·`select-runtime`과 8단계의 최종 평가는 이후 성능 검증 때 진행합니다. 짧은 평가 결과를 오호출 목표 통과나 최종 모델로 표시하지 않으며, 개발용 20~30시간·최종 100시간 등의 평가 기준은 그대로 유지합니다.

출력의 `manifest.jsonl`에 학습에 쓰지 않은 연속 비호출 평가 목록을 합쳐 `data/eval/development.jsonl`을 만듭니다. **성능 검증 단계에서는 20~30시간**, 동작 확인용에서는 소량을 사용해 채점 흐름을 확인할 수 있습니다. [평가 목록 작성법](DATA_FORMAT.md#평가-자료-목록)에 형식과 합치는 예시가 있습니다.

```powershell
koww evaluate --project projects/bora/project.json --bundle projects/bora/models/candidate_selected --manifest projects/bora/data/eval/development.jsonl --output projects/bora/reports/dev_selected
koww tune --project projects/bora/project.json --bundle projects/bora/models/candidate_selected --evaluation projects/bora/reports/dev_selected --output projects/bora/reports/gates_selected --thresholds 0.5 0.7 0.85 0.9 0.95 0.98 0.9862500429153442 0.995 --consecutive 1 2 3
koww select-runtime --project projects/bora/project.json --bundle projects/bora/reports/gates_selected/bundle
```

`evaluate`는 모델별 호출 성공률·오호출·지연을 계산합니다. `tune`은 저장된 점수를 재사용해 감지 기준을 비교하므로 같은 오디오를 신경망으로 다시 계산하지 않습니다. `threshold`는 감지로 인정할 최소 점수, `consecutive`는 그 점수를 연속으로 넘어야 하는 프레임 수입니다.

평가 보고서의 `adoption.passed`가 목표 통과 여부입니다. 목표 미달이면 이유를 확인하고 필요한 부분을 보완하세요. 표본 모델의 임계값을 새 호출어에 그대로 확정하지 않습니다. 기준은 [학습 계획](../plan.md#5-평가-목표)에서 확인합니다.

## 8. 모델·설정 확정 후 최종 평가하기

개발용 평가를 통과한 모델과 실행 설정을 고정합니다.

```powershell
koww freeze --project projects/bora/project.json --bundle projects/bora/reports/gates_selected/bundle --development-report projects/bora/reports/gates_selected/evaluation/report.json
```

이후 최종 평가용 음색을 생성·검수합니다. 검수 저장 후 서버를 종료하세요.

```powershell
koww references --project projects/bora/project.json --split final
koww review --project projects/bora/project.json --queue projects/bora/data/references/final/references.jsonl --kind references --port 8790
```

```powershell
koww generate --project projects/bora/project.json --split final
koww prepare-eval --project projects/bora/project.json --output projects/bora/data/eval/final_paired_v1 --split final --originals 5000 --noise-sections 5
```

이 단계에서 만든 호출어 목록과 **별도 연속 비호출 100시간 이상**을 합쳐 `data/eval/final.jsonl`을 준비합니다. 학습·개발용 자료와 음색·원본을 공유하지 않습니다.

```powershell
koww evaluate --project projects/bora/project.json --bundle projects/bora/reports/gates_selected/bundle --stage final --manifest projects/bora/data/eval/final.jsonl --output projects/bora/reports/final_v1
```

최종 평가 자료는 모델 선택이나 설정 조정에 사용하지 않습니다. 최종 결과를 보고 개선하려면 새 실험과 새로운 독립 최종 평가 자료를 사용하세요.

실제 사람의 호출 성능을 확인할 때는 해당 음성을 최종 평가에만 사용하고 합성 결과와 따로 집계합니다. 화자·거리·방향·마이크 조건과 표본 수를 기록합니다. 자료를 확보하지 않았다면 실제 사람 평가를 미검증으로 남깁니다.

## 9. 결과 사이트에서 확인하기

```powershell
koww site --evaluation projects/bora/reports/final_v1 --output projects/bora/reports/final_v1/site
koww serve --directory projects/bora/reports/final_v1/site --port 8791
```

[http://127.0.0.1:8791/](http://127.0.0.1:8791/)에서 모델 순위와 실제 시험 오디오를 확인하세요. 재생·통계 읽는 방법은 [결과 사이트 안내](TRAINING_RESULTS_SITE.md)에 있습니다.

## 추가 설정과 재사용

### 승인한 합성 음색 가져오기

이미 이 라이브러리로 검수한 이전 프로젝트가 있을 때만 사용합니다. 예시의 `projects/previous/project.json`을 실제 프로젝트 경로로 바꾸세요.

```powershell
koww import-references --project projects/bora/project.json --source-project projects/previous/project.json --split train
koww import-references --project projects/bora/project.json --source-project projects/previous/project.json --split validation
```

참조는 원래의 학습/개발/최종 구분을 유지하며, 검수 기록과 파일 해시가 확인되어야 합니다. 실제 사람 참조는 허용하지 않습니다. 최종 평가용 참조 재사용도 모델·설정 확정 후에만 가능합니다.

### 세부 설정·증강 횟수 비교

처음 프로젝트를 만들 때 `--advanced`를 붙이면 모델 구조·학습률 등을 선택할 수 있습니다. 현재 설정 확인은 다음과 같습니다.

```powershell
koww settings --project projects/bora/project.json --advanced
```

R1과 R3을 비교하려면 원본 수와 다른 조건을 고정하고 설정을 별도 파일에 저장합니다.

```powershell
koww configure --project projects/bora/project.json --output projects/bora/project_R1.json --non-interactive --rounds 1
koww configure --project projects/bora/project.json --output projects/bora/project_R3.json --non-interactive --rounds 3
koww features --project projects/bora/project_R1.json --role positive
koww features --project projects/bora/project_R3.json --role positive
```

각 설정의 `pool_features.positive`에 생성된 경로를 사용해 `compare-rounds`로 확인하고, 같은 학습량·Negative·배치·seed로 별도 학습합니다. 필요하면 같은 방식으로 R5를 추가합니다. R 비교와 원본 수 비교는 다른 실험입니다.

학습량을 바꾸면 기본 학습률 변화 기간과 저장 시점도 연동됩니다. 같은 기간에서 학습량만 비교하려면 `--schedule-horizon`을 지정합니다. 자세한 통제 기준과 지원 설정은 [학습 계획](../plan.md)에 있습니다.

### Python에서 직접 학습한 모델 사용하기

```python
from korean_openwakeword import WakeWordDetector

detector = WakeWordDetector("projects/bora/reports/gates_selected/bundle")
result = detector.predict_file("test.wav")
print(result["events"])
```

입력은 16kHz·모노·16비트 PCM WAV입니다. 기본 설정에서는 시작 후 2초의 준비 구간이 필요합니다. 실시간 입력은 같은 형식의 80ms 음성 배열(1,280개 샘플)을 `process`에 전달하고 새 세션에서는 `reset`합니다.

모델 구조·배치·VAD(음성 구간 감지)·노이즈 제거의 지원 범위와 [이용조건](../THIRD_PARTY_NOTICES.md)을 확인하세요.
