# 한국어 호출어 학습 가이드

이 가이드는 **새 호출어를 생성하고 모델을 학습한 뒤, 평가 결과를 사이트로 확인하는 과정**을 안내합니다. 예시는 ‘보라야’이며, 명령은 저장소 폴더에서 실행합니다.

처음 사용하는 경우 `koww web`으로 [웹 학습 화면](WEB_GUIDE.md)을 여세요. 자료 준비·검수·생성·학습·평가·마이크 시험을 한 화면에서 진행하며, 아래 개별 명령은 직접 자료와 실험을 관리할 때 사용합니다.

먼저 [README의 기본 설치](../README.md#설치)를 마친 뒤, [학습 데이터 준비의 음성 생성·학습 도구](../README.md#학습-데이터-준비)를 추가로 설치하세요. 아래 명령은 Windows PowerShell 기준입니다. 창을 다시 열었다면 다음을 실행합니다.

```powershell
. .\scripts\Use-ProjectStorage.ps1
.venv\Scripts\Activate.ps1
```

## 시작 전에 알아둘 용어

| 용어 | 의미 |
| --- | --- |
| Positive | 인식해야 할 호출어 음성 |
| Negative | 호출어가 아닌 대화·음악·환경음 |
| Hard Negative(유사어) | 호출어와 발음이 비슷하지만 다른 표현. 학습 정답은 비호출 |
| 참조 음성 | Qwen TTS가 음색을 따라 생성할 때 사용하는 합성 음성 |
| Feature(특징) | 음성을 모델이 학습할 수 있는 숫자 배열로 바꾼 자료 |
| 개발용 평가(Development) | 모델과 감지 설정을 고를 때 사용하는 평가 |
| 최종 평가(Final) | 모델과 설정을 확정한 뒤 별도 자료로 확인하는 평가 |

**호출어는 단독으로 생성**합니다. 뒤에 명령을 붙이지 않으며, 개인의 실제 호출 녹음은 학습·설정 조정·TTS 참조에 사용하지 않습니다.

## 1. 프로젝트 만들기

```powershell
koww init --wakeword 보라야 --id bora --output projects/bora --interactive --validation-voices 30
koww check
koww prepare-frontend --project projects/bora/project.json
koww storage --project projects/bora/project.json
```

첫 명령에서 **TTS 원본 수 / 원본당 증강 횟수 / 모델당 학습량 / 일반 비호출 시간 / 한국어 비호출 시간 / 환경소음 시간**을 선택합니다. [README의 세 단계](../README.md#학습-규모-선택) 중 처음에는 **동작 확인용**을 권장합니다. 값 없이 Enter를 누르면 상용 모델용 설정이 적용됩니다.

| 단계 | 호출어 원본 / 증강 / 모델당 학습량 | 일반 비호출 / 한국어 비호출 자료 |
| --- | --- | --- |
| 동작 확인용 | 1,000개 / R1 / 10,000회 | 2,000 / 1시간 |
| 시험 모델용 | 20,000개 / R3 / 50,000회 | 2,000 / 약 667시간 |
| 상용 모델용 | 100,000개 / R5 / 100,000회 | 7,000 / 3,000시간 |

세 단계 모두 CNN128 모델 3개와 같은 학습 방법을 사용합니다. 총 학습량은 모델당 학습량의 3배입니다. 작은 두 단계의 수치는 작업량을 줄인 시작 예시이며 성능 검증을 마친 설정은 아닙니다. 동작 확인용의 완료 기준은 **새 호출어를 실제 생성·증강·학습하고, 만든 모델로 감지·평가 결과를 얻는 것**입니다. 설치 확인이나 동봉 모델 실행만으로 이 과정을 대신하지 않습니다.

학습 설정을 비교·선정했어도 **위 데이터량의 최적값을 확인한 것은 아닙니다.** 새 호출어의 자료 규모는 [공식 자료·공개 사례와 데이터량 선정 기준](../plan.md#데이터량의-근거와-선택-원칙)을 검토해 선택합니다.

자료 시간은 준비 목표입니다. 시험 모델용의 일반·한국어 비호출은 **75:25**이며 정확한 한국어 목표는 666⅔시간입니다. 상용 모델용은 **일반 7,000시간·한국어 3,000시간(70:30)**으로 구성합니다. **동작 확인용은 한국어 비호출을 1시간만 준비**해 학습 절차를 확인합니다. 자료 시간과 별개로 학습 배치의 일반·한국어 비호출 비중은 세 단계 모두 **75:25**입니다. 시간 선택만으로 자료를 다운로드하거나 자르지는 않습니다. 동작 확인용·시험 모델용의 소음 재료는 [공식 예제 안내](DATASETS.md#공식-자동-학습-예제의-증강-재료)에 따라 **AudioSet `bal_train09.tar` 1개·FMA 음악 1시간**을 사용하며, 잔향 재료는 **세 단계 모두 MIT RIR `train` 전체**를 사용합니다. 원본 비호출 오디오도 증강에 재사용할 수 있으며 소음·잔향 재료는 별도 배치 항목으로 세지 않습니다.

세 단계의 배치는 **일반 768 / 한국어 256 / 호출어 100 / 유사어 100 = 1,224개**입니다. 유사어는 별도 자료로 준비하며, 자료가 없으면 임의의 일반 비호출로 대체하거나 해당 배치를 생략하지 않습니다.

상용 모델용의 일반 7,000시간은 **AudioSet 혼합 음원 3,000시간·MLS 다국어 낭독 3,000시간·FMA full 음악 1,000시간**, 한국어 3,000시간은 **YODAS2 한국어 웹 음성**으로 계획합니다. 원본 다운로드와 연결 방법은 [상용 비호출 구성](DATASETS.md#상용-모델용-비호출-구성)을 참고하세요. 이 계획만으로 자료가 내려받아지거나 사용 승인되는 것은 아닙니다.

설정은 `project.json`, 예상 특징 수·용량·학습량은 `reports/setup_summary.json`에 저장됩니다. 예상 용량에는 호출어 특징 배열만 포함되므로 원본 음원·비호출 자료·평가 오디오 공간도 확보하세요. `init`은 설정만 만들고 학습을 시작하지 않습니다.

컴퓨터의 CPU·GPU 학습 속도를 간단히 비교하려면 다음을 실행합니다.

```powershell
koww benchmark --output projects/bora/reports/cpu_gpu.json
```

이 결과는 짧은 학습 연산의 비교입니다. 다운로드나 전체 음성 생성 시간까지 나타내지는 않습니다. 큰 작업은 대표 자료의 실제 처리 속도와 메모리 사용량도 확인하세요.

## 2. 합성 음색 준비·검수

처음에는 Qwen TTS로 참조 음성을 만듭니다. 이 가이드의 프로젝트는 학습용100개·공통 개발용30개·최종 평가용10개의 음색 후보를 분리합니다. `init --validation-voices 30`이 개발용 수량을 정하며 웹의 공통 평가도30계열을 사용합니다. 후보 개수만으로 실제 음색 다양성이 보장되지는 않으므로 자연스러운 한국어인지, 지나치게 비슷한 음색이 반복되는지 들어봅니다.

빠른 명령 기능 시험이라면 학습용 `references`에 `--limit 4`, 개발용 명령에 `--limit 2`를 붙여 소수 음색으로 실행할 수 있습니다. **웹의 공통 평가 v4는 학습 규모와 관계없이 개발용30계열을 따로 준비**하며, 소수 음색 시험을 공통 평가 완료로 기록하지 않습니다.

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

CUDA·BF16·SDPA로 단독 호출을 생성할 때는 GPU 가속을 자동 비교합니다. 각 참조 음색의 첫 발화를 같은 난수 상태로 반복해 학습용 PCM·품질 검사·CPU/GPU 난수 상태가 모두 같은지 확인합니다. 초기 준비 비용을 제외한 재실행이 기본 경로보다 최소 5% 빨라야 가속을 사용하며, 출력이 다르거나 충분히 빠르지 않으면 기본 결과를 사용합니다. 비교용 반복 음성은 학습 자료에 추가하지 않습니다. 확인 결과와 실측 시간은 프로젝트의 `runtime/tts_execution.json`에 저장합니다. 기본 경로를 직접 비교하려면 생성 명령에 `--tts-backend eager`를 붙입니다. 출력이 같다는 확인은 음성의 발음 검수를 대신하지 않습니다.

유사어 원본도 Qwen으로 만들 수 있습니다. 아래 명령은 **후보만 생성**하며, 대표 발음 검수와 자료 사용 승인 전에는 학습에 넣지 않습니다. `--count`는 원본 수이고, 배치의 유사어 100개는 매 학습 단계의 추출량입니다.

**호출어 자체나 호출어의 동일한 이름·낱말이 통째로 들어간 표현(호격 생략·변경 포함)은 유사어에서 제외합니다.** ‘보라야’의 ‘보라’는 제외하고, 다른 낱말인 ‘보리야’·‘보람아’·‘돌아와’는 사용할 수 있습니다. 공통 음절이나 부분 문자열만으로 제외하지 않습니다. 같은 표현을 서로 다른 음색·발음으로 생성하는 것은 가능하지만, 동일 WAV를 복제해 수량을 채우지는 않습니다.

```powershell
koww generate-hard --project projects/bora/project.json --phrases 보리야 보람아 돌아와 --count 100
```

공개 소음·잔향과 유사어를 한 번에 대표 검수할 때는 원본 목록과 대표 목록을 먼저 고정합니다. 각 데이터셋·소리 범주와 새 유사어의 표현·음색 조합을 대표 목록에 포함하고, `review --kind content --pool-manifest 원본목록.jsonl`로 엽니다. 화면에서 대표 음원을 듣고 **자료 묶음 사용 동의**를 직접 체크해야 전체 목록을 승인합니다. 듣지 않은 원본은 개별 청취 완료로 기록하지 않습니다. 폐기·재검수가 있으면 먼저 해결하고 다시 승인합니다.

폐기한 표현은 원본 목록과 승인 범위에서도 제외합니다. 승인된 원본이 75개여도 학습 배치에서는 100개를 무작위 복원 추출할 수 있습니다. 원본 수와 배치 추출량을 구분하고, 배치를 맞추려고 폐기한 표현이나 복제 WAV를 추가하지 않습니다.

학습 규모와 관계없이 **명령의 빠른 기능 시험만 따로 할 때**는 개발용 생성 수를 아래처럼 100개로 제한할 수 있습니다. 기본 공통 평가는 5,000개이며, 학습용과 다른 개발용 음색·자료를 사용하는 원칙은 유지합니다.

```powershell
koww generate --project projects/bora/project.json --split validation --limit 100
```

100개 시험은 기본 개발용 목표 5,000개를 충족한 것으로 기록하지 않습니다. 성능을 검증하는 단계로 넘어갈 때는 `--limit` 없이 별도 실행해 목표 수량을 채웁니다.

각 생성 폴더의 `summary.json`에서 `target`(목표 수), `technical_pass`(자동 검사 통과 수), `full_target_reached`(목표 달성 여부)를 확인하세요. 실패한 항목은 정해진 재시도 후 기록에 남습니다.

중단된 생성은 같은 명령을 다시 실행하면 검증된 완료 항목을 재사용합니다. `--limit`는 추가 생성 수가 아니라 그 실행의 총 목표 수입니다.

## 5. 비호출·소음·잔향 자료 연결하기

[데이터 준비 안내](DATASETS.md)에 따라 공개 자료를 내려받고 압축을 풉니다. [공개 음원 자동 준비](DATA_FORMAT.md#공개-음원-자동-준비)의 `index-data → prepare-materials → review → admit-materials` 순서로 변환·분리·검수·연결합니다. 자료 목록과 파일 해시는 명령이 기록하므로 직접 작성할 필요가 없습니다. 여러 소음 묶음은 `connect-materials`로 함께 연결합니다.

자동 연결 결과는 다음 프로젝트 목록에 저장됩니다. 외부에서 만든 목록을 연결하려면 [자료 목록 작성법](DATA_FORMAT.md)을 참고하세요.

| 프로젝트 파일 | 넣을 자료 |
| --- | --- |
| `data/negative/korean.jsonl` | 학습용 한국어 중심 비호출 음성 |
| `data/negative/hard.jsonl` | 호출어와 발음이 비슷한 다른 표현의 학습용 음성 |
| `data/materials/noise.jsonl` | 학습용 환경음·음악·겹치는 대화 |
| `data/materials/rir.jsonl` | 학습용 잔향 자료 |
| `data/materials/validation_noise.jsonl` | 개발용 평가의 세 종류 배경음 |
| `data/materials/final_noise.jsonl` | 최종 평가용으로 따로 분리한 배경음 |

공개 일반 비호출 특징 파일(Generic)을 쓴다면 다운로드한 파일을 연결합니다.

```powershell
koww import-generic --project projects/bora/project.json --features shared/openwakeword_features_ACAV100M_2000_hrs_16bit.npy --output projects/bora/features/generic_negative
```

이 파일은 약 17.28GB이며 처음 연결할 때 전체 파일을 검사합니다. 원본 WAV가 없으므로 이 자료에 소음을 추가했다고 기록하지 않습니다.

`admit-materials --pools korean_negative --connect`는 원본을 복사하지 않는 2초 구간 목록까지 연결합니다. 아래 수동 `windows` 명령은 외부에서 작성한 기존 원본 목록을 연결할 때만 필요합니다.

```powershell
koww audit --project projects/bora/project.json
koww windows --project projects/bora/project.json --pool korean_negative --output projects/bora/data/negative/korean_windows.jsonl
```

수동 목록을 사용했다면 완료 후 `project.json`의 `manifests.korean_negative`를 **`data/negative/korean_windows.jsonl`**로 바꿉니다. 자동 연결 경로는 이 수정이 필요하지 않습니다. 학습을 시작한 뒤에는 입력을 변경하지 않습니다.

학습 전에는 **처음 선택한 일반 비호출·한국어 비호출 목표 시간**을 확인합니다. 상용 모델용은 각각 7,000시간 / 3,000시간이며 작은 단계는 1단계 표의 시간을 명시적으로 선택합니다. 공개 Generic 파일은 약 2,000시간만 포함하므로 상용 모델용에는 더 큰 자료가 필요합니다. 원본 일반 비호출 목록을 준비했다면 `windows --pool generic_negative`와 `features --role generic_negative`로 특징을 만들 수 있습니다. 공개 파일을 복제해 7,000시간으로 계산하거나 여러 특징 배열이 자동 통합된다고 가정하지 않습니다. 호출어 원본 수만 줄여도 자료 목표 시간이 자동으로 줄어들지는 않습니다.

유사어는 [유사어 자료 안내](DATA_FORMAT.md#유사어-학습-자료)에 따라 내용·출처·검수 기록을 준비합니다. 긴 원본은 다음으로 고정 2초 목록을 만들고, `project.json`의 `manifests.hard_negative`를 `data/negative/hard_windows.jsonl`로 바꿉니다.

```powershell
koww windows --project projects/bora/project.json --pool hard_negative --output projects/bora/data/negative/hard_windows.jsonl
```

## 6. 특징 만들기·모델 학습하기

```powershell
koww features --project projects/bora/project.json --role korean_negative
koww features --project projects/bora/project.json --role hard_negative
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
koww prepare-eval --project projects/bora/project.json --output projects/bora/data/eval/paired_v4 --split validation --originals 5000 --noise-sections 1 --noise-selection balanced_sources --reverb-originals 1000 --reverb-selection metadata_axes
```

공통 평가 v4는30음색·호출 원본5,000개, 비호출30시간을 [평가 자료 기준](DATASETS.md#성능-평가용-배경-음원)에 따라 준비합니다. 잔향을 추가하지 않은 Clean와 세 배경범주×SNR20·15·10·5·0dB의80,000회에, 고르게 고른1,000개 원본의 일반·강한 잔향32,000회를 더해 **112,000개 WAV**를 만듭니다. 두 잔향 강도에 같은1,000개를 쓰며, 대응하는 잔향 없는 조건은 기본80,000회에 포함됩니다. 대표 원본 수는 시험 비용과 다양성을 고려한 선택입니다. `rir_validation`에는 학습과 분리한 가상 공간20개·RIR80개가 필요합니다. SNR이 낮을수록 소음이 강하며, 같은 원본의 배경·구간은 SNR별로 고정합니다. 비교할 모든 모델에 같은 평가 WAV·목록을 재사용하세요. 기존 v1/v2/v3 프로젝트나 완료 결과는 자동 변경하지 않습니다. [저장 공간](STORAGE_POLICY.md)도 확인합니다.

기존 Clean·소음 조건 표는 **잔향 추가 없는 기본80,000회**를 기준으로 유지합니다. 잔향 결과는 별도 표에 실제 표본 수와 함께 표시하므로 새로운 잔향 시험이 기존 성능 기준을 바꾸지 않습니다.

학습 규모와 관계없이 **빠른 기능 시험만 선택할 때**는 아래처럼 원본100개로 일부를 실행할 수 있습니다. 이는 기본 공통 평가 전체를 완료한 결과가 아닙니다.

```powershell
koww prepare-eval --project projects/bora/project.json --output projects/bora/data/eval/quick_v1 --split validation --originals 100 --noise-sections 1 --noise-selection balanced_sources
```

빠른 기능 시험은 아래 `evaluate`의 저장까지 확인하고 마칠 수 있습니다. 전체 공통 평가와 감지 설정 선택은 정해진 표본으로 진행하며 독립 최종 평가는8단계에서 따로 수행합니다. 짧은 결과를 오호출 목표 통과나 최종 성능으로 표시하지 않습니다.

출력의 `manifest.jsonl`에 준비·승인한 **일곱 범주·고유 비호출30시간**의 목록을 합쳐 `data/eval/development.jsonl`을 만듭니다. 같은 기본 원본을 호출어 혼합과 배경 단독 시험에 사용하며 자료 시간을 이중 계산하지 않습니다. 기존193개 배경과 별도 낭독 예약분은 새 공통 평가에 연결하지 않습니다. 오호출 시험 초기화로 제외되는 시간은 같은 자료 목록의 미사용 원음으로 필요한 만큼 보충하며, 호출어 혼합에는 추가하지 않습니다. 기본30시간·보충량·검수 승인·실제 채점 가능한 시간을 구분하고, 짧은 음원을 반복해 시간을 채우지 마세요. 웹은 같은 준비 절차를 연결하며 부족분을 표시합니다. [평가 목록 작성법](DATA_FORMAT.md#평가-자료-목록)에 형식과 합치는 예시가 있습니다.

```powershell
koww evaluate --project projects/bora/project.json --bundle projects/bora/models/candidate_selected --manifest projects/bora/data/eval/development.jsonl --output projects/bora/reports/dev_selected --workers auto
koww site --evaluation projects/bora/reports/dev_selected --output projects/bora/reports/dev_selected/site
koww serve --directory projects/bora/reports/dev_selected/site --port 8791
```

[http://127.0.0.1:8791/](http://127.0.0.1:8791/)에서 단일 모델과 앙상블을 비교하고 Ctrl+C로 서버를 종료합니다. 사용할 후보를 선택한 뒤 감지 설정을 비교하세요.

```powershell
koww select-candidate --project projects/bora/project.json --bundle projects/bora/models/candidate_selected --evaluation projects/bora/reports/dev_selected --candidate member3_seed20260930 --output projects/bora/models/selected_v1
koww tune --project projects/bora/project.json --bundle projects/bora/models/selected_v1/bundle --evaluation projects/bora/models/selected_v1/evaluation --output projects/bora/reports/gates_selected --thresholds 0.5 0.7 0.85 0.9 0.95 0.98 0.9862500429153442 0.995 --consecutive 1 2 3
koww select-runtime --project projects/bora/project.json --bundle projects/bora/reports/gates_selected/bundle
```

`evaluate`는 모델별 호출 성공률·오호출·지연을 계산합니다. `tune`은 저장된 점수를 재사용해 감지 기준을 비교하므로 같은 오디오를 신경망으로 다시 계산하지 않습니다. `threshold`는 감지로 인정할 최소 점수, `consecutive`는 그 점수를 연속으로 넘어야 하는 프레임 수입니다.

`--workers auto`는 전체 오디오 단위로 CPU 기준 계산을 나누며, CPU·여유 RAM과 작업 수에 따라 최대4개 작업자를 사용합니다. 프레임 점수는 디스크에 저장하고32세션마다 재개 상태를 남깁니다. 중단한 평가만 같은 입력·모델·설정에서 원래 명령에 `--resume`을 붙여 이어갈 수 있습니다. 작업자 수를 바꿔도 점수 계산은 같으며 완료 결과는 덮어쓰지 않습니다. 실제131개 음원의 검증·시작 포함 실측은 순차36.4초·4개19.1초였고, 전체112,000회의 예상 시간으로 단정하지 않습니다.

`candidate_adoptions`는 모든 단일 모델·앙상블 각각의 통과 여부이며 `adoption`은 지정한 후보의 판정입니다. `select-candidate`의 위 ID는 예시이므로 실제 결과에 표시된 ID로 바꾸세요. `mean`을 고르면 세 모델 평균, 단일 ID를 고르면 해당 모델 하나만 실행 묶음에 들어갑니다. 저장된 점수와 같은 채점 오디오를 재사용하며 자동 재학습은 없습니다. 목표 미달이면 이유를 확인하고 필요한 부분을 보완하세요. 표본 모델의 임계값을 새 호출어에 그대로 확정하지 않습니다. 기준은 [학습 계획](../plan.md#5-평가-목표)에서 확인합니다.

## 8. 모델·설정 확정 후 최종 평가하기

개발용 평가를 통과한 모델과 실행 설정을 고정합니다.

```powershell
koww freeze --project projects/bora/project.json --bundle projects/bora/reports/gates_selected/bundle --development-report projects/bora/reports/gates_selected/evaluation/report.json
```

동결 후에는 학습·공통 평가 입력 목록과 실제 개발 평가 자료가 해시로 고정됩니다. Final용 배경은 `project.json`이 지정한 `noise_final` 목록에 실제 출처·원본 경로·해시·`split: final`을 넣어 별도로 연결합니다. 학습·공통 평가 자료를 재분류하지 않습니다. 미리 예약한 Final 목록이 동결 전에 비어 있지 않았다면 해당 목록도 고정되므로 변경할 수 없습니다. [자료 형식](DATA_FORMAT.md)을 참고하세요.

이후 별도 최종 평가용 음색을 생성·검수합니다. 검수 저장 후 서버를 종료하세요.

```powershell
koww references --project projects/bora/project.json --split final
koww review --project projects/bora/project.json --queue projects/bora/data/references/final/references.jsonl --kind references --port 8790
```

```powershell
koww generate --project projects/bora/project.json --split final
koww prepare-eval --project projects/bora/project.json --output projects/bora/data/eval/final_paired_v1 --split final --originals 5000 --noise-sections 5 --noise-selection balanced_sources
```

이 단계에서 만든 호출어 목록과 **별도 연속 비호출100시간 이상**을 합쳐 `data/eval/final.jsonl`을 준비합니다. 학습·공통 평가와 음색·원본·알려진 화자·음악가·공간을 공유하지 않습니다. 생성된 혼합 오디오는 `components`에 호출어·배경의 원본 정보를 보존합니다. 다른 도구로 섞은 Final 오디오도 이 출처 정보를 제공해야 합니다.

채점 전에 실제 입력 분리를 검사합니다. 파일명이 달라도 같은 원본 ID·해시·동일 길이 PCM이 겹치면 거부합니다. 원본을 알 수 없는 Generic 특징이나 미확인 화자 등의 검사 한계도 보고합니다.

```powershell
koww audit-final --project projects/bora/project.json --manifest projects/bora/data/eval/final.jsonl --output projects/bora/reports/final_source_audit_v1.json
```

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

### 마이크로 모델 시험하기

새로 학습하지 않고도 기본 설치에 포함된 ‘춘식아’ 모델을 마이크로 시험할 수 있습니다. 추가 패키지를 설치하지 않고 실행합니다.

```powershell
koww mic
```

직접 학습한 모델은 `manifest.json`이 있는 모델 묶음 폴더를 지정합니다. 아래 경로를 자신의 `package` 결과 폴더로 바꾸세요.

```powershell
koww mic --bundle projects/bora/models/candidate_selected
```

1. [http://127.0.0.1:8792/](http://127.0.0.1:8792/)을 열고 사용할 마이크를 선택합니다.
2. **시작**을 누르고 브라우저의 마이크 권한 요청을 허용합니다. 허용 전에는 장치 이름이 표시되지 않을 수 있습니다.
3. 준비 표시가 끝난 뒤(기본 2초) 화면에 나온 호출어를 말합니다.
4. 입력 크기와 각 모델의 점수·감지 여부·감지 횟수를 확인합니다. 모델 3개가 있는 묶음은 세 점수의 평균에 따른 앙상블 결과도 보여줍니다.
5. **중지**를 누르면 마이크 입력을 종료합니다. 다시 시작하면 모델의 입력 이력과 준비 시간이 초기화됩니다.

마이크 시험은 **기존 모델의 반응을 확인하는 기능**입니다. 임계값·연속 프레임·초기 준비·재감지 간격은 모델 묶음의 실행 설정을 그대로 사용합니다. 개인 음성은 녹음 파일로 저장하지 않으며, 학습·임계값 조정·TTS 참조로 사용하지 않습니다.

자동 음량 보정은 하지 않습니다. 브라우저의 자동 음량 조절·소음 제거·에코 제거를 끄도록 요청하고 실제 장치 설정을 화면에 표시합니다. 장치나 운영체제에서 별도로 적용하는 음성 처리는 다를 수 있습니다.

서버가 실행되는 동안 터미널을 열어 두세요. 종료는 **Ctrl+C**, 재실행은 같은 `mic` 명령입니다. 포트를 바꾸려면 `--port 8793`처럼 지정합니다. 권한이 거부되면 브라우저의 사이트 권한에서 마이크를 허용하고, 다른 장치가 선택되었거나 입력이 없다면 마이크를 다시 선택해 시작하세요.

### Python에서 직접 학습한 모델 사용하기

```python
from korean_openwakeword import WakeWordDetector

detector = WakeWordDetector("projects/bora/reports/gates_selected/bundle")
result = detector.predict_file("test.wav")
print(result["events"])
```

입력은 16kHz·모노·16비트 PCM WAV입니다. 기본 설정에서는 시작 후 2초의 준비 구간이 필요합니다. 실시간 입력은 같은 형식의 80ms 음성 배열(1,280개 샘플)을 `process`에 전달하고 새 세션에서는 `reset`합니다.

모델 구조·배치·VAD(음성 구간 감지)·노이즈 제거의 지원 범위와 [이용조건](../THIRD_PARTY_NOTICES.md)을 확인하세요.
