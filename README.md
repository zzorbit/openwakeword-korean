# Korean openWakeWord

**영어 호출어에 특화된 openWakeWord를 한국어 호출어 전용으로 수정한 라이브러리입니다.**

이 프로젝트는 **[openWakeWord](https://github.com/dscripka/openWakeWord)(David Scripka 및 기여자)를 기반으로 개발한 파생 프로젝트**입니다. 원 프로젝트의 공통 음성 특징 추출 기능을 사용하며, 한국어 Qwen TTS 데이터 생성·소음 증강·판별기 학습·평가·결과 사이트를 위한 절차를 추가했습니다. 원 프로젝트와 별도로 개발·관리합니다.

## 예시 모델과 성능

설치 후, 동봉된 ‘춘식아’ 예시 모델을 음성 파일이나 실시간 마이크 입력으로 시험할 수 있습니다.

| 호출어 | 모델 구성 | 조용한 환경 | 소음 평균 | SNR 0 dB 평균 | 오호출/시간 |
| --- | --- | ---: | ---: | ---: | ---: |
| 춘식아 | CNN128 앙상블 · R5 | **100%**<br>5,000/5,000회 | **99.18%**<br>5,951/6,000회 | **89.73%**<br>1,346/1,500회 | **0.083회**<br>2회/24.03시간 |

합성 호출 음성의 개발용 평가 결과이며, 상세 조건과 검증 범위는 [표본 모델 설명서](korean_openwakeword/resources/sample_chunsik/MODEL_CARD.md)에 있습니다.

## 설치

**Git과 Python 3.10–3.13**을 준비하세요. 아래 명령은 **Windows PowerShell** 기준입니다.

```powershell
git clone https://github.com/zzorbit/openwakeword-korean.git
cd openwakeword-korean
. .\scripts\Use-ProjectStorage.ps1
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install .
python -m pip install openwakeword==0.6.0 --no-deps
koww check
```

`koww check`는 필요한 패키지의 설치 여부·버전과 GPU 사용 가능 여부를 확인합니다. 학습이나 모델 성능 시험을 시작하는 명령은 아닙니다. openWakeWord는 설치 호환성을 위해 위 명령처럼 별도로 설치합니다.

`Use-ProjectStorage.ps1`은 다운로드 캐시와 임시 파일을 저장소 폴더 안에 모아 저장하도록 설정합니다. 저장 위치를 바꾸려면 [저장 위치 안내](docs/STORAGE_POLICY.md)를 참고하세요.

## 웹에서 학습·시험하기

처음 사용한다면 설치 후 아래 명령 하나로 시작하세요.

```powershell
koww web
```

[학습 화면](http://127.0.0.1:8793/)에서 **호출어·작업 규모**를 정하고 **만들고 준비 시작**을 누릅니다. 자료와 결과는 `models/model_0001`, `models/model_0002`처럼 번호가 붙은 폴더에 자동 저장됩니다. 저장될 전체 경로는 화면에 표시됩니다. 필요한 학습 도구가 없으면 화면에서 CPU/GPU용 도구를 선택해 설치할 수 있습니다. Windows에서는 `scripts\Start-Web.cmd`를 더블클릭해도 같은 화면이 열립니다.

데스크톱에서는 **대시보드 안에서 작업 화면을 전환**합니다. 진행 상태와 주요 버튼을 확인하면서 자료·검수·결과·마이크 시험을 열 수 있습니다. 긴 목록과 기록은 해당 영역에서 확인하고, 작은 화면에서는 읽기 편하게 배치가 바뀝니다.

음성 생성·모델 학습·공통 평가는 **GPU를 우선 사용**합니다. 공통 평가는 GPU 실행을 확인한 뒤 배치로 처리하고, 임계값 근처 음원은 CPU로 다시 확인합니다. GPU 실행에 실패하면 원인을 표시하고 멈추며 CPU로 자동 전환하지 않습니다. CPU 평가는 직접 선택한 경우에만 실행합니다. 평가 화면에서 완료 수와 진행률을 확인할 수 있습니다.

**자료 다운로드·전처리 → 필요한 음원 검수 → 호출어 생성·증강 → 모델 학습·평가 → 평가 오디오·마이크 시험**이 이어집니다. 검수는 이 화면 안에서 저장하며, 명령이나 파일 경로를 단계마다 다시 입력하지 않습니다. 현재 단계 후 일시정지할 수 있고, 서버나 컴퓨터를 다시 켠 뒤 같은 명령으로 열어 **이어서 진행**을 누르면 됩니다.

새 모델은 **감지 기준(임계값) 0.5·연속 감지 1회**로 평가를 시작합니다. 공통 평가 후에는 **임계값 0.01–0.99를 0.01 간격으로, 연속 감지 1–6회를 모두 비교**합니다. 깨끗한 환경과 반응속도 기준을 지키고 **오호출이 시간당 1회 이하**인 후보 중 강한 소음과 잔향에서 잘 감지하는 설정을 모델마다 추천합니다. 이 추천 기준과 정식 채택 목표는 구분해 표시합니다. 저장된 점수를 재사용하며, 기존 모델과 평가 기록은 보존합니다. 조건을 만족하는 후보가 없으면 **추천 없음**으로 표시하고 절충값을 따로 보여줍니다.

화면에 표시한 공개 자료와 용량을 먼저 확인하세요. 새 웹 프로젝트는 **선택한 규모에 맞춰 자료를 자동 선정·다운로드·전처리**하고, 제공 학습·평가 배경 자료는 자동 검사 후 연결합니다. 기준 목소리는 자동 품질 검사를 통과하면 다음 단계로 넘어가며, 듣기는 선택 사항입니다. 이어서 **학습용 목소리마다 호출어 1개**를 듣고 발음을 확인합니다. 검수량은 **동작 확인용 20개 / 시험 모델용 100개 / 상용 모델용 500개**입니다. 직접 추가한 자료의 검수도 유지합니다. 실제 준비 시간이 부족하거나 이용 조건을 확인해야 하면 이유를 표시하고 멈춥니다. 자세한 사용법은 [웹 학습 안내](docs/WEB_GUIDE.md)를 참고하세요.

새 웹 프로젝트는 공개 학습 음원을 **학습에 사용하는 16kHz 모노 PCM16과 동일한 무손실 FLAC**으로 보관합니다. 새 대규모 비호출 원본과 중복 WAV만 검증 후 정리하며, 기존 자료·공유 기본 묶음·생성 TTS·평가 WAV는 보존합니다. FMA 음악은 한 곡씩 변환·검증하면서 다음 곡 1개만 미리 받습니다. **학습부터 독립 최종 평가까지 필요한 총 저장 용량은 아직 확정되지 않았습니다.** 항목별 참고 계산과 미확정 항목은 [저장 방식과 용량 계산](docs/STORAGE_POLICY.md#상용-모델용-저장-용량-계산)에 정리했습니다.

세 학습 규모 모두 **같은 공통 평가 기준**을 사용합니다. 한국어 대화·인터뷰·낭독, 다른 언어, 음악·환경·전자음의 **일곱 범주·비호출 30시간**으로 통일하고, 평가 음색 30계열·호출 원본 2,000개로 **소음·잔향을 포함한 48,000회 호출 시험**을 구성합니다. 잔향을 추가하지 않은 32,000회와, 같은 대표 원본 500개에 두 강도의 잔향을 적용한 16,000회를 구분해 보고합니다. 같은 배경 자료를 호출어에 섞거나 단독으로 재생해 감지율과 오호출을 확인합니다. 기존 설정과 완료 결과는 변경하지 않습니다. 실제 준비량과 채점 가능한 시간은 구분하며, **독립 최종 평가는 학습·공통 평가와 분리한 자료**로 모델·설정 동결 후 진행합니다. [평가 자료 안내](docs/DATASETS.md#성능-평가용-배경-음원)를 참고하세요.

아래는 자료와 명령을 직접 관리할 때 사용하는 설명입니다. 웹 화면에서도 같은 라이브러리 기능과 학습 설정을 사용합니다.

## 학습 데이터 준비

### 음성 생성·학습 도구 설치

호출어 음성을 만드는 **Qwen TTS**와 **모델 학습 도구**를 설치합니다. 먼저 [공식 설치 안내](https://pytorch.org/get-started/previous-versions/)의 **PyTorch 2.8.0** 항목에서 자신의 CPU/GPU 환경에 맞는 **`torch==2.8.0`과 `torchaudio==2.8.0`을 함께** 설치하세요. 두 패키지의 버전과 CPU/CUDA 배포판을 맞춰야 합니다.

```powershell
python -m pip install ".[train,tts]"
koww check --tts
```

### 필요한 학습 자료

학습에는 다음 자료가 필요합니다. **새 호출어 음성은 Qwen TTS(음성 합성 모델)로 호출어만 단독으로 생성**하며, 나머지 공개 자료는 여러 호출어 학습에 재사용할 수 있습니다.

| 자료 | 역할 |
| --- | --- |
| 호출어 음성(Positive) | 모델이 인식해야 할 한국어 호출어 |
| 일반 비호출 자료(Negative) | 호출어가 아닌 대화·음악·환경음으로 오호출을 줄이는 학습 자료 |
| 한국어 비호출 자료 | 한국어 대화·방송·보컬 음성으로 한국어 사용 환경의 오호출을 줄이는 학습 자료 |
| 유사어 자료(Hard Negative) | 호출어와 발음이 비슷한 다른 표현으로 오인식을 줄이는 별도 비호출 학습 자료 |
| 소음 증강 재료 | 호출어와 일반 음성에 섞는 환경음·음악·말소리. 비호출 원본 오디오도 재사용 가능 |
| 잔향 자료(RIR) | 방이나 공간에 따른 울림을 재현하는 재료 |

### 학습 규모 선택

준비할 자료의 양과 학습 작업량을 먼저 정하세요. 아래 표는 모델의 목적에 맞추어 임의대로 정한 설정 예시입니다. 아래 설정값은 최적값으로 입증된 수치는 아니며 모델 학습 후에 결과물에 따라 설정값을 조정하여 더 좋은 결과물을 얻을 수 있습니다. 선택한 값은 다음 **호출어 학습 시작** 단계에서 입력합니다.

| 선택 항목 | 동작 확인용 | 시험 모델용 | 상용 모델용 |
| --- | ---: | ---: | ---: |
| 호출어 TTS 원본 수 | 1,000개 | 20,000개 | 100,000개 |
| 학습용 기준 목소리 | 20개 | 100개 | 500개 |
| 유사어 한 문구당 음원 수 | 20개 | 100개 | 500개 |
| 원본당 증강 횟수 | 1회(R1) | 3회(R3) | 5회(R5) |
| 모델당 학습량 | 10,000회 | 50,000회 | 100,000회 |
| 일반 비호출 자료 | 2,000시간 | 2,000시간 | 3,500시간 |
| 한국어 비호출 자료 | 1시간 | 약 667시간 | 1,500시간 |
| 소음 증강 재료 | 공식 예제 | 공식 예제 | 비호출 자료 재사용 |
| 잔향 증강 재료 | MIT RIR | MIT RIR | MIT RIR |

기준 목소리는 호출어 음성을 만들 때 따라 할 합성 목소리입니다. 선택한 수량 안에서 **성별·연령·음색·말투·발성 강도를 가능한 고르게 배분**하고, 속도와 억양도 분산합니다. 연령은 아동·청소년·젊은 성인·중년·고령으로 나눕니다. **공통 평가용 30개는 학습용과 분리**하며 같은 방식으로 구성합니다. 목소리 수는 시작값이며, 공식 권장량이나 검증된 최적값은 아닙니다.

**유사어 문구 수는 사용자가 정합니다.** 각 문구를 학습용 목소리마다 한 번씩 생성하며, 전체 유사어 음원 수는 **문구 수 × 문구당 음원 수**입니다. 문구당 20·100·500개는 이 라이브러리의 시작값이며, 공식 권장량이나 검증된 최적값은 아닙니다. 기존에 생성한 음원과 검수 기록은 유지합니다.

유사어는 **문구당 대표 1개만 필수 검수**합니다. 문구가 6종이면 생성한 120개 중 6개를 듣습니다. 전체 음원의 자동 검사는 유지하며, **유사어 전체 듣기**는 필요할 때 선택할 수 있습니다. 듣지 않은 나머지는 대표 검수에 근거해 사용하고, 개별 청취 완료로 기록하지 않습니다.

소음 증강 재료의 **공식 예제**는 **AudioSet 일부 + FMA 음악 1시간**을 뜻합니다. 잔향 재료는 **세 단계 모두 MIT RIR의 `train` 전체**를 사용합니다. 출처와 준비 범위는 [증강 재료 안내](docs/DATASETS.md#공식-자동-학습-예제의-증강-재료)를 참고하세요.

상용 모델용은 학습용 일반·한국어 비호출 자료의 **원본 오디오를 소음 증강에도 재사용**합니다. 소음 전용 자료를 별도로 내려받지 않으며, 원본 없는 특징 자료와 평가용 음원은 재사용 대상에서 제외합니다.

상용 모델용의 일반 3,500시간은 **AudioSet 생활·환경·말소리 혼합 1,500시간 / MLS 다국어 낭독 1,500시간 / FMA full 음악·보컬·연주곡 500시간**, 한국어 1,500시간은 **YODAS2 한국어 웹 음성**으로 계획합니다. FMA full은 전체 ZIP 대신 **필요한 곡만 원본 MP3로 선택 다운로드**합니다. 새 상용 웹 프로젝트는 정상 변환·PCM 검증을 마친 새 MP3만 정리하며, 품질 검사에서 거부한 원곡은 보존합니다. 기존 프로젝트와 개별 `download-fma` 명령의 원본 보관 방식은 유지합니다. 다운로드와 연결 방법은 [상용 비호출 구성](docs/DATASETS.md#상용-모델용-비호출-구성)을 참고하세요. 이 수치는 준비 목표이며 다운로드·검수 완료량이 아닙니다.

동작 확인용·시험 모델용의 일반 비호출은 [openWakeWord 공개 특징 자료](https://huggingface.co/datasets/davidscripka/openwakeword_features)의 약 2,000시간짜리 파일을 사용합니다. **원본 오디오가 아니라, 학습에 바로 넣을 수 있도록 미리 숫자로 변환한 자료**이며 아래의 `koww download-generic` 명령으로 받습니다.

<details>
<summary>자료 규모와 증강 횟수 읽는 법</summary>

표의 시간은 학습 전에 준비할 자료의 양입니다. 시험 모델용은 일반 2,000시간 + 한국어 약 667시간(정확히는 666⅔시간), 상용 모델용은 **일반 3,500시간 + 한국어 1,500시간 = 총 5,000시간(70:30)**입니다. 자료 시간과 별개로 세 단계의 일반·한국어 비호출 배치는 **75:25**를 유지합니다. 한국어 비중과 자료 시간은 이 라이브러리에서 선택한 구성으로, 공식에서 제시한 최적값은 아닙니다.

호출어 원본 수는 **증강 전 TTS 수량**입니다.

**R1·R3·R5는 원본 한 개당 각각 1·3·5개의 변형을 만드는 설정**입니다. 원본 100,000개라면 R1은 100,000개, R5는 500,000개의 호출어 학습 자료를 준비합니다. 증강 횟수를 늘려도 매회 사용하는 자료 수와 모델당 학습량은 그대로이며, 더 다양한 변형 중에서 자료를 뽑게 됩니다.

비호출 자료는 **시간과 종류의 다양성**으로 관리합니다. 소음 증강에는 AudioSet·음악 같은 원본 오디오를 사용하며, 일반·한국어 비호출 자료의 **원본 오디오도 재사용할 수 있습니다.** 같은 자료를 두 용도로 사용해도 비호출 총시간에 다시 더하지 않습니다. 원본 오디오 없이 사전 계산된 특징만 받은 자료는 비호출 학습에만 사용합니다. 잔향(RIR)은 서로 다른 공간의 자료를 연결합니다.

목표 시간은 이미 준비한 자료를 재사용할 때도 적용합니다. **웹은 규모에 맞춰 자료를 준비하며, CLI의 `init`은 설정만 저장합니다.** 직접 명령으로 준비하려면 [데이터 준비 안내](docs/DATASETS.md)를 따르세요. 정식 학습 전에는 검사와 사용 조건 확인을 거쳐 연결한 비호출 시간이 목표를 충족했는지 확인합니다. 소음 증강 재료는 별도 비호출 배치 항목으로 추가하지 않습니다.

**모델 구조·학습 방법의 설정 비교와 데이터량의 최적화는 구분합니다.** 위의 호출어 수와 비호출 시간은 최적량이 입증된 값이 아닙니다. 새 호출어의 데이터량은 공식 안내와 공개 학습 사례를 참고해 선택합니다. 근거와 비교 방법은 [데이터량 선정 기준](plan.md#데이터량의-근거와-선택-원칙)에 있습니다.

작업량을 줄여도 Qwen 모델과 공개 사전 계산 자료의 최초 다운로드 크기는 줄어들지 않습니다.

</details>

새 동작 확인용·시험 모델용은 공식 예제의 **같은 음악 120곡과 목록 파일만 선택 다운로드**합니다. 음악·목록 전송 상한은 합계 약 **116MiB**이며, 기존 전체 ZIP 두 개 약 **7.49GiB**를 모두 받지 않습니다. 일반 비호출·한국어·공통 평가 자료는 별도로 필요합니다. 원본 음질·곡 구성은 유지하고 곡별 검증과 이어받기를 지원합니다.

미리 확정된 학습·평가 자료는 **하나의 다운로드 대기열**에서 받고, 전송 상태에 맞춰 동시 처리 수를 자동 조절합니다. 화면에서 자료별 대기·다운로드·검사·완료 상태를 확인할 수 있습니다. 추가 선정이 필요한 대용량 원본은 목록과 저장 공간이 준비된 뒤 이어서 받으며, 파일별 이어받기·해시 검증·이후 처리 순서는 유지합니다. 자세한 범위와 수동 설정은 [웹 학습 안내](docs/WEB_GUIDE.md#다운로드-속도와-동시-처리)를 참고하세요.

### 공개 자료 다운로드

사용 가능한 공개 자료 목록과 다운로드 예시입니다.

```powershell
koww datasets
koww download-data --dataset musan --output shared/musan.tar.gz
koww download-data --dataset rir --output shared/rirs_noises.zip
koww download-data --dataset zeroth --output shared/zeroth_korean.tar.gz
koww download-generic --output shared/openwakeword_features_ACAV100M_2000_hrs_16bit.npy
```

내려받은 자료를 학습에 연결하는 절차는 [데이터 준비 안내](docs/DATASETS.md)와 [라이브러리 가이드](docs/LIBRARY_GUIDE.md)를 따르세요.

## 호출어 학습 시작

**학습 설정 → 호출어 생성·검수 → 자료 연결·증강 → 모델 학습 → 평가** 순서로 진행합니다. 실제 학습을 시작하는 명령은 **`koww train`**입니다. 아래는 전체 흐름이며, 검수·자료 목록 작성의 세부 명령은 [학습 가이드](docs/LIBRARY_GUIDE.md)를 따르세요.

아래는 호출어 **‘보라야’**의 학습 설정을 직접 명령으로 만드는 예시입니다. `보라야`는 호출어, `bora`는 설정을 구분하는 영문 이름입니다. 웹에서는 저장 위치가 자동으로 정해지며, 아래 명령 예시는 `models/bora`를 사용합니다.

### 1. 학습 설정

`init`에서 [앞서 선택한 학습 규모](#학습-규모-선택)의 **호출어 원본 수·증강 횟수·모델당 학습량·일반 및 한국어 비호출 시간**을 입력합니다. 값을 입력하지 않고 Enter를 누르면 **상용 모델용 기본값**을 사용합니다. 현재 설정 화면은 환경소음 재료의 목표 시간도 묻습니다. 이 시간은 준비 계획용이며, 공개 자료를 자동으로 내려받거나 선택하지 않습니다.

```powershell
koww init --wakeword 보라야 --id bora --output models/bora --interactive --validation-voices 30
koww prepare-frontend --project models/bora/project.json
koww status --project models/bora/project.json
```

`init`은 설정을 만드는 단계이며 음성 생성이나 학습을 바로 시작하지 않습니다. `prepare-frontend`는 학습에 필요한 공통 음성 처리 모델을 준비하고, `status`는 준비 상태를 확인합니다.

입력한 값은 `models/bora/project.json`에 저장됩니다. 이후 명령은 이 설정을 읽으므로 학습량을 매번 다시 입력하지 않습니다.

CNN128·AdamW 등은 초기 표본 모델 학습 결과에서 선정한 **권장 시작값**입니다. 이 시작값은 최적값으로 입증된 수치는 아니며 모델 학습 후에 결과물에 따라 설정값을 조정하여 더 좋은 결과물을 얻을 수 있습니다. 직접 변경하려면 `init` 명령에 `--advanced`를 추가하세요. 새 호출어와 선택한 작업량의 성능은 학습 후 평가로 확인합니다.

<details>
<summary>상용 모델용 값을 한 번에 지정하는 명령</summary>

위의 대화형 `init` 대신 아래 명령으로 설정값을 저장할 수 있습니다.

```powershell
# 상용 모델용
koww init --wakeword 보라야 --id bora --output models/bora --non-interactive --originals 100000 --rounds 5 --updates 100000 --generic-hours 3500 --korean-hours 1500 --validation-voices 30
```

값은 직접 바꿀 수 있습니다. 이미 만든 프로젝트에 `init`을 다시 실행하지 말고, 다른 규모의 실험은 새 폴더 또는 [별도 설정 파일](docs/LIBRARY_GUIDE.md#추가-설정과-재사용)을 사용하세요.

</details>

### 2. 호출어 생성·자료 연결

먼저 [기준 목소리 준비·자동 검사](docs/LIBRARY_GUIDE.md#2-기준-목소리-준비자동-검사)와 [목소리별 호출어 발음 확인](docs/LIBRARY_GUIDE.md#3-목소리별-호출어-발음-확인하기)을 진행합니다. **학습용 목소리별 발음 검수와 생성 정책 승인 후** 다음 명령으로 학습용과 개발용 호출어를 생성합니다.

```powershell
koww generate --project models/bora/project.json --split train
koww generate --project models/bora/project.json --split validation
```

웹의 제공 학습·평가 배경 자료는 자동 검사 후 연결합니다. 직접 추가한 공개 음원은 **`prepare-materials`로 변환·자료 분리·검수 목록을 준비**하고, 검수 화면에서 저장한 답변을 **`admit-materials`로 학습에 연결**합니다. 한국어 비호출은 고정 2초 구간까지 자동으로 준비합니다. 출처·언어·화자 정보는 제공자 메타데이터를 사용합니다.

아래는 직접 준비한 Zeroth 한국어 자료에서 동작 확인용 1시간을 연결하는 예입니다. 웹의 제공 자료를 사용할 때는 이 검수 절차를 따로 반복하지 않습니다. `shared/zeroth/train_data_01`은 실제로 압축을 푼 음원 폴더로 바꾸세요.

```powershell
koww index-data --dataset zeroth --input shared/zeroth/train_data_01 --output shared/zeroth_index.jsonl --source-url https://www.openslr.org/40/ --license CC-BY-4.0
koww prepare-materials --project models/bora/project.json --input shared/zeroth/train_data_01 --source-index shared/zeroth_index.jsonl --hours 1 --output models/bora/prepared/korean_v1
koww review --project models/bora/project.json --queue models/bora/prepared/korean_v1/representatives.jsonl --pool-manifest models/bora/prepared/korean_v1/train_sources.jsonl --kind content
# 브라우저에서 검수·저장한 뒤 Ctrl+C로 서버를 종료합니다.
koww admit-materials --project models/bora/project.json --preparation models/bora/prepared/korean_v1/preparation_receipt.json --pools korean_negative --connect
```

소음·잔향 준비와 여러 자료 묶음의 연결은 [자료 준비 안내](docs/DATA_FORMAT.md#공개-음원-자동-준비)에 있습니다. 개발용 자료는 원본·화자·공간 단위로 분리해 예약하며, 실제 확보 시간과 부족분은 준비 결과에 표시합니다. 일반 비호출의 공개 특징 자료는 기존 `import-generic` 명령으로 연결합니다.

유사어는 사용자가 고른 문구로 별도 생성하고 자료 목록과 특징 배열을 준비합니다. 한 문구당 생성량은 [학습 규모 표](#학습-규모-선택)를 따릅니다. 배치의 100개는 매회 뽑는 수이며, 전체 생성량과는 별개입니다. 준비 방법은 [유사어 자료 안내](docs/DATA_FORMAT.md#유사어-학습-자료)를 따릅니다.

### 3. 증강·모델 학습

호출어 생성과 자료 연결을 마친 뒤 다음을 실행합니다.

```powershell
koww features --project models/bora/project.json --role korean_negative
koww features --project models/bora/project.json --role hard_negative
koww features --project models/bora/project.json --role positive
```

`features`는 소음·잔향·음량 변형을 적용하고 음성을 학습용 숫자 배열로 바꿉니다. 호출어 자료는 **원본 수 × 증강 횟수**만큼 준비합니다. 동작 확인용·시험 모델용에서 일반 비호출의 사전 계산 특징을 연결했다면 다시 추출하지 않습니다.

상용 모델용처럼 **일반 비호출을 원본 오디오로 준비·승인해 연결했다면**, 학습 전에 이 자료의 특징도 생성합니다. 웹 화면은 이 단계를 자동으로 실행합니다.

```powershell
koww features --project models/bora/project.json --role generic_negative
```

준비된 자료로 **실제 모델 학습을 시작**하고, 학습 결과를 사용 가능한 모델 묶음으로 정리합니다.

```powershell
koww train --project models/bora/project.json --output models/bora/runs/selected_budget
koww package --project models/bora/project.json --run models/bora/runs/selected_budget --output models/bora/models/candidate_selected
```

`train`은 저장된 설정에 따라 모델을 학습하고, `package`는 모델 파일·공통 음성 처리 모델·실행 설정을 한 폴더에 모읍니다.

**학습 1회**는 준비된 자료에서 일반 비호출 768개·한국어 비호출 256개·호출어 100개·유사어 100개, 총 **1,224개**를 무작위로 뽑아 모델의 판단을 정답과 비교하고, 모델을 한 번 수정하는 과정입니다. 일반·한국어 비호출 배치는 **75:25**이며, 유사어는 호출어와 발음이 비슷하지만 호출어가 아닌 별도 비호출 자료입니다. **모델당 100,000회**는 이 과정을 모델 하나마다 100,000번 반복한다는 뜻입니다. 전체 자료를 매번 모두 사용하지 않으며, 같은 자료가 여러 번 뽑힐 수 있습니다.

기본 모델 3개를 각각 학습하므로 상용 모델용의 총 학습량은 **300,000회**입니다. 큰 작업 전에 `koww benchmark --output models/bora/reports/cpu_gpu.json`으로 CPU와 GPU의 학습 속도를 비교할 수 있습니다.

### 4. 성능 평가

[개발용 평가 자료 준비 안내](docs/LIBRARY_GUIDE.md#7-개발용-평가감지-기준-선택하기)에 따라 소음별 호출어 오디오와 연속 비호출 평가 목록을 준비한 뒤 실행합니다.

```powershell
koww evaluate --project models/bora/project.json --bundle models/bora/models/candidate_selected --manifest models/bora/data/eval/development.jsonl --output models/bora/reports/dev_selected
koww site --evaluation models/bora/reports/dev_selected --output models/bora/reports/dev_selected/site
koww serve --directory models/bora/reports/dev_selected/site --port 8791
```

[http://127.0.0.1:8791/](http://127.0.0.1:8791/)에서 단일 모델과 앙상블의 성공률·오호출·지연, **각 후보의 기준 통과 여부**를 비교합니다. 확인 후 Ctrl+C로 서버를 종료하세요. 앙상블이 항상 더 좋다고 가정하지 않습니다. 사용할 후보는 평가에 표시된 ID로 직접 선택합니다.

```powershell
# 아래 ID는 선택 예시입니다. 실제 결과를 보고 원하는 후보를 지정하세요.
koww select-candidate --project models/bora/project.json --bundle models/bora/models/candidate_selected --evaluation models/bora/reports/dev_selected --candidate member3_seed20260930 --output models/bora/models/selected_v1
koww mic --bundle models/bora/models/selected_v1/bundle
```

선택은 기존 평가 점수를 재사용하며, 단일 모델을 골랐다면 실제 실행 묶음에도 그 모델만 넣습니다. 감지 기준 비교·동결·[별도 최종 평가](docs/LIBRARY_GUIDE.md#8-모델설정-확정-후-최종-평가하기)는 [학습 가이드](docs/LIBRARY_GUIDE.md)를 따르세요. 평가 기준을 통과하지 않은 후보도 시험할 수 있지만 Final 동결은 할 수 없습니다.

공통 평가를 마치면 **성능 결과**에서 모델별 추천 감지 설정과 그 결과를 확인합니다. 최초 평가의 통계·오디오 반응은 별도로 보존합니다. **마이크 시험**에서는 모든 모델을 각자의 추천값으로 함께 시험하거나, 직접 감지 기준과 연속 감지 수를 조절할 수 있습니다. 변경값은 해당 시험에만 적용됩니다. 동봉된 예시 모델과 기존 학습의 감지 설정은 유지합니다.

## 마이크로 모델 시험

기본 설치만 마쳤다면 다음 명령으로 동봉된 ‘춘식아’ 모델을 시험할 수 있습니다. 추가 패키지는 필요하지 않습니다.

```powershell
koww mic
```

[http://127.0.0.1:8792/](http://127.0.0.1:8792/)을 열어 마이크를 선택하고 **시작**을 누르세요. 마이크 권한을 허용하고 준비 표시가 끝나면(기본 2초) 호출어를 말합니다. 각 모델과 앙상블의 점수·감지 여부·감지 횟수를 확인할 수 있습니다.

직접 학습한 모델 묶음을 시험하려면 경로를 지정합니다.

```powershell
koww mic --bundle models/bora/models/candidate_selected
```

모델의 감지 기준은 바꾸지 않으며, 음성은 파일로 저장하거나 학습에 사용하지 않습니다. **중지**를 누르면 마이크 입력을 종료합니다. 서버를 유지하려면 터미널을 열어 두고, 종료할 때는 Ctrl+C를 누르세요. 상세 사용법은 [마이크 시험 안내](docs/LIBRARY_GUIDE.md#마이크로-모델-시험하기)에 있습니다.

## Python에서 모델 사용

```python
from korean_openwakeword import WakeWordDetector

detector = WakeWordDetector()  # 동봉된 ‘춘식아’ 모델 사용
result = detector.predict_file("test.wav")
print(result["events"])  # 호출어가 감지된 시점 목록
```

입력 파일은 **16kHz·모노·16비트 PCM 형식의 WAV**를 사용합니다. 기본 설정의 초기 준비 구간은 2초이므로 호출어가 그 이후에 나오도록 준비하세요. 직접 학습한 모델 사용법과 실시간 입력은 [라이브러리 가이드](docs/LIBRARY_GUIDE.md)를 참고하세요.

## 자세한 안내

- [라이브러리 가이드](docs/LIBRARY_GUIDE.md): 새 호출어 학습의 전체 절차와 명령
- [데이터 준비 안내](docs/DATASETS.md): 공개 자료의 출처·다운로드·이용조건
- [저장 위치 안내](docs/STORAGE_POLICY.md): 다운로드·캐시·학습 자료의 저장 위치
- [학습 계획](plan.md): 기본 설정, 비교 실험과 평가 기준
- [표본 모델 설명서](korean_openwakeword/resources/sample_chunsik/MODEL_CARD.md): 모델 구성과 성능 결과
- [검수 기록](docs/VALIDATION.md): 설치·실행 확인 범위
- [이용조건·출처](THIRD_PARTY_NOTICES.md): 코드·모델·음원별 이용조건

## 라이선스

소스 코드는 원 프로젝트와 같은 [Apache-2.0](LICENSE)으로 배포합니다. 모델·음원·학습 데이터에는 별도 이용조건이 적용되므로 [이용조건·출처](THIRD_PARTY_NOTICES.md)를 확인하세요.
