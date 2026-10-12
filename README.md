# Korean openWakeWord

**영어 호출어에 특화된 openWakeWord를 한국어 호출어 전용으로 수정한 라이브러리입니다.** 초기 표본 모델 학습에서 선정한 설정을 기본값으로 제공하며, 새 호출어마다 별도 프로젝트를 만들어 학습합니다.

이 프로젝트는 **[openWakeWord](https://github.com/dscripka/openWakeWord)(David Scripka 및 기여자)를 기반으로 개발한 파생 프로젝트**입니다. 원 프로젝트의 공통 음성 특징 추출 기능을 사용하며, 한국어 Qwen TTS 데이터 생성·소음 증강·판별기 학습·평가·결과 사이트를 위한 절차를 추가했습니다. 원 프로젝트와 별도로 개발·관리합니다.

## 예시 모델과 성능

설치 후, 동봉된 ‘춘식아’ 모델로 음성 파일의 호출어 감지를 시험할 수 있습니다.

| 호출어 | 모델 구성 | 조용한 환경 | 소음 평균 | SNR 0 dB 평균 | 오호출/시간 |
| --- | --- | ---: | ---: | ---: | ---: |
| 춘식아 | CNN128 앙상블 · R5 | **100%**<br>5,000/5,000회 | **99.18%**<br>5,951/6,000회 | **89.73%**<br>1,346/1,500회 | **0.083회**<br>2회/24.03시간 |

합성 호출 음성의 개발용 평가 결과이며, 상세 조건과 검증 범위는 [표본 모델 설명서](korean_openwakeword/resources/sample_chunsik/MODEL_CARD.md)에 있습니다.

## 설치

**Git과 Python 3.10~3.13**을 준비하세요. 아래 명령은 **Windows PowerShell** 기준이며, 저장소에 접근할 수 있는 GitHub 계정이 필요합니다.

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

**새 호출어를 생성·학습하려면** 다음을 추가로 설치하세요. GPU를 사용할 경우 먼저 [공식 설치 안내](https://pytorch.org/get-started/previous-versions/)의 **PyTorch 2.8.0** 항목에서 자신의 환경에 맞는 GPU용 PyTorch를 설치합니다.

```powershell
python -m pip install ".[train,tts]"
koww check
```

## 호출어 학습 시작

아래는 호출어 **‘보라야’**를 학습할 프로젝트를 만드는 예시입니다. `보라야`는 호출어, `bora`는 영문 프로젝트 이름입니다. 원하는 호출어와 프로젝트 이름으로 바꿔 사용하세요. 생성된 파일은 `projects/bora`에 저장됩니다.

```powershell
koww init --wakeword 보라야 --id bora --output projects/bora --interactive
koww prepare-frontend --project projects/bora/project.json
koww status --project projects/bora/project.json
```

첫 번째 명령에서 **학습 작업량과 준비할 자료의 목표 시간**을 선택합니다. 처음 사용한다면 **동작 확인용**으로 시작하세요. 값 없이 Enter를 누르면 **전체 규모**의 기본값이 적용됩니다.

| 선택 항목 | 동작 확인용 | 중간 규모 | 전체 규모 |
| --- | ---: | ---: | ---: |
| 호출어 TTS 원본 수 | 1,000개 | 20,000개 | 100,000개 |
| 원본당 증강 횟수 | 1회(R1) | 3회(R3) | 5회(R5) |
| 모델당 학습량 | 1,000회 | 20,000회 | 50,000회 |
| 일반 비호출 자료 목표 | 10시간 | 500시간 | 2,000시간 |
| 한국어 비호출 자료 목표 | 1시간 | 50시간 | 200시간 |
| 환경소음 재료 목표 | 1시간 | 10시간 | 30시간 |

**동작 확인용**은 새 호출어의 생성·증강·학습·감지 흐름을 적은 작업량으로 시험하는 단계입니다. **중간 규모**는 본격적인 학습 전에 발음과 소음별 성능을 살펴보는 단계이고, **전체 규모**는 초기 표본 모델에서 사용한 작업량입니다. 작은 두 단계는 비용을 줄인 시작 예시이며, 새 호출어의 성능이나 상용 사용 가능 여부는 별도 평가로 확인합니다.

호출어 원본 수는 **증강 전 TTS 수량**, 학습량은 **모델 한 개의 가중치를 갱신하는 횟수**입니다. 비호출·소음은 **시간과 종류의 다양성**으로 관리합니다. 전체 규모에서는 한국어 비호출 200~300시간, 환경소음 30~50시간부터 시작해 필요하면 늘립니다. 잔향(RIR)은 서로 다른 공간의 자료를 연결합니다.

목표 시간은 이미 준비한 자료를 재사용할 때도 적용합니다. **시간을 입력한다고 자료가 자동으로 다운로드되거나 그 길이로 잘리지는 않습니다.** 실제 파일과 사용 구간은 [데이터 준비 안내](docs/DATASETS.md)에 따라 연결합니다. 정식 학습 전에는 선택한 비호출 목표 시간을 충족했는지 확인하고, 환경소음 시간은 준비 목표로 관리하면서 종류의 다양성을 별도로 확인합니다. 환경소음은 증강 재료이므로 별도 비호출 배치 항목으로 추가하지 않습니다.

**R1·R3·R5는 원본 한 개당 각각 1·3·5개의 변형을 만드는 설정**입니다. 세 단계 모두 기본 모델 3개를 각각 학습하므로 총 학습량은 각각 3,000 / 60,000 / 150,000회입니다. 작업량을 줄여도 Qwen 모델과 공개 사전 계산 자료의 최초 다운로드 크기는 줄어들지 않습니다.

`init`은 설정을 만드는 단계이며 음성 생성이나 학습을 바로 시작하지 않습니다. `prepare-frontend`는 학습에 필요한 공통 음성 처리 모델을 준비하고, `status`는 준비 상태를 확인합니다.

<details>
<summary>단계별 값을 한 번에 지정하는 명령</summary>

위의 대화형 `init` 대신 아래에서 **한 명령만** 선택하세요. 생성·학습 과정은 [학습 가이드](docs/LIBRARY_GUIDE.md)를 따릅니다.

```powershell
# 동작 확인용
koww init --wakeword 보라야 --id bora --output projects/bora --non-interactive --originals 1000 --rounds 1 --updates 1000 --generic-hours 10 --korean-hours 1 --noise-hours 1

# 중간 규모
koww init --wakeword 보라야 --id bora --output projects/bora --non-interactive --originals 20000 --rounds 3 --updates 20000 --generic-hours 500 --korean-hours 50 --noise-hours 10

# 전체 규모
koww init --wakeword 보라야 --id bora --output projects/bora --non-interactive --originals 100000 --rounds 5 --updates 50000 --generic-hours 2000 --korean-hours 200 --noise-hours 30
```

값은 직접 바꿀 수 있습니다. 이미 만든 프로젝트에 `init`을 다시 실행하지 말고, 다른 규모의 실험은 새 폴더 또는 [별도 설정 파일](docs/LIBRARY_GUIDE.md#추가-설정과-재사용)을 사용하세요.

</details>

CNN128·AdamW 등은 초기 표본 모델 학습 결과에서 선정한 **권장 시작값**입니다. 모델 구조와 학습 방법은 기본값으로 시작하고, 직접 변경하려면 `init` 명령에 `--advanced`를 추가하세요. 새 호출어와 선택한 작업량의 성능은 학습 후 평가로 확인합니다.

[라이브러리 가이드](docs/LIBRARY_GUIDE.md)의 명령을 따라 **대표 음성 생성·검수 → 전체 음성 생성 → 데이터 준비·증강 → 모델 학습 → 평가·감지 기준 선택 → 최종 평가·결과 사이트 생성** 순서로 진행합니다. 큰 작업 전에 `koww benchmark --output projects/bora/reports/cpu_gpu.json`으로 CPU와 GPU의 학습 속도를 비교할 수 있습니다.

## 학습 데이터 준비

학습에는 다음 자료가 필요합니다. **새 호출어 음성은 Qwen TTS(음성 합성 모델)로 호출어만 단독으로 생성**하며, 나머지 공개 자료는 여러 호출어 학습에 재사용할 수 있습니다.

| 자료 | 역할 |
| --- | --- |
| 호출어 음성(Positive) | 모델이 인식해야 할 한국어 호출어 |
| 비호출 자료(Negative) | 호출어가 아닌 대화·음악·환경음으로 오호출을 줄이는 학습 자료 |
| 소음 자료 | 호출어와 일반 음성에 섞어 다양한 소음 조건을 만드는 재료 |
| 잔향 자료(RIR) | 방이나 공간에 따른 울림을 재현하는 재료 |

사용 가능한 공개 자료 목록과 다운로드 예시입니다. 내려받은 자료를 학습에 연결하는 절차는 [데이터 준비 안내](docs/DATASETS.md)와 [라이브러리 가이드](docs/LIBRARY_GUIDE.md)를 따르세요.

```powershell
koww datasets
koww download-data --dataset musan --output shared/musan.tar.gz
koww download-data --dataset rir --output shared/rirs_noises.zip
koww download-data --dataset zeroth --output shared/zeroth_korean.tar.gz
koww download-generic --output shared/openwakeword_features_ACAV100M_2000_hrs_16bit.npy
```

검수한 합성 음색은 재사용하고, 대량 생성 음성에는 무음·깨짐 등의 자동 품질 검사를 적용합니다. 모든 오디오를 음성 인식 모델로 전사하거나 하나씩 청취하는 방식은 기본 절차가 아닙니다. 개인의 호출 녹음은 학습·설정 조정·음성 복제 참조로 사용하지 않습니다.

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
