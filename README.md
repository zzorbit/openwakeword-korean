# Korean openWakeWord

**영어 호출어에 특화된 openWakeWord를 한국어 호출어 전용으로 수정한 라이브러리입니다.** 초기 표본 모델 학습에서 선정한 설정을 기본값으로 제공하며, 새 호출어마다 별도 프로젝트를 만들어 학습합니다.

이 프로젝트는 **[openWakeWord](https://github.com/dscripka/openWakeWord)(David Scripka 및 기여자)를 기반으로 개발한 파생 프로젝트**입니다. 원 프로젝트의 공통 음성 특징 추출 기능을 사용하며, 한국어 Qwen TTS 데이터 생성·소음 증강·판별기 학습·평가·결과 사이트를 위한 절차를 추가했습니다. 원 프로젝트와 별도로 개발·관리합니다.

표본 모델은 **‘춘식아’ CNN128 앙상블 모델**입니다. 독립적으로 학습한 CNN128 모델 3개의 호출 점수를 평균해 감지 여부를 판단합니다. 각 원본 음성에 5회 증강을 적용하는 R5 설정으로 학습했으며, 모델 파일·실행 설정·모델 설명서·평가 오디오를 함께 제공합니다.

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
koww doctor
```

`koww doctor`는 설치된 패키지와 GPU 사용 가능 여부를 보여줍니다. openWakeWord는 설치 호환성을 위해 위 명령처럼 별도로 설치합니다. **기본 설치만으로 표본 모델을 실행하고 결과 사이트를 볼 수 있습니다.**

`Use-ProjectStorage.ps1`은 다운로드 캐시와 임시 파일을 저장소 폴더 안에 모아 저장하도록 설정합니다. 저장 위치를 바꾸려면 [저장 위치 안내](docs/STORAGE_POLICY.md)를 참고하세요.

**새 호출어를 생성·학습하려면** 다음을 추가로 설치하세요. GPU를 사용할 경우 먼저 [공식 설치 안내](https://pytorch.org/get-started/previous-versions/)의 **PyTorch 2.8.0** 항목에서 자신의 환경에 맞는 GPU용 PyTorch를 설치합니다.

```powershell
python -m pip install ".[train,tts]"
koww doctor
```

## 예시 모델과 결과 사이트

다음 명령은 ‘춘식아’ 표본 모델을 복사하고, 동봉된 오디오를 채점해 결과 사이트를 만듭니다.

```powershell
koww sample --output examples/chunsik_model
koww infer --audio examples/chunsik_model/audio/validation_standalone_003495_a0__clean.wav
koww demo --output examples/chunsik_demo
koww serve --directory examples/chunsik_demo/site --port 8791
```

브라우저에서 [http://127.0.0.1:8791/](http://127.0.0.1:8791/)을 여세요. 소음 수준을 선택해 오디오를 연속으로 들으면서 감지·누락과 모델별 순위를 확인할 수 있습니다. 채점과 청취에는 같은 오디오를 사용합니다.

**사이트를 보는 동안 서버를 실행한 PowerShell 창을 열어 두세요.** 이미 만든 사이트를 다시 열 때는 마지막 `koww serve` 명령만 실행하면 됩니다. 모델과 사이트를 새로 만들 때는 비어 있는 출력 폴더를 사용합니다.

## 호출어 학습 시작

아래는 **‘보라야’**를 학습할 프로젝트를 만드는 예시입니다. `보라야`는 호출어, `bora`는 영문 프로젝트 이름입니다. 원하는 호출어와 프로젝트 이름으로 바꿔 사용하세요. 생성된 파일은 `projects/bora`에 저장됩니다.

```powershell
koww init --wakeword 보라야 --id bora --output projects/bora --interactive
koww prepare-frontend --project projects/bora/project.json
koww status --project projects/bora/project.json
```

첫 번째 명령에서 작업량 세 가지를 선택합니다. Enter를 누르면 권장 시작값을 사용합니다.

| 선택 항목 | 의미 | 권장 시작값 |
| --- | --- | --- |
| 호출어 음성 수 | 증강 전에 TTS로 생성할 원본 오디오 수 | 100,000개 |
| 증강 횟수 | 원본마다 소음·잔향·음량 조건을 바꾼 학습 샘플 수 | 5회(R5) |
| 모델당 학습량 | 모델 한 개의 가중치를 갱신하는 횟수 | 50,000회 |

**R1·R3·R5는 원본 한 개당 각각 1·3·5개의 변형을 만드는 설정**입니다. 기본 구성은 모델 3개를 각각 학습하므로, 모델당 50,000회를 선택하면 전체 학습량은 150,000회입니다. 컴퓨터 자원과 필요한 성능에 맞춰 작업량을 선택하세요.

`init`은 설정을 만드는 단계이며 음성 생성이나 학습을 바로 시작하지 않습니다. `prepare-frontend`는 학습에 필요한 공통 음성 처리 모델을 준비하고, `status`는 준비 상태를 확인합니다.

선택 화면 없이 작업량을 직접 지정할 수도 있습니다.

```powershell
# 원본 20,000개, 원본당 3회 증강, 모델당 20,000회 학습
koww init --wakeword 보라야 --id bora --output projects/bora_small --non-interactive --originals 20000 --rounds 3 --updates 20000
```

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
