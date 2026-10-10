# Korean openWakeWord

**영어 호출어에 특화된 openWakeWord를 한국어 호출어 전용으로 수정한 라이브러리입니다.** ‘춘식아’ 개발에서 검증한 설정과 재사용 절차를 기본값으로 제공합니다. 다른 한국어 호출어는 별도 프로젝트에서 학습합니다.

표본 모델은 **‘춘식아’ CNN128 앙상블 모델**입니다. 독립적으로 학습한 CNN128 모델 3개의 호출 점수를 평균해 감지 여부를 판단합니다. 각 원본 음성에 5회 증강을 적용하는 R5 설정으로 학습했으며, 가중치·공유 frontend·실행 설정·모델 카드·평가 오디오를 함께 제공합니다.

## 설치

Python 3.10~3.13. 현재 실제 검증 환경은 Windows / Python 3.12 / openWakeWord 0.6.0 / PyTorch 2.8.0 CUDA 12.8 / ONNX Runtime 1.30.0입니다. 다른 OS·Python 조합의 성능까지 검증한 것은 아닙니다.

GitHub 계정에 이 비공개 저장소 접근 권한이 있는 환경에서:

```powershell
git clone https://github.com/zzorbit/openwakeword-korean.git
cd openwakeword-korean
```

```powershell
. .\scripts\Use-ProjectStorage.ps1
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install .
python -m pip install openwakeword==0.6.0 --no-deps
koww doctor
```

openWakeWord는 Windows의 TFLite 의존성 설치 문제를 피하도록 별도 설치합니다. 이 라이브러리는 ONNX 경로를 사용하고 필요한 ONNX 실행 의존성을 직접 선언합니다. Linux에서도 동일 ONNX 경로를 사용할 수 있으나 검증 장치 외에서 동일 성능을 보장하지 않습니다. 추론만 할 때 Torch/Qwen은 필요 없습니다.

위 스크립트는 설치·모델 다운로드에 필요한 캐시와 임시 파일을 프로젝트 폴더 안에 저장하도록 설정합니다. 저장 위치를 변경하거나 여러 학습 프로젝트에서 자료를 공유하는 방법은 [저장 위치 안내](docs/STORAGE_POLICY.md)를 참고하세요.

학습·생성 기능을 사용할 경우:

```powershell
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install ".[train,tts,dev]"
```

CUDA 환경에 맞는 PyTorch 배포를 선택합니다. 이미 정상인 환경은 위 명령으로 불필요하게 재설치하지 않습니다. 배포 wheel을 쓰면 `.` 대신 wheel 파일을 지정합니다. `python -m korean_openwakeword`도 `koww`와 같습니다.

## 예시 모델과 결과 사이트

```powershell
koww sample --output examples/chunsik_model
koww infer --audio examples/chunsik_model/audio/validation_standalone_003495_a0__clean.wav
koww demo --output examples/chunsik_demo
koww serve --directory examples/chunsik_demo/site --port 8791
```

`http://127.0.0.1:8791/`에서 통계와 같은 WAV를 듣습니다. 고정 SNR을 선택하면 소음 범주·구간을 순서대로 연속 재생합니다. 단일 모델 순위와 앙상블은 분리합니다. demo는 작은 Development 재현이며 전체 학습 통계를 복사하지 않습니다. 동봉된 `examples/library_demo_site`는 재계산 없이 열 수 있는 참고 결과입니다.

```python
import numpy as np
from korean_openwakeword import WakeWordDetector

detector = WakeWordDetector()  # 동봉 춘식아 묶음, CPU ONNX
result = detector.predict_file("test.wav")  # mono16kHz PCM16
# 실시간 입력: 80ms PCM16마다 공유 frontend + 세 판별기
frame_result = detector.process(np.zeros(1280, dtype=np.int16))
detector.reset()  # 새 세션의 상태 초기화
```

## 호출어 학습 시작

```powershell
koww init --wakeword 보라야 --id bora --output projects/bora --interactive
koww prepare-frontend --project projects/bora/project.json
koww benchmark --output projects/bora/reports/cpu_gpu.json
koww status --project projects/bora/project.json
```

초기에는 **TTS 원본 수 / 증강 횟수 R1·R3·R5 / 모델당 학습 update 수**를 선택합니다. Enter를 누르면 기존 제안인 100,000개 / R5 / 50,000회를 사용합니다. 선택 후 실제 feature 수·예상 용량·세 모델의 총 학습량을 보여줍니다. `init`은 설정만 만들며 생성·학습을 바로 시작하지 않습니다. 터미널에서는 기본적으로 선택 화면을 열고, 스크립트·파이프 입력에서는 질문 없이 실행합니다.

```powershell
# 작업량을 직접 지정 (20k × R3 = 60k feature, 기본 3모델 × 20k = 총 60k update)
koww init --wakeword 보라야 --id bora --output projects/bora_small --non-interactive --originals 20000 --rounds 3 --updates 20000
# 모델 구조·optimizer·학습률 등은 필요할 때만 세부 설정
koww init --wakeword 다솜아 --id dasom --output projects/dasom --interactive --advanced
koww settings --project projects/bora_small/project.json --advanced
# 기존 설정을 보존하고 같은 폴더에 새 실험 설정 저장
koww configure --project projects/bora_small/project.json --output projects/bora_small/project_R1.json --non-interactive --rounds 1
```

CNN128·AdamW 등은 초기 표본 모델 학습 결과에서 선정한 **권장 시작값**입니다. 모든 한국어 호출어의 최적값을 보증하지는 않습니다. 작업량을 줄이면 시간·용량을 줄일 수 있지만 성능도 달라질 수 있으므로 같은 평가로 확인합니다. 학습량을 바꾸면 기본 schedule·checkpoint도 연동되며, 비교 실험의 공통 horizon은 별도 지정할 수 있습니다.

[라이브러리 가이드](docs/LIBRARY_GUIDE.md)에 작업량 선택·합성 참조 재사용·파일럿·선택한 수량 생성·공개 Negative 계약·R1/R3/R5·학습·임계값 선택·Final 동결·사이트 제작의 전체 명령을 정리했습니다. 한글 호출어, 데이터 경로, 프로젝트 ID를 바꾸면 날짜별 코드를 고칠 필요가 없습니다. 기본 프로파일은 시작값이며 새 호출어의 Development에서 임계값을 확인합니다.

## 학습 데이터 준비

**새 호출어 음성은 사용자가 Qwen TTS로 생성**합니다. 비호출·소음·잔향은 공개 데이터셋에서 내려받아 여러 호출어에 재사용합니다.

```powershell
koww datasets
koww download-data --dataset musan --output shared/musan.tar.gz
koww download-data --dataset rir --output shared/rirs_noises.zip
koww download-data --dataset zeroth --output shared/zeroth_korean.tar.gz
koww download-generic --output shared/openwakeword_features_ACAV100M_2000_hrs_16bit.npy
```

[데이터 다운로드 안내](docs/DATASETS.md)에 실제 사용한 YODAS 한국어 자료의 고정 shard, DNS 환경음, 출처·용량·이용조건·분할·내용 승인 절차를 정리했습니다. 다운로드 완료는 학습 승인과 다르며, 새 호출어가 포함된 것으로 확인된 비호출 구간은 제외합니다. 전수 STT를 기본 실행하지 않습니다.

## 포함된 원칙

- 호출어 **단독** Positive, 사람 음성 학습·튜닝·복제 참조 금지.
- 승인한 참조/청취 근거 재사용, 전수 독립 STT 금지, 기술 QC와 청취 구분.
- 실제 N×R 검증, source·family 분할, hash-bound cache와 중단 재개.
- GPU 메모리 기반 pool cache, CPU/GPU 측정, 결과를 확인한 feature 가속.
- 고정 768/256/50/0 batch, 실제 누적 노출량 기록.
- Development 점수 재사용으로 threshold 선택, Final 동결, 같은 WAV 채점·청취.

최신 기준은 [plan.md](plan.md)입니다. GitHub에는 설치 가능한 `korean_openwakeword/`, 시험, 가이드, 표본 모델과 결과 화면을 제공합니다. 날짜별 과거 실험·대용량 원본·캐시·개인 환경 기록은 개발 작업 폴더에 보존하며 설치 의존성에서 제외합니다.

소스 코드 이용조건과 모델·음원 조건은 다릅니다. [이용조건·출처](THIRD_PARTY_NOTICES.md)를 확인하세요. 현재 배포 대상은 GitHub 비공개 저장소이며 PyPI에는 게시하지 않습니다. 검증 범위는 [검수 기록](docs/VALIDATION.md)을 참고하세요.
