# 학습 데이터 다운로드와 재사용

**호출어 음성은 Qwen TTS로 새로 생성**합니다. 비호출·소음·잔향 자료는 공개 데이터셋에서 내려받아 여러 호출어 학습에 재사용할 수 있습니다.

이 문서는 자료를 구하는 방법을 안내합니다. 내려받은 음원을 학습에 연결하는 방법은 [자료 목록 작성법](DATA_FORMAT.md), 전체 순서는 [학습 가이드](LIBRARY_GUIDE.md)를 참고하세요.

## 어떤 자료를 받나요?

| 용도 | 공개 자료 | 참고할 점 |
| --- | --- | --- |
| 일반 비호출 특징(Generic) | [openWakeWord features](https://huggingface.co/datasets/davidscripka/openwakeword_features) | 제공자가 미리 계산한 숫자 배열. 명목 약 2,000시간이며 원본 WAV와 행별 출처는 없음 |
| 한국어 비호출 음성 | [YODAS 한국어](https://huggingface.co/datasets/espnet/yodas), [Zeroth Korean](https://www.openslr.org/40/) | 대화·음성 콘텐츠를 확보해 한국어 간섭 조건에 사용 |
| 환경음·음악 | [DNS-Challenge](https://github.com/microsoft/DNS-Challenge), [MUSAN](https://www.openslr.org/17/) | 종류가 다양한 자료를 선택. 음악과 환경소음 시간은 따로 기록 |
| 잔향(RIR) | [RIRS_NOISES](https://www.openslr.org/28/) | 공간의 울림을 재현. 잔향 파일 길이를 환경소음 시간으로 계산하지 않음 |
| 별도 평가용 한국어 음성 | [Seoul Corpus](https://www.openslr.org/113/) | 학습 원본과 분리해 평가 배경·연속 비호출에 사용 |

공개되어 있어도 자료마다 이용조건이 다릅니다. 출처와 제공자의 라이선스·원문 안내를 보존하세요. 코드의 Apache-2.0 라이선스를 음원 전체에 적용하지 않습니다. Generic은 제공자 CC BY-NC-SA 4.0, YODAS는 CC BY 3.0, Zeroth는 CC BY 4.0, Seoul은 CC BY-NC 2.0으로 기록되어 있습니다. MUSAN·DNS·RIRS는 개별 원출처 조건도 확인합니다. 자세한 조건은 [이용조건·출처](../THIRD_PARTY_NOTICES.md)에 있습니다.

## 기본 자료 다운로드

저장소 폴더에서 명령을 실행합니다. `shared/`는 여러 호출어에서 공유할 자료를 두는 예시이며, 원하는 데이터 폴더로 바꿀 수 있습니다. SSD나 HDD 중 특정 장치를 사용할 필요는 없습니다.

```powershell
koww datasets
koww download-generic --output shared/openwakeword_features_ACAV100M_2000_hrs_16bit.npy
koww download-data --dataset zeroth --output shared/zeroth_korean.tar.gz
koww download-data --dataset musan --output shared/musan.tar.gz
koww download-data --dataset rir --output shared/rirs_noises.zip
```

`koww datasets`는 자료의 출처·버전·파일 크기·이용조건을 보여줍니다. 기본 다운로드 파일의 크기는 다음과 같습니다. 압축 해제한 음원과 학습 결과의 용량은 별도입니다.

| 파일 | 다운로드 크기 |
| --- | --- |
| Generic 특징 | 약 17.28GB |
| Zeroth | 약 10.34GB |
| MUSAN | 약 11.09GB |
| RIRS_NOISES | 약 1.31GB |

GB는 10억 바이트 기준입니다. 중단된 다운로드는 같은 명령으로 이어받습니다. 완료 시 크기와 SHA256을 확인하고 다운로드 기록을 남기며, 정확한 파일이 이미 있으면 다시 받지 않습니다.

## 한국어 음성 추가 확보: YODAS

기본 한국어 비호출 목표는 약 200~300시간입니다. Zeroth만으로 이 목표를 채울 수는 없으므로 필요한 경우 YODAS 한국어 음성 콘텐츠를 함께 사용합니다.

라이브러리에는 초기 표본 모델에서 확보한 **ko000 압축 파일 30개의 고정 버전·해시·크기**가 기록되어 있습니다. 목록은 `koww datasets`, 원본 기록은 [public_datasets.json](../korean_openwakeword/resources/public_datasets.json)에서 확인합니다.

파일 한 개를 받는 예시입니다.

```powershell
koww download-data --dataset yodas_00000002 --output shared/yodas/ko000_00000002.tar.gz
```

필요한 파일부터 받고, 실제 사용 가능한 고유 시간을 합산하세요. 30개를 모두 내려받으려면 아래 코드를 `download_yodas.py`로 저장한 뒤 실행합니다. 전체 목록을 받는 대용량 작업입니다.

```python
from pathlib import Path
from korean_openwakeword.downloads import dataset_catalog, download_dataset

for key, asset in dataset_catalog()["assets"].items():
    if key.startswith("yodas_"):
        download_dataset(key, Path("shared/yodas") / asset["filename"])
```

제공자의 압축 음원을 사용하며 YouTube 영상을 따로 수집하지 않습니다. 같은 영상의 발화를 학습·평가 양쪽에 나누지 않고, 영상 ID를 화자 ID라고 기록하지 않습니다. 제공된 전사는 재사용할 수 있지만 모든 음원을 별도로 STT하는 과정은 기본 절차가 아닙니다.

## 다양한 환경소음 확보: DNS

팬·에어컨·청소기·도로·차량·비·바람·키보드·문·그릇·충격음·생활 기계음 등 **소음 종류의 다양성**을 먼저 확보하세요. 환경소음은 고다양성 30~50시간에서 시작하고 평가 결과에 따라 100~200시간까지 확대합니다.

[고정 버전의 소음 폴더](https://github.com/microsoft/DNS-Challenge/tree/70f19285c36cca4df2338f9248775ddc50980c6b/datasets/noise)에서 필요한 자료를 선택합니다. 같은 종류의 긴 녹음만 추가해 시간을 채우지 않습니다. 원출처·원본 ID·이용조건도 보존하세요.

아래는 다운로드 방법을 확인할 수 있는 짧은 팬 소음 예시입니다. 이 한 파일만으로 소음 자료 준비가 끝나는 것은 아닙니다.

```python
from korean_openwakeword.downloads import download_checked

download_checked(
    "https://media.githubusercontent.com/media/microsoft/DNS-Challenge/70f19285c36cca4df2338f9248775ddc50980c6b/datasets/noise/fan_Freesound_validated_100355_0.wav",
    "shared/dns/fan_100355.wav",
    "a3d2ede479606cae392b5332973ed1a1a6e9f0d4176e030bb22920379e3cd5ba",
    320044,
)
```

음성이 포함된 간섭은 한국어 자료를 우선 사용합니다. 일반 환경음에는 언어 기준을 적용하지 않습니다.

## 다운로드 후 해야 할 일

1. 압축을 풀고 원출처·이용조건을 보존합니다. 압축 도구는 대상 폴더 밖으로 파일이 나가지 않게 사용합니다.
2. 원본 녹음·영상·화자·방 단위로 학습/개발/최종 평가 자료를 분리합니다. 같은 원본의 절단본이나 복사본도 같은 원본으로 관리합니다.
3. 파일 품질과 대표 구간의 내용을 확인합니다. 새 호출어가 실제로 포함된 비호출 구간은 제외합니다.
4. [자료 목록](DATA_FORMAT.md)에 파일 경로·해시·출처·분할·검수 결과를 기록합니다.
5. [학습 가이드](LIBRARY_GUIDE.md#5-비호출소음잔향-자료-연결하기)의 `audit`·`windows`·`import-generic` 순서로 프로젝트에 연결합니다.

같은 소리가 여러 파일·증강 결과에 반복되어도 고유 원본 시간은 늘어나지 않습니다. 음악을 한국어 대화 시간으로, 잔향을 환경소음 시간으로 계산하지 않습니다. 확보하지 않은 방송·보컬 범주를 확보한 것처럼 기록하지 않습니다.

<details>
<summary>초기 표본 모델에서 사용한 한국어 자료 규모</summary>

정책에 따라 선별한 원본은 YODAS 약 185.7시간, Zeroth 약 42.8시간이었습니다. 실제 학습용 고정 구간이 대표한 고유 시간은 약 221.3시간입니다. 이 수량은 이전 준비 결과이며 다운로드만으로 자동 확보되는 승인 시간은 아닙니다.

</details>
