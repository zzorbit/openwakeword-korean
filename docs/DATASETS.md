# 학습 데이터 다운로드와 재사용

호출어 Positive는 사용자가 **Qwen TTS로 새로 생성**한다. 비호출·소음·잔향 원본은 아래 공개 자료에서 확보한다. 이전 호출어에서 검증한 자료와 승인 근거는 재사용할 수 있지만, 새 호출어가 들어 있는 것으로 확인된 구간은 Negative에서 제외한다. 합성 Train / Validation / Final 음색 계열과 배경 원본은 서로 분리한다.

## 실제 춘식아에 사용한 출처

| 자료 | 출처 | 사용 범위 |
|---|---|---|
| Generic | [openWakeWord features](https://huggingface.co/datasets/davidscripka/openwakeword_features) | ACAV100M 기반 precomputed array. 제공자 명목 약2,000h, 원본 WAV·행별 source ID는 제공되지 않음 |
| 한국어 Negative | [YODAS ko000](https://huggingface.co/datasets/espnet/yodas) + [Zeroth Korean](https://www.openslr.org/40/) | 춘식아에서 정책 선별 원본 약185.7h +42.8h. 실제 고정 window가 대표한 고유 시간은 약221.3h |
| 환경음·음악 | [DNS-Challenge](https://github.com/microsoft/DNS-Challenge) + [MUSAN](https://www.openslr.org/17/) | 환경소음 1차 고다양성30~50h, 음악 별도. 자료 전체 시간을 환경음 시간으로 더하지 않음 |
| RIR | [RIRS_NOISES](https://www.openslr.org/28/) | 공간·잔향 증강. RIR 길이를 환경소음 시간으로 세지 않음 |
| 평가용 한국어 대화 | [Seoul Corpus](https://www.openslr.org/113/) | 독립 배경/연속 Negative에 사용. 해당 녹음은 Train에 재사용하지 않음 |

출처 설명과 개별 음원의 이용조건을 함께 보존한다. Generic은 제공자 CC BY-NC-SA4.0, YODAS는 제공자 CC BY3.0, Zeroth는 CC BY4.0, Seoul은 CC BY-NC2.0으로 기록했다. MUSAN/DNS/RIRS는 포함된 개별 원출처 notice를 확인한다. 이 안내가 새로운 재배포·상업 이용 허가를 부여하지 않는다. [THIRD_PARTY_NOTICES](../THIRD_PARTY_NOTICES.md)를 따른다.

## 저장 위치·다운로드

먼저 원하는 HDD 작업 폴더에서 설치한다. PowerShell은 `. .\scripts\Use-ProjectStorage.ps1`을 적용한다. 압축 원본과 압축 해제본이 함께 존재하므로 archive bytes만 보고 여유 공간을 정하지 않는다. 학습 feature와 평가 WAV 용량도 별도다. 원본은 라이브러리 저장소가 아닌 `shared/` 또는 별도 데이터 드라이브에 둔다.

```powershell
koww datasets
koww download-generic --output shared/openwakeword_features_ACAV100M_2000_hrs_16bit.npy
koww download-data --dataset zeroth --output shared/zeroth_korean.tar.gz
koww download-data --dataset musan --output shared/musan.tar.gz
koww download-data --dataset rir --output shared/rirs_noises.zip
```

현재 pin 기준 Generic17.28GB / Zeroth10.34GB / MUSAN11.09GB / RIRS1.31GB이며 단위는 decimal GB다. 다운로드는 `.part`에 저장하고 재실행 시 이어받는다. SHA256·크기 검증 후 완료 파일과 receipt를 만든다. 이미 정확한 파일이 있으면 재다운로드하지 않는다. 원시 자료를 자동 승인하거나 새 실험을 시작하지 않는다.

### YODAS 한국어

`korean_openwakeword/resources/public_datasets.json`에는 실제 확보했던 ko000 audio archive30개의 URL·revision·SHA256·크기가 있다. 전체 ko000을 받지 않고 필요한 shard만 사용했다. ID·이용조건·버전은 `koww datasets`로 확인한다.

```powershell
koww download-data --dataset yodas_00000002 --output shared/yodas/ko000_00000002.tar.gz
```

같은30개를 모두 받으려면 아래를 실행한다. **대용량 다운로드를 시작하는 명령**이며 목표 시간이 부족한 경우 필요한 shard만 먼저 받는다.

```python
# download_yodas.py로 저장한 뒤 python download_yodas.py
from pathlib import Path
from korean_openwakeword.downloads import dataset_catalog, download_dataset
for key, asset in dataset_catalog()["assets"].items():
    if key.startswith("yodas_"):
        download_dataset(key, Path("shared/yodas") / asset["filename"])
```

배포자 제공 audio archive를 사용하며 YouTube 영상을 따로 수집하지 않는다. 같은 영상의 모든 utterance는 같은 split으로 묶는다. 영상 ID를 화자 ID로 간주하지 않는다. 자막의 기존 문장에서 호출어 포함 여부를 점검할 수 있지만 전수 독립 STT를 새로 수행하지 않는다. 자막 구간은 서로 겹칠 수 있으므로 파일 길이 합계를 고유 원본 시간으로 주장하지 않는다.

### DNS 환경소음

춘식아의 고정 revision은 `70f19285c36cca4df2338f9248775ddc50980c6b`이며 `datasets/noise/`의 필요한 종류를 사용했다. Freesound 선별 음원과 AudioSet 유래 음원은 원출처·원본 ID·이용조건을 함께 남긴다. 전체 저장소와 모든 challenge 자료를 한꺼번에 받는 것이 기본 절차는 아니다.

공식 [고정 버전 폴더](https://github.com/microsoft/DNS-Challenge/tree/70f19285c36cca4df2338f9248775ddc50980c6b/datasets/noise)에서 필요한 파일을 고른다. 짧은 fan 예시는 아래 고정 URL로 받을 수 있다.

```python
from korean_openwakeword.downloads import download_checked
download_checked(
    "https://media.githubusercontent.com/media/microsoft/DNS-Challenge/70f19285c36cca4df2338f9248775ddc50980c6b/datasets/noise/fan_Freesound_validated_100355_0.wav",
    "shared/dns/fan_100355.wav",
    "a3d2ede479606cae392b5332973ed1a1a6e9f0d4176e030bb22920379e3cd5ba",
    320044,
)
```

팬·에어컨·청소기·도로·차량·비·바람·키보드·문·그릇·충격·기계음 등 종류와 실제 음향 분포를 먼저 확보한다. 한국어 competing speech는 YODAS/Zeroth의 적절한 별도 원본 구간으로 구성한다. 짧은 fan 예제 한 파일만으로 소음 Pool 목표가 충족되지는 않는다.

## 다운로드 다음 단계

1. checksum 확인 후 압축을 데이터 폴더에 푼다. 상대 경로 이탈·symlink를 허용하지 않는 신뢰 가능한 archive 도구를 사용하고 원본 archive와 notice를 보존한다.
2. 원본 ID·영상/화자/방 ID를 먼저 분할한다. 동일 원본의 절단본·이름만 다른 복사본을 다른 split에 넣지 않는다. RIRS의 MUSAN 유래 소음 중복도 같은 원본으로 관리한다.
3. 실제 파형 기술 검사와 대표 내용 검수를 진행한다. 승인하지 않은 후보는 manifest의 학습 사용 대상으로 확정하지 않는다. 표본 청취를 전수 청취로 기록하지 않는다.
4. [라이브러리 가이드](LIBRARY_GUIDE.md#공개-오디오-manifest와-내용-승인)의 JSONL에 path/SHA256/source URL/license/split/admission을 기록한다. 파형 Negative는 `koww windows`로 고정2초 구간을 준비한다. Generic feature는 `koww import-generic`으로 연결한다.
5. `koww audit` → features → train → package → Development/tune → freeze → Final → site 순서로 진행한다.

YODAS의 장르, Zeroth의 낭독 음성, MUSAN의 음악을 실제로 보유한 범주대로 기록한다. 확보하지 않은 한국어 방송/보컬 시간을 임의로 채우지 않는다. 학습 중 같은 자료가 여러 번 샘플링되거나 여러 rounds로 증강되어도 고유 원본 시간은 늘어나지 않는다.
