# 학습·평가 자료 목록 작성법

이 문서는 다운로드한 공개 음원을 프로젝트에 연결하는 방법을 설명합니다. 음원을 내려받는 절차는 [데이터 준비 안내](DATASETS.md), 전체 작업 순서는 [학습 가이드](LIBRARY_GUIDE.md)를 참고하세요.

## 자료 목록이란?

자료 목록(manifest)은 **어떤 파일을 어디에 사용할지 적는 JSONL 파일**입니다. JSONL은 한 줄에 하나의 JSON 객체를 저장하는 형식입니다. 호출어 자료 목록은 생성 명령이 만들지만, 공개 자료 목록은 사용하는 원본의 출처와 검수 결과를 넣어 준비해야 합니다.

경로는 `project.json`이 있는 폴더를 기준으로 적습니다. 예를 들어 `data/negative/source_0001.wav`는 `projects/bora/data/negative/source_0001.wav`를 가리킵니다. 여러 프로젝트에서 공유하는 음원은 절대 경로로 연결할 수도 있습니다.

| 필드 | 적을 내용 |
| --- | --- |
| `sample_id` | 목록에서 중복되지 않는 항목 이름 |
| `path` | 실제 음원 파일 경로 |
| `sha256` | 파일 내용이 바뀌지 않았는지 확인할 SHA256 값 |
| `source_id` / `canonical_source_id` | 원본 녹음·영상의 식별자. 자른 파일들도 같은 원본이면 같은 식별자 사용 |
| `split` | `train`(학습), `validation`(개발용 평가), `final`(최종 평가) |
| `source_url` / `license` | 원출처 주소와 이용조건 |
| `provenance` | 공개 자료는 `public_dataset` |
| `category` | 환경음 `environment`, 음악·전자음 `music`, 겹치는 대화 `competing_speech`, 잔향 `rir` |
| `speech` / `language` | 발화 유무와 실제 언어 |
| `admission` | 내용 검수 후 반환된 승인 파일의 경로·해시 |

화자 정보가 있으면 `canonical_speaker_id`, 방 정보가 있으면 `canonical_room_id`를 추가합니다. 같은 원본·화자·음색·방이 학습과 평가 양쪽에 들어가지 않도록 분리합니다. 원본 화자 정보가 없으면 임의의 화자 ID를 만들어 분리가 입증된 것처럼 기록하지 않습니다.

## 파일 해시 확인하기

Windows PowerShell 예시입니다. 출력한 값을 자료 목록의 `sha256`에 넣습니다.

```powershell
((Get-FileHash -LiteralPath 'projects/bora/data/negative/source_0001.wav' -Algorithm SHA256).Hash).ToLower()
```

다운로드 완료 기록(receipt)은 파일이 정확히 내려받아졌다는 근거입니다. 내용 검수 승인과는 별개입니다.

## 공개 음원의 대표 내용 검수

자료가 설명한 범주에 맞고 사용 가능한지 대표 구간을 듣습니다. 깨짐·부적절한 내용·실제로 포함된 호출어 등 문제가 있는 원본은 제외합니다. 모든 음원을 새로 STT하거나 모든 학습 구간을 하나씩 청취하지는 않습니다.

`projects/bora/review/public_queue.jsonl`에 다음 형태의 검수 항목을 준비합니다. 아래 JSON은 읽기 쉽게 줄을 나누었으며, **실제 JSONL에는 항목 하나를 한 줄로 저장**합니다. 경로·해시·출처·이용조건은 실제 값으로 바꾸세요.

```json
{
  "sample_id": "public_0001",
  "path": "review/previews/source_0001.wav",
  "sha256": "실제_미리듣기_파일_SHA256",
  "category": "competing_speech",
  "provenance": "public_dataset",
  "source_url": "실제_원출처_URL",
  "license": "실제_이용조건",
  "reviewed_source": {
    "path": "data/negative/source_0001.wav",
    "sha256": "실제_원본_파일_SHA256"
  }
}
```

`path`는 들을 대표 구간, `reviewed_source`는 그 구간이 나온 원본입니다. 대표 파일을 원본과 연결해야 해당 원본을 사용할 수 있습니다.

```powershell
koww review --project projects/bora/project.json --queue projects/bora/review/public_queue.jsonl --kind content --port 8790
```

[http://127.0.0.1:8790/](http://127.0.0.1:8790/)에서 검수하고 저장합니다. 반환 결과의 `admission` 객체를 보존한 원본 항목에 넣으세요. 이 객체는 변경되지 않는 승인 기록을 가리킵니다. 최신 화면 요약인 `review/content_admission.json`을 대신 연결하면 후속 검수 때 내용이 바뀔 수 있습니다.

검수하지 않은 자료에 승인 결과를 임의로 만들어 넣지 않습니다. 폐기한 원본은 학습 목록에서 제외합니다.

## 학습 자료 목록 예시

한국어 비호출 원본을 `data/negative/korean.jsonl`에 넣는 예시입니다.

```json
{
  "sample_id": "ko_0001",
  "source_id": "corpus:recording_0001",
  "canonical_source_id": "corpus:recording_0001",
  "canonical_speaker_id": "corpus:speaker_001",
  "split": "train",
  "path": "data/negative/source_0001.wav",
  "sha256": "실제_원본_파일_SHA256",
  "provenance": "public_dataset",
  "source_url": "실제_원출처_URL",
  "license": "실제_이용조건",
  "category": "competing_speech",
  "language": "Korean",
  "speech": true,
  "admission": {
    "path": "review/evidence/content_admission_실제ID.json",
    "sha256": "실제_승인파일_SHA256"
  }
}
```

학습용 소음·잔향 목록에도 출처·검수 필드를 사용합니다. 순수 환경음은 `speech: false`로 기록하고 발화 대비 소음 비율을 억지로 계산하지 않습니다. 환경소음·음악·한국어 대화가 섞이는 조건을 모두 준비하세요.

개발용·최종 평가용 배경음은 각각 `validation_noise.jsonl`, `final_noise.jsonl`에 해당 `split`으로 기록합니다. 학습용 자료를 이름만 바꾸거나 다른 구간으로 잘라 평가용에 넣지 않습니다.

긴 원본에서 여러 구간을 사용하면 원본 식별자를 유지하고 `start_s`, `end_s`로 범위를 적습니다. 기존에 잘린 음원은 부모 녹음 식별자 `canonical_recording_id` 또는 `parent_source_id`, 부모 녹음에서의 시작 위치 `parent_start_s`도 남깁니다. 겹치는 구간을 중복 시간으로 합산하지 않습니다.

목록을 준비한 뒤 다음으로 파일·출처·자료 분리를 확인합니다.

```powershell
koww audit --project projects/bora/project.json
```

## 유사어 학습 자료

유사어(Hard Negative)는 호출어와 발음이 비슷한 **다른 표현**입니다. 공개 음원의 해당 구간을 고르거나 Qwen TTS로 별도 생성한 음성을 `data/negative/hard.jsonl`에 연결합니다. 호출어 전체가 포함된 표현은 비호출 정답으로 넣지 않습니다. 개인의 호출 녹음이나 실제 사람 음성을 TTS 참조로 사용하지 않습니다.

공개 자료는 위 비호출 예시와 같은 출처·승인 필드를 사용하고, **`text`에 실제 유사 표현**, **`speech: true`**, **`split: "train"`**을 추가합니다. 합성 자료는 `provenance: "qwen_tts"`, `human_reference_used: false`, 학습용 `voice_family_id`를 기록하고, 출처·생성 근거와 내용 검수 결과를 보존합니다. 합성 음색도 개발용·최종 평가용 계열과 분리합니다.

공개 또는 합성 유사어의 대표 검수 목록을 `koww review --kind content`로 확인한 뒤, 반환된 승인 기록을 해당 원본의 `admission`에 연결합니다. 새 유사어 생성 방식은 대표 발음을 먼저 검수하고 전체 음원을 다시 STT하지 않습니다. `init`은 빈 목록만 만들며 유사어를 자동 수집·생성·승인하지 않습니다.

긴 원본은 `koww windows --pool hard_negative`로 고정 2초 목록을 준비하고, `manifests.hard_negative`를 그 목록으로 바꿉니다. 이어서 `koww features --role hard_negative`로 다른 비호출과 같은 계열의 증강을 적용합니다. 준비된 음성이 비어 있으면 학습을 시작할 수 없습니다.

배치의 유사어 **100개는 매회 추출량**이며 전체 원본 수가 아닙니다. 원본 수·표현별 분포·누적 추출량을 따로 기록하고, 일반·한국어 비호출과 중복되는 원본 시간은 다시 더하지 않습니다.

## 평가 자료 목록

`prepare-eval`이 만드는 `manifest.jsonl`은 소음 조건별 호출어 평가 목록입니다. 여기에 연속 비호출 평가 목록을 합칩니다.

연속 비호출 항목 예시입니다. 평가 음원은 **16kHz·모노·16비트 PCM WAV**로 준비합니다. 형식을 변환했다면 변환한 파일의 해시를 기록합니다.

```json
{
  "sample_id": "dev_negative_0001",
  "source_id": "corpus:dev_recording_0001",
  "canonical_source_id": "corpus:dev_recording_0001",
  "split": "validation",
  "kind": "negative",
  "category": "competing_speech",
  "path": "data/eval/negative/dev_0001.wav",
  "sha256": "실제_평가파일_SHA256",
  "provenance": "public_dataset",
  "source_url": "실제_원출처_URL",
  "license": "실제_이용조건",
  "calls": []
}
```

같은 비호출 파일을 반복하거나 이어붙여 독립 평가 시간을 늘리지 않습니다. 최종 평가에서는 새로운 원본을 쓰고 `split`을 `final`로 지정합니다.

호출어 평가 항목은 `kind: "positive"`이며 `calls`에 실제 호출 시작·끝(`start_s`, `end_s`)과 원본 발화 식별자(`original_utterance_id`)가 있어야 합니다. 자동 생성한 목록의 호출 경계를 임의로 바꾸지 마세요.

다음 Python 코드를 `combine_development.py`로 저장해 실행하면 두 목록을 합칩니다. 비호출 목록을 먼저 준비하세요. 이미 내용이 있는 출력 파일은 덮어쓰지 않으며, 큰 목록도 한 줄씩 처리합니다.

```python
from pathlib import Path

positive = Path("projects/bora/data/eval/paired_v1/manifest.jsonl")
negative = Path("projects/bora/data/eval/development_negative.jsonl")
output = Path("projects/bora/data/eval/development.jsonl")

for source in (positive, negative):
    if not source.is_file():
        raise FileNotFoundError(source)
if output.exists() and output.stat().st_size:
    raise FileExistsError("새 출력 파일 이름을 사용하세요.")
with output.open("w", encoding="utf-8") as target:
    for source in (positive, negative):
        with source.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    target.write(line.rstrip("\n") + "\n")
```

`init`이 빈 평가 파일을 만들어 두므로 위 예시는 그 파일을 채웁니다. **이미 평가에 사용한 파일에는 다시 실행하지 않습니다.**

최종 평가 목록도 최종 호출어·비호출 파일을 사용해 같은 방식으로 준비합니다. 실제 사람 호출을 추가할 때는 `provenance: "human"`, `speaker_id`, `distance_m`, `direction`, `microphone`을 기록하고 최종 평가에만 사용합니다.
