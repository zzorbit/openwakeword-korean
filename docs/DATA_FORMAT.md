# 학습·평가 자료 목록 작성법

이 문서는 다운로드한 공개 음원을 프로젝트에 연결하는 방법을 설명합니다. 음원을 내려받는 절차는 [데이터 준비 안내](DATASETS.md), 전체 작업 순서는 [학습 가이드](LIBRARY_GUIDE.md)를 참고하세요.

## 제공 배경 자료의 자동 연결

웹의 제공 학습·평가 배경 자료는 자동 검사 후 연결합니다. 고정 제공 출처의 선정·검증·제외 기준은 라이브러리 제작 단계에서 관리하며, 호출어를 만들 때마다 같은 배경을 다시 청취하도록 요구하지 않습니다. 직접 추가한 자료와 새로 생성한 합성 참조·호출어·유사어는 검수합니다.

`library_fixed_sources_v1`은 라이브러리가 고정한 카탈로그의 출처·버전·파일 pin과 실제 준비 자료가 일치하고 자동 기술·분할 검사를 통과했을 때만 연결 근거를 만듭니다. 이 근거는 정책·파일 해시·검사 결과를 보존하고 `human_listening: false`로 기록합니다. 사람의 청취 답변이나 묶음 사용 동의를 생성하지 않으며, 모든 원본의 내용을 사람이 확인했다는 뜻이 아닙니다. 파일·출처 조건이 달라지거나 검사에 실패한 자료를 통과한 것으로 처리하지 않습니다.

아래 수동 명령은 사용자가 직접 추가한 음원을 준비·검수·연결하는 경로입니다. 웹의 고정 제공 자료를 사용할 때는 반복할 필요가 없습니다. 기존 프로젝트의 승인 기록과 완료 자료는 자동 변경하지 않습니다.

## 공개 음원 자동 준비

직접 추가한 자료는 아래 명령으로 자료 목록을 자동으로 만드세요. 직접 JSONL을 작성하는 형식 설명은 문서 뒤의 고급 사용 안내입니다.

**1. 제공자 메타데이터 연결**

Zeroth는 배포본의 화자 폴더와 `*.trans.txt`를 읽습니다. 전사 모델을 새로 실행하지 않습니다.

```powershell
koww index-data --dataset zeroth --input shared/zeroth/train_data_01 --output shared/zeroth_index.jsonl --source-url https://www.openslr.org/40/ --license CC-BY-4.0
```

다른 자료가 음원별 출처·범주·이용조건을 가진 JSONL 또는 JSON의 `files`/`tracks` 목록을 제공하면 `index-data --dataset manifest --metadata 제공자목록.json`으로 연결합니다. CSV/JSONL의 음원별 정보를 `--source-index`로 직접 사용할 수도 있습니다. 여러 범주가 섞인 자료는 각 음원의 실제 범주를 기록하세요. 언어·화자·방을 소리에서 자동 추정하지 않습니다.

공식 예제의 소음·잔향 자료도 원출처 메타데이터를 직접 읽습니다. 아래 경로는 다운로드·압축 해제 위치에 맞게 바꾸세요.

```powershell
koww index-data --dataset audioset --input shared/audioset/audio/bal_train --metadata shared/audioset/balanced_train_segments.csv --ontology shared/audioset/ontology.json --output shared/audioset_index.jsonl
koww index-data --dataset fma --input shared/fma_small --metadata shared/fma_metadata/tracks.csv --output shared/fma_index.jsonl
koww index-data --dataset mit_rir --input shared/mit_rir/16khz --output shared/mit_rir_index.jsonl
```

AudioSet는 제공자 라벨로 범주와 원본 영상·구간을, FMA는 개별 곡의 이용조건·곡 ID를 읽습니다. 이용조건이 없는 곡과 같은 곡의 중복 파일은 제외 이유를 기록합니다. 말소리 유무와 언어가 제공되지 않았으면 미확인으로 유지합니다.

MIT에 실제 공간 대응표가 없다면 전체 자료를 하나의 보수적인 공간 그룹으로 묶습니다. **각 파일이 독립된 방이라고 주장하지 않습니다.** 이 경우 `prepare-materials --validation-fraction 0`으로 학습용 전체를 준비하고, 공간 평가용 자료는 별도 출처에서 확보하세요. 실제 공간 대응표가 있으면 `index-data --room-metadata`로 연결합니다.

**2. 변환·분리·대표 검수 목록 만들기**

```powershell
koww prepare-materials --project models/bora/project.json --input shared/zeroth/train_data_01 --source-index shared/zeroth_index.jsonl --hours 1 --output models/bora/prepared/korean_v1
```

원본을 보존하고 16kHz·모노·PCM16 음원, 기술 검사, 학습 분량, 개발용 예약 자료, 대표 청취 목록을 만듭니다. `--hours`는 고유 학습 분량이며 같은 파일을 복제해 채우지 않습니다. 자료가 부족하면 `target_met: false`와 실제 시간을 보고합니다.

분할을 먼저 고정한 뒤 학습 분할 안에서 원본 녹음 단위로 필요한 분량을 선택합니다. 마지막 녹음 길이만큼 목표 시간을 넘을 수 있습니다. 전체 원본의 해시·메타데이터·파일 헤더는 확인하고, 실제 변환·기술 검사는 선택한 학습 자료와 필요한 개발용 예약 자료에만 수행합니다. 선택하지 않은 원본을 검사·청취·승인 완료로 기록하지 않습니다.

같은 녹음·화자·공간은 같은 분할에 둡니다. 기본 개발용 예약 비율은 10%이며 실제 비율은 그룹 크기에 따라 달라질 수 있습니다. 화자를 알 수 없는 같은 자료셋의 음성은 하나의 그룹으로 취급합니다. 별도 평가 자료를 확보해 학습용만 준비하려면 `--validation-fraction 0`을 지정하세요. Final 음원은 이 경로에서 열지 않습니다.

폴더 전체의 출처와 범주가 동일하다면 `--dataset-id`, `--source-url`, `--license`, `--category`, `--category-basis`, `--language`, `--speech` 또는 `--no-speech`로 지정할 수 있습니다. 잔향은 `--category rir`로 준비하고, 제공자가 명시한 실제 `canonical_room_id`를 음원별 목록에 넣습니다. 파일 이름마다 다른 방이라고 가정하지 않습니다.

잔향 검수 화면에는 고정 시험음을 해당 RIR에 통과시킨 소리를 제공합니다. 울림을 듣기 위한 미리듣기이며 학습용 RIR을 바꾸거나 성능 평가 오디오로 사용하지 않습니다. RIR은 시간 대신 파일 수와 공간 다양성으로 관리합니다.

**3. 검수 답변 적용·학습 자료 연결**

```powershell
koww review --project models/bora/project.json --queue models/bora/prepared/korean_v1/representatives.jsonl --pool-manifest models/bora/prepared/korean_v1/train_sources.jsonl --kind content
# 대표 음원을 듣고 자료 묶음 사용 동의·저장을 완료합니다. Ctrl+C로 서버를 종료합니다.
koww admit-materials --project models/bora/project.json --preparation models/bora/prepared/korean_v1/preparation_receipt.json --pools korean_negative --connect
```

저장된 답변을 서버에서 읽어 승인 목록을 연결합니다. 폐기한 대표의 원본은 제외하고 재검수 항목이 남으면 중단합니다. 대표 검수와 묶음 사용 동의는 듣지 않은 모든 음원의 개별 청취를 뜻하지 않습니다. 일반·한국어 비호출은 고정 2초 구간까지 준비하므로 `project.json`을 직접 바꿀 필요가 없습니다.

소음 재료로도 쓸 한국어 원본이라면 `--pools korean_negative noise`를 지정할 수 있습니다. 같은 원본을 두 역할로 재사용해도 고유 시간을 다시 더하지 않습니다.

**여러 자료셋 병합**

AudioSet·음악을 각각 준비·검수했다면 `admit-materials --pools noise`로 승인 목록만 만들고, 다음으로 함께 연결합니다. 이 예시의 경로는 각 준비 결과에 맞추세요.

```powershell
koww connect-materials --project models/bora/project.json --admissions models/bora/prepared/audioset_v1/admitted/admission_receipt.json models/bora/prepared/music_v1/admitted/admission_receipt.json --pools noise
```

출처·승인·해시·교차 분할을 다시 확인한 뒤 비어 있는 학습 목록에 한 번 연결합니다. 이미 시작한 실험은 바꾸지 않습니다. 중단된 준비·승인·연결은 같은 입력과 명령으로 재개하며, 기존 결과와 내용이 다르면 새 출력 폴더를 사용합니다.

`validation_reserve.jsonl`은 학습과 분리한 개발용 원본 목록입니다. 이를 평가용으로 사용할 때도 다른 원본과의 분리를 확인하고 아래 평가 목록 형식을 따릅니다. 학습 자료의 승인으로 Final 사용을 승인하지 않습니다.

## 자료 목록이란?

자료 목록(manifest)은 **어떤 파일을 어디에 사용할지 적는 JSONL 파일**입니다. JSONL은 한 줄에 하나의 JSON 객체를 저장하는 형식입니다. 호출어는 생성 명령, 공개 음원은 위 준비·승인 명령이 목록을 만듭니다. 직접 작성하거나 외부 자료를 연결할 때는 아래 필드를 사용합니다.

경로는 `project.json`이 있는 폴더를 기준으로 적습니다. 예를 들어 `data/negative/source_0001.wav`는 `models/bora/data/negative/source_0001.wav`를 가리킵니다. 여러 프로젝트에서 공유하는 음원은 절대 경로로 연결할 수도 있습니다.

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
| `canonical_speaker_ids` | 제공자가 여러 참여 화자를 명시했다면 모든 화자 ID. 주 화자만 확인해 분리하지 않음 |
| `admission` | 고정 제공 정책 또는 직접 추가한 자료의 내용 검수로 반환된 연결 근거 파일의 경로·해시 |

화자 정보가 있으면 `canonical_speaker_id`, 방 정보가 있으면 `canonical_room_id`를 추가합니다. 같은 원본·화자·음색·방이 학습과 평가 양쪽에 들어가지 않도록 분리합니다. 원본 화자 정보가 없으면 임의의 화자 ID를 만들어 분리가 입증된 것처럼 기록하지 않습니다.

## 파일 해시 확인하기

Windows PowerShell 예시입니다. 출력한 값을 자료 목록의 `sha256`에 넣습니다.

```powershell
((Get-FileHash -LiteralPath 'models/bora/data/negative/source_0001.wav' -Algorithm SHA256).Hash).ToLower()
```

다운로드 완료 기록(receipt)은 파일이 정확히 내려받아졌다는 근거입니다. 고정 제공 정책의 자동 검사·연결 근거나 직접 추가한 자료의 내용 검수 승인과는 별개입니다.

## 공개 음원의 대표 내용 검수

직접 추가한 자료가 설명한 범주에 맞고 사용 가능한지 대표 구간을 듣습니다. 깨짐·부적절한 내용·실제로 포함된 호출어 등 문제가 있는 원본은 제외합니다. 모든 음원을 새로 STT하거나 모든 학습 구간을 하나씩 청취하지는 않습니다.

`models/bora/review/public_queue.jsonl`에 다음 형태의 검수 항목을 준비합니다. 아래 JSON은 읽기 쉽게 줄을 나누었으며, **실제 JSONL에는 항목 하나를 한 줄로 저장**합니다. 경로·해시·출처·이용조건은 실제 값으로 바꾸세요.

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
koww review --project models/bora/project.json --queue models/bora/review/public_queue.jsonl --kind content --port 8790
```

[http://127.0.0.1:8790/](http://127.0.0.1:8790/)에서 검수하고 저장합니다. 반환 결과의 `admission` 객체를 보존한 원본 항목에 넣으세요. 이 객체는 변경되지 않는 승인 기록을 가리킵니다. 최신 화면 요약인 `review/content_admission.json`을 대신 연결하면 후속 검수 때 내용이 바뀔 수 있습니다.

검수하지 않은 자료에 승인 결과를 임의로 만들어 넣지 않습니다. 폐기한 원본은 학습 목록에서 제외합니다.

## 학습 자료 목록 예시

독립 최종 평가의 배경·비호출·잔향도 실제 `path`·`sha256`·원출처·이용조건과 일관된 원본 ID를 기록하고 `split`은 `final`로 지정합니다. 같은 원본의 파일명·구간·형식이 바뀌어도 `canonical_recording_id`를 바꾸지 않습니다. 알려진 화자·음악가·공간 정보가 있으면 각각 `canonical_speaker_id`·`canonical_artist_id`·`canonical_room_id`로 보존합니다. 모르는 신원은 추측해 채우지 않습니다.

`prepare-eval`은 혼합 오디오에 호출어·배경 원본의 `components`를 자동 기록합니다. 외부에서 만든 Final 혼합 오디오도 `role: call`과 `role: background`의 구성 자료를 제공해야 합니다. 각 구성 자료에 원본 경로·해시·출처·분할·음색 등 실제 메타데이터를 넣습니다. 잔향을 적용했다면 `role: rir`의 원본·공간 근거도 포함합니다. 구성 원본을 학습·공통 평가와 대조하므로 완성 WAV의 이름만 바꿔 Final로 사용할 수 없습니다.

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

학습용 소음·잔향 목록에도 출처와 연결 근거를 기록하며, 직접 추가한 자료에는 검수 필드를 사용합니다. 순수 환경음은 `speech: false`로 기록하고 발화 대비 소음 비율을 억지로 계산하지 않습니다. 환경소음·음악·한국어 대화가 섞이는 조건을 모두 준비하세요.

개발용·최종 평가용 배경음은 각각 `validation_noise.jsonl`, `final_noise.jsonl`에 해당 `split`으로 기록합니다. 학습용 자료를 이름만 바꾸거나 다른 구간으로 잘라 평가용에 넣지 않습니다.

긴 원본에서 여러 구간을 사용하면 원본 식별자를 유지하고 `start_s`, `end_s`로 범위를 적습니다. 기존에 잘린 음원은 부모 녹음 식별자 `canonical_recording_id` 또는 `parent_source_id`, 부모 녹음에서의 시작 위치 `parent_start_s`도 남깁니다. 겹치는 구간을 중복 시간으로 합산하지 않습니다.

목록을 준비한 뒤 다음으로 파일·출처·자료 분리를 확인합니다.

```powershell
koww audit --project models/bora/project.json
```

## 유사어 학습 자료

유사어(Hard Negative)는 호출어와 발음이 비슷한 **다른 표현**입니다. 공개 음원의 해당 구간을 고르거나 Qwen TTS로 별도 생성한 음성을 `data/negative/hard.jsonl`에 연결합니다. 호출어 전체가 포함된 표현은 비호출 정답으로 넣지 않습니다. 개인의 호출 녹음이나 실제 사람 음성을 TTS 참조로 사용하지 않습니다.

문구 수는 사용자가 정하며, Qwen 생성량은 **한 문구당 동작 확인용 20개 / 시험 모델용 100개 / 상용 모델용 500개**입니다. 각 문구를 학습용 목소리마다 한 번씩 생성하고, 전체 수량은 **문구 수 × 문구당 생성량**으로 계산합니다. 문구 수·문구당 수량·전체 수량을 구분해 기록하며, 기존 총량 방식의 음원·검수·승인 기록은 보존합니다.

새 웹 합성 유사어는 **문구당 대표 1개를 실제로 청취**하고, 그 결과로 같은 문구의 나머지 음원을 사용 승인합니다. 전체 생성 음원의 기술 검사는 유지합니다. 대표 선정 정책·seed·원본 목록·대표 목록·실제 답변의 해시를 보존하며, 직접 들은 대표와 대표 검수에 근거해 연결한 원본 수를 구분합니다. **유사어 전체 듣기**는 읽기 전용 선택 기능으로 검수·승인 기록을 자동 생성하지 않습니다. 기존 문구·목소리 조합별 검수의 원본 목록·답변·승인 기록은 변경하지 않습니다. 공개 자료의 대표 검수 기준은 그대로 유지합니다.

공개 자료는 위 비호출 예시와 같은 출처·승인 필드를 사용하고, **`text`에 실제 유사 표현**, **`speech: true`**, **`split: "train"`**을 추가합니다. 합성 자료는 `provenance: "qwen_tts"`, `human_reference_used: false`, 학습용 `voice_family_id`를 기록하고, 출처·생성 근거와 내용 검수 결과를 보존합니다. 합성 음색도 개발용·최종 평가용 계열과 분리합니다.

공개 또는 합성 유사어의 대표 검수 목록을 `koww review --kind content`로 확인한 뒤, 반환된 승인 기록을 해당 원본의 `admission`에 연결합니다. 새 유사어 생성 방식은 대표 발음을 먼저 검수하고 전체 음원을 다시 STT하지 않습니다. `init`은 빈 목록만 만들며 유사어를 자동 수집·생성·승인하지 않습니다.

긴 원본은 `koww windows --pool hard_negative`로 고정 2초 목록을 준비하고, `manifests.hard_negative`를 그 목록으로 바꿉니다. 이어서 `koww features --role hard_negative`로 다른 비호출과 같은 계열의 증강을 적용합니다. 준비된 음성이 비어 있으면 학습을 시작할 수 없습니다.

배치의 유사어 **100개는 매회 추출량**이며 전체 원본 수가 아닙니다. 원본 수·표현별 분포·누적 추출량을 따로 기록하고, 일반·한국어 비호출과 중복되는 원본 시간은 다시 더하지 않습니다.

## 평가 자료 목록

`prepare-eval`이 만드는 `manifest.jsonl`은 소음 조건별 호출어 평가 목록입니다. 여기에 연속 비호출 평가 목록을 합칩니다.

새 평가에는 `--noise-selection balanced_sources`로 소음의 세부 종류와 원본을 균형 있게 선택하세요. 같은 원본을 여러 구간으로 복제해도 선택 비중을 늘리지 않습니다. 선택한 배경과 구간은 SNR별로 고정하며, 실제 사용 분포는 같은 폴더의 `noise_coverage.json`에 기록합니다. 웹 화면은 이 정책을 사용합니다. 옵션을 생략한 CLI/API의 `legacy` 정책은 과거 선택 순서를 재현하기 위한 호환 경로입니다.

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

공통 평가의 원본 비호출에는 `eval_bucket`으로 `korean_conversation`, `korean_interview`, `korean_read`, `other_speech`, `music`, `environment`, `electronic`을 구분합니다. `canonical_recording_id`와 원본 구간 `source_start_s`·`source_end_s`를 기록해 중복 없이 시간을 계산합니다. 증강 횟수나 모델 수를 곱한 시간은 자료 확보량이 아닙니다.

### 평가용 잔향 자료

평가 잔향은 `rir_validation` 또는 `rir_final` 목록에 연결하며, 비호출 음원 목록에 넣지 않습니다. 웹 화면은 공개 자료를 다음 형식으로 준비합니다.

| 필드 | 의미 |
| --- | --- |
| `category`, `split` | `rir`, `validation` 또는 `final` |
| `path`, `sha256` | 실제 모노 16kHz RIR 파일과 해시 |
| `canonical_source_id`, `canonical_room_id` | 개별 응답과 제공자 공간 식별자 |
| `reverb_group` | `moderate` 또는 `strong` |
| `rt60_s`, `reverb_group_basis` | 잔향 시간과 분류 근거 |
| `acoustic_source_type` | `simulated` 등 실제 출처의 종류 |
| `provenance`, `source_url`, `license` | 공개 자료 출처와 이용 조건 |

생성된 시험 WAV에는 `reverb_condition: none/moderate/strong`과 사용한 RIR 해시·공간 식별자가 남습니다. 소음 점수 하락은 같은 울림 조건의 Clean과 비교하고, 잔향 없는 Clean과의 차이도 별도 기록합니다. 시뮬레이션 공간을 실제 측정 공간으로 표시하지 않습니다.

호출어 평가 항목은 `kind: "positive"`이며 `calls`에 실제 호출 시작·끝(`start_s`, `end_s`)과 원본 발화 식별자(`original_utterance_id`)가 있어야 합니다. 자동 생성한 목록의 호출 경계를 임의로 바꾸지 마세요.

다음 Python 코드를 `combine_development.py`로 저장해 실행하면 두 목록을 합칩니다. 비호출 목록을 먼저 준비하세요. 이미 내용이 있는 출력 파일은 덮어쓰지 않으며, 큰 목록도 한 줄씩 처리합니다.

```python
from pathlib import Path

positive = Path("models/bora/data/eval/paired_v1/manifest.jsonl")
negative = Path("models/bora/data/eval/development_negative.jsonl")
output = Path("models/bora/data/eval/development.jsonl")

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
