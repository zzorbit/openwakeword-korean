# 다른 한국어 호출어를 학습하는 방법

전체 기준은 [plan.md](../plan.md), 시작값은 [default_profile.json](../korean_openwakeword/resources/default_profile.json)이다. 아래 예시는 `보라야`다. 명령은 저장소 루트에서 실행한다. `.venv`를 활성화하지 않았으면 `koww` 대신 `.venv\Scripts\python.exe -m korean_openwakeword`를 사용한다.

## 프로젝트·자원

HDD 작업 폴더에서 `. .\scripts\Use-ProjectStorage.ps1`을 먼저 실행하면 pip·모델 다운로드·임시 파일도 그 폴더에 저장한다. 새 `koww` API의 기본 캐시는 각 프로젝트의 `cache`이며 명시적 공유 HDD 캐시도 지원한다. `koww storage --project projects/bora/project.json`으로 위치를 확인한다. [저장 정책](STORAGE_POLICY.md)을 따른다. `.codex`에 대용량 자료를 저장하지 않는다.

```powershell
koww init --wakeword 보라야 --id bora --output projects/bora --interactive
koww doctor
koww prepare-frontend --project projects/bora/project.json
koww benchmark --output projects/bora/reports/resources.json
```

**처음 선택할 것은 작업량 세 가지**다. 증강 전 TTS 원본 수 N, 원본당 증강 횟수 R1/R3/R5, 모델당 optimizer update 수 U를 선택한다. 기존 제안은 100,000 / R5 / 50,000이며 강제값이 아니다. Enter로 유지하거나 원본·update를 직접 입력한다. CNN128·AdamW·학습률·배분은 그대로 유지하고 세부 설정을 열 때만 바꾼다. 실제 N×R 수, Positive feature payload 용량, 모델 수와 U×모델 수 총량을 `reports/setup_summary.json`에 저장한다. 시간은 그 컴퓨터에서 측정하기 전 확정하지 않는다.

```powershell
koww init --wakeword 보라야 --id bora --output projects/bora_small --non-interactive --originals 20000 --rounds 3 --updates 20000
koww init --wakeword 다솜아 --id dasom --output projects/dasom --interactive --advanced
koww settings --project projects/bora_small/project.json
koww settings --project projects/bora_small/project.json --advanced
```

터미널의 기본 `init`은 선택 화면을 연다. 파이프/비대화형 입력은 질문 없이 값과 기본값을 저장한다. 선택 화면을 강제로 열려면 `--interactive`, 자동화는 `--non-interactive`다. `--advanced`는 대화형 세부 화면이며 자동화의 세부 변경은 `--architecture cnn64 --models 1 --optimizer AdamW --learning-rate 0.0001`처럼 명시한다. 지원 CNN 폭·MLP 구조, optimizer·dropout·WD·Positive weight·schedule·device 등을 변경할 수 있다. 상세 seed·backend·배분은 `project.json`에 있으며 지원 밖 배분/VAD/NS/층 구조는 별도 구현과 검증이 필요하다.

U를 바꾸면 기본 horizon=U / warmup=2% U / checkpoint=약20%·40%·100% U가 된다. 같은 horizon에서 학습량만 비교하려면 `--schedule-horizon 50000` 등을 명시한다. U는 epoch나 feature 수가 아니라 **모델 한 개의 optimizer 갱신 수**다. 기본 세 모델은 같은 횟수로 각각 독립 학습한다. 예를 들어 N20k/R3/U20k는 60k Positive feature, 세 모델 합계60k update이고 feature payload 약0.343GiB다. WAV·Negative·journal·checkpoint 비용은 별도다. 배치당 Positive50·Negative1024 복원 추출이므로 원본/feature가 모두 똑같이 노출되는 것은 아니다.

기존 설정 변경은 원본 옆의 **새 JSON snapshot**에 저장한다:

```powershell
koww configure --project projects/bora/project.json --output projects/bora/project_R1.json --non-interactive --rounds 1
koww configure --project projects/bora/project.json --output projects/bora/project_20k_updates.json --non-interactive --updates 20000 --schedule-horizon 50000
koww configure --project projects/bora/project.json --output projects/bora/project_custom.json --interactive --advanced
```

같은 이름을 덮어쓰거나 다른 폴더로 상대 경로를 바꾸는 것은 거부한다. N/R 변경은 별도의 Positive cache 경로로 연결하고 U/모델 수가 달라진 기본 run 경로도 구분한다. 추가 세부 변경은 명시적 `--output` run을 쓴다. 이미 실행 중인 작업·기존 모델·receipt를 고치지 않는다. Final 동결 후 변경은 새 프로젝트에서 한다. 작은 작업량도 검수·출처·split·Negative 요구를 통과하면 정식 실험할 수 있으나 성능 통과·Final 완료로 자동 간주하지 않는다.

모든 기본 데이터 경로는 `project.json` 기준 상대 경로다. 재사용하는 대규모 불변 자료의 외부 절대 경로도 허용하되 모델 묶음·결과 사이트에는 상대 경로로 복사한다. 예시 모델을 새 단어의 모델로 바꾸어 쓰지 않는다. 학습 중 config를 수정하지 않는다.

`benchmark`는 고정 CNN128 batch의 CPU/GPU 처리량과 초기 logits 차이를 측정한다. 전체 디스크 IO·전송·학습 결과 동등성을 증명하는 시험은 아니다. 큰 작업은 대표 실제 입력의 feature qualification·cache 적재 시간·VRAM/RAM·초기 실제 update throughput도 확인한다. CUDA 학습과 CPU 추론은 별도 기준 경로다.

## 합성 참조 → 새 호출어 파일럿 → 원본 생성

`voices.json`에 Train100 / Validation10 / Final10개의 **후보** 계열을 미리 분리한다. 서로 다른 ID만으로 음향 다양성이 입증되지는 않는다. 생성 스타일·속도·억양의 실제 결과를 확인한다. Qwen 모델 revision은 초기 프로파일에 고정되어 있다. 모델을 바꾸면 별도 정책을 검증한다.

이미 이 라이브러리로 승인한 합성 뱅크가 있으면 같은 split의 참조만 재사용한다. 이전 프로젝트의 해시와 청취 evidence가 필요하고 사람 참조는 거부한다.

```powershell
koww import-references --project projects/bora/project.json --source-project projects/previous/project.json --split train
koww import-references --project projects/bora/project.json --source-project projects/previous/project.json --split validation
```

새 뱅크가 필요하면:

```powershell
koww references --project projects/bora/project.json --split train
koww review --project projects/bora/project.json --queue projects/bora/data/references/train/references.jsonl --kind references --port 8790
```

검수 서버는 현재 프로세스에서 실행한다. 브라우저에서 `http://127.0.0.1:8790/`를 연다. 생성된 전체 큐를 연속으로 듣고 문제 번호만 폐기/재검수한다. 끝까지 재생된 항목만 자동 보존되며 **검수 완료·저장**이 서버 파일에 직접 반영된다. 큐를 합쳐 제공하려면 JSONL들을 연결한 별도 큐를 만들 수 있고 각 WAV path는 프로젝트 기준을 유지한다. 이미 만든 큐를 임의 세션으로 잘게 나누지 않는다. 서버 종료 후 같은 명령으로 다시 열 수 있다.

Validation 참조도 같은 방식으로 별도 split에서 생성·승인한다. Final 참조는 후보 metadata만 남기고 동결 전 생성·청취하지 않는다. 새 호출어 파일럿:

```powershell
koww generate --project projects/bora/project.json --pilot --limit 24
koww review --project projects/bora/project.json --queue projects/bora/data/pilot/train/candidates.jsonl --kind pilot --port 8790
```

clone의 발성은 승인한 참조를 따른다. `--method design`은 지원되는 다른 생성 방식이며 pilot과 production에서 같은 방법을 써야 한다. 파일럿 예외가 있으면 자동으로 기존 정책을 승인하지 않는다. 생성 방식/참조/Qwen 설정을 수정할 경우 새 프로젝트 또는 버전 output을 만들고 해당 변경만 다시 확인한다.

승인 후:

```powershell
koww generate --project projects/bora/project.json --split train
koww generate --project projects/bora/project.json --split validation
```

`--limit`는 추가 수량이 아니라 해당 실행의 총 slot 목표다. 같은 명령 재실행은 계약이 같을 때 성공 slot을 재사용한다. 슬롯당 최대 3회 기술 QC 재시도 후 문제를 남긴다. `summary.json`의 `full_target_reached`, 시도·기술 통과·정책 승인 수량을 확인한다. 무음/깨짐/길이/중복/전체 호출 창 적합성을 검사하며 독립 STT는 실행하지 않는다. 생성 정책 승인은 개별 100k의 사람 청취 완료가 아니다. 생성 prompt와 reference는 호출어 뒤 명령을 붙이지 않는다.

## 공개 오디오 manifest와 내용 승인

먼저 [데이터 다운로드 안내](DATASETS.md)의 공개 자료를 내려받고 출처·라이선스·분할을 기록한다. 공개 한국어 대화·방송/보컬 자료는 한국어 Negative에 우선 넣고, 환경음은 noise 재료로 관리한다. 프로젝트가 원시 데이터 전체를 함께 배포하지 않는다. 이미 보유한 공개 자료는 복사하지 않아도 된다.

JSONL은 **한 줄에 하나의 JSON object**다. Negative 예시:

```json
{"sample_id":"ko_0001","source_id":"corpus:recording_0001","canonical_source_id":"corpus:recording_0001","canonical_speaker_id":"corpus:speaker_001","split":"train","path":"data/negative/source_0001.wav","sha256":"실제_파일_SHA256","provenance":"public_dataset","source_url":"https://원본_출처","license":"원본_이용조건","category":"competing_speech","language":"Korean","speech":true,"admission":{"path":"review/evidence/content_admission_실제ID.json","sha256":"실제_승인파일_SHA256"}}
```

noise에도 같은 출처 필드를 쓰고 category는 `environment`, `music`, `competing_speech`다. RIR은 `rir`, 필요하면 `canonical_room_id`를 추가한다. noise/RIR은 훈련 split만 train material manifest에 넣는다. 별도의 `noise_validation` / `noise_final` manifest에는 각 held-out split을 사용한다. 한 원본을 잘라 다른 split에 넣거나 SHA만 다르다고 별개 원본으로 취급하지 않는다.

대표 내용 검수·공개 metadata를 바탕으로 다음 형태의 승인 근거가 필요하다:

```json
{"state":"accepted","content_review":"accepted","accepted_source_sha256":["검증된_원본_SHA256"],"evidence":"검수/출처_근거","scope":"공개 metadata와 대표 내용 검사; 전수 전사 아님"}
```

실제 승인 전에 이 예시를 accepted 결과로 복사하지 않는다. `koww review --kind content --queue ...`도 지원한다. 대표 preview WAV를 검수하면 queue row의 `reviewed_source:{path,sha256}`로 원본을 묶는다. 원본 전체를 다시 STT하거나 모든 window를 사람이 듣지 않는다. 서버가 반환하는 `admission`의 **불변 evidence 경로와 해시**를 해당 원본 manifest에 넣는다. `review/content_admission.json`은 최신 화면 요약이며 다음 검수에서 바뀔 수 있으므로 새 manifest는 여기에 묶지 않는다. 폐기한 원본은 학습 manifest에서 제거한다. Positive 생성 정책도 불변 snapshot을 사용하므로 Validation/Final 참조 승인으로 이전 학습 자료가 무효화되지 않는다.

```powershell
koww audit --project projects/bora/project.json
koww windows --project projects/bora/project.json --pool korean_negative --output projects/bora/data/negative/korean_windows.jsonl
```

생성된 window manifest로 `project.json`의 `manifests.korean_negative`를 바꾼다. 파일은 그대로 두고 구간만 지정한다. Generic 파형을 쓰면 동일 방법으로 고정 window를 만든다. 순수 환경음 Negative는 speech=false로 기록하고 발화 SNR을 계산하지 않는다.

공개 precomputed Generic을 사용할 때:

```powershell
koww download-generic --output shared/openwakeword_features_ACAV100M_2000_hrs_16bit.npy
koww import-generic --project projects/bora/project.json --features shared/openwakeword_features_ACAV100M_2000_hrs_16bit.npy --output projects/bora/features/generic_negative
```

약 17.28GB 다운로드와 전체 hash/유한값 검사는 시간이 든다. 기존 검증 파일을 재사용할 수 있다. 명목 2,000h·5,625,000행이고 원본 ID가 없어 모든 중복·분할을 입증하지 못한다. 이 한계는 metadata에 남긴다. 파형 Generic을 직접 준비하면 제공자 bridge 없이 동일 frontend feature를 사용한다.

## 선택 작업량의 Feature·학습

선택한 N·R·U 그대로 진행하는 명령이다. cache 경로와 수량은 초기 설정에서 가져오므로 파일 이름을 직접 맞출 필요가 없다.

```powershell
koww features --project projects/bora/project.json --role korean_negative
koww features --project projects/bora/project.json --role positive
koww train --project projects/bora/project.json --output projects/bora/runs/selected_budget
koww package --project projects/bora/project.json --run projects/bora/runs/selected_budget --output projects/bora/models/candidate_selected
```

### R 효과를 따로 비교할 때

이전 참조 100k의 cache 경로를 직접 지정하는 예시는 다음과 같다. 모든 R은 같은 N에서 생성한다. 선택 R만 실행하는 것과 R 효과를 통제 비교한 것은 구분한다.

```powershell
koww features --project projects/bora/project.json --role korean_negative
koww features --project projects/bora/project.json --role positive --rounds 1 --output projects/bora/features/positive_R1
koww features --project projects/bora/project.json --role positive --rounds 3 --output projects/bora/features/positive_R3
koww compare-rounds projects/bora/features/positive_R1 projects/bora/features/positive_R3
koww features --project projects/bora/project.json --role positive --rounds 5 --output projects/bora/features/positive_R5
koww verify-cache projects/bora/features/positive_R5
```

파형 Generic이면 `features --role generic_negative`도 실행한다. Positive와 speech Negative는 공통 noise/RIR/gain 분포를 사용한다. SNR은 호출 활성 구간을 기준으로 하고, 같은 원본의 round별 RNG를 달리한다. 실제 생성 수·PCM·행 수를 검증하며 R별 독립 cache를 사용한다. 환경소음은 추가 batch class가 아니다.

위 비교 명령의 Positive output은 직접 지정한 cache 경로다. `--output`을 생략하면 선택한 `pool_features.positive`를 사용하고, 선언한 R과 다른 `--rounds`는 `features/positive_N{N}_R{R}`에 따로 만든다. `configure --rounds ...`의 snapshot이 가리키는 cache와 실제 생성 경로를 일치시킨다. 직접 지정한 경로를 쓰면 학습 시작 전 snapshot의 `pool_features.positive`를 그 경로로 지정한다. R1/R3 각각 config의 snapshot을 별도 파일로 남기고 **같은 선택 N·Negative·배분·update·schedule·seed·noise 분포를 고정**한다. 공통 cache를 읽는 것은 허용하지만 원본을 수정하지 않는다. 직접 수정 없이 snapshot으로 비교 cache를 만들려면:

```powershell
koww configure --project projects/bora/project.json --output projects/bora/project_R1.json --non-interactive --rounds 1
koww configure --project projects/bora/project.json --output projects/bora/project_R3.json --non-interactive --rounds 3
koww features --project projects/bora/project_R1.json --role positive
koww features --project projects/bora/project_R3.json --role positive
```

정식 학습은 승인된 cache·선택 N과 실제 N×R 일치·Generic 2,000h·한국어 200h 등의 선행 규모를 확인한다. 100k는 이전 참조 작업량이며 작은 N의 R3/R5를 일괄 금지하지 않는다. 작은 예산의 정식 실험과 미승인 fixture의 기능 시험은 다르다. 기능 시험에는 `features --smoke`와 `train --smoke`를 쓰며 성능 또는 Final 근거로 승격하지 않는다. 학습 중 checkpoint의 optimizer/sampler/RNG와 실제 누적 draw를 저장한다. receipt는 선택 작업량과 실제 노출량을 함께 기록한다. 이미 완료된 output은 덮어쓰지 않는다. 중단 후 같은 config/코드/입력으로 `--resume`을 사용한다. `.writer.lock`이 남았으면 PID·생성 시각·실제 명령을 확인한 후 죽은 작업의 lock만 정리한다.

## Development·임계값·Final

```powershell
koww prepare-eval --project projects/bora/project.json --output projects/bora/data/eval/paired_v1 --split validation --originals 100 --noise-sections 5
```

같은 원본·동일 소음 구간으로 Clean와 SNR20/15/10/5/0의 실제 WAV를 만든다. 레벨·peak headroom·SNR을 기록하고 말소리가 너무 작아지면 오류로 남긴다. score와 청취에 같은 WAV를 사용한다. 이 결과 manifest의 Positive와 **source-separated intact 연속 Negative 20~30h**를 하나의 Development JSONL로 합친다. 100원본은 파일럿 크기이며 최종 비교에는 계획의 Validation 규모를 적용한다.

평가 row는 `kind`, `category`, `split`, 실제 path/SHA, provenance가 필요하다. Positive의 `calls`는 호출별 `start_s`,`end_s`,`original_utterance_id`를 포함한다. 첫 호출은 2초 warmup 뒤에 와야 한다. Negative에는 `calls:[]`. 임의 반복·이어붙인 파일을 독립 시간으로 세지 않는다.

```powershell
koww evaluate --project projects/bora/project.json --bundle projects/bora/models/candidate_selected --manifest projects/bora/data/eval/development.jsonl --output projects/bora/reports/dev_selected
koww tune --project projects/bora/project.json --bundle projects/bora/models/candidate_selected --evaluation projects/bora/reports/dev_selected --output projects/bora/reports/gates_selected --thresholds 0.5 0.7 0.85 0.9 0.95 0.98 0.9862500429153442 0.995 --consecutive 1 2 3
koww select-runtime --project projects/bora/project.json --bundle projects/bora/reports/gates_selected/bundle
koww site --evaluation projects/bora/reports/gates_selected/evaluation --output projects/bora/reports/gates_selected/site
```

`tune`는 동일 hash-bound scores를 재생하며 신경망을 다시 계산하지 않는다. 선언한 grid 안에서 목표 통과 → Clean → balanced noise → FA 순으로 후보를 고른다. 미달이면 `adoption.passed=false`와 이유를 남기며 Final 동결을 허용하지 않는다. 기존 예시의 임계값을 새 호출어에 그대로 확정하지 않는다.

```powershell
koww freeze --project projects/bora/project.json --bundle projects/bora/reports/gates_selected/bundle --development-report projects/bora/reports/gates_selected/evaluation/report.json
koww references --project projects/bora/project.json --split final
koww generate --project projects/bora/project.json --split final
```

Final 참조의 승인 또는 승인 뱅크 재사용은 동결 후 같은 참조 provenance 규칙을 따른다. Final을 Train/Validation에 다시 넣지 않는다. Final Positive·noise·100h 이상 연속 Negative manifest를 준비한 뒤:

```powershell
koww evaluate --project projects/bora/project.json --bundle projects/bora/reports/gates_selected/bundle --stage final --manifest projects/bora/data/eval/final.jsonl --output projects/bora/reports/final_v1
koww site --evaluation projects/bora/reports/final_v1 --output projects/bora/reports/final_v1/site
koww serve --directory projects/bora/reports/final_v1/site --port 8791
```

실제 다수 화자 Final은 별도 provenance `human` 및 화자·거리·방향·마이크 metadata로 집계한다. 개인 마이크 시험을 중단한 현재 프로젝트에서는 이를 합성 결과로 대체해 완료라고 표시하지 않는다. 공개 사람 음성도 학습·튜닝·TTS 참조로 자동 전환하지 않는다.

## 출력과 Python API

`features`는 feature 배열·증강 journal·계약·metadata, `train`은 checkpoint·로그·완료 receipt·ONNX, `package`는 상대 경로 모델 묶음, `evaluate`는 report·clips·score cache·receipt를 남긴다. 결과 사이트는 단일 모델 성능순·앙상블 별도·고정 SNR 연속 청취·동일 WAV 다운로드를 제공한다.

```python
from korean_openwakeword import create_project, WakeWordDetector
from korean_openwakeword.features import build_features
from korean_openwakeword.train import train
from korean_openwakeword.evaluate import evaluate
from korean_openwakeword.site import export_site

project = create_project("projects/newword", "다솜아", "dasom",
                         positive_originals=20000, augmentation_rounds=3, updates=20000)
# 출처·파일럿·원본·재료 승인 이후 build_features(project, ...), train(project, ...)
detector = WakeWordDetector("path/to/completed/bundle")
result = detector.predict_file("mono_16k_pcm16.wav")
```

함수와 CLI는 같은 구현을 사용한다. baseline의 batch 변경, VAD/NS, 지원 외 구조는 명시적 별도 실험으로 구현·검증해야 한다. 모델/음원 이용조건은 [THIRD_PARTY_NOTICES](../THIRD_PARTY_NOTICES.md)를 따른다.
