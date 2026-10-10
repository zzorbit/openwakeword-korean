# 한국어 openWakeWord 학습 라이브러리 · 통합 계획

갱신: 2026-10-11. 앞으로 다른 한국어 호출어를 학습할 때 적용할 기준 문서다. 과거 실험과 config·receipt·모델은 개발 작업 폴더에 보존하고 게시용 라이브러리와 분리한다.

## 목표와 현재 결과

**불특정 다수의 목소리로 부르는 한국어 호출어를 다양한 소음에서도 안정적으로 인식하고 오호출은 적은 openWakeWord 모델을 만든다.** 재사용 가능한 코드·기본값·데이터 계약·평가·결과 사이트를 함께 제공한다.

패키지는 `korean_openwakeword`, 명령은 `koww`다. 호출어·경로·설정은 프로젝트별 `project.json`에서 지정한다. 기존 `wakeword_lab`과 날짜별 scripts는 이력으로 보존하며 새 패키지의 실행 의존성에서 분리한다. 설치는 [README](README.md), 전체 명령과 데이터 작성은 [라이브러리 가이드](docs/LIBRARY_GUIDE.md)를 따른다.

표본은 **춘식아 CNN128 × 3 / R5 / 점수 평균**이다. 선택 후 동일 seed 재학습·CPU ONNX 재현을 확인한 기존 가중치를 그대로 동봉한다. 고정 Development에서 세 반복 묶음 모두 Clean 100%, 소음 균등 평균 99.1111%, 최악 관측 오호출 0.0832/h였다. **독립 Final 100h·실제 다수 화자 보증은 아니다.** [모델 카드](korean_openwakeword/resources/sample_chunsik/MODEL_CARD.md)와 [라이브러리 검수 범위](docs/VALIDATION.md)를 구분해 읽는다. 작은 동봉 demo의 분모는 전체 Development와 다르다.

## 1. 일곱 영역의 기본값과 비교 원칙

실행 기준은 [default_profile.json](korean_openwakeword/resources/default_profile.json)이다. **초기 선택은 작업량, 세부 설정은 학습 방법**으로 구분한다. 예시 가중치·임계값을 새 호출어의 검증 결과로 간주하지 않는다. CNN128 등의 권장값은 이번 춘식아 고정 Development에서 확인한 안정적인 시작값이며, 모든 한국어 호출어의 최적 설정이 입증된 것은 아니다.

### 초기 선택과 세부 설정

`koww init`을 터미널에서 실행하면 먼저 다음 세 항목을 선택한다. Enter는 기존 작업량 제안을 사용하고 각 항목을 독립적으로 바꿀 수 있다. 자동화에서는 `--originals`, `--rounds`, `--updates`와 `--non-interactive`로 지정한다. 입력이 터미널이 아니면 질문을 띄우지 않는다. Python API도 같은 선택을 지원한다.

| 처음 선택할 항목 | 선택 방법 | 기존 작업량 제안 |
|---|---|---|
| Positive TTS 원본 수 N | 양의 정수; 20k/50k/100k 또는 직접 입력 | 100,000개 |
| 원본당 증강 횟수 R | R1 / R3 / R5 | R5 |
| 모델당 학습량 U | optimizer update 수; 10k/20k/50k 또는 직접 입력 | 50,000회 |

설정 직후 실제 `N×R` 행 수·Positive float32 feature 예상 용량·모델 수·총 `U×모델 수` 학습량·학습률 horizon·저장 지점을 보여주고 `reports/setup_summary.json`에 남긴다. WAV·Negative·journal·checkpoint 용량은 이 feature 용량에 포함하지 않는다. 정확한 시간은 해당 컴퓨터의 실측 처리량을 확보한 뒤 추정한다. 자원을 자동 감지했다고 작업량이나 성능을 자동 확정하지 않는다.

예를 들어 **20,000개 / R3 / 모델당 20,000회**는 Positive 60,000행, 기본 세 모델 합계 60,000 update다. 같은 원본·같은 계산 조건으로 장치만 바꾸는 CPU/GPU 선택과 달리, N·R·U를 줄이면 학습 결과가 달라질 수 있다. 작은 예산도 승인된 자료로 정식 실험할 수 있지만 품질 달성은 평가 후 판단한다. 데이터 승인·Negative 규모·Validation/Final 분리·성능 목표를 함께 낮추지 않는다.

CNN128·AdamW·학습률·dropout·배분·증강 분포 등은 기본값을 유지한다. `--advanced` 선택 화면 또는 세부 CLI 인자로 지원 구조·optimizer·학습률·정규화·schedule·모델 1/3개·CPU/GPU를 바꿀 수 있다. 지원 범위 밖 구조·층 구성은 별도 구현·검증이 필요하다. `koww settings --project ... --advanced`로 전체 값을 확인한다. 세 모델을 한 모델로 줄이는 것도 결과와 실행 비용에 영향을 줄 수 있다.

기본 작업량 선택에서 U를 바꾸면 **horizon=U, warmup=2% U, 저장=약 20%/40%/100% U**를 함께 변경한다. 학습량 비교에서 같은 horizon을 유지하려면 세부 설정 `--schedule-horizon`을 명시한다. 기존 설정 변경은 `koww configure --project ... --output 같은_폴더의_새.json`으로 별도 snapshot에 저장한다. 기존 config·cache·모델·receipt를 덮어쓰지 않으며 Final 동결 이후에는 새 프로젝트를 쓴다.

| 영역 | 고정할 시작값 | 부족한 경우 비교 |
|---|---|---|
| 모델 구조·크기 | CNN128, Conv1D 3층, 16×96 입력, 독립 seed 3개 | CNN64/128, MLP32/128/256. 종류·폭의 효과 분리 |
| 데이터·증강 | 호출어 단독 TTS N·R을 초기 선택; 100k/R5 제안. 공개 Negative·소음·RIR | N 효과와 R1/R3/R5 효과를 별도 비교 |
| 배치·샘플링 | Generic 768 / 한국어 256 / Positive 50 / Hard 0, pool별 독립 RNG 균등 복원 추출 | recall·FA 문제를 근거로 별도 실험 |
| 학습 방법 | AdamW, LR 0.0001, WD 0.01, Positive BCE weight 1, dropout 0.1, cosine + 2% warmup | LR·optimizer·가중치·dropout·WD·schedule 중 한 축씩 |
| 학습량·선택 | U를 초기 선택; 50k 제안, horizon·warmup·checkpoint 연동 | 같은 horizon의 예산 비교; Development로 선택 |
| 평가·감지 | threshold 0.9862500429153442, 연속 2프레임, VAD/NS off | 새 호출어 Development에서 threshold·프레임; VAD/NS는 별도 검증 |
| 실행·재현 | 검증된 CUDA float32 학습, CPU ONNX 공유 frontend 추론 | 장치·정밀도·backend의 속도와 결과를 따로 검증 |

seed는 20260928 / 20260929 / 20260930. 세 점수를 순서대로 float64 합산 → 평균 → float32 변환한다. 최댓값 결합과 다르며 frontend는 한 번 계산한다. 전체 batch는 **1,074**, Negative 합계는 **1,024**다.

backend: cuDNN deterministic on / benchmark off, CUDA matmul TF32 off / cuDNN TF32 on, MHA fastpath off. 저장된 float16 feature의 float32 변환과 AMP 학습은 다르다. AMP·GPU 추론을 자동 도입하지 않는다. R5는 이번 데이터의 선택값이며 보편적인 단조 성능 향상을 뜻하지 않는다.

기존 실험·검수·cache를 재사용하고 필요한 축만 보완한다. 안정 API에 포함하지 않은 GRU/LSTM/CRNN/Transformer/VAD/NS 과거 실험을 지원 기능처럼 표시하지 않는다. Batch 변경과 Targeted Hard Negative도 기본 실행에 몰래 추가하지 않는다.

## 2. 데이터 규모와 역할

새 호출어 Positive는 사용자가 Qwen TTS로 생성한다. 비호출·환경소음·RIR은 [공개 자료 다운로드 안내](docs/DATASETS.md)의 고정 출처에서 받고 기존 검증 자료를 재사용한다. 다운로드 완료와 내용 승인·학습 사용 판정을 구분한다. 승인 파일은 불변 snapshot에 묶고 최신 UI 요약을 후속 검수에서 바꾸어도 기존 입력 해시는 보존한다. 이미 만들어진 참조의 부분 생성은 같은 전체 뱅크 계약으로 이어서 확대한다.

| Pool | 목표 | 역할 |
|---|---|---|
| Positive Train | 초기 선택 N; 참조 작업량 100,000개 | 호출어 단독; 가능하면 분포가 비슷한 20k/50k/100k nested subset |
| Positive Validation | 약 5,000개 | 모델·실행 설정 선택 |
| Positive Synthetic Final | 약 5,000개 | 모델·실행 설정 동결 후 평가 |
| Generic Negative | 약 2,000h 이상 | 일반 음성·여러 언어·음악·환경음 |
| 한국어 특화 Negative | 약 200~300h | 한국어 대화·방송·보컬 중심 |
| 환경소음 재료 | 1차 고다양성 30~50h, 필요 시 100~200h | Positive·파형이 있는 speech Negative 증강; 독립 batch class 아님 |
| RIR | 출처·방·분할을 관리한 공개 자료 | 공간·잔향 조건 |
| Hard Negative | 초기 0개 | 반복 FP가 확인된 표현만 후속 도입 |
| 개발 연속 Negative | 20~30h | FP·실행 설정 선택 |
| 독립 Final 연속 Negative | 100h 이상 | 동결 설정의 오호출 평가 |

한국어 대화 150~250h와 보컬·방송 50~100h는 예시 구성이다. 합계 200~300h에 맞춰 조정하며 상한을 합산해 새 필수량을 만들지 않는다. 환경소음은 팬·에어컨·청소기·차량·도로·비·바람·키보드·문·그릇·충격·생활 기계음 등 다양성을 먼저 확보한다. 음성이 포함된 간섭은 한국어를 우선 샘플링한다.

파일 수를 Positive:Negative 1:1로 맞추지 않는다. 전체 시간·실제 batch 비율·누적 draw·서로 다른 feature/원본의 노출량을 기록한다.

공개 Generic precomputed는 파형 자료와 구분한다. 제공자 명목 2,000h를 검증된 독립 원본 시간으로 표시하지 않는다. 정해진 공개 파일·전체 해시·shape·유한값·frontend 자산을 확인하는 좁은 legacy bridge만 지원한다. 행별 원본 ID·과거 추출기 좌표 동일성·평가 중복을 완전히 입증할 수 없는 한계를 명시한다. 파형이 없는 feature에 noise augmentation을 적용했다고 기록하지 않는다.

## 3. 생성·검수에서 반복하지 않을 실수

1. **Positive는 호출어 하나만** 생성한다. ‘불 켜줘’ 등 뒤따르는 명령을 붙이지 않는다. 감지 후 pre-roll을 STT로 넘기는 제품 기능은 별도다.
2. 실제 사람 호출 녹음은 학습·튜닝·TTS 참조에 사용하지 않는다. 참조 뱅크도 Qwen 합성 음성만 허용한다.
3. 승인한 합성 참조·정책·청취 근거를 재사용한다. 새 호출어 발음이나 변경한 생성 방식의 대표 파일럿만 청취한다. 이미 생성한 큐는 한 번에 제공하고 부담만을 이유로 검수를 잘게 나누지 않는다.
4. **선택한 Positive 전체와 Negative 전체의 독립 STT를 하지 않는다.** QC는 유한값·깨짐·무음·clipping·길이·중복·전체 호출이 feature 창에 들어가는지 확인한다. 자동 음량 경계는 발음·의미 검수가 아니다. 반복적인 의심 사례만 표적 분석한다.
5. Negative는 공개 출처·라이선스·언어·내용·중복·분할·대표 내용 검수를 확인한다. 드문 이름의 혼입 가능성만을 이유로 전수 ASR/VAD 전사를 기본 단계로 만들지 않는다. 공개 전사는 재사용하고 실제 FP를 분석한다.
6. 생성 시도·자동 QC·정책 승인·개별 사람 청취를 구분한다. 듣지 않은 음원을 청취 완료로 기록하지 않는다. 저장된 검수 결과는 직접 읽으며 복사·붙여넣기를 요구하지 않는다.

음색·말투·속도·억양·음량감·짧게/길게 부르기의 분포를 기록한다. 100개 ID가 서로 다른 100명 음색을 보장하지 않는다. 파일럿에서 음향적으로 비슷한 후보의 과다 반복을 검사하고, 20k/50k/100k subset의 분포도 맞춘다. clone 스타일은 실제 참조 발성에 의존하며 지원하지 않는 instruct를 적용했다고 표시하지 않는다. 단독/명령 포함 분포는 **100% / 0%**다.

‘멀리 있는 상대를 부르는 듯한 큰 발성·강한 투사 스타일’은 발성이다. 실제 거리는 RIR·입력 음량·마이크 수음으로 별도 관리한다. 0.5m/1m/3m 등의 수음은 적절한 장치가 있을 때 대표 조건으로 검증한다.

Train / Validation / Final의 합성 음색·참조 계열을 완전히 분리한다. 원본 녹음·화자·RIR 방·noise source·동일 해시의 교차 분할을 막는다. Final 음색은 Train과 Validation에서 금지하고 Final 점수로 설정을 고르지 않는다.

## 4. 실행 순서와 완료 조건

1. **새 프로젝트:** `init`에서 N·R·U 선택 → `doctor` → CPU/GPU probe. 호출어·상대 경로 config와 작업량 요약을 만들고 재사용 자산의 근거를 확인한다. 기존 모델이 있으면 재학습 전 Paired Test. 없으면 미측정으로 기록한다.
2. **파일럿 → 확대:** 참조 다양성과 단독 호출 발음을 확인한다. 문제가 있는 정책만 수정·재검증한 뒤 선택한 N개를 생성한다. N이 적어도 음색·발성 다양성은 유지한다. 실측 throughput·시도·통과·재사용·승인·청취를 구분한다.
3. **Negative·소음 준비:** 출처 분할·내용을 검증하고 고정 2초 window manifest를 만든다. 원본 WAV를 window마다 복사하지 않는다. speech Negative와 Positive에 같은 계열의 noise/RIR/gain을 적용한다. 순수 소음 Negative에는 발화 SNR을 정의하지 않는다.
4. **N×R 검증:** 실제 generator 출력·feature 행·원본별 round/seed/PCM·cache 분리를 확인한다. 100k 기준 R1=100k / R3=300k / R5=500k이며 다른 N도 실제 `N×R`이어야 한다. 정식 feature 생성·학습은 선언 N과 실제 원본/행 수가 맞아야 한다. 공식 `augmentation_rounds` 값만 믿지 않는다. 공통 round는 동일 결과를 재현한다.
5. **통제 학습:** R 효과를 주장하려면 선택한 **같은 N**에서 R1/R3 필수 비교, R5 필요 시. R5 선택으로 새 호출어의 비교가 자동 완료되지 않는다. 단일 작업량 실행과 통제 비교 완료를 구분한다. Negative·batch·update·schedule·seed·소음 분포를 고정한다. 원본 양과 R 효과를 동시에 바꾸지 않는다. 과거 100k 실험의 통제·결과는 보존한다.
6. **Development 선택:** Clean·세 소음 범주·각 SNR·연속 Negative를 평가한다. 프레임 점수 cache로 threshold/프레임을 비교해 신경망 계산을 반복하지 않는다. 목표를 모두 만족한 후보에서 Clean 유지·balanced noise recall·FA·실행 비용을 비교한다. 부족한 축만 보완한다.
7. **동결 → Final:** 모델·실행 설정·config를 해시로 동결한 뒤 Final을 연다. 합성·실제 사람 결과를 분리한다. Final을 보고 개선하면 새 버전과 새 독립 Final이 필요하다.
8. **사이트 제공:** 실제 평가에서 통계·동일 WAV를 내보낸다. 기능 시험을 정식 N0-R1/R3/R5 또는 성능 검증으로 세지 않는다.

checkpoint는 optimizer·sampler·CPU/CUDA/NumPy/Python RNG·update를 함께 저장한다. 변경된 코드·모델·입력·배분·precision을 같은 cache나 checkpoint로 이어 쓰지 않는다. 재개는 마지막 hash-bound commit부터 한다.

## 5. 평가 목표

| 항목 | 기준 |
|---|---|
| Clean | ≥98%, 같은 호출어의 사전 기준 대비 유지 |
| 소음 | 환경 / 음악·전자음 / 한국어 competing speech 각각 SNR10 ≥95%, SNR5 ≥90% |
| Paired | 같은 호출·같은 소음 구간, Clean / 20 / 15 / 10 / 5 / 0dB. 최대 raw score·Clean 대비 하락·감지·지연 |
| FP | 관측 FA ≤0.1/h. 이벤트 수·유효 시간·범주·Poisson 불확실성 함께 보고 |
| 지연 | 호출 끝 기준 p95 ≤500ms; warmup·cooldown·프레임·post-roll 고정 |
| 반복 | seed 변동·조건별 실제 표본·원본/화자 단위 불확실성 |

SNR0은 한계 진단으로 분리한다. 평균으로 약한 조건을 숨기지 않고 같은 Negative의 모델별 평가 시간을 합산하지 않는다. 계속 높은 score가 rising-edge FA만 낮게 보일 수 있으므로 high/qualified 점유 시간도 확인한다.

실제 사람 Final의 원래 목표는 **20~30명 이상 / 총 600~1,000회**, 각 화자 Clean·환경·음악·competing speech 각각 가능한 범위에서 최소 5회다. 대표 거리·방향·마이크를 균형 배분하고 조건별 실제 수량·성공률을 보고한다. 모든 세부 조합을 강제하지 않는다. 현재 사용자는 저품질 마이크로 개인 녹음 시험을 중단했다. 합성 시험으로 대체했다고 실제 다수 화자 검증 완료로 표시하지 않는다. 적절한 자료가 없으면 미검증으로 남긴다.

## 6. 큰 작업의 효율·재현

- 시작 전에 입력 수·bytes·디스크/GPU/RAM 병목·병렬성·cache 재사용·중단 재개를 확인한다. 대표 입력으로 준비/전송/계산/저장 시간을 측정한다. CPU로 오래 돌린 뒤에야 GPU를 검토하는 방식을 반복하지 않는다.
- 검증된 GPU 학습을 기본 후보로 둔다. VRAM pool cache는 원래 dtype·index·float32 batch가 같은지 확인하고 chunk 복사한다. 넘치는 pool은 mmap gather로 처리한다. 실시간 단일 프레임 추론의 GPU 우위를 가정하지 않는다.
- 독립 2초 feature의 초기 불필요 embedding 생략은 실제 입력·edge case의 bitwise 동일성·속도를 통과한 경우만 사용한다. 실패하면 CPU 기준 경로. GPU 신경망의 미세한 차이를 bitwise 동일이라고 발표하지 않는다.
- 공유 frontend·저장 score를 재사용한다. 긴 noise WAV 전체의 반복 decode, feature 행마다 fsync, 반복적인 cache 복사를 피한다. 변경된 입력은 새 namespace에 둔다.
- 외장 F: 대량 읽기·쓰기·랜덤 읽기를 겹치지 않는다. 대용량 원본·feature·모델·평가 WAV·다운로드·임시 파일은 사용자가 선택한 프로젝트 드라이브에 둔다. `.codex`·사용자 기본 C 캐시로 보내지 않는다. SSD는 사용자가 명시적으로 선택한 경우에만 실측 병목·용량 한도·기간·원본 보존·해시·종료 후 복귀/정리를 기록해 사용한다. [저장 정책](docs/STORAGE_POLICY.md)과 `koww storage`로 위치를 확인한다.
- ETA는 최근 처리량과 해당 단계의 남은 수량으로 산정한다. load·다운로드·후속 평가를 구분하며 로그만으로 생존·완료를 단정하지 않는다.

## 7. 결과 사이트·라이브러리 완료

**단일 모델은 좋은 성능 순으로 정렬**하고 앙상블은 별도로 표시한다. 조건별 분모·FP 시간·최대 score·Clean 대비 하락·지연을 제공한다. 고정 SNR에서 원본 → 소음 범주 → 구간 순서로 끝까지 연속 재생하고 특정 번호로 이동할 수 있게 한다. 시간별 score 그래프는 넣지 않는다.

사용자가 듣는 WAV와 채점 WAV는 동일 해시여야 한다. 별도 크게 듣기로 다른 음원을 보여주지 않는다. 레벨을 고치면 새 입력 버전으로 다시 채점한다. 과거 전체 통계를 작은 demo의 결과로 복사하지 않는다. `koww site`는 상대 경로 HTML/JS/CSS/WAV를 출력한다. [사이트 규칙](docs/TRAINING_RESULTS_SITE.md)과 [동봉 예제](examples/library_demo_site/index.html)를 제공한다.

라이브러리 완료 근거는 설치 가능한 wheel·portable archive·다른 호출어 기능 시험·N×R/재개 검증·예시 모델 score/event 재현·사이트 재생이다. 새 호출어의 목표 달성은 그 호출어의 실제 Development/Final로 별도 판단한다. 원시 대규모 데이터·개인 환경·venv·과거 cache는 배포물에서 제외하고 [개별 이용조건](THIRD_PARTY_NOTICES.md)을 보존한다.
