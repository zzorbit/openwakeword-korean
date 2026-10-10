# 춘식아 · 라이브러리 예시 모델

최종 선택·동일 seed 재학습·CPU ONNX 재현을 완료한 **CNN128 × 3 / R5 / 점수 평균** 묶음이다. 세 ONNX는 기존 선택 모델과 바이트가 동일하며 공통 frontend는 한 번 실행한다. 다른 호출어를 학습할 때 이 가중치를 결과로 복사하지 않고 새로운 판별기를 학습한다.

## 학습과 실행

- Qwen 호출어 단독 원본 100,000개, Positive R5 500,000 feature 행.
- Batch Generic 768 / 한국어 Negative 256 / Positive 50 / Hard 0.
- AdamW, LR 0.0001, decay 0.01, Positive loss weight 1, dropout 0.1, cosine 2% warmup.
- 각 seed 50,000 update. seed 20260928 / 20260929 / 20260930.
- 공유 openWakeWord 0.6.0 ONNX frontend, mono 16kHz PCM16, 80ms 입력.
- 정해진 순서로 float64 합산 → 평균 → float32. Threshold 0.9862500429153442, 연속 2프레임, warmup 2초, cooldown 1초, VAD/NS off.

## 확인된 성능의 범위

고정 **Development** 자료의 독립 seed 묶음 3회 결과다. 같은 시험 자료를 반복 사용했으므로 평가 시간을 합산하지 않는다.

| 지표 | 관측 결과 |
|---|---|
| Clean | 각 묶음 5,000/5,000, 100% |
| 소음 평균 | 세 범주 × SNR20/15/10/5 균등 평균 99.1111% |
| 묶음별 소음 평균 | 99.1833% / 98.9833% / 99.1667% |
| 오호출 | 동일 24.0329h에서 2 / 2 / 1회, 최악 0.0832/h |
| 한국어 competing speech SNR5 | 평균 91.1333%, 최저 90.2% |

이 모델은 **독립 Final 100시간과 실제 다수 화자 검증을 완료한 배포 보증 모델이 아니다.** 다른 호출어·마이크·장치에서 이 성능을 보장하지 않는다. 동봉 오디오는 위 Development의 소규모 재현 예제이며 전체 통계의 분모와 구분한다.

## 파일과 출처

`manifest.json`의 모든 경로는 이 폴더 기준 상대 경로이고, 각 ONNX SHA256을 검증한다. `audio_manifest.json`은 동봉 Development WAV의 원본 ID·호출 경계·해시를 기록한다. 과거 전체 실험 기록은 개발 작업 폴더에 보존한다. 이 저장소는 표본 모델과 재현용 입력을 제공한다.

openWakeWord 코드와 공유 embedding backbone의 상위 출처는 [공식 저장소](https://github.com/dscripka/openWakeWord)다. upstream 코드는 Apache-2.0이며 공식 배포 pretrained 모델에는 별도 모델 이용조건이 있다. Qwen·Generic feature·공개 한국어/소음 자료도 각각의 출처 조건을 따른다. 이 예시 묶음에 일괄적인 상업 이용 허가를 부여하지 않는다. `THIRD_PARTY_NOTICES.md`를 확인한다. GitHub 비공개 저장소에 제공하며 PyPI에 게시하지 않는다.
