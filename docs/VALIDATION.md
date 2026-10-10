# 라이브러리 검수 범위

이 문서는 설치·실행 품질을 설명하며 새 호출어의 성능 평가를 대신하지 않는다.

## 확인 대상

- 기존 코드 전체의 학습·생성·검수·입력 계약·상태·추론·평가·Final 잠금과 문서 경로 검토.
- Qwen 단독 호출·합성 참조·부분 생성 확대·불변 승인 근거·반복 참조 재사용. 실제 사람 음성 참조와 전수 독립 STT를 허용하지 않음.
- 공개 자료 다운로드의 중단 재개·Range·크기·SHA256·손상 파일 거부. 전체 대용량 원본을 다시 받지 않고 기존 receipt와 작은 실제 음원을 사용.
- N×R·중첩 round 일치·증강 sampling 재사용·Negative 승인 hash 조회·원본/영상/화자/방 split·겹치는 parent 시간.
- 새로운 한국어 호출어 프로젝트의 실제 소형 학습·ONNX export·중단 재개·샘플링/RNG 복원·모델 묶음. 이 기능 시험의 입력은 fixture이며 호출 성공률 입증으로 사용하지 않음.
- 기존 춘식아 세 가중치와 공유 frontend의 SHA256 보존·동일 오디오 실제 추론·결과 화면의 WAV/점수 대응.
- 별도 작업 폴더에서 배포 wheel 설치·CLI·feature·학습·package·추론·결과 사이트 실행. 날짜별 실험 코드와 원래 작업 폴더에 의존하지 않는지 확인.

GitHub Actions는 Python3.12의 Windows/Linux에서 동일 portable 시험과 패키지 build를 실행하도록 구성한다. CUDA 대규모 TTS/100k 재학습·실제 다수 화자·100h 독립 Final을 매 commit마다 실행하는 CI는 아니다. 로컬 과거 구현과의 세 parity 시험은 개발 작업 폴더에서만 실행하며 CI는 이를 명시적으로 제외한다.

## 확인된 성능과 한계

표본은 원래 춘식아 CNN128세 모델의 평균이며 가중치와 실행 기준을 바꾸지 않았다. 고정 Development 세 반복 묶음의 Clean100%, 소음 균등 평균99.1111%, 최악 관측FA0.0832/h는 [모델 카드](../korean_openwakeword/resources/sample_chunsik/MODEL_CARD.md)의 해당 분모로 읽는다.

이 통계는 독립100h Final이나 실제 불특정 다수의 음성을 검증한 보증이 아니다. 다른 한국어 호출어는 새 TTS·모델·Development·동결된 Final을 진행해야 한다. 동봉 결과 사이트는 작은 재현 자료이며 과거 전체 통계를 복사한 새 성능 시험이 아니다.

## 0.3.0 게시 전 확인 결과

- Windows / Python3.12 로컬 전체 시험 **57개 통과**, 실패·skip 없음.
- 별도 폴더에 wheel을 설치하고 과거 실험 코드 import를 차단한 시험 **54개 통과**. 로컬 과거 구현과의 parity3개만 의도적으로 제외했다. 23개 라이브러리 모듈 모두 설치한 wheel에서 불러왔다. 기존 의존성은 재사용했다.
- 실제 새 호출어 소형3모델 학습·ONNX 변환·중단 재개·R1/R3/R5 행 수·임계값 replay·Final 잠금 확인. 합성 fixture/Qwen API 모형 시험을 새 대규모 음성 생성 또는 성능 입증으로 세지 않는다.
- DNS 고정 fan 파일320,044bytes를 실제 다운로드하고 SHA256을 확인. 대용량 공개 자료34개의 다운로드 pin은 기존 검증 receipt에서 가져왔다.
- 예시 춘식아3가중치·frontend2파일의 SHA256, 결과 사이트20WAV의 채점 해시 일치, 상대 문서 링크·개인 경로/토큰 제외 확인.
- 브라우저에서 SNR10 환경음→음악→한국어 대화의 자동 연속 재생과 일시정지 확인.
- wheel과 source archive build 확인. PyTorch2.8의 기존 ONNX exporter는 deprecation/trace 경고가 있지만 변환·추론·재개 시험은 통과했다.

기계적으로 읽을 수 있는 근거는 [publication_checks.json](../validation/publication_checks.json)에 있다. GitHub의 새 Windows/Linux 의존성 설치 결과는 저장소 **Actions**에서 확인한다. 이 문서의 로컬 시험 결과를 모든 OS·CUDA·Qwen 조합의 검증 완료로 읽지 않는다.
