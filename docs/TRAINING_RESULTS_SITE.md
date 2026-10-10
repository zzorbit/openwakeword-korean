# 학습 결과 사이트

`koww site --evaluation 완료_평가_폴더 --output 결과_사이트_폴더`로 내보낸다.
`koww serve --directory 결과_사이트_폴더 --port 8791`로 열거나 정적 index.html을 연다.
[동봉 예제](../examples/library_demo_site/index.html)는 작은 Development 재현이며 새 학습 결과가 아니다.

- 단일 모델은 호출 성공률 순, 앙상블은 별도 표시.
- Clean·소음 종류·SNR·실제 표본·Negative 유효 시간·FA/hour·지연을 구분.
- SNR을 고르면 원본→배경→소음 구간 순서로 자동 연속 재생. 이전/다음·특정 항목·일시정지/재개 지원.
- 시간별 점수 그래프는 넣지 않음.
- 청취·다운로드 WAV는 실제 채점 WAV와 동일 SHA256. 숨은 재생 gain 없음.
- 레벨을 고치면 입력 버전을 바꾸고 다시 채점.
- 합성·실제 사람 결과, 작은 demo·전체 Development·독립 Final의 분모를 분리.

평가→임계값 선택→동결→Final 순서는 [plan.md](../plan.md)와 [라이브러리 가이드](LIBRARY_GUIDE.md)를 따른다.
