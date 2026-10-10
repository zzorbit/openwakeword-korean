# 예시 모델·오디오 출처

모델 이용조건과 소스 Apache-2.0 조건은 구분한다. 이 묶음에 일괄 상업 이용 허가를 부여하지 않는다. OpenWakeWord의 [공식 저장소](https://github.com/dscripka/openWakeWord)와 Qwen 모델 카드를 따른다. 공유 frontend의 고정 release는 v0.5.1이다.

[AUDIO_SOURCES.json](AUDIO_SOURCES.json)은 동봉 환경음(DNS/Freesound CC0), 음악(MUSAN과 개별 음악 라이선스), 한국어 대화(Seoul Corpus CC BY-NC 2.0)의 source ID·출처를 보존한다. 음악의 원문 notice는 `licenses/MUSAN_FMA_LICENSE.txt`와 `licenses/MUSAN_JAMENDO_LICENSE.txt`다. Apache 원문은 `licenses/Apache-2.0.txt`다.

Seoul Corpus attribution: Yun, W., Yoon, K., Park, S., Lee, J., Cho, S., Kang, D., Byun, K., Hahn, H., Kim, J., The Korean Corpus of Spontaneous Speech, Journal of the Korean Society of Speech Sciences 7(2), 103–109, 2015. https://www.openslr.org/113/ . 비상업 조건을 유지한다. 음악은 OpenSLR17 MUSAN 원문과 source별 notice를 함께 확인한다.

오디오는 기존 Development 원본의 resample 또는 소음 혼합 결과이며 호출 Positive는 Qwen 합성이다. 개인 호출 녹음과 Final이 아니다. 모델/음원은 코드 라이선스만으로 공개·상업 배포 조건이 해결되었다고 주장하지 않는다.
