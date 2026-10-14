# 코드·모델·음원 출처와 이용조건

이 라이브러리의 소스 코드는 [Apache-2.0](LICENSE)으로 제공합니다. **모델·음원·학습 데이터는 각각의 출처 조건을 확인해야 합니다.**

| 사용할 대상 | 확인할 문서 |
| --- | --- |
| 라이브러리 소스 코드 | [LICENSE](LICENSE) |
| 제공하는 ‘춘식아’ 모델 | [모델 설명서](korean_openwakeword/resources/sample_chunsik/MODEL_CARD.md)와 아래 학습 자료 출처 |
| 동봉 평가 오디오 | 아래 음원별 출처와 예시 모델 폴더의 원문 라이선스 |
| 직접 내려받을 학습 자료 | [데이터 준비 안내](docs/DATASETS.md)와 제공자의 이용조건 |

일부 동봉 음원에는 비상업 조건이 있습니다. 코드 라이선스를 근거로 모델·음원 전체가 상업 이용 가능하다고 판단하지 않습니다. 대규모 원시 데이터와 생성용 Qwen 가중치는 함께 배포하지 않습니다.

## openWakeWord와 공유 모델

- [dscripka/openWakeWord](https://github.com/dscripka/openWakeWord), David Scripka와 기여자. 코드 Apache-2.0. 라이브러리는 0.6.0 ONNX 기능에 의존한다.
- `melspectrogram.onnx`, `embedding_model.onnx`: [공식 v0.5.1 release](https://github.com/dscripka/openWakeWord/releases/tag/v0.5.1)의 고정 SHA256 자산. 상위 README는 Google speech embedding backbone을 Apache-2.0으로 설명하고 공식 배포 wakeword 모델에는 별도 CC BY-NC-SA 4.0 조건을 설명한다. 우리 판별기를 공식 pretrained wakeword 모델로 오인하지 않는다.
- ONNX Runtime, NumPy, SciPy, PyTorch 등 설치 의존성은 각 상위 프로젝트 조건을 따른다. 라이브러리가 그 바이너리를 재배포하지 않는다.

## Qwen·학습 입력·예시 모델

- [Qwen3-TTS VoiceDesign](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign), [Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base): 모델 카드와 상위 라이선스를 따른다. 생성용 가중치는 wheel에 넣지 않는다. 참조는 합성 음성만 사용한다.
- Generic feature: [davidscripka/openwakeword_features](https://huggingface.co/datasets/davidscripka/openwakeword_features), 고정 revision과 전체 파일 해시로 재사용한다. 제공자에 기록된 이용조건은 CC BY-NC-SA 4.0이다. ACAV 기반 제공자 데이터의 명목 시간·행별 출처 한계를 metadata에 남긴다. 전체 파일은 동봉하지 않는다.
- 한국어 학습 Negative: [ESPnet YODAS](https://huggingface.co/datasets/espnet/yodas)의 ko000 분할(제공자 CC BY 3.0)과 [Zeroth Korean / OpenSLR40](https://www.openslr.org/40/)(CC BY 4.0). 원시 음성은 동봉하지 않으며 제공자의 원본 attribution을 보존한다. 실제 다운로드 pin과 구체적인 재사용 절차는 [DATASETS.md](docs/DATASETS.md)와 [공개 자료 목록](korean_openwakeword/resources/public_datasets.json)에 있다.
- 증강 입력: [MUSAN](https://www.openslr.org/17/), [DNS-Challenge](https://github.com/microsoft/DNS-Challenge), [RIRS_NOISES](https://www.openslr.org/28/). corpus 설명과 각 원출처 notice를 함께 보존하며 모두 한 가지 라이선스로 간주하지 않는다.
- 춘식아의 세 CNN 판별기는 이 프로젝트에서 학습한 모델이다. 코드 Apache 조건만으로 원시 공개 데이터와 학습 모델의 모든 상업 이용 조건이 해결되었다고 주장하지 않는다. 출처별 조건을 유지하며 공개 배포/상업 적용 시 별도 모델 조건을 확인한다. [모델 카드](korean_openwakeword/resources/sample_chunsik/MODEL_CARD.md)에 성능 범위와 재현을 명시한다.

## 동봉 Development 오디오

소규모 재현 WAV와 파생 noisy WAV만 동봉한다. 전부 `audio_manifest.json`의 SHA로 채점·청취에 동일하게 사용한다. 일부 음원은 **비상업 조건**이 있으므로 wheel 전체를 일괄 상업 이용 가능 음원 묶음으로 취급하지 않는다. attribution metadata는 예시 모델 폴더의 `AUDIO_SOURCES.json`에 있다.

- 환경음: [Microsoft DNS-Challenge](https://github.com/microsoft/DNS-Challenge)의 Freesound CC0 선별 자료. 동봉 구간은 source `freesound:323547`(door), `freesound:264156`(squeaky chair)이며 URL·revision을 보존한다.
- 음악: [MUSAN / OpenSLR17](https://www.openslr.org/17/)에서 `music-fma-0113`, **“Don't You Worry” by Tricky Diesel**, 및 `music-jamendo-0047`의 구간. corpus attribution과 개별 원출처 조건을 함께 따른다. `licenses/MUSAN_FMA_LICENSE.txt`, `licenses/MUSAN_JAMENDO_LICENSE.txt`에 상위 원문 notice를 보존한다. 노이즈 구간 선택·혼합·resample 등 변환 metadata는 audio manifest에 남긴다.
- 한국어 대화: [Seoul Corpus / OpenSLR113](https://www.openslr.org/113/), **CC BY-NC 2.0**. recordings `s09f17f1`(resample된 intact Negative), `s15m22m2`(noisy 평가에 사용한 구간). Yun, W., Yoon, K., Park, S., Lee, J., Cho, S., Kang, D., Byun, K., Hahn, H., Kim, J., “The Korean Corpus of Spontaneous Speech”, Journal of the Korean Society of Speech Sciences 7(2), 103–109, 2015. 사용 범위는 공개 background/Negative 및 평가다.
- 호출 Positive는 프로젝트의 Qwen 합성 ‘춘식아’ 발화이며 개인 호출 녹음은 아니다.

이 파일은 출처 기록이다. 미확인된 권리를 추가로 부여하는 문서가 아니며 라이선스 원문과 데이터 제공자의 조건을 보존한다.

## Qwen CUDA 가속 구성요소

`tts_cuda_predictor.py`와 `tts_cuda_sampling.py`는 [faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts)의 MIT 코드에서 가져왔습니다. 원본 commit과 파일 해시는 `resources/faster_qwen3_tts_UPSTREAM.json`, 라이선스는 `resources/faster_qwen3_tts_LICENSE.txt`에 보존합니다. 한국어 라이브러리 adapter는 기본 Qwen 호출과 같은 출력인지 별도로 검증합니다.
