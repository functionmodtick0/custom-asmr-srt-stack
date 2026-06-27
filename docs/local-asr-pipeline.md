# 로컬 ASMR ASR 파이프라인

작성일: 2026-06-27

## 목적

일본 ASMR/동인음성 전사는 외부 API 없이 로컬에서 처리한다.

이 문서는 현재 구현된 고품질 로컬 경로, 실험 결과, 다음 작업 계획을 기록한다. 제품 범위와 데이터 계약은 `docs/product-decisions.md`가 기준이고, 이 문서는 ASR 파이프라인의 세부 구현/평가 기록이다.

## 현재 기본 방향

현재 고품질 경로는 다음 순서를 따른다.

```text
원본 오디오
-> 16-bit PCM WAV 정규화
-> L/R/MIX 채널 생성
-> MIX energy 기반 speech chunking
-> Qwen3-ASR 로컬 전사
-> L/R energy 기반 channel attribution
-> master.json 저장
-> eval-transcript로 품질 측정
```

핵심 결정:

- ASR 텍스트는 `MIX`에서 만든다.
- L/R 단독 ASR은 기본 경로로 쓰지 않는다.
- L/R은 발화 위치 판정에 사용한다.
- `text`에는 `[L]`, `[R]` 같은 라벨을 넣지 않는다.
- 모든 channel 정보는 `segment.channel`에만 저장한다.
- 번역은 이 파이프라인에서 하지 않는다.

## 로컬 Qwen ASR 런타임

기본 adapter:

```text
local-qwen-asr
```

권장 모델:

```text
Qwen/Qwen3-ASR-1.7B
```

Qwen runtime은 `qwen-asr`가 `transformers==4.57.6`을 강하게 고정하므로 root venv와 분리한다.

```bash
uv venv .casrt/qwen-asr-venv --python 3.12
uv pip install --python .casrt/qwen-asr-venv/bin/python -e .
uv pip install --python .casrt/qwen-asr-venv/bin/python qwen-asr
```

실행 예:

```bash
CASRT_QWEN_ASR_WORKER_COMMAND='.casrt/qwen-asr-venv/bin/python -m custom_asmr_srt_stack.qwen_asr_worker' \
CASRT_QWEN_ASR_DEVICE_MAP=cuda:0 \
CASRT_QWEN_ASR_DTYPE=bfloat16 \
uv run casrt project transcribe PROJECT_ID \
  --adapter local-qwen-asr \
  --model-id Qwen/Qwen3-ASR-1.7B
```

Qwen worker는 JSON Lines subprocess protocol을 사용한다. worker import, model load, inference, response contract 오류는 fallback 없이 실패로 표시한다.

## 오디오 전처리

로컬 ASR worker는 추론 직전에 조용한 PCM16 WAV를 bounded gain으로 보정한다.

현재 값:

```text
target RMS: -24.0 dBFS
max peak: -3.0 dBFS
max gain: 4.0x
```

이 보정은 ASMR의 저음량/속삭임 구간에서 모델 입력을 안정화하기 위한 것이다. 클리핑을 숨기는 fallback이 아니라, 피크 제한이 있는 좁은 범위의 전처리다.

## Speech Chunking

`local-qwen-asr`는 분석 단계의 180초 chunk를 그대로 쓰지 않고, MIX 채널에서 energy 기반 speech interval을 다시 만든다.

현재 값:

```text
threshold_dbfs: -48.0
window_ms: 100
min_silence_ms: 800
min_speech_ms: 200
pad_ms: 400
```

이 결정의 이유:

- 10초 통째 입력은 텍스트는 어느 정도 맞아도 segment timing이 거칠다.
- ForcedAligner 단독보다 먼저 발화 단위 chunking을 해야 timing이 안정된다.
- WebUI에는 이 값을 옵션으로 노출하지 않는다. 필요하면 config/env 또는 developer setting으로 분리한다.

## Channel Attribution

Qwen은 `MIX`를 전사한다. 생성된 speech segment에 대해 같은 시간 범위의 L/R RMS를 비교해 channel을 판정한다.

현재 값:

```text
L/R 확정 기준: 6.0 dB 이상 차이
```

동작:

- L이 R보다 6dB 이상 크면 `channel: "L"`
- R이 L보다 6dB 이상 크면 `channel: "R"`
- 차이가 작으면 `channel: "MIX"`

이 기준은 보수적이다. 채널을 틀리게 확정하는 것보다 `MIX`로 남기는 쪽을 우선한다.

## ForcedAligner 상태

후보:

```text
Qwen/Qwen3-ForcedAligner-0.6B
```

실행은 다음 env로 켠다.

```bash
CASRT_QWEN_ASR_ALIGNER_MODEL_ID=Qwen/Qwen3-ForcedAligner-0.6B
```

현재 판단:

- 기본 경로로 승격하지 않는다.
- 10초 실데이터 crop에서 일부 timestamp가 초반으로 잘 맞지 않고, token duration이 0인 항목이 있었다.
- ForcedAligner는 VAD/speech chunking이 안정된 뒤 segment 내부 보정 또는 debug alignment 용도로 다시 평가한다.

## 평가 Harness

CLI:

```bash
uv run casrt eval-transcript reference.srt candidate.json --json -o eval.json
```

현재 측정값:

- speech text strict CER
- speech text practical CER
- segment index 기준 mean start/end/boundary error
- L/R channel accuracy
- candidate `needs_review` 비율

strict CER는 공백만 제거한다.

practical CER는 현재 다음을 정규화한다.

- Unicode NFKC
- 공백 제거
- punctuation/symbol 제거

practical CER는 자막 실용 비교용이다. 원문 보존 품질은 strict CER를 같이 본다.

## 10초 실데이터 실험

입력:

```text
.casrt/experiments/upload-real-crop/01-front10.wav
```

참조:

```text
やばっ!見つかっちゃったぁ…。
ね、魔女ちゃん、こいつ強い?えっと…。
```

결과 요약:

| 경로 | segment 수 | CER | mean boundary error |
| --- | ---: | ---: | ---: |
| Qwen3-ASR 1.7B, 10초 통째 | 1 | 20.6% | 3600.0ms |
| Qwen3-ASR 1.7B + ForcedAligner | 1 | 20.6% | 3300.0ms |
| Qwen3-ASR 1.7B + energy chunking | 2 | 23.5% | 302.5ms |
| Qwen3-ASR 1.7B + energy chunking + channel attribution | 2 | 23.5% | 302.5ms |

energy chunking + channel attribution 출력:

```text
00:00.600-00:04.200  L
やば、見つかっちゃった。

00:04.700-00:09.700  MIX
ねね魔女ちゃんこいつ強い？えっと。
```

해석:

- timing은 energy chunking으로 크게 개선됐다.
- strict CER는 punctuation/表記差 때문에 높게 나온다.
- 첫 segment는 L/R energy 차이가 충분해서 L로 확정됐다.
- 둘째 segment는 L/R 차이가 6dB 미만이라 MIX로 남았다.

## 다음 작업 계획

1. Gold set 구축
   - `/data/uploads`, `/data/outputs`에서 30초~2분 단위 reference set을 만든다.
   - CER, timing error, channel accuracy, human edit count를 기록한다.

2. 일본어 평가 정규화 확장
   - strict/practical CER는 분리됐다.
   - 다음 단계에서는 장음/감탄/소형 kana 차이를 별도 옵션으로 추가할지 평가한다.

3. VAD 후보 추가
   - 현재 energy splitter는 baseline이다.
   - Silero VAD 또는 TEN VAD를 비교 후보로 붙인다.
   - VAD도 WebUI 옵션으로 노출하지 않고 고정/내부 설정으로 둔다.

4. Channel attribution 튜닝
   - 현재 6dB threshold는 보수적 baseline이다.
   - gold set 기준으로 threshold와 MIX 유지 비율을 조정한다.
   - 필요하면 segment별 channel confidence를 debug metadata로만 저장한다.

5. Forced alignment 재평가
   - VAD chunk가 안정된 뒤 Qwen3-ForcedAligner를 segment 내부 timing 보정에 다시 사용한다.
   - word/char alignment는 번역용 JSON에 넣지 않는다.

6. 모델 비교
   - `Qwen/Qwen3-ASR-1.7B`를 주력으로 둔다.
   - `Qwen/Qwen3-ASR-0.6B`는 속도/저사양 후보로 비교한다.
   - Gemma 4 E4B는 general multimodal baseline으로만 유지한다.
   - Whisper 계열 도메인 fine-tune은 제품 기본이 아니라 비교 baseline으로만 본다.

## 문서화 규칙

파이프라인 변경, 모델 선택, threshold, 평가 결과, 다음 작업 계획은 채팅에만 남기지 않는다. 변경이 확정되면 이 문서 또는 관련 제품 문서에 반영한다.
