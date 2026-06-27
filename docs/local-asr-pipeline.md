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
min_silence_ms: 500
min_speech_ms: 200
pad_ms: 200
```

`CASRT_VAD_COMMAND`가 설정되어 있으면 energy splitter 대신 고정 VAD command의 interval을 사용한다.

```bash
CASRT_VAD_COMMAND='python3 path/to/vad.py' \
  uv run casrt project transcribe PROJECT_ID \
  --adapter local-qwen-asr \
  --model-id Qwen/Qwen3-ASR-1.7B
```

VAD command contract:

- stdin: `{ audio_file, audio_info }`
- stdout: `{ intervals: [{ start_ms, end_ms }] }`
- interval은 정렬되어야 하고 서로 겹치면 안 된다.
- interval이 audio duration을 넘거나 malformed이면 fallback하지 않고 실패한다.
- WebUI/CLI 옵션으로 노출하지 않는다.

내장 energy splitter는 다음 env로만 내부 튜닝할 수 있다.

```text
CASRT_QWEN_ENERGY_THRESHOLD_DBFS
CASRT_QWEN_ENERGY_WINDOW_MS
CASRT_QWEN_ENERGY_MIN_SILENCE_MS
CASRT_QWEN_ENERGY_MIN_SPEECH_MS
CASRT_QWEN_ENERGY_PAD_MS
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

여러 샘플은 gold set manifest로 묶어서 평가한다.

```json
{
  "format": "custom-asmr-eval-manifest-v1",
  "cases": [
    {
      "id": "front10",
      "reference": "refs/front10.srt",
      "candidate": "outputs/qwen-front10.json",
      "candidate_id": "qwen-energy"
    }
  ]
}
```

```bash
uv run casrt eval-manifest gold.json --json -o eval-suite.json
```

현재 측정값:

- speech text strict CER
- speech text practical CER
- segment index 기준 mean start/end/boundary error
- time-overlap 기준 `timing_time_aligned` mean start/end/boundary error
- boundary sample 수, max boundary error, 250ms/500ms 이내 boundary ratio
- L/R/MIX channel confusion
- candidate MIX 유지 비율
- index 기준 `channel` 및 time-overlap 기준 `channel_time_aligned` L/R channel accuracy
- candidate `needs_review` 비율

strict CER는 공백만 제거한다.

practical CER는 현재 다음을 정규화한다.

- Unicode NFKC
- 공백 제거
- punctuation/symbol 제거

practical CER는 자막 실용 비교용이다. 원문 보존 품질은 strict CER를 같이 본다.

manifest summary는 case별 평균이 아니라 전체 edit distance/reference characters와 전체 paired/boundary/comparable segment 수 기준으로 가중 집계한다. 짧은 clip과 긴 clip이 같은 비중을 갖지 않게 하기 위한 결정이다. 품질 threshold 판단은 segment split 차이에 덜 취약한 `timing_time_aligned`와 `channel_time_aligned`를 우선 사용한다.

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

## 120초 품질 루프

일자: 2026-06-28

입력:

```text
data/uploads/01.淫魔＆魔女との遭遇.wav 앞 120초
```

reference:

```text
data/outputs/eval-csv-srt-01-full.srt에서 120초 crop
```

현재 실사용 후보 기준:

- practical CER: 5~10% 이하
- time-aligned 500ms boundary ratio: 90% 이상
- 명확한 L/R 구간 channel accuracy: 85~90% 이상

자동 gate 예시:

```bash
uv run casrt eval-transcript ref.master.json candidate.master.json \
  --max-practical-cer 0.10 \
  --min-time-aligned-500ms-ratio 0.90 \
  --min-channel-time-aligned-accuracy 0.85 \
  --max-channel-time-aligned-mix-ratio 0.50
```

결과:

| 후보 | segments | practical CER | time-aligned 500ms ratio | channel time-aligned accuracy | 판단 |
| --- | ---: | ---: | ---: | ---: | --- |
| Qwen3-ASR 1.7B, energy 800/400 | 7 | 29.3% | 10.0% | 100.0% on 2 comparable | 불합격: 과도한 병합 |
| Qwen3-ASR 1.7B, energy 500/200 | 26 | 21.7% | 25.0% | 66.7% | 불합격 |
| Qwen3-ASR 1.7B, energy 500/100 | 26 | 22.8% | 27.1% | 66.7% | 불합격 |
| Qwen3-ASR 1.7B, energy 500/200 + context | 26 | 46.5% | 25.0% | 66.7% | 불합격: context hallucination |
| Qwen3-ASR 1.7B, energy 500/200 + ForcedAligner | 26 | 21.7% | 31.2% | 83.3% | 불합격: timing/text 부족 |
| stable-ts baseline | 25 | 7.8% | 56.5% | n/a | text 합격, timing 불합격 |
| neosophie/Qwen3-ASR-1.7B-JA, energy 500/200 | 26 | 20.4% | 25.0% | 66.7% | 불합격 |
| neosophie/Qwen3-ASR-1.7B-JA, full 120s | 1 | 28.6% | 4.0% | n/a | 불합격: 과도한 병합 |
| neosophie/Qwen3-ASR-1.7B-JA, energy 1500/200 | 1 | 27.3% | 4.0% | n/a | 불합격: 과도한 병합 |

확장 gold 결과:

| 후보 | cases | reference segments | candidate segments | practical CER | time-aligned 500ms ratio | channel time-aligned accuracy | 판단 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| neosophie/Qwen3-ASR-1.7B-JA, 01/04/07 front120 | 3 | 74 | 81 | 29.6% | 29.5% | 73.1% | 불합격 |
| mistralai/Voxtral-Mini-4B-Realtime-2602, 01/04/07 front120 | 3 | 74 | 44 | 40.0% | 28.7% | 63.6% | 불합격 |
| google/gemma-4-E4B-it, 4-bit local-transformers, MIX-first, 01/04/07 front120 | 3 | 74 | 81 | 42.3% | 29.5% | 73.1% | 불합격 |
| google/gemma-4-E4B-it, 8-bit local-transformers, MIX-first, 01/04/07 front120 | 3 | 74 | 81 | 46.1% | 29.5% | 73.1% | 불합격 |
| zhifeixie/Mega-ASR, routed, MIX-first, 01/04/07 front120 | 3 | 74 | 68 | 30.9% | 28.4% | 69.6% | 불합격 |
| zhifeixie/Mega-ASR, base-only threshold 1.1, MIX-first, 01/04/07 front120 | 3 | 74 | 64 | 30.8% | 27.3% | 68.2% | 불합격 |
| zhifeixie/Mega-ASR, forced LoRA, MIX-first, 01/04/07 front120 | 3 | 74 | 81 | 77.6% | 29.5% | 73.1% | 불합격: LoRA가 ASMR에서 악화 |
| stable-ts baseline, 01/04/07 front120 | 3 | 74 | 60 | 16.1% | 56.7% | n/a | 불합격: text/timing 부족, MIX-only |

case별 practical CER:

| 후보 | case | practical CER | time-aligned 500ms ratio | channel time-aligned accuracy |
| --- | --- | ---: | ---: | ---: |
| Neosophie | 01-front120 | 20.4% | 25.0% | 66.7% |
| Neosophie | 04-front120 | 21.3% | 37.0% | 60.0% |
| Neosophie | 07-front120 | 53.0% | 26.9% | 80.0% |
| Voxtral Mini Realtime | 01-front120 | 22.4% | 22.7% | 66.7% |
| Voxtral Mini Realtime | 04-front120 | 20.8% | 37.0% | 60.0% |
| Voxtral Mini Realtime | 07-front120 | 89.0% | 0.0% | n/a |
| Gemma 4 E4B 4-bit | 01-front120 | 30.9% | 25.0% | 66.7% |
| Gemma 4 E4B 4-bit | 04-front120 | 30.9% | 37.0% | 60.0% |
| Gemma 4 E4B 4-bit | 07-front120 | 72.6% | 26.9% | 80.0% |
| Gemma 4 E4B 8-bit | 01-front120 | 27.7% | 25.0% | 66.7% |
| Gemma 4 E4B 8-bit | 04-front120 | 31.4% | 37.0% | 60.0% |
| Gemma 4 E4B 8-bit | 07-front120 | 90.2% | 26.9% | 80.0% |
| Mega-ASR routed | 01-front120 | 23.0% | 22.7% | 66.7% |
| Mega-ASR routed | 04-front120 | 20.1% | 37.0% | 60.0% |
| Mega-ASR routed | 07-front120 | 55.8% | 25.0% | 75.0% |
| Mega-ASR base-only threshold 1.1 | 01-front120 | 23.5% | 22.7% | 66.7% |
| Mega-ASR base-only threshold 1.1 | 04-front120 | 20.1% | 37.0% | 60.0% |
| Mega-ASR base-only threshold 1.1 | 07-front120 | 54.6% | 21.4% | 72.7% |
| Mega-ASR forced LoRA | 01-front120 | 66.2% | 25.0% | 66.7% |
| Mega-ASR forced LoRA | 04-front120 | 81.3% | 37.0% | 60.0% |
| Mega-ASR forced LoRA | 07-front120 | 88.4% | 26.9% | 80.0% |
| stable-ts baseline | 01-front120 | 7.8% | 56.5% | n/a |
| stable-ts baseline | 04-front120 | 7.3% | 60.9% | n/a |
| stable-ts baseline | 07-front120 | 39.0% | 52.4% | n/a |

stable-ts에 L/R energy attribution만 붙인 channel 진단:

| attribution threshold | practical CER | time-aligned 500ms ratio | channel time-aligned accuracy | comparable segments | candidate MIX ratio |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3dB | 16.1% | 56.7% | 68.2% | 22 | 17.9% |
| 6dB | 16.1% | 56.7% | 65.0% | 20 | 23.9% |
| 10dB | 16.1% | 56.7% | 76.9% | 13 | 58.2% |

window 단위 dominant fraction attribution도 01/04/07 front120 stable-ts baseline에서 실험했다. 100ms window, active threshold -60dBFS, margin 1~10dB, dominant fraction 35~75% sweep 기준 최고 channel time-aligned accuracy는 71.4%였고, segment 전체 RMS 10dB 방식의 76.9%보다 낮았다. 따라서 window 방식은 기본 구현으로 승격하지 않는다.

결정:

- Qwen 내장 energy 기본값은 `min_silence_ms=500`, `pad_ms=200`으로 낮춘다.
- `CASRT_QWEN_ASR_CONTEXT`에 긴 glossary를 그대로 넣는 방식은 기본값으로 쓰지 않는다. 짧은 구간에서 glossary 전체를 출력하는 hallucination이 발생했다.
- Qwen3-ForcedAligner는 channel/timing을 일부 개선했지만 120초 gold 기준으로 기본 승격하지 않는다.
- 현재 Qwen3-ASR 1.7B 경로만으로는 품질 기준을 만족하지 못한다.
- `neosophie/Qwen3-ASR-1.7B-JA`는 다운로드 재시도 후 점수화했다. 120초 gold 기준 Qwen3-ASR 1.7B보다 약간 낫지만 practical CER 20.4%라 기본 승격하지 않는다.
- 01/04/07 front120 확장 gold에서도 Neosophie/Qwen3-ASR-JA는 practical CER 29.6%라 기본 승격하지 않는다. 특히 07의 whisper/침대 ASMR 구간에서 텍스트 인식이 크게 무너졌다.
- Neosophie full-window와 1.5초 silence 병합 실험은 text와 timing이 모두 악화됐다. 이 샘플에서는 chunk를 길게 잡는 것이 해결책이 아니다.
- `Qwen/Qwen3-ASR-1.7B-hf`는 Hugging Face metadata상 `automatic-speech-recognition`, `ja` 지원, `transformers` 모델이다. 현재 pinned Transformers는 `qwen3_asr` 아키텍처를 인식하지 못한다. 공식 Transformers main `5.13.0.dev0` ephemeral runtime에서는 `qwen3_asr` support를 확인했지만, weight 다운로드가 Xet/일반 HTTP 양쪽에서 장시간 진행돼 2026-06-28 루프에서는 점수화하지 못했다.
- `mistralai/Voxtral-Mini-4B-Realtime-2602`는 remote model code 없이 Transformers `VoxtralRealtimeForConditionalGeneration`으로 로딩됐다. 8.9GB weight는 단일 HF stream이 느려 HTTP range 8조각 병렬 다운로드로 확보했다. 30초 smoke와 01/04 일부 텍스트는 Qwen보다 자연스러웠지만, 07 whisper/침대 ASMR에서 chunked 입력은 대부분 빈 출력이었고 120초 full-window 입력도 앞부분만 출력해 기본 승격하지 않는다.
- `mistralai/Voxtral Mini Transcribe 2.0`는 Mistral API batch transcription 제품으로 확인됐고 open-weight 로컬 checkpoint는 확인하지 못했다. 외부 API는 제품 방향이 아니므로 기본 경로에서 제외한다.
- `google/gemma-4-E4B-it`는 공식 오디오 입력을 지원하고 5초 smoke에서 유의미한 전사를 반환했다. 그러나 01/04/07 front120 확장 gold에서 4-bit practical CER 42.3%, 8-bit practical CER 46.1%로 기준을 크게 벗어났다. 8-bit는 01 smoke와 01 case를 조금 개선했지만 07 whisper/침대 ASMR에서 반복 hallucination이 발생해 전체 지표가 악화됐다. 따라서 기본 승격하지 않는다.
- Gemma E4B 실험 산출물은 `/tmp/casrt-quality/gemma-e4b-4bit-bounded-results`, `/tmp/casrt-quality/gemma-e4b-8bit-bounded-results`, report는 `/tmp/casrt-quality/gemma-e4b-4bit-bounded-3case-report.json`, `/tmp/casrt-quality/gemma-e4b-8bit-bounded-3case-report.json`에 있다.
- `zhifeixie/Mega-ASR`는 2026-05 공개 Qwen3-ASR-1.7B 기반 robust ASR 후보이며, noisy/reverberant/clipped/band-limited/overlapping 등 어려운 실제 녹음에서 empty output, omission, repetition, hallucination을 줄이는 것을 목표로 한다. ASMR 전용은 아니지만 현재 07 whisper/침대 구간 실패 양상과 맞닿아 있으므로 다음 우선 모델 실험으로 둔다. 공식 runtime은 `xzf-thu/Mega-ASR` repository 코드와 checkpoint 배치를 요구하므로 `/tmp` 격리 환경에서 실행한다.
- Mega-ASR runtime은 실행 전 `gpt-5.4 xhigh` subagent가 정적 보안 검토했다. Verdict는 `PASS_WITH_CONSTRAINTS`다. 허용 범위는 `/tmp` 별도 venv, `/tmp` HF cache, Hugging Face allowlist download, Transformers backend만, `infer.py`/`evaluate_wer.py`만, vLLM/webui/training/wandb 금지, safetensors-only checkpoint 강제다. `adapter_model.bin`, `.pt`, `.pth` 또는 router non-safetensors checkpoint를 읽게 되면 미승인으로 간주한다.
- Mega-ASR 정적 검토에서 확인한 고위험 지점은 `lora_switch.py`의 `adapter_model.bin` fallback `torch.load`와 `router.py`의 non-safetensors `torch.load(weights_only=False)`다. 따라서 실험 전 `mega-asr-merged/adapter_model.safetensors`와 `audio_quality_router/best_acc_model.safetensors` 존재를 확인하고 unsafe fallback 파일은 사용하지 않는다.
- Mega-ASR는 `/tmp/casrt-quality/mega-asr-venv`, `HF_HOME=/tmp/casrt-quality/hf-home-mega-asr`, offline env에서 점수화했다. Checkpoint에는 unsafe pickle fallback 대상 파일이 없고, `adapter_model.safetensors`와 `best_acc_model.safetensors`를 확인했다.
- Mega-ASR 5초 smoke는 `やば、見つかっちゃった。`를 반환했고 router는 `use_lora=False`, degraded probability 0.0559였다. 그러나 01/04/07 front120 routed practical CER는 30.9%라 Neosophie 29.6%보다 약간 낮고 기준에 크게 못 미친다. threshold 1.1로 base-only에 가깝게 만든 경로도 30.8%로 유의미한 개선이 없다. forced LoRA는 77.6%로 크게 악화되어 ASMR 기본 경로로 쓰지 않는다.
- Mega-ASR 산출물은 `/tmp/casrt-quality/mega-asr-results/routed`, `/tmp/casrt-quality/mega-asr-results/base-threshold-1p1`, `/tmp/casrt-quality/mega-asr-results/force-lora`에 있다. Report는 각각 `routed-3case-report.json`, `base-threshold-1p1-3case-report.json`, `force-lora-3case-report.json`이다.
- `Atotti/llm-jp-4-8b-speech-asr`는 일본어 ASR 특화 8B 후보지만 model card상 `speech_llm_ja` 패키지(`git+https://github.com/Atotti/ja-speech-llm.git`)가 필요하다. 현재 설치된 Transformers `5.12.1`와 official main `5.13.0.dev0` 모두 `LlamaForSpeechLM`을 노출하지 않는다. 원격/외부 패키지 코드를 실행해야 하므로 사용자 명시 승인 전에는 자동 검증하지 않는다.
- `AutoArk-AI/ARK-ASR-3B`와 `CohereLabs/cohere-transcribe-03-2026`는 최신 로컬 후보지만 model card metadata에 `custom_code`가 있다. Cohere는 gated 모델이다. 외부 모델 저장소 코드를 실행하는 `trust_remote_code=True`는 기본 실험 경로로 쓰지 않고, 사용자 명시 승인이나 first-party package 지원이 있을 때만 검증한다.
- stable-ts/Whisper계 baseline은 현재 후보 중 text가 가장 좋지만 3-case practical CER 16.1%로 기준 10%를 넘고, time-aligned 500ms ratio도 56.7%로 기준 90%에 못 미친다. L/R energy attribution을 후처리로 붙여도 channel accuracy가 85%에 도달하지 않는다. 따라서 제품 기본 경로로 승격하지 않고 품질 상한 비교용으로만 유지한다.
- 2026년 공개 파이프라인 조사에서 WhisperJAV는 ASMR/VR/whisper 콘텐츠에 `fidelity` pipeline과 `aggressive` sensitivity를 권장한다. 또한 ChronosJAV는 Qwen ASR, anime-whisper, Kotoba처럼 timestamp 없는 모델의 text generation과 timestamp alignment를 분리한다. 이 방향은 모델 단독 교체보다 VAD/scene detection/alignment를 분리해서 검증해야 함을 뒷받침한다.
- `TransWithAI/Whisper-Vad-EncDec-ASMR-onnx`는 Whisper encoder 기반 VAD이며 공개 discussion에서 일본어 ASMR 약 500시간으로 학습됐다고 설명된다. ASR 모델이 아니므로 text CER를 직접 개선하지는 않지만, energy splitter보다 ASMR whisper boundary를 더 잘 잡는지 `CASRT_VAD_COMMAND` 후보로 비교한다.
- vocal separation은 무조건 적용하지 않는다. WhisperJAV README는 blanket denoise/vocal separation이 Whisper log-Mel feature를 망가뜨릴 수 있다고 경고한다. 반면 WhisperJAV issue에서는 강한 BGM/환경음이 있을 때 UVR/MDX/Demucs류 분리의 필요성이 제기됐다. 따라서 BGM/SFX가 강한 case에서만 별도 실험으로 둔다.
- 다음 개선은 ASMR-trained VAD wrapper, scene-aware chunking, forced alignment 재평가 순서로 검증한다.

## 다음 작업 계획

1. Gold set 운영
   - gold set manifest CLI는 추가됐다.
   - `/data/uploads`, `/data/outputs`에서 30초~2분 단위 reference case를 늘린다.
   - CER, timing error, channel accuracy, human edit count를 manifest report로 기록한다.

2. 일본어 평가 정규화 확장
   - strict/practical CER는 분리됐다.
   - 다음 단계에서는 장음/감탄/소형 kana 차이를 별도 옵션으로 추가할지 평가한다.

3. VAD 후보 추가
   - VAD command hook은 추가됐다.
   - 현재 energy splitter 500/200은 fallback-free baseline이다.
   - `TransWithAI/Whisper-Vad-EncDec-ASMR-onnx`, Silero VAD, TEN VAD wrapper를 command로 붙여 gold set에서 비교한다. 단, 현재 120초 결과는 VAD만으로 품질 기준을 만족할 가능성이 낮으므로 text ASR 후보 검증을 우선한다.
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
   - `zhifeixie/Mega-ASR`는 검증 완료 후보지만 기본 승격하지 않는다.
   - `neosophie/Qwen3-ASR-1.7B-JA`는 검증 완료 후보지만 기본 승격하지 않는다.
   - `Qwen/Qwen3-ASR-1.7B-hf`는 공식 Transformers release가 `qwen3_asr`를 포함하거나 weight를 안정적으로 내려받을 수 있으면 다시 비교한다.
   - `mistralai/Voxtral-Mini-4B-Realtime-2602`는 remote code 없이 검증했지만 07 whisper 구간에서 실패해 기본 승격하지 않는다.
   - `Atotti/llm-jp-4-8b-speech-asr`는 ASR 특화 일본어 후보지만 third-party runtime package가 필요하므로 사용자 명시 승인 후 비교한다.
   - `AutoArk-AI/ARK-ASR-3B`와 `CohereLabs/cohere-transcribe-03-2026`는 성능 후보로 남기되, custom code/gated 접근 조건을 먼저 해결해야 한다.
   - `Qwen/Qwen3-ASR-0.6B`는 속도/저사양 후보로 비교한다.
   - Gemma 4 E4B는 공식 오디오 입력과 smoke 전사는 성공했지만 01/04/07 front120 gold 기준을 만족하지 못해 기본 승격하지 않는다.
   - Whisper 계열 도메인 fine-tune은 제품 기본이 아니라 비교 baseline으로만 본다.

## 2026 공개 조사 출처

- Mega-ASR model card: https://huggingface.co/zhifeixie/Mega-ASR
- Mega-ASR runtime repository: https://github.com/xzf-thu/Mega-ASR
- WhisperJAV README: https://github.com/meizhong986/WhisperJAV
- WhisperJAV vocal separation issue: https://github.com/meizhong986/WhisperJAV/issues/224
- ASMR-trained Whisper VAD discussion: https://github.com/CrispStrobe/CrispASR/issues/36
- Qwen3-ASR user report thread: https://www.reddit.com/r/LocalLLaMA/comments/1rq118c/qwen3_asr_seems_to_outperform_whisper_in_almost/

## 문서화 규칙

파이프라인 변경, 모델 선택, threshold, 평가 결과, 다음 작업 계획은 채팅에만 남기지 않는다. 변경이 확정되면 이 문서 또는 관련 제품 문서에 반영한다.
