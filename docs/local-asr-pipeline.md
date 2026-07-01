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

## Human-Reviewed Reference Workflow

제품 품질 판단은 사람이 검수한 기준본만 사용한다. 현재 01/04/07 front120 pseudo-gold는 stable-ts 산출물에서 만든 상대 비교용 기준이고, 모델 승격 근거가 아니다.

검수 흐름:

```text
후보 transcript 생성
-> 사람이 SRT 또는 master JSON에서 text, timing, channel을 검수
-> casrt freeze-reference로 reference master JSON 고정
-> gold manifest에 reference_type=human-reviewed 기록
-> eval-manifest 품질 gate로 모델/heuristic 변경 판단
```

명령:

```bash
uv run casrt freeze-reference reviewed.srt -o refs/front120.master.json --json
```

`freeze-reference`는 시간순 정렬, stable id 재부여, `needs_review=false` 저장만 수행한다. 검수 여부를 자동 판정하지 않으므로 사람이 검수하지 않은 pseudo-gold는 `reference_type=pseudo-gold`로만 기록한다.

## 로컬 Qwen ASR 런타임

기본 adapter:

```text
local-qwen-asr
```

권장 모델:

```text
Qwen/Qwen3-ASR-1.7B
```

외부 runtime이나 downloaded tooling을 실행하는 실험/benchmark는 repo id 대신 고정 snapshot directory를 model id로 넘긴다. 우리 저장소의 일반 wrapper, 테스트, 문서 변경은 subagent 보안 검토 대상이 아니며 behavior test와 자체 리뷰로 검증한다.

큰 local snapshot은 `/tmp`에만 두지 않는다. `/tmp`는 재부팅이나 정리 작업으로 사라질 수 있으므로 다운로드 staging이나 삭제되어도 되는 실험 출력에만 사용한다. 재다운로드를 피하기 위해 gitignored `.casrt/models/` 또는 persistent Hugging Face snapshot cache를 쓰고, digest report는 `.casrt/model-digests/`에 둔다. benchmark에는 그 exact directory와 digest report를 기록한다. 과거 문서의 `/tmp/casrt-quality...` 경로는 실행 provenance일 뿐 장기 보존 원본이 아니다.

```text
Qwen3-ASR-1.7B snapshot: /home/brain-offloaded/.cache/huggingface/hub/models--Qwen--Qwen3-ASR-1.7B/snapshots/7278e1e70fe206f11671096ffdd38061171dd6e5
Qwen3-ForcedAligner-0.6B snapshot: /home/brain-offloaded/.cache/huggingface/hub/models--Qwen--Qwen3-ForcedAligner-0.6B/snapshots/c7cbfc2048c462b0d63a45797104fc9db3ad62b7
```

Qwen runtime은 `qwen-asr`가 `transformers==4.57.6`을 강하게 고정하므로 root venv와 분리한다.

```bash
uv venv .casrt/qwen-asr-venv --python 3.12
uv pip install --python .casrt/qwen-asr-venv/bin/python -e .
uv pip install --python .casrt/qwen-asr-venv/bin/python qwen-asr==0.0.6
```

허용 package fingerprint:

```text
qwen-asr version: 0.0.6
qwen_asr-0.0.6.dist-info/RECORD SHA-256: 56454a099599cb3c86fd96347baa86269cc62e0d9eced004eeb2faa26b3a8a7c
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

## 로컬 Qwen HF ASR 런타임

HF-native Qwen3-ASR는 qwen-asr package 경로와 분리된 adapter를 사용한다.

```text
local-qwen-hf-asr
```

실행 조건:

```text
CASRT_LOCAL_WORKER_ENV_MODE=offline
CASRT_QWEN_HF_ASR_REQUIRE_LOCAL_MODEL_PATH=1
CASRT_QWEN_HF_ASR_LOCAL_FILES_ONLY=1
CASRT_QWEN_HF_ASR_DISABLE_NETWORK=1
```

worker는 `AutoProcessor`와 `AutoModelForMultimodalLM`을 사용하고, model load는 local snapshot path + `local_files_only=True` + `trust_remote_code=False` + `use_safetensors=True`로 고정한다. Qwen HF ASR는 transcript text만 반환하므로 chunk 전체 timing과 `needs_review=true`를 반환하고, timing 품질은 후속 VAD/alignment 평가에서 본다.

2026-06-30 최신 후보 상태:

- 모델: `Qwen/Qwen3-ASR-1.7B-hf`
- revision: `057a3b044fcd31c433e7971ab40d68d20e7eae6d`
- local dir: `.casrt/models/qwen3-asr-1.7b-hf-057a3b044fcd31c433e7971ab40d68d20e7eae6d` (moved to persistent cache on 2026-07-01)
- digest report: `.casrt/model-digests/qwen3-asr-1.7b-hf-057a3b044fcd31c433e7971ab40d68d20e7eae6d-digest.json`
- snapshot SHA-256: `9c5e214252ebc2be3d989c83bddc1dc7c8981389e8c27fb99f27516a1dfa556c`
- `model.safetensors` SHA-256: `2db53c7d81bd9b8cbc6a074e89be2c968a0d373fb4ee68bb1b1e14f7042dfee1`, size 4,076,193,080 bytes.
- root Transformers 5.12.1 smoke: `qwen3_asr` architecture unknown, fail closed.
- Transformers main venv: `5.13.0.dev0`, commit `45b004d7bb505a258542d1965b0f9e0d8b03b89d`.
- 5s smoke: `やば、見つかっちゃった。`
- 01/04/07 front120 pseudo-gold benchmark: practical CER 29.4%, time-aligned 500ms ratio 27.3%, channel time-aligned accuracy 68.2%, review effort 75/75 segments. Output dir: `/tmp/casrt-quality.Q5OdDf/qwen-hf-asr-transformers-main`, report: `/tmp/casrt-quality.Q5OdDf/qwen-hf-asr-transformers-main-3case-report.json`, review pack: `/tmp/casrt-quality.Q5OdDf/review-pack-qwen-hf-asr-transformers-main`.
- 결정: HF-native Qwen3-ASR는 local adapter로 유지하지만 기본 ASMR 경로로 승격하지 않는다. Timestamp 없는 full-chunk output 때문에 alignment/review burden이 크고, text도 기존 Qwen/Neosophie 계열을 이기지 못했다.

외부 runtime benchmark의 local-only 실행 조건:

```text
CASRT_LOCAL_WORKER_ENV_MODE=offline
CASRT_COHERE_ASR_DISABLE_NETWORK=1
CASRT_QWEN_ASR_REQUIRE_LOCAL_MODEL_PATH=1
CASRT_QWEN_ASR_LOCAL_FILES_ONLY=1
CASRT_QWEN_ASR_DISABLE_NETWORK=1
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
HF_DATASETS_OFFLINE=1
WANDB_MODE=disabled
```

- `CASRT_LOCAL_WORKER_ENV_MODE=offline`은 local worker subprocess 환경에서 token/proxy류 env를 제거하고 offline flags를 주입한다.
- `CASRT_QWEN_ASR_REQUIRE_LOCAL_MODEL_PATH=1`은 `model_id`와 `CASRT_QWEN_ASR_ALIGNER_MODEL_ID`가 존재하는 local directory가 아니면 실패시킨다.
- `CASRT_QWEN_ASR_LOCAL_FILES_ONLY=1`은 Transformers model load kwargs에 `local_files_only=True`, `trust_remote_code=False`를 붙인다.
- `CASRT_QWEN_ASR_DISABLE_NETWORK=1`은 Qwen worker process 내부 Python socket 생성을 차단한다.
- 실험 산출물은 고정 `/tmp/casrt-quality`를 재사용하지 않고, 매 실행마다 새 `0700` directory 아래에 둔다. 원본 fixture tree에는 쓰지 않는다.

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
max_chunk_ms: unset
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
CASRT_QWEN_ENERGY_MAX_CHUNK_MS
```

`CASRT_QWEN_ENERGY_MAX_CHUNK_MS`는 긴 energy interval을 고정 길이 이하로 자르는 내부 실험 옵션이다. 기본값은 unset이며 WebUI/CLI 모델 선택 옵션으로 노출하지 않는다.

이 결정의 이유:

- 10초 통째 입력은 텍스트는 어느 정도 맞아도 segment timing이 거칠다.
- ForcedAligner 단독보다 먼저 발화 단위 chunking을 해야 timing이 안정된다.
- WebUI에는 이 값을 옵션으로 노출하지 않는다. 필요하면 config/env 또는 developer setting으로 분리한다.

## ASR Text Cleanup

로컬 일본어 ASR worker는 모델 출력 텍스트를 segment 저장 전에 정리한다.

현재 동작:

- common prefix(`Transcription:`, `Transcript:`, `文字起こし:`, `書き起こし:`)를 제거한다.
- 일본어 문자 사이의 불필요한 공백과 punctuation 주변 공백을 압축한다.
- 앞뒤의 비일본어 noise를 제거한다.
- 정리 후 일본어 문자가 하나도 없으면 hallucination segment로 보고 버린다.

이 필터는 punctuation-only/English-only 같은 비일본어-only 출력에 한정한다. 일본어 문자가 포함된 segment는 여기서 버리지 않고 평가와 human review에서 다룬다.

## Channel Attribution

Qwen은 `MIX`를 전사한다. 생성된 speech segment에 대해 같은 시간 범위의 L/R RMS를 비교해 channel을 판정한다. 같은 구현은 기존 SRT/master 후처리 CLI에서도 사용한다.

현재 값:

```text
L/R 확정 기준: 8.0 dB 이상 차이
quieter side gate: -40.0 dBFS 이하
```

동작:

- L이 R보다 8dB 이상 크고 R이 -40dBFS 이하이면 `channel: "L"`
- R이 L보다 8dB 이상 크고 L이 -40dBFS 이하이면 `channel: "R"`
- 차이가 작거나 양쪽이 모두 충분히 active이면 `channel: "MIX"`

이 기준은 보수적이다. 채널을 틀리게 확정하는 것보다 `MIX`로 남기는 쪽을 우선한다.

기존 transcript 후처리:

```bash
uv run casrt attribute-channels audio.wav candidate.master.json -o candidate.attributed.master.json --json
```

이 명령은 `MIX` speech segment만 relabel하며, 이미 `L`/`R`인 segment와 speech가 아닌 segment는 바꾸지 않는다. mono audio나 L/R을 만들 수 없는 audio는 실패한다. `--threshold-db`와 `--quiet-channel-max-dbfs`는 benchmark 재현용 CLI 옵션이고 WebUI에는 노출하지 않는다.

## ForcedAligner 상태

후보:

```text
Qwen/Qwen3-ForcedAligner-0.6B
```

실행은 다음 env로 켠다.

```bash
CASRT_QWEN_ASR_ALIGNER_MODEL_ID=Qwen/Qwen3-ForcedAligner-0.6B
CASRT_QWEN_ASR_MIN_ALIGNED_DURATION_MS=80
```

`CASRT_QWEN_ASR_MIN_ALIGNED_DURATION_MS`보다 짧은 aligned span은 비현실적인 timestamp contract 위반으로 보고 해당 clip bounds로 되돌린다. 2026-06-30 forced aligner 3-case 실험에서 1ms segment가 관측되어 추가했다. 기본값은 80ms이며 WebUI/CLI 옵션으로 노출하지 않는다.

기존 master 텍스트를 재정렬하는 generic aligner command도 제공한다.

```bash
CASRT_ALIGNER_ENV_MODE=offline \
CASRT_QWEN_ALIGNER_REQUIRE_LOCAL_MODEL_PATH=1 \
CASRT_QWEN_ALIGNER_LOCAL_FILES_ONLY=1 \
CASRT_QWEN_ALIGNER_DISABLE_NETWORK=1 \
CASRT_ALIGNER_COMMAND='.casrt/qwen-asr-venv/bin/python -m custom_asmr_srt_stack.qwen_aligner_worker --model-id /path/to/Qwen3-ForcedAligner-0.6B/snapshot'
```

이 command는 `{ audio_file, master }`를 받아 speech segment별 clip을 만들고 `Qwen3ForcedAligner.align(audio, text, language)`로 segment 내부 start/end를 갱신한다. text, channel, kind는 변경하지 않는다. 실행은 local snapshot path, offline env scrub, network-disabled Python socket guard, `local_files_only=True`, `trust_remote_code=False` 조건에서만 허용한다. worker는 `CASRT_ALIGNER_ENV_MODE=offline`, `CASRT_QWEN_ALIGNER_REQUIRE_LOCAL_MODEL_PATH=1`, `CASRT_QWEN_ALIGNER_LOCAL_FILES_ONLY=1`, `CASRT_QWEN_ALIGNER_DISABLE_NETWORK=1`이 모두 없으면 실패한다. `qwen-asr` package version, RECORD hash, RECORD에 기록된 각 설치 파일 hash, `qwen_asr` import origin도 고정값과 다르면 실패한다.

Generic Qwen aligner worker는 두 가지 bounded fallback을 가진다. `CASRT_QWEN_ALIGNER_MIN_ALIGNED_DURATION_MS=80`보다 짧은 span은 비현실적인 timestamp로 보고 원래 segment timing을 유지한다. `CASRT_QWEN_ALIGNER_MIN_COVERAGE_RATIO=0.5`보다 원 segment coverage가 낮은 span도 과도한 trim으로 보고 원래 timing을 유지한다. 둘 다 UI/CLI 옵션으로 노출하지 않고 env 계약으로만 남긴다.

2026-06-30 정적 보안 재검토 결과:

- reviewer: `gpt-5.4 xhigh` subagent
- scope: `qwen_aligner_worker.py`, `alignment.py` offline env, tests, docs
- verdict: `PASS`
- 허용 조건: offline env, local path-only model id, `local_files_only=True`, `trust_remote_code=False`, Python socket network guard, `qwen-asr==0.0.6` RECORD hash, per-file RECORD hash, import origin 검증
- caution: Python socket guard는 OS-level egress control이 아니며, 실패 요약에는 local path 같은 운영 정보가 남을 수 있다.

2026-06-30 coverage guard 변경 정적 보안 재검토:

- reviewer: `gpt-5.4 xhigh` subagent
- scope: commit `9723036`, `CASRT_QWEN_ALIGNER_MIN_COVERAGE_RATIO` guard
- verdict: `PASS`
- 판단: 실행 경계, local path-only, `local_files_only`, `trust_remote_code=False`, socket guard, package/hash/import-origin 검증, no-traceback contract를 약화하지 않는다. env 값이 잘못되면 fail-open이 아니라 `ValueError`로 실패한다.

2026-06-30 실제 로딩/추론 smoke:

- command: `.casrt/qwen-asr-venv/bin/python -m custom_asmr_srt_stack.qwen_aligner_worker --model-id /home/brain-offloaded/.cache/huggingface/hub/models--Qwen--Qwen3-ForcedAligner-0.6B/snapshots/c7cbfc2048c462b0d63a45797104fc9db3ad62b7`
- env: `CASRT_ALIGNER_ENV_MODE=offline`, `CASRT_QWEN_ALIGNER_REQUIRE_LOCAL_MODEL_PATH=1`, `CASRT_QWEN_ALIGNER_LOCAL_FILES_ONLY=1`, `CASRT_QWEN_ALIGNER_DISABLE_NETWORK=1`
- model snapshot digest report: `/tmp/casrt-quality.Q5OdDf/qwen3-forced-aligner-snapshot-digest.json`
- model snapshot SHA-256: `5b0efb9cbc06d49988d4593c5d8bc52947ff0dfc20731e230dddb1fe0f8f2573`
- input: `/tmp/casrt-quality.Q5OdDf/01-front120.wav` + first segment from `ref-01-front120.master.json`
- result: `seg_000001` moved from `980-3800ms` to `1460-2660ms`
- output: `/tmp/casrt-quality.Q5OdDf/qwen-aligner-smoke-output.json`

기존 transcript를 benchmark용으로 재정렬할 때는 `align-transcript`를 사용한다.

```bash
CASRT_ALIGNER_COMMAND='.casrt/qwen-asr-venv/bin/python -m custom_asmr_srt_stack.qwen_aligner_worker --model-id /path/to/Qwen3-ForcedAligner-0.6B/snapshot' \
  uv run casrt align-transcript audio.wav candidate.master.json -o candidate.aligned.master.json --json
```

2026-06-30 `align-transcript` CLI smoke:

- input: `/tmp/casrt-quality.Q5OdDf/qwen-aligner-smoke-input.master.json`
- output: `/tmp/casrt-quality.Q5OdDf/qwen-aligner-smoke-cli-output.master.json`
- result: `seg_000001` moved from `980-3800ms` to `1460-2660ms`, matching direct worker smoke
- note: sandboxed run used `UV_CACHE_DIR=/tmp/casrt-uv-cache` to avoid default uv cache write errors.
- 2026-06-30 `--diagnostics-output` no-op real-data smoke: input `/tmp/casrt-quality.Q5OdDf/stable-ts-cli-attributed-quiet8/01-front120.master.json`, audio `/tmp/casrt-quality.Q5OdDf/01-front120.wav`, output `/tmp/casrt-quality.Q5OdDf/alignment-diagnostics-smoke/01-front120.aligned.master.json`, diagnostics `/tmp/casrt-quality.Q5OdDf/alignment-diagnostics-smoke/01-front120.alignment-diagnostics.json`. Result: `segments=25`, `changed_segments=0`, `review_flag_changes=0`, `max_boundary_delta_ms=0`; output master is byte-identical to input.

현재 판단:

- 기본 경로로 승격하지 않는다.
- 10초 실데이터 crop에서 일부 timestamp가 초반으로 잘 맞지 않고, token duration이 0인 항목이 있었다.
- 2026-06-30 01/04/07 front120에서는 text를 바꾸지 않고 time-aligned 500ms와 channel time-aligned accuracy를 개선했지만 practical CER가 여전히 높고 1ms span이 관측됐다.
- ForcedAligner는 duration guard 적용 후 다시 평가한다.

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
- speech text Japanese relaxed CER
- segment index 기준 mean start/end/boundary error
- time-overlap 기준 `timing_time_aligned` mean start/end/boundary error
- boundary sample 수, max boundary error, 250ms/500ms 이내 boundary ratio
- L/R/MIX channel confusion
- candidate MIX 유지 비율
- index 기준 `channel` 및 time-overlap 기준 `channel_time_aligned` L/R channel accuracy
- candidate `needs_review` 비율
- segment 단위 `review_effort`: practical text mismatch, channel mismatch, 500ms 초과 timing mismatch, missing reference, extra candidate
- case별 `review_effort.items`: human review와 heuristic 개선이 어느 segment를 봐야 하는지 알 수 있도록 reasons와 reference/candidate text/channel/timing을 보존한다.
- `casrt review-effort eval-suite.json --json -o review-effort.json`: suite/single report에서 `custom-asmr-review-effort-v1` 수정 큐를 추출한다. manifest case context와 timing delta를 보존하므로 다음 human review 또는 heuristic 개선 순서를 정하는 기본 산출물이다.
- `casrt review-pack review-effort.json --audio-map audio-map.json -o review-pack --json`: 수정 큐 item별 audio clip과 `custom-asmr-review-pack-v1` index를 만든다. pseudo-gold 비교에서 발견한 실패 구간을 사람이 빠르게 듣고 human-reviewed reference로 승격하기 위한 표준 산출물이다.

strict CER는 공백만 제거한다.

practical CER는 현재 다음을 정규화한다.

- Unicode NFKC
- 공백 제거
- punctuation/symbol 제거

practical CER는 자막 실용 비교용이다. 원문 보존 품질은 strict CER를 같이 본다.

Japanese relaxed CER는 practical CER에 더해 장음류 문자 `ー〜～`를 제거한다. ASMR 속삭임의 발화 길이 표기 차이를 분리해 보기 위한 보조 metric이며, 모델 승격 gate와 `review_effort`는 계속 practical CER를 사용한다.

2026-06-30 일본어 ASMR relaxed normalization 후보 실험:

- 후보: current practical에 더해 장음 부호 `ー〜～` 제거, 소형 kana를 대형 kana로 치환, 또는 둘 다 적용.
- 01/04/07 front120 pseudo-gold 기준 stable-ts CLI attributed quiet8: current 16.1%, 장음 제거 15.5%, 소형 kana 치환 16.1%, 둘 다 15.4%.
- Qwen HF ASR Transformers main: current 29.4%, 장음 제거 27.5%, 소형 kana 치환 29.4%, 둘 다 27.5%.
- 결정: 장음 제거는 ASMR 발화 길이 차이 노이즈를 줄이지만 실제 표기 차이까지 숨길 수 있다. 기존 품질 gate와 historical report의 의미를 바꾸지 않기 위해 current practical CER 기본값은 유지한다. `text_japanese_relaxed`를 별도 metric으로 추가하고 기본 gate로 쓰지는 않는다.
- 2026-06-30 구현 smoke: `/tmp/casrt-quality.Q5OdDf/stable-ts-cli-attributed-quiet8-3case-gold.json`를 새 report `/tmp/casrt-quality.Q5OdDf/stable-ts-cli-attributed-quiet8-3case-report-relaxed.json`로 재평가했다. Summary는 practical CER 16.1%, Japanese relaxed CER 15.5%다. 기존 report와 새 report를 같이 넣은 compare output `/tmp/casrt-quality.Q5OdDf/eval-comparison-old-new-relaxed.json`에서 old report의 `japanese_relaxed_cer`는 `null`, new report는 `0.1547`로 표시되어 기존 report 비교 호환성을 확인했다.

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
- unresolved candidate `needs_review`: 0%

자동 gate 예시:

```bash
uv run casrt eval-transcript ref.master.json candidate.master.json --product-gate
```

모델 승격용 manifest 평가는 reference authority도 gate로 강제한다.

```bash
uv run casrt eval-manifest gold.json --product-gate
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

확장 pseudo-gold 결과:

주의: 현재 01/04/07 front120 reference는 `/home/brain-offloaded/vscode/asmr/whisperx-webui/data/outputs/eval-csv-srt-*-full.srt`에서 만든 pseudo-gold다. 해당 CSV/SRT의 `source_backend`는 stable-ts이며, 사람 검수 ground truth가 아니다. 따라서 아래 수치는 실제 정확도라기보다 stable-ts 계열 pseudo-reference와의 일치도다. stable-ts CSV channel을 candidate로 다시 넣으면 practical CER 0%, timing 100%, channel 93.1%가 나오므로 이것은 benchmark leakage로 간주하고 모델 승격 근거로 쓰지 않는다.

| 후보 | cases | reference segments | candidate segments | practical CER | time-aligned 500ms ratio | channel time-aligned accuracy | 판단 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Qwen/Qwen3-ASR-1.7B, energy 500/200, 01/04/07 front120 | 3 | 74 | 81 | 29.5% | 29.5% | 73.1% | 불합격 |
| Qwen/Qwen3-ASR-1.7B + Qwen3-ForcedAligner, energy 500/200, 01/04/07 front120 | 3 | 74 | 81 | 29.5% | 36.6% | 75.0% | 불합격: timing/channel 개선, text 미달, 1ms span 관측 |
| Qwen/Qwen3-ASR-1.7B + Qwen3-ForcedAligner guard80, energy 500/200, 01/04/07 front120 | 3 | 74 | 81 | 29.5% | 36.1% | 75.0% | 불합격: 80ms 미만 span 제거, text 미달 |
| Qwen/Qwen3-ASR-1.7B, energy 500/200 + max10000, 01/04/07 front120 | 3 | 74 | 83 | 29.3% | 30.1% | 72.0% | 불합격: 미세 개선, channel 악화 |
| Qwen/Qwen3-ASR-1.7B, energy 500/200 + max6000, 01/04/07 front120 | 3 | 74 | 96 | 30.3% | 30.1% | 72.0% | 불합격: text 악화 |
| neosophie/Qwen3-ASR-1.7B-JA, 01/04/07 front120 | 3 | 74 | 81 | 29.6% | 29.5% | 73.1% | 불합격 |
| neosophie/Qwen3-ASR-1.7B-JA + ASMR ONNX VAD default, 01/04/07 front120 | 3 | 74 | 47 | 30.2% | 17.6% | 73.7% | 불합격: timing 악화 |
| neosophie/Qwen3-ASR-1.7B-JA + ASMR ONNX VAD t035-pad400-sil400, 01/04/07 front120 | 3 | 74 | 16 | 33.4% | 7.0% | 76.9% | 불합격: 과도한 병합 |
| neosophie/Qwen3-ASR-1.7B-JA + ASMR ONNX VAD hybrid rescue500, 01/04/07 front120 | 3 | 74 | 109 | 31.0% | 31.1% | 73.1% | 불합격: text 악화 |
| mistralai/Voxtral-Mini-4B-Realtime-2602, 01/04/07 front120 | 3 | 74 | 44 | 40.0% | 28.7% | 63.6% | 불합격 |
| google/gemma-4-E4B-it, 4-bit local-transformers, MIX-first, 01/04/07 front120 | 3 | 74 | 81 | 42.3% | 29.5% | 73.1% | 불합격 |
| google/gemma-4-E4B-it, 8-bit local-transformers, MIX-first, 01/04/07 front120 | 3 | 74 | 81 | 46.1% | 29.5% | 73.1% | 불합격 |
| zhifeixie/Mega-ASR, routed, MIX-first, 01/04/07 front120 | 3 | 74 | 68 | 30.9% | 28.4% | 69.6% | 불합격 |
| zhifeixie/Mega-ASR, base-only threshold 1.1, MIX-first, 01/04/07 front120 | 3 | 74 | 64 | 30.8% | 27.3% | 68.2% | 불합격 |
| zhifeixie/Mega-ASR, forced LoRA, MIX-first, 01/04/07 front120 | 3 | 74 | 81 | 77.6% | 29.5% | 73.1% | 불합격: LoRA가 ASMR에서 악화 |
| stable-ts baseline, 01/04/07 front120 | 3 | 74 | 60 | 16.1% | 56.7% | n/a | 불합격: text/timing 부족, MIX-only |

2026-06-30 `review_effort` 재평가:

- Qwen/Qwen3-ASR-1.7B + Qwen3-ForcedAligner, 01/04/07 front120: `review_effort.segments_needing_edit=77`, ratio 100.0%. pseudo-gold 기준에서도 전 구간 수정 대상이라 기본 승격 불가다. Report: `/tmp/casrt-quality.Q5OdDf/qwen17-align-review-effort-report.json`.
- stable-ts CSV channel leakage candidate: `review_effort.segments_needing_edit=2`, ratio 2.7%. 낮은 값은 같은 stable-ts 계열 reference와 candidate를 비교한 누수 결과이므로 모델 품질 근거가 아니고, reference authority gate 필요성을 확인하는 값이다. Report: `/tmp/casrt-quality.Q5OdDf/stable-ts-csv-channel-review-effort-report.json`.
- stable-ts CSV channel + secured Qwen3-ForcedAligner re-alignment: practical CER 0.0%, time-aligned 500ms ratio 62.8%, channel time-aligned accuracy 86.2%, `review_effort.segments_needing_edit=48`, ratio 64.9%. 같은 stable-ts 계열 pseudo-reference 기준으로는 timing/review_effort가 크게 악화되므로 기본 승격하지 않는다. Report: `/tmp/casrt-quality.Q5OdDf/stable-ts-csv-channel-qwen-aligner-3case-report.json`.
- 2026-06-30부터 `review_effort.items`는 `casrt review-effort`로 별도 JSON 추출한다. 이 산출물은 모델 승격 근거가 아니라, 사람이 볼 다음 수정 후보와 heuristic 실패 패턴을 고르는 작업 큐다.
- 2026-06-30 `casrt review-effort` 실제 추출:
  - Qwen/Qwen3-ASR-1.7B + Qwen3-ForcedAligner: `/tmp/casrt-quality.Q5OdDf/qwen17-align-review-effort-items.json`, `item_count=77`, `reason_counts={text: 70, timing: 65, channel: 36, missing_reference: 2, extra_candidate: 3}`. Text 오류가 대부분이므로 alignment만으로 구제할 수 없는 후보로 본다.
  - stable-ts CSV channel + Qwen3-ForcedAligner: `/tmp/casrt-quality.Q5OdDf/stable-ts-csv-channel-qwen-aligner-review-effort-items.json`, `item_count=48`, `reason_counts={timing: 47, text: 4, channel: 4}`. Text는 보존되지만 Qwen aligner가 segment span을 자주 줄여 pseudo-reference 기준 timing review가 크게 늘어난다.
- 2026-06-30 existing artifact 재계산에서 stable-ts CSV channel + Qwen aligner에 coverage fallback을 적용하면 threshold 0.5 기준 `review_effort` 48 -> 33, time-aligned 500ms 62.8% -> 75.7%로 개선된다. threshold 0.9는 `review_effort` 6, time-aligned 96.6%까지 올라가지만 원 timing을 대부분 보존하는 값이라 기본 guard로 쓰지 않는다. 제품 기본값은 과도한 trim만 막는 0.5다.
- 2026-06-30 실제 Qwen aligner coverage05 재실행:
  - candidate dir: `/tmp/casrt-quality.Q5OdDf/stable-ts-csv-channel-qwen-aligner-coverage05`
  - manifest: `/tmp/casrt-quality.Q5OdDf/stable-ts-csv-channel-qwen-aligner-coverage05-3case-gold.json`
  - report: `/tmp/casrt-quality.Q5OdDf/stable-ts-csv-channel-qwen-aligner-coverage05-3case-report.json`
  - review queue: `/tmp/casrt-quality.Q5OdDf/stable-ts-csv-channel-qwen-aligner-coverage05-review-effort-items.json`
  - result: practical CER 0.0%, time-aligned 500ms ratio 75.7%, channel time-aligned accuracy 89.7%, `review_effort.segments_needing_edit=33`, ratio 44.6%, `reason_counts={timing: 32, text: 3, channel: 3}`.
  - 판단: coverage guard는 실제 worker 실행에서도 over-trim을 줄였지만 stable-ts 원본 pseudo-reference의 `review_effort=2`, 500ms 99.3%보다 여전히 나쁘다. 따라서 stable-ts 계열 후보에는 Qwen aligner를 기본 적용하지 않는다.
- 2026-06-30 `casrt review-pack` 실제 생성:
  - audio map: `/tmp/casrt-quality.Q5OdDf/review-audio-map.json`
  - Qwen/Qwen3-ASR-1.7B + Qwen3-ForcedAligner pack: `/tmp/casrt-quality.Q5OdDf/review-pack-qwen17-align`, clips 77개.
  - stable-ts CSV channel + Qwen3-ForcedAligner pack: `/tmp/casrt-quality.Q5OdDf/review-pack-stable-ts-qwen-aligner`, clips 48개.
  - stable-ts CSV channel + Qwen3-ForcedAligner coverage05 pack: `/tmp/casrt-quality.Q5OdDf/review-pack-stable-ts-qwen-aligner-coverage05`, clips 33개.
  - stable-ts CLI attributed 6dB pack: `/tmp/casrt-quality.Q5OdDf/review-pack-stable-ts-cli-attributed`, clips 66개.
  - stable-ts CLI attributed 8dB quiet-side pack: `/tmp/casrt-quality.Q5OdDf/review-pack-stable-ts-cli-attributed-quiet8`, clips 64개.
  - stable-ts CLI attributed 10dB pack: `/tmp/casrt-quality.Q5OdDf/review-pack-stable-ts-cli-attributed-th10`, clips 61개.
  - 여섯 pack 모두 `custom-asmr-review-pack-v1` index와 `clips/*.wav` 생성을 확인했다.
  - priority queue pack: input `/tmp/casrt-quality.Q5OdDf/stable-ts-cli-attributed-quiet8-review-effort-priority.json`, output `/tmp/casrt-quality.Q5OdDf/review-pack-stable-ts-cli-attributed-quiet8-priority`, clips 64개. `priority_rank=1`이 `clips/000001__01-front120__text__seg_000003__seg_000004.wav`, `priority_rank=64`가 `clips/000064__04-front120__channel__seg_000002__seg_000002.wav`로 들어가 review-effort 우선순서와 score/rank가 pack index에 보존됨을 확인했다.
- 2026-06-30 WebUI review-pack viewer smoke:
  - server: `uv run casrt serve --port 5174`
  - load API input: `/tmp/casrt-quality.Q5OdDf/review-pack-stable-ts-cli-attributed-quiet8-priority`
  - result: `clip_count=64`, first item `priority_rank=1`, `clip_url` 생성 확인.
  - first clip GET: `200 audio/wav 1962284 bytes` for `clips/000001__01-front120__text__seg_000003__seg_000004.wav`.
  - 판단: pack 생성은 CLI가 담당하고 WebUI는 local path로 priority 검수 큐를 열어 clip을 재생하는 viewer 역할만 한다. 추가 threshold/model 옵션은 노출하지 않는다.
- 2026-06-30 WebUI review case set smoke:
  - server: `uv run casrt serve --port 5174`
  - load API input: `/tmp/casrt-quality.Q5OdDf/all8-front120-review-cases`
  - result: `kind=review-case-set`, `case_count=8`, first case `id=01-front120-existing-srt`, `segments=10`, `review_count=2`.
  - first case audio GET: `200 audio/wav 23040044 bytes`.
  - 동작: case list는 전체 `needs_review` flag 수와 flag가 남은 case를 표시한다. Case click은 reference master를 기존 segment editor에 붙이고, edit/save는 reference master JSON과 `case-index.json` count를 갱신한다. `검수 완료`는 현재 `needs_review` segment를 false로 바꾸고 다음 검수 segment로 이동한다. `case 목록`/`다음 case` 이동은 pending save를 flush한다.
  - 판단: human-reviewed gold 제작을 돕는 편집 경로이며, 검수 완료 판정은 하지 않는다.
- 2026-06-30 real SRT import audit:
  - `/home/brain-offloaded/vscode/asmr/whisperx-webui/data/outputs/02.敗北確定えっちバトル-c28f819996c9400cb05ec6ccbea1849f.srt` 같은 실제 산출물은 `[L:SPEAKER_01]`, `[R:SPEAKER_00]` prefix를 사용한다.
  - SRT import는 compound channel/speaker label을 본문 text에서 제거하고 channel metadata만 보존하도록 확장했다. `[SPEAKER_00]` 단독 prefix도 번역 대상 text에서 제거한다.
  - smoke output: `/tmp/casrt-quality.Q5OdDf/real-02-compound-label-smoke.master.json`; `rg "SPEAKER|\\[L|\\[R|\\[MIX|\\[LR"` 결과 없음, 첫 segment `channel=L`, 둘째 segment `channel=R`.
  - 목적: 외부 번역용 JSON에 speaker/channel label이 섞이지 않게 유지한다.

case별 practical CER:

| 후보 | case | practical CER | time-aligned 500ms ratio | channel time-aligned accuracy |
| --- | --- | ---: | ---: | ---: |
| Qwen3-ASR 1.7B energy 500/200 | 01-front120 | 20.4% | 25.0% | 66.7% |
| Qwen3-ASR 1.7B energy 500/200 | 04-front120 | 21.2% | 37.0% | 60.0% |
| Qwen3-ASR 1.7B energy 500/200 | 07-front120 | 52.7% | 26.9% | 80.0% |
| Qwen3-ASR 1.7B + ForcedAligner | 01-front120 | 21.7% | 31.2% | 83.3% |
| Qwen3-ASR 1.7B + ForcedAligner | 04-front120 | 20.4% | 43.2% | 50.0% |
| Qwen3-ASR 1.7B + ForcedAligner | 07-front120 | 51.8% | 36.0% | 78.6% |
| Qwen3-ASR 1.7B + ForcedAligner guard80 | 01-front120 | 21.7% | 31.2% | 80.0% |
| Qwen3-ASR 1.7B + ForcedAligner guard80 | 04-front120 | 20.4% | 43.2% | 50.0% |
| Qwen3-ASR 1.7B + ForcedAligner guard80 | 07-front120 | 51.8% | 34.6% | 80.0% |
| Qwen3-ASR 1.7B max10000 | 01-front120 | 21.3% | 27.1% | 60.0% |
| Qwen3-ASR 1.7B max10000 | 04-front120 | 20.4% | 37.0% | 60.0% |
| Qwen3-ASR 1.7B max10000 | 07-front120 | 51.8% | 26.9% | 80.0% |
| Qwen3-ASR 1.7B max6000 | 01-front120 | 21.9% | 20.8% | 60.0% |
| Qwen3-ASR 1.7B max6000 | 04-front120 | 22.7% | 41.3% | 60.0% |
| Qwen3-ASR 1.7B max6000 | 07-front120 | 51.5% | 28.8% | 80.0% |
| Neosophie | 01-front120 | 20.4% | 25.0% | 66.7% |
| Neosophie | 04-front120 | 21.3% | 37.0% | 60.0% |
| Neosophie | 07-front120 | 53.0% | 26.9% | 80.0% |
| Neosophie + ASMR ONNX VAD default | 01-front120 | 20.4% | 17.4% | 75.0% |
| Neosophie + ASMR ONNX VAD default | 04-front120 | 22.0% | 17.4% | 66.7% |
| Neosophie + ASMR ONNX VAD default | 07-front120 | 54.3% | 18.2% | 75.0% |
| Neosophie + ASMR ONNX VAD t035-pad400-sil400 | 01-front120 | 28.2% | 8.3% | n/a |
| Neosophie + ASMR ONNX VAD t035-pad400-sil400 | 04-front120 | 20.4% | 4.3% | n/a |
| Neosophie + ASMR ONNX VAD t035-pad400-sil400 | 07-front120 | 57.6% | 8.3% | 76.9% |
| Neosophie + ASMR ONNX VAD hybrid rescue500 | 01-front120 | 22.1% | 28.0% | 57.1% |
| Neosophie + ASMR ONNX VAD hybrid rescue500 | 04-front120 | 21.8% | 39.1% | 75.0% |
| Neosophie + ASMR ONNX VAD hybrid rescue500 | 07-front120 | 55.2% | 26.9% | 80.0% |
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

| attribution setting | practical CER | time-aligned 500ms ratio | channel time-aligned accuracy | comparable segments | candidate MIX ratio |
| ---: | ---: | ---: | ---: | ---: | ---: |
| threshold-only 3dB | 16.1% | 56.7% | 68.2% | 22 | 17.9% |
| threshold-only 6dB | 16.1% | 56.7% | 65.0% | 20 | 23.9% |
| threshold-only 10dB | 16.1% | 56.7% | 76.9% | 13 | 58.2% |
| 8dB + quiet <= -40dBFS | 16.1% | 56.7% | 68.8% | 16 | 40.3% |

2026-06-30 `casrt attribute-channels` 재현:

- threshold-only 6dB historical run: input dir `/tmp/casrt-quality.Q5OdDf/stable-ts-cli-mix`, output dir `/tmp/casrt-quality.Q5OdDf/stable-ts-cli-attributed`, report `/tmp/casrt-quality.Q5OdDf/stable-ts-cli-attributed-3case-report.json`. Result: practical CER 16.1%, time-aligned 500ms 56.7%, channel time-aligned accuracy 65.0%, candidate MIX ratio 23.9%, review effort 66/74, channel edits 41.
- 10dB sweep: output dir `/tmp/casrt-quality.Q5OdDf/stable-ts-cli-attributed-th10`, report `/tmp/casrt-quality.Q5OdDf/stable-ts-cli-attributed-th10-3case-report.json`. Result: practical CER 16.1%, time-aligned 500ms 56.7%, channel time-aligned accuracy 76.9%, candidate MIX ratio 58.2%, review effort 61/74, channel edits 28.
- 8dB + quiet-side -40dBFS default: output dir `/tmp/casrt-quality.Q5OdDf/stable-ts-cli-attributed-quiet8`, report `/tmp/casrt-quality.Q5OdDf/stable-ts-cli-attributed-quiet8-3case-report.json`, review queue `/tmp/casrt-quality.Q5OdDf/stable-ts-cli-attributed-quiet8-review-effort-items.json`, review pack `/tmp/casrt-quality.Q5OdDf/review-pack-stable-ts-cli-attributed-quiet8`. Result: practical CER 16.1%, time-aligned 500ms 56.7%, channel time-aligned accuracy 68.8%, candidate MIX ratio 40.3%, review effort 64/74, channel edits 36.
- 결정: 현재 기본값은 8dB + quiet-side -40dBFS gate다. 기존 6dB threshold-only보다 review effort가 66 -> 64로 줄었고 MIX ratio 40.3%로 50% gate 안에 남는다. 10dB threshold-only는 wrong L/R를 더 줄이지만 MIX ratio가 50% gate를 넘으므로 기본 승격하지 않는다.
- 2026-06-30 `--diagnostics-output` smoke: output dir `/tmp/casrt-quality.Q5OdDf/stable-ts-cli-attributed-quiet8-diagnostics`. 01/04/07 front120 MIX master에 대해 diagnostics JSON을 생성했고 attributed master는 기존 quiet8 output과 byte-identical이다. Reason counts는 `below_threshold=23`, `left_dominant=14`, `right_dominant=19`, `quieter_side_active=4`다.
- 2026-06-30 `compare-evals` smoke: outputs `/tmp/casrt-quality.Q5OdDf/eval-comparison-current.json` and gated `/tmp/casrt-quality.Q5OdDf/eval-comparison-current-gated.json`. Compared Qwen HF ASR, stable-ts quiet8, stable-ts 10dB, stable-ts + Qwen aligner reports. Ranking by review effort put 10dB first (`61/74`, ratio 82.4%), quiet8 second (`64/74`, ratio 86.5%), Qwen aligner third (`69/74`, ratio 93.2%), Qwen HF ASR fourth (`75/75`, ratio 100%). With product gates, all candidates fail; 10dB additionally fails MIX ratio gate at 58.2%, so default remains quiet8.
- 2026-06-30 Japanese relaxed CER 포함 재비교: output `/tmp/casrt-quality.Q5OdDf/eval-comparison-current-relaxed-gated.json`. Ranking은 기존과 동일하게 10dB, quiet8, Qwen aligner, Qwen HF 순서다. Japanese relaxed CER는 stable-ts 계열 15.5%, Qwen HF 27.5%로 practical CER보다 낮지만, 모든 후보가 practical CER, time-aligned 500ms, channel accuracy, review effort gate를 실패한다. 10dB는 MIX ratio 58.2%도 실패하므로 default는 계속 quiet8이다.
- 2026-06-30 unresolved candidate review gate 포함 재비교: output `/tmp/casrt-quality.Q5OdDf/eval-comparison-current-relaxed-review-gated.json`. `--max-candidate-review-ratio 0.00`을 추가해도 ranking은 10dB, quiet8, Qwen aligner, Qwen HF 순서로 유지된다. stable-ts 계열은 `candidate_review_ratio=0.0`이지만 기존 product gate를 실패한다. Qwen HF ASR는 `candidate_review_ratio=1.0`이라 timestamp/alignment 미확정 후보로도 실패한다.
- 2026-06-30 `--product-gate` preset smoke: output `/tmp/casrt-quality.Q5OdDf/eval-comparison-current-product-gate.json`. Ranking은 기존과 동일하며 모든 후보가 `reference_type 'pseudo-gold' != 'human-reviewed'`를 포함해 실패한다. stable-ts 계열은 candidate review gate는 통과하지만 practical CER, timing, channel/review-effort gate를 실패하고, Qwen HF는 candidate review ratio 100%도 함께 실패한다.
- 2026-06-30 priority review queue smoke: input `/tmp/casrt-quality.Q5OdDf/stable-ts-cli-attributed-quiet8-3case-report-relaxed.json`, output `/tmp/casrt-quality.Q5OdDf/stable-ts-cli-attributed-quiet8-review-effort-priority.json`. Result: `sort=priority_score_desc`, `item_count=64`, `reason_counts={text:46, channel:36, timing:41, missing_reference:7}`. Top item은 `01-front120` `seg_000003` vs `seg_000004`, reasons `text/channel/timing`, score `6590.22`로 사람이 먼저 들을 큰 복합 실패를 큐 상단에 올렸다.
- 2026-06-30 `sweep-channel-attribution` smoke: input manifest `/tmp/casrt-quality.Q5OdDf/stable-ts-cli-mix-3case-gold.json`, audio map `/tmp/casrt-quality.Q5OdDf/review-audio-map.json`, output `/tmp/casrt-quality.Q5OdDf/channel-sweep-smoke-guard`. Settings: 8dB/-40dBFS changed 33/60 segments, review effort 64/74, channel time-aligned accuracy 68.8%, MIX ratio 40.3%; 10dB/-40dBFS changed 20/60, review effort 60/74, channel time-aligned accuracy 75.0%, MIX ratio 62.7%. 10dB+quiet lowers edit count but violates the 50% MIX ratio gate, so default remains 8dB+quiet-side -40dBFS.
- 2026-07-01 `sweep-channel-attribution --product-gate` smoke: input manifest `/tmp/casrt-quality.Q5OdDf/stable-ts-cli-mix-3case-gold.json`, audio map `/tmp/casrt-quality.Q5OdDf/review-audio-map.json`, output `/tmp/casrt-channel-sweep-gate.OcMn59/sweep`. `comparison.json`과 `index.json`에 `quality_gate.preset=local-asmr-v1`가 저장된다. 두 setting 모두 `reference_type=pseudo-gold`와 text/timing/edit gate 때문에 실패한다. th8/-40은 channel accuracy 68.8%지만 MIX ratio 40.3%로 MIX gate는 통과한다. th10/-40은 channel accuracy 75.0%와 edit ratio 81.1%로 상대 개선이 있으나 MIX ratio 62.7%가 gate를 깨므로 기본값은 계속 8dB+quiet-side -40dBFS다.
- Qwen3-ForcedAligner를 6dB `stable-ts-cli-attributed` 후보에 적용한 실험은 output dir `/tmp/casrt-quality.Q5OdDf/stable-ts-cli-attributed-qwen-aligner`, report `/tmp/casrt-quality.Q5OdDf/stable-ts-cli-attributed-qwen-aligner-3case-report.json`에 있다. Result: practical CER 16.1%, time-aligned 500ms 47.0%, channel time-aligned accuracy 65.0%, candidate MIX ratio 25.8%, review effort 69/74, timing edits 51. 원 stable-ts CLI attributed의 timing/review effort보다 나빠 기본 경로로 쓰지 않는다.

window 단위 dominant fraction attribution도 01/04/07 front120 stable-ts baseline에서 실험했다. 100ms window, active threshold -60dBFS, margin 1~10dB, dominant fraction 35~75% sweep 기준 최고 channel time-aligned accuracy는 71.4%였고, segment 전체 RMS 10dB 방식의 76.9%보다 낮았다. 따라서 window 방식은 기본 구현으로 승격하지 않는다.

결정:

- Qwen 내장 energy 기본값은 `min_silence_ms=500`, `pad_ms=200`으로 낮춘다.
- `CASRT_QWEN_ENERGY_MAX_CHUNK_MS`는 추가했지만 기본값으로 켜지 않는다. `max10000`은 official Qwen 3-case에서 practical CER 29.5% -> 29.3%, time-aligned 500ms 29.5% -> 30.1%로 미세 개선했지만 channel accuracy가 73.1% -> 72.0%로 떨어졌다. `max6000`은 practical CER 30.3%로 악화됐다.
- `CASRT_QWEN_ASR_CONTEXT`에 긴 glossary를 그대로 넣는 방식은 기본값으로 쓰지 않는다. 짧은 구간에서 glossary 전체를 출력하는 hallucination이 발생했다.
- Qwen3-ForcedAligner는 official Qwen 3-case에서 text를 바꾸지 않고 time-aligned 500ms 29.5% -> 36.6%, channel time-aligned 73.1% -> 75.0%로 개선했다. `CASRT_QWEN_ASR_MIN_ALIGNED_DURATION_MS=80` guard 적용 후 80ms 미만 span은 제거됐고 time-aligned 500ms는 36.1%, channel time-aligned는 75.0%다. practical CER 29.5%가 여전히 기준 미달이므로 text 병목은 별도 모델/전처리/후처리로 풀어야 한다.
- 2026-06-30 audit에서 01/04/07 front120 reference가 stable-ts 기반 pseudo-gold임을 확인했다. `eval-manifest`는 `reference_type`과 `reference_notes`를 report에 보존한다. 제품 기본 모델 승격은 `reference_type=human-reviewed` gold에서 다시 판단해야 한다.
- 현재 Qwen3-ASR 1.7B 경로만으로는 품질 기준을 만족하지 못한다.
- `neosophie/Qwen3-ASR-1.7B-JA`는 다운로드 재시도 후 점수화했다. 120초 gold 기준 Qwen3-ASR 1.7B보다 약간 낫지만 practical CER 20.4%라 기본 승격하지 않는다.
- 01/04/07 front120 확장 gold에서도 Neosophie/Qwen3-ASR-JA는 practical CER 29.6%라 기본 승격하지 않는다. 특히 07의 whisper/침대 ASMR 구간에서 텍스트 인식이 크게 무너졌다.
- Neosophie full-window와 1.5초 silence 병합 실험은 text와 timing이 모두 악화됐다. 이 샘플에서는 chunk를 길게 잡는 것이 해결책이 아니다.
- `Qwen/Qwen3-ASR-1.7B-hf`는 Hugging Face metadata상 `automatic-speech-recognition`, `ja` 지원, `transformers` 모델이다. 2026-06-30 root Transformers 5.12.1에서는 `qwen3_asr` 아키텍처를 인식하지 못해 fail closed했고, Transformers main `5.13.0.dev0` commit `45b004d7bb505a258542d1965b0f9e0d8b03b89d` venv에서 5초 smoke와 01/04/07 front120 benchmark를 완료했다. 결과는 practical CER 29.4%, time-aligned 500ms 27.3%, channel time-aligned 68.2%, review effort 75/75라 기본 승격하지 않는다.
- `mistralai/Voxtral-Mini-4B-Realtime-2602`는 remote model code 없이 Transformers `VoxtralRealtimeForConditionalGeneration`으로 로딩됐다. 8.9GB weight는 단일 HF stream이 느려 HTTP range 8조각 병렬 다운로드로 확보했다. 30초 smoke와 01/04 일부 텍스트는 Qwen보다 자연스러웠지만, 07 whisper/침대 ASMR에서 chunked 입력은 대부분 빈 출력이었고 120초 full-window 입력도 앞부분만 출력해 기본 승격하지 않는다.
- `mistralai/Voxtral Mini Transcribe 2.0`는 Mistral API batch transcription 제품으로 확인됐고 open-weight 로컬 checkpoint는 확인하지 못했다. 외부 API는 제품 방향이 아니므로 기본 경로에서 제외한다.
- `google/gemma-4-E4B-it`는 공식 오디오 입력을 지원하고 5초 smoke에서 유의미한 전사를 반환했다. 그러나 01/04/07 front120 확장 gold에서 4-bit practical CER 42.3%, 8-bit practical CER 46.1%로 기준을 크게 벗어났다. 8-bit는 01 smoke와 01 case를 조금 개선했지만 07 whisper/침대 ASMR에서 반복 hallucination이 발생해 전체 지표가 악화됐다. 따라서 기본 승격하지 않는다.
- Gemma E4B 실험 산출물은 `/tmp/casrt-quality/gemma-e4b-4bit-bounded-results`, `/tmp/casrt-quality/gemma-e4b-8bit-bounded-results`, report는 `/tmp/casrt-quality/gemma-e4b-4bit-bounded-3case-report.json`, `/tmp/casrt-quality/gemma-e4b-8bit-bounded-3case-report.json`에 있다.
- `CohereLabs/cohere-transcribe-03-2026`는 2026년 2B local ASR 후보이며 일본어 포함 14개 언어를 지원한다. 공식 card는 Transformers native, safetensors, no timestamps/diarization, VAD 필요를 명시한다. Root Transformers 5.12.1에 Cohere ASR class가 있어 `local-cohere-asr` adapter를 구현한다. 다만 gated/custom_code repo이므로 실제 download/evaluation은 exact revision pin과 `casrt model digest` report 기록 후, local snapshot path + `trust_remote_code=False` + `local_files_only=True` + `use_safetensors=True` 조건에서만 한다.
- Cohere exact revision은 `b1eacc2686a3d08ceaae5f24a88b1d519620bc09`로 확인했다. `model.safetensors` LFS SHA-256은 `987bd3e141c7bfdb5a78f5db11397ee7737308357e6cc0a3f36a4979b158137a`, size는 4,131,862,976 bytes다. 2026-06-30 anonymous download는 gated 403으로 실패했다. 사용자가 HF access를 승인한 뒤 같은 revision을 받아 `casrt model digest`를 기록해야 평가 가능하다.
- 2026-06-30 live HF metadata refresh:
  - `microsoft/VibeVoice-ASR`는 `automatic-speech-recognition`, `transformers`, `safetensors`, `ja` tag가 있고 exact revision은 `d0c9efdb8d614685062c04425d91e01b6f37d944`다. Config architecture는 `VibeVoiceForASRTraining`, model type은 `vibevoice`, BF16 parameter count는 8.67B다. 현재 repo env의 Transformers 5.12.1에는 `VibeVoiceForASRTraining`가 없어 바로 실행하지 않는다.
  - `microsoft/VibeVoice-ASR-HF`는 `audio-text-to-text`, `transformers`, `safetensors`, `ja` tag가 있고 exact revision은 `f22241c2062b3b25272bf117397e03d73381037a`다. HF metadata상 `AutoModel`을 가리키지만 현재 repo env에는 VibeVoice 전용 audio-text-to-text class가 없어 바로 실행하지 않는다.
  - `OpenMOSS-Team/MOSS-Transcribe-preview-2B`는 2026-06-26 공개된 2.4B safetensors ASR 후보이고 exact revision은 `c98175cb20e48bd9be4e95f6c85f2af18899f780`다. 그러나 metadata에 `custom_code`가 있고 language tag가 `en` 중심이라 일본 ASMR 우선순위는 낮다. 실행하려면 외부 model code 검토가 먼저 필요하다.
  - `cstr/MOSS-Transcribe-preview-2B-GGUF`는 2026-06-30 공개 GGUF 변환이고 language tag는 `en`, `zh`다. 일본어 tag가 없고 GGUF runtime은 별도 실행 경로가 필요하므로 현재 로컬 ASMR 우선 후보가 아니다.
  - `XiaomiMiMo/MiMo-V2.5-ASR`는 2026-04-24 revision `98641d537df521ac6df05f74090475694d9510b7`의 ASR 후보지만 language tag가 `zh`, `en`, `yue`이고 일본어 tag가 없다. 일본 ASMR 후보 우선순위에서 제외한다.
- 2026-07-01 live HF metadata refresh:
  - `ibm-granite/granite-speech-4.1-2b`: revision `de575db64086f84fdc79da4932d1076e965bc546`, tags `transformers`, `safetensors`, `granite_speech`, `automatic-speech-recognition`, `ja`, license Apache-2.0. Model card는 2026-04-29 release, Japanese ASR support, native `transformers>=4.52.1`, and Japanese-tailored synthetic data를 명시한다. 현재 repo env Transformers 5.12.1에서 `transformers.models.granite_speech`와 `AutoModelForSpeechSeq2Seq` import가 가능해 `local-granite-asr` adapter를 추가했다. Persistent cache는 `.casrt/models/granite-speech-4.1-2b-de575db64086f84fdc79da4932d1076e965bc546`, digest report는 `.casrt/model-digests/granite-speech-4.1-2b-de575db64086f84fdc79da4932d1076e965bc546-digest.json`, snapshot SHA-256은 `67c7d69184b53bae7a2bec077fbc88d8695a72f043fd70831f4e4830dc4752ca`다. 실제 evaluation은 이 exact local snapshot digest 기준으로 수행한다.
  - `local-granite-asr`는 Granite Plus timestamp prompt도 같은 worker로 실험한다. `CASRT_GRANITE_ASR_PARSE_TIMESTAMPS=1`이면 `[T:N]` centisecond tag를 unwrap해 speech segment timing으로 쓰고 `_` silence marker로 segment를 split/trim한다. Tag가 없으면 기존 chunk-bound segment로 남기고 `needs_review=true`는 유지한다. 이 env는 내부 실험 경로이고 WebUI/CLI 옵션으로 노출하지 않는다.
  - Granite runtime note: `AutoProcessor` 생성 시 `GraniteSpeechFeatureExtractor`가 `torchaudio`를 요구한다. 첫 실제 smoke는 model load 후 `torchaudio` missing으로 실패했고, `local` extra에 `torchaudio`를 추가한 뒤 `torch 2.12.1+cu130` / `torchaudio 2.11.0+cu130` import smoke와 실제 전사를 통과했다.
  - 2026-07-01 Granite 10초 smoke: input `.casrt/experiments/upload-real-crop/01-front10.wav`, project root `.casrt/experiments/granite-smoke/projects`, project `52cf1cc9379a484e97cb866a3ec48399`, command env `CASRT_LOCAL_WORKER_ENV_MODE=offline CASRT_GRANITE_ASR_MAX_NEW_TOKENS=128`. Result: 3 speech segments, first text `やばい、見つかっちゃった`, all `needs_review=true`.
  - 2026-07-01 Granite 01/04/07 front120 pseudo-gold benchmark before non-Japanese hallucination filter: regenerated references/audio under `.casrt/experiments/granite-front120-eval`, manifest `.casrt/experiments/granite-front120-eval/granite-front120-3case-manifest.json`, report `.casrt/experiments/granite-front120-eval/reports/granite-front120-3case-report.json`. Summary: reference segments 60, candidate segments 69, practical CER 24.7%, Japanese relaxed CER 23.8%, time-aligned 500ms ratio 23.7%, candidate MIX ratio 54.4%, candidate review ratio 100%, review effort 67 segments / 100%.
  - 2026-07-01 Granite non-Japanese hallucination filter benchmark: manifest `.casrt/experiments/granite-front120-eval/granite-filtered-front120-3case-manifest.json`, report `.casrt/experiments/granite-front120-eval/reports/granite-filtered-front120-3case-report.json`. Summary: reference segments 60, candidate segments 61, practical CER 23.6%, Japanese relaxed CER 22.5%, time-aligned 500ms ratio 21.8%, candidate MIX ratio 56.4%, candidate review ratio 100%, review effort 62 segments / 100%. The filter removed punctuation-only/English-only hallucinations and slightly improved text/review effort, but `--product-gate` still failed on practical CER, timing, unavailable L/R channel accuracy, MIX ratio, review effort, and unresolved review flags. Granite is useful as a 2026 local candidate but does not beat the current best practical baseline and is not promoted.
  - 2026-07-01 Granite filtered + Qwen3-ForcedAligner benchmark: manifest `.casrt/experiments/granite-front120-eval/granite-filtered-qwen-aligner-front120-3case-manifest.json`, report `.casrt/experiments/granite-front120-eval/reports/granite-filtered-qwen-aligner-front120-3case-report.json`, aligner diagnostics under `.casrt/experiments/granite-front120-eval/reports/*qwen-aligner.diagnostics.json`. Summary: reference segments 60, candidate segments 61, practical CER 23.6%, Japanese relaxed CER 22.5%, time-aligned 500ms ratio 32.7%, candidate MIX ratio 54.5%, candidate review ratio 100%, review effort 62 segments / 100%. Diagnostics changed 18/22, 19/21, and 15/18 segments for cases 01/04/07, with max boundary deltas 1140ms, 2060ms, and 1420ms. The aligner is useful for timing but does not change text/channel uncertainty and is not enough to promote Granite.
  - `ibm-granite/granite-speech-4.1-2b-plus`: revision `1454e6e1e33845ca9280ff65f52cf1141ba6e6e2`, tags `transformers`, `safetensors`, `granite_speech_plus`, multilingual ASR지만 HF card language metadata에 `ja`가 없다. Local snapshot은 `.casrt/models/granite-speech-4.1-2b-plus-1454e6e1e33845ca9280ff65f52cf1141ba6e6e2`, digest report는 `.casrt/model-digests/granite-speech-4.1-2b-plus-1454e6e1e33845ca9280ff65f52cf1141ba6e6e2-digest.json`, snapshot SHA-256은 `1ef78d5809fbf87e6d2b7cad64aab4b7cb35a460683f988db09e83f216644326`다.
  - 2026-07-01 Granite Plus 10초 prompt sweep: input `.casrt/experiments/upload-real-crop/01-front10.wav`, output `.casrt/experiments/granite-plus-smoke/01-front10-prompt-sweep.json`. Card timestamp prompt는 3 segments, raw `_ [T:90]いやば [T:146] ...`, 900~5190ms까지만 유의미했다. Japanese timestamp prompt도 3 segments, 0~3860ms까지만 유의미했다. Plain ASR prompt는 1 segment로 전체 10초를 뭉갰다.
  - 2026-07-01 Granite Plus timestamp 01/04/07 front120 benchmark: manifest `.casrt/experiments/granite-front120-eval/granite-plus-ts-front120-3case-manifest.json`, report `.casrt/experiments/granite-front120-eval/reports/granite-plus-ts-front120-3case-report.json`, product gate report `.casrt/experiments/granite-front120-eval/reports/granite-plus-ts-front120-3case-product-gate-report.json`. Summary: reference segments 60, candidate segments 94, practical CER 84.1%, Japanese relaxed CER 35.3%, time-aligned 500ms ratio 22.2%, candidate MIX ratio 53.7%, candidate review ratio 100%, review effort 68 segments / 98.6%. Timestamp tags produce more segment boundaries, but text hallucination/repetition and over-fragmentation make it worse than Granite base and Qwen aligner. Do not promote.
  - `efwkjn/cohere-asr-ja`: revision `8f1794e22b802731bdbf8ce53ff08f96a5af2bb4`, tags `safetensors`, `cohere_asr`, `custom_code`, `ja`, base model `CohereLabs/cohere-transcribe-03-2026`. Current Transformers 5.12.1 has `cohere_asr`, but metadata includes `custom_code`; execution priority는 official Cohere snapshot과 Granite 이후로 둔다.
  - `AutoArk-AI/ARK-ASR-3B`: revision `1e28271b79edc97635783bea65abc89195a09ed3`, tags include `ja`, `safetensors`, `custom_code`; current Transformers 5.12.1 has no `arkasr`, so still external code/runtime review 대상이다.
- `zhifeixie/Mega-ASR`는 2026-05 공개 Qwen3-ASR-1.7B 기반 robust ASR 후보이며, noisy/reverberant/clipped/band-limited/overlapping 등 어려운 실제 녹음에서 empty output, omission, repetition, hallucination을 줄이는 것을 목표로 한다. ASMR 전용은 아니지만 현재 07 whisper/침대 구간 실패 양상과 맞닿아 있으므로 다음 우선 모델 실험으로 둔다. 공식 runtime은 `xzf-thu/Mega-ASR` repository 코드와 checkpoint 배치를 요구하므로 `/tmp` 격리 환경에서 실행한다.
- Mega-ASR runtime은 실행 전 `gpt-5.4 xhigh` subagent가 정적 보안 검토했다. Verdict는 `PASS_WITH_CONSTRAINTS`다. 허용 범위는 `/tmp` 별도 venv, `/tmp` HF cache, Hugging Face allowlist download, Transformers backend만, `infer.py`/`evaluate_wer.py`만, vLLM/webui/training/wandb 금지, safetensors-only checkpoint 강제다. `adapter_model.bin`, `.pt`, `.pth` 또는 router non-safetensors checkpoint를 읽게 되면 미승인으로 간주한다.
- Mega-ASR 정적 검토에서 확인한 고위험 지점은 `lora_switch.py`의 `adapter_model.bin` fallback `torch.load`와 `router.py`의 non-safetensors `torch.load(weights_only=False)`다. 따라서 실험 전 `mega-asr-merged/adapter_model.safetensors`와 `audio_quality_router/best_acc_model.safetensors` 존재를 확인하고 unsafe fallback 파일은 사용하지 않는다.
- Mega-ASR는 `/tmp/casrt-quality/mega-asr-venv`, `HF_HOME=/tmp/casrt-quality/hf-home-mega-asr`, offline env에서 점수화했다. Checkpoint에는 unsafe pickle fallback 대상 파일이 없고, `adapter_model.safetensors`와 `best_acc_model.safetensors`를 확인했다.
- Mega-ASR 5초 smoke는 `やば、見つかっちゃった。`를 반환했고 router는 `use_lora=False`, degraded probability 0.0559였다. 그러나 01/04/07 front120 routed practical CER는 30.9%라 Neosophie 29.6%보다 약간 낮고 기준에 크게 못 미친다. threshold 1.1로 base-only에 가깝게 만든 경로도 30.8%로 유의미한 개선이 없다. forced LoRA는 77.6%로 크게 악화되어 ASMR 기본 경로로 쓰지 않는다.
- Mega-ASR 산출물은 `/tmp/casrt-quality/mega-asr-results/routed`, `/tmp/casrt-quality/mega-asr-results/base-threshold-1p1`, `/tmp/casrt-quality/mega-asr-results/force-lora`에 있다. Report는 각각 `routed-3case-report.json`, `base-threshold-1p1-3case-report.json`, `force-lora-3case-report.json`이다.
- `Atotti/llm-jp-4-8b-speech-asr`는 일본어 ASR 특화 8B 후보지만 model card상 `speech_llm_ja` 패키지(`git+https://github.com/Atotti/ja-speech-llm.git`)가 필요하다. 현재 설치된 Transformers `5.12.1`와 official main `5.13.0.dev0` 모두 `LlamaForSpeechLM`을 노출하지 않는다. 원격/외부 패키지 코드를 실행해야 하므로 사용자 명시 승인 전에는 자동 검증하지 않는다.
- `AutoArk-AI/ARK-ASR-3B`는 최신 로컬 후보지만 model card metadata에 `custom_code`가 있다. 외부 모델 저장소 코드를 실행하는 `trust_remote_code=True`는 기본 실험 경로로 쓰지 않고, 사용자 명시 승인이나 first-party package 지원이 있을 때만 검증한다.
- stable-ts/Whisper계 baseline은 현재 후보 중 text가 가장 좋지만 3-case practical CER 16.1%로 기준 10%를 넘고, time-aligned 500ms ratio도 56.7%로 기준 90%에 못 미친다. L/R energy attribution을 후처리로 붙여도 channel accuracy가 85%에 도달하지 않는다. 따라서 제품 기본 경로로 승격하지 않고 품질 상한 비교용으로만 유지한다. 다만 human-reviewed gold를 만들 때 우선 검수할 후보는 stable-ts CLI attributed 계열이다. 8dB + quiet-side gate가 기본 channel attribution pack이고, 10dB는 MIX ratio가 높아 기본값은 아니지만 review effort가 64 -> 61로 낮아 사람 검수 시작점으로 비교할 수 있다.
- 2026년 공개 파이프라인 조사에서 WhisperJAV는 ASMR/VR/whisper 콘텐츠에 `fidelity` pipeline과 `aggressive` sensitivity를 권장한다. 또한 ChronosJAV는 Qwen ASR, anime-whisper, Kotoba처럼 timestamp 없는 모델의 text generation과 timestamp alignment를 분리한다. 이 방향은 모델 단독 교체보다 VAD/scene detection/alignment를 분리해서 검증해야 함을 뒷받침한다.
- `TransWithAI/Whisper-Vad-EncDec-ASMR-onnx`는 Whisper encoder 기반 VAD이며 공개 discussion에서 일본어 ASMR 약 500시간으로 학습됐다고 설명된다. ASR 모델이 아니므로 text CER를 직접 개선하지는 않지만, energy splitter보다 ASMR whisper boundary를 더 잘 잡는지 `CASRT_VAD_COMMAND` 후보로 비교한다.
- ASMR Whisper ONNX VAD는 외부 `inference.py`를 실행하지 않고 `casrt vad whisper-asmr-onnx`로 직접 구현한다. 입력은 CASRT VAD command stdin contract를 따르고 출력은 `{ intervals }`만 반환한다. 전처리는 16kHz mono, 30초 chunk, WhisperFeatureExtractor, ONNX Runtime, sigmoid activation, hysteresis postprocess로 제한한다.
- ASMR Whisper ONNX VAD 실행 전 `gpt-5.4 xhigh` subagent가 정적 보안 검토했다. Verdict는 `PASS_WITH_CONSTRAINTS`다. 조건은 전용 venv, `model.onnx`/`model_metadata.json` 두 파일만 있는 전용 모델 디렉터리, SHA-256 기록, CPU-only `--force-cpu --num-threads 1`, 외부 `inference.py` 실행 금지, HF/API/W&B token 제거, VAD subprocess timeout, metadata/shape fail-closed 검증이다. ORT가 custom op/external tensor/unexpected provider를 요구하거나 모델 디렉터리에 extra file이 있으면 중단한다.
- ONNX Runtime session metadata는 input shape를 `['s6', 80, 3000]`, output shape를 `[1, 1500]`, provider를 `CPUExecutionProvider`로 노출했다. `s6`는 symbolic batch dim으로 보고 이 축만 허용하며, feature/time/output shape는 metadata 계약 그대로 고정 검증한다.
- ASMR Whisper ONNX VAD 파일은 `/tmp/casrt-quality/whisper-vad-asmr-onnx-model-v1`에 두 파일만 저장했고 SHA-256은 `/tmp/casrt-quality/whisper-vad-asmr-onnx-model-v1.sha256`에 있다. `model.onnx`는 `cd47513515766d57f740e3094440dbbca9ab87e026b9cf21540d7ad588c0e047`, `model_metadata.json`은 `aeb23b4d032b38e8fe36d6eb350c91f1ae751e0ce11813633ab9533ada4c55b3`다.
- VAD coverage 단독 비교에서 default ONNX VAD는 recall 87.5%, precision 87.2%, interval 47개였고, tuned `threshold=0.35,pad=400,min_silence=400`는 recall 93.6%, precision 83.0%, interval 16개였다. Energy 500/200 baseline은 recall 91.3%, precision 85.0%, interval 69개다. Tuned VAD는 coverage만 보면 좋아 보이지만 실제 ASR에서는 chunk가 과도하게 병합되어 timing과 text가 악화됐다.
- Neosophie/Qwen3-ASR-JA에 ONNX VAD를 붙인 실제 ASR 산출물은 `/tmp/casrt-quality/projects-neosophie-onnx-vad-default`, `/tmp/casrt-quality/projects-neosophie-onnx-vad-t035-pad400`, `/tmp/casrt-quality/projects-neosophie-onnx-vad-hybrid-rescue500`에 있다. Report는 `/tmp/casrt-quality/neosophie-onnx-vad-default-3case-report.json`, `/tmp/casrt-quality/neosophie-onnx-vad-t035-pad400-3case-report.json`, `/tmp/casrt-quality/neosophie-onnx-vad-hybrid-rescue500-3case-report.json`이다. 결론은 ASMR ONNX VAD를 단독 chunker로 기본 교체하지 않는 것이다.
- `--energy-rescue-min-ms 500` hybrid는 coverage recall 95.5%와 time-aligned 500ms 31.1%로 energy baseline보다 timing은 조금 높였지만 practical CER가 31.0%로 악화됐다. 따라서 hybrid도 기본 승격하지 않는다.
- vocal separation은 무조건 적용하지 않는다. WhisperJAV README는 blanket denoise/vocal separation이 Whisper log-Mel feature를 망가뜨릴 수 있다고 경고한다. 반면 WhisperJAV issue에서는 강한 BGM/환경음이 있을 때 UVR/MDX/Demucs류 분리의 필요성이 제기됐다. 따라서 BGM/SFX가 강한 case에서만 별도 실험으로 둔다.
- 다음 개선은 forced alignment 재평가, channel attribution 재평가 순서로 검증한다.

## 다음 작업 계획

1. Gold set 운영
   - gold set manifest CLI는 추가됐다.
   - `casrt slice-case`는 긴 원본 audio와 SRT/master에서 matching WAV/master eval case를 자르고 timestamp를 0 기준으로 rebase한다. 경계에서 잘린 segment는 `needs_review=true`로 표시한다.
   - `casrt prepare-review-cases`는 여러 slice plan을 한 번에 처리해 `audio-map.json`, `case-index.json`, `audio/*.wav`, `references/*.master.json`을 만들고, 모든 case에 candidate가 있으면 `eval-manifest.json`도 만든다.
   - `casrt review-case-status`는 준비된 `case-index.json`에서 audio/reference/candidate 파일 존재 여부, 실제 segment/review count, stale index count, 남은 reference `needs_review` case를 다시 계산한다. 모델 승격 전에는 `--fail-on-issues --fail-on-review`로 운영 gate를 걸 수 있지만, human-reviewed 여부 자체는 추정하지 않는다.
   - `casrt save-review-case-reference`는 WebUI 없이도 편집한 단일 SRT/master를 prepared case reference에 저장하고 `case-index.json` count를 갱신한다. Reference authority는 바꾸지 않는다.
   - `casrt freeze-case-references`는 사람이 검수한 prepared reference들을 batch로 stable id와 `needs_review=false` 상태로 고정하고 새 case set을 만든다. 실검수 여부를 자동 판정하지 않으므로 pseudo-gold smoke에는 `reference_type=pseudo-gold`를 사용한다. Human-reviewed 승격 전에는 `--fail-on-review`를 사용해 남은 flag가 output으로 고정되는 것을 막는다.
   - `casrt build-eval-manifest`는 candidate가 있는 `case-index.json`에서 `custom-asmr-eval-manifest-v1`을 다시 만든다. 사람이 reference를 수정한 뒤에는 `--reference-type human-reviewed --fail-on-review`로 manifest를 만들고, 이어서 `eval-manifest --require-reference-type human-reviewed`로 품질 gate를 실행한다.
   - 2026-06-30 실데이터 smoke: `/home/brain-offloaded/vscode/asmr/whisperx-webui/data/uploads/01.淫魔＆魔女との遭遇.wav`와 `eval-01-full-stable-ts.srt`에서 0~60000ms를 잘라 `/tmp/casrt-quality.Q5OdDf/slice-case-smoke/01-front60.wav`와 `/tmp/casrt-quality.Q5OdDf/slice-case-smoke/01-front60.master.json`을 생성했다. Result: duration 60000ms, segments 14, review_count 1.
   - 2026-06-30 `prepare-review-cases` 실데이터 smoke: plan `/tmp/casrt-quality.Q5OdDf/prepare-review-cases-smoke-plan.json`, output `/tmp/casrt-quality.Q5OdDf/prepare-review-cases-smoke`. 01/04/07 front60 3개 case를 생성했고 result는 `case_count=3`, total `review_count=2`, audio duration은 모두 60000ms다. Case별 segments/review_count는 01: 14/1, 04: 9/0, 07: 11/1이다.
   - 2026-06-30 `review-case-status` 실데이터 smoke: input `/tmp/casrt-quality.Q5OdDf/prepare-review-cases-smoke/case-index.json`, output `/tmp/casrt-quality.Q5OdDf/prepare-review-cases-smoke/status.json`. Result: `case_count=3`, `candidate_case_count=0`, `reference_type_counts={pseudo-gold: 3}`, `missing_file_count=0`, `case_issue_count=0`, `reference_review_count=2`, `cases_needing_review=[01-front60, 07-front60]`.
   - 2026-06-30 `build-eval-manifest` 실데이터 smoke: plan `/tmp/casrt-quality.Q5OdDf/build-eval-manifest-smoke-plan.json`, output dir `/tmp/casrt-quality.Q5OdDf/build-eval-manifest-smoke-cases`, rebuilt manifest `/tmp/casrt-quality.Q5OdDf/build-eval-manifest-smoke-cases/eval-manifest.rebuilt.json`. Result: `case_count=1`, `candidate_case_count=1`, `reference_type=pseudo-gold`, `reference_review_count=1`, `missing_file_count=0`, `case_issue_count=0`. `--reference-type human-reviewed --fail-on-review`는 expected failure로 `review_count=1`을 막았고 output file을 만들지 않았다.
   - 2026-06-30 `freeze-case-references` 실데이터 smoke: input `/tmp/casrt-quality.Q5OdDf/build-eval-manifest-smoke-cases/case-index.json`, output `/tmp/casrt-quality.Q5OdDf/freeze-case-references-smoke`, `reference_type=pseudo-gold`로 실행했다. Result: `case_count=1`, frozen `review_count=0`, generated `audio-map.json`, `case-index.json`, `eval-manifest.json`. `review-case-status --fail-on-review`는 `reference_review_count=0`, `missing_file_count=0`, `case_issue_count=0`으로 통과했고, generated manifest 평가 report는 `/tmp/casrt-quality.Q5OdDf/freeze-case-references-smoke/eval-report.json`에 저장했다.
   - 2026-06-30 8개 실제 원본 front120 review case set 생성:
     - plan: `/tmp/casrt-quality.Q5OdDf/all8-front120-review-cases-plan.json`
     - output: `/tmp/casrt-quality.Q5OdDf/all8-front120-review-cases`
     - source: `/home/brain-offloaded/vscode/asmr/whisperx-webui/data/uploads`의 01~08 wav와 기존 `/data/outputs` SRT 산출물.
     - result: `case_count=8`, `reference_type_counts={pseudo-gold: 8}`, `missing_file_count=0`, `case_issue_count=0`, `reference_review_count=15`.
     - case별 segments/review_count: 01 `10/2`, 02 `10/1`, 03 `11/2`, 04 `12/2`, 05 `10/2`, 06 `11/2`, 07 `10/2`, 08 `8/2`.
     - `review-case-status --fail-on-review`는 report 출력 후 expected failure로 `review_count=15`를 반환했다.
     - 2026-07-01 진행률 field smoke: output `/tmp/casrt-review-progress-status.json`, `reference_review_count=15`, `reference_review_case_count=8`, `reference_review_clear_case_count=0`.
     - 2026-07-01 `save-review-case-reference` 복사본 smoke: copied case set `/tmp/casrt-save-review-case-smoke.c7G3HG/cases`, command saved `01-front120-existing-srt` reference to itself, result `segments=10`, `review_count=2`; follow-up status stayed `reference_review_count=15`, `case_issue_count=0`.
     - 2026-07-01 `freeze-case-references --fail-on-review` 실데이터 expected failure: input all8 case set, `reference_type=human-reviewed`, output `/tmp/casrt-freeze-fail-on-review-smoke`; failed with `reference review_count=15` and did not create output directory.
     - 2026-07-01 durable recreation: plan `.casrt/experiments/all8-front120-review-cases-plan.json`, output `.casrt/experiments/all8-front120-review-cases`, status `.casrt/experiments/all8-front120-review-cases/status.json`. Source SRT는 2025-12-22 hash outputs `01-251e...`, `02-ae45...`, `03-7363...`, `04-84f8...`, `05-0a9d...`, `06-803d...`, `07-1402...`, `08-4be4...`다. Result: `case_count=8`, `reference_type_counts={pseudo-gold: 8}`, `missing_file_count=0`, `case_issue_count=0`, `reference_review_count=15`, `reference_review_case_count=8`, `reference_review_clear_case_count=0`. `review-case-status --fail-on-issues --fail-on-review` failed as expected with `review_count=15`.
     - 판단: 모델 승격용 gold가 아니라, 사람이 WebUI에서 audio/reference를 열어 검수하고 `freeze-case-references --reference-type human-reviewed`로 올릴 시작점이다.
   - `/data/uploads`, `/data/outputs`에서 30초~2분 단위 reference case를 늘릴 때는 `custom-asmr-case-slice-plan-v1` plan으로 재현 가능하게 기록한다.
   - 사람이 검수한 단일 파일은 `casrt freeze-reference`, prepared case set은 `casrt freeze-case-references`로 stable id와 `needs_review=false`를 고정한다.
   - manifest에 `reference_type=human-reviewed`와 검수 메모를 기록한다.
   - CER, timing error, channel accuracy, human edit count를 manifest report로 기록한다.
   - `review-effort` export는 `priority_score` 내림차순 큐로 운영한다. missing/extra/text/timing/channel 실패를 큰 것부터 들어 human-reviewed gold 제작 시간을 줄인다.

2. 일본어 평가 정규화 확장
   - strict/practical CER는 분리됐다.
   - 다음 단계에서는 장음/감탄/소형 kana 차이를 별도 옵션으로 추가할지 평가한다.

3. VAD 후보 추가
   - VAD command hook은 추가됐다.
   - `casrt vad whisper-asmr-onnx` command는 추가됐다.
   - 현재 energy splitter 500/200은 fallback-free baseline이다.
   - `TransWithAI/Whisper-Vad-EncDec-ASMR-onnx`는 단독 chunker로 비교했고 기본 교체하지 않는다.
   - `--energy-rescue-min-ms 500` hybrid도 비교했고 기본 교체하지 않는다.
   - Silero VAD, TEN VAD wrapper는 ASMR ONNX VAD보다 후순위로 둔다.
   - VAD도 WebUI 옵션으로 노출하지 않고 고정/내부 설정으로 둔다.

4. Channel attribution 튜닝
   - 현재 8dB threshold + quiet-side -40dBFS gate는 보수적 baseline이다.
   - `casrt attribute-channels --diagnostics-output`은 segment별 L/R dBFS와 판정 이유를 JSON으로 저장해 사람이 channel threshold 실패 패턴을 확인할 수 있게 한다.
   - `casrt sweep-channel-attribution`은 eval manifest와 audio map으로 threshold/quiet-side setting을 반복 적용하고 setting별 candidate, eval report, comparison을 만든다. 기본값 변경은 sweep output과 human-reviewed gold gate를 보고 별도 결정한다.
   - gold set 기준으로 threshold와 MIX 유지 비율을 조정한다.
   - 필요하면 segment별 channel confidence를 debug metadata로만 저장한다.

5. Forced alignment 재평가
   - VAD chunk가 안정된 뒤 Qwen3-ForcedAligner를 segment 내부 timing 보정에 다시 사용한다.
   - word/char alignment는 번역용 JSON에 넣지 않는다.

6. 모델 비교
   - `Qwen/Qwen3-ASR-1.7B`를 주력으로 둔다.
   - `zhifeixie/Mega-ASR`는 검증 완료 후보지만 기본 승격하지 않는다.
   - `neosophie/Qwen3-ASR-1.7B-JA`는 검증 완료 후보지만 기본 승격하지 않는다.
   - `Qwen/Qwen3-ASR-1.7B-hf`는 Transformers main에서 검증했지만 기본 승격하지 않는다. 공식 release에 `qwen3_asr`가 들어오면 runtime 안정성만 재확인하고, 품질 재평가는 human-reviewed gold가 늘어난 뒤에 한다.
   - `microsoft/VibeVoice-ASR`와 `microsoft/VibeVoice-ASR-HF`는 일본어 tag가 있는 최신 로컬 후보지만 현재 repo env의 Transformers 5.12.1에서 전용 class가 없어 보류한다. 공식 release 지원 또는 별도 runtime 검토 후 exact revision local snapshot으로만 평가한다.
   - `mistralai/Voxtral-Mini-4B-Realtime-2602`는 remote code 없이 검증했지만 07 whisper 구간에서 실패해 기본 승격하지 않는다.
   - `ibm-granite/granite-speech-4.1-2b`는 native Transformers/safetensors/ja 후보로 `local-granite-asr` adapter, exact snapshot download, `casrt model digest`, 10초 smoke, 01/04/07 front120 pseudo-gold 평가를 완료했다. Non-Japanese hallucination filter 후 practical CER 23.6%, Qwen3-ForcedAligner 적용 후 time-aligned 500ms 32.7%지만 text/review gate 실패로 기본 승격하지 않는다. Granite Plus timestamp prompt도 practical CER 84.1%로 악화되어 기본 승격하지 않는다.
   - `Atotti/llm-jp-4-8b-speech-asr`는 ASR 특화 일본어 후보지만 third-party runtime package가 필요하므로 사용자 명시 승인 후 비교한다.
   - `AutoArk-AI/ARK-ASR-3B`, `CohereLabs/cohere-transcribe-03-2026`, `OpenMOSS-Team/MOSS-Transcribe-preview-2B`는 성능 후보로 남기되, custom code/gated/runtime 접근 조건을 먼저 해결해야 한다.
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
- VibeVoice-ASR model card: https://huggingface.co/microsoft/VibeVoice-ASR
- VibeVoice-ASR-HF model card: https://huggingface.co/microsoft/VibeVoice-ASR-HF
- MOSS Transcribe preview model card: https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-preview-2B
- MOSS Transcribe GGUF model card: https://huggingface.co/cstr/MOSS-Transcribe-preview-2B-GGUF
- MiMo V2.5 ASR model card: https://huggingface.co/XiaomiMiMo/MiMo-V2.5-ASR
- Granite Speech 4.1 2B model card: https://huggingface.co/ibm-granite/granite-speech-4.1-2b
- Granite Speech 4.1 2B Plus model card: https://huggingface.co/ibm-granite/granite-speech-4.1-2b-plus
- Cohere ASR Japanese fine-tune model card: https://huggingface.co/efwkjn/cohere-asr-ja
- ARK-ASR-3B model card: https://huggingface.co/AutoArk-AI/ARK-ASR-3B

## 문서화 규칙

파이프라인 변경, 모델 선택, threshold, 평가 결과, 다음 작업 계획은 채팅에만 남기지 않는다. 변경이 확정되면 이 문서 또는 관련 제품 문서에 반영한다.
