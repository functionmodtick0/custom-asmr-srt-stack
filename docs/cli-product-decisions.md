# CLI 제품 결정 기록

작성일: 2026-06-27

## 목표

WebUI가 제공하는 모든 제품 기능을 CLI에서도 사용할 수 있게 한다.

CLI의 핵심 약속은 다음이다.

```text
파일 경로 입력 -> project 생성/갱신 -> 전사/재전사 -> JSON/SRT 내보내기
```

CLI는 WebUI와 별도의 파이프라인을 만들지 않는다. 동일한 core 모델, project 저장소, audio 분석, transcription adapter, alignment boundary, JSON/SRT 변환 로직을 재사용한다.

## 범위

CLI는 다음 WebUI 기능과 동등해야 한다.

- 오디오 파일로 project 생성
- SRT로 project 생성
- `master.json`으로 project 생성
- 오디오 분석 및 L/R/MIX 채널 생성
- 모델 설정 검증
- 전체 project 전사
- 선택 segment 재전사
- `master.json` 내보내기
- `translation.json` 내보내기
- `translated.json`을 병합한 SRT 내보내기
- 원문 SRT 내보내기
- project 상태 확인
- reference/candidate 전사 결과 평가
- gold set manifest 기준 batch 평가

번역 기능은 제공하지 않는다.

## 명령 계약

### Project 생성

오디오에서 project 생성:

```bash
uv run casrt project create-audio input.wav
```

SRT에서 project 생성:

```bash
uv run casrt project create-srt input.srt
```

master JSON에서 project 생성:

```bash
uv run casrt project create-master master.json
```

기본 출력은 사람이 읽을 수 있는 한 줄 요약이며, 자동화용으로 `--json`을 지원한다.

```json
{
  "project_id": "...",
  "metadata": {}
}
```

### Project 상태

```bash
uv run casrt project show PROJECT_ID
```

상태는 metadata, segment count, review count를 포함한다.

### 오디오 분석

```bash
uv run casrt project analyze PROJECT_ID
```

동작:

- 오디오를 WAV로 정규화한다.
- stereo WAV는 `L`, `R`, `MIX` 채널 파일을 만든다.
- mono WAV는 `MIX` 채널 파일을 만든다.
- chunk interval을 project metadata에 저장한다.

### 모델 설정 검증

```bash
uv run casrt model validate \
  --adapter openai-compatible \
  --endpoint-url http://127.0.0.1:8000/v1 \
  --model-id gemma-4-e4b \
  --api-key ...
```

이 명령은 필수 필드와 adapter 계약만 검증한다. 실제 모델 호출은 하지 않는다.

로컬 Transformers worker를 사용할 때는 endpoint URL을 입력하지 않는다.

```bash
uv run casrt model validate \
  --adapter local-transformers \
  --model-id google/gemma-4-E4B-it
```

로컬 Qwen ASR worker를 사용할 때도 endpoint URL을 입력하지 않는다.

```bash
uv run casrt model validate \
  --adapter local-qwen-asr \
  --model-id Qwen/Qwen3-ASR-1.7B
```

### 전체 전사

```bash
uv run casrt project transcribe PROJECT_ID \
  --adapter openai-compatible \
  --endpoint-url http://127.0.0.1:8000/v1 \
  --model-id gemma-4-e4b \
  --api-key ...
```

동작:

- project가 아직 분석되지 않았다면 실패한다.
- OpenAI-compatible/Gemini endpoint는 L/R 채널이 있으면 각각 전사하고, mono/MIX만 있으면 MIX를 전사한다.
- 로컬 ASR adapter인 `local-transformers`와 `local-qwen-asr`는 L/R이 있어도 MIX-first로 전사한다.
- 로컬 ASR adapter는 silence/energy 기반 chunk interval별로 MIX 오디오를 잘라 모델에 보낸다.
- `local-transformers` adapter는 worker 모델의 audio limit을 고려해 chunk를 30초 이하 subchunk로 다시 자른다.
- 로컬 ASR adapter가 반환한 MIX segment는 L/R energy 기반 channel attribution을 적용한다.
- 모델이 반환한 chunk-relative timing을 원본 timeline timing으로 offset한다.
- 결과를 시간순으로 정렬하고 stable segment id를 다시 부여한다.
- `master.json`을 project에 저장한다.
- `CASRT_ALIGNER_COMMAND`가 설정되어 있으면 고정 aligner hook을 실행한다.

로컬 Transformers worker:

```bash
CASRT_TRANSFORMERS_QUANTIZATION=4bit \
  uv run casrt project transcribe PROJECT_ID \
  --adapter local-transformers \
  --model-id google/gemma-4-E4B-it
```

동작:

- `casrt`가 내부적으로 `python -m custom_asmr_srt_stack.transformers_worker` subprocess를 시작한다.
- worker와 JSON Lines로 통신한다.
- worker는 모델을 lazy load하고 같은 CLI/WebUI 프로세스 안에서 재사용한다.
- `CASRT_TRANSFORMERS_QUANTIZATION=4bit` 또는 `8bit`가 설정되면 runtime quantization을 사용하고, Gemma 4 audio path가 깨지지 않도록 `lm_head`와 `model.audio_tower`는 quantization에서 제외한다.
- 잘못된 quantization 값은 full precision fallback으로 넘기지 않고 오류로 표시한다.
- worker generation은 기본 `max_new_tokens=256`으로 제한한다. `CASRT_TRANSFORMERS_MAX_NEW_TOKENS`로 조정할 수 있지만 WebUI 옵션으로 노출하지 않는다.
- worker import, model load, inference, response contract 오류는 실패로 표시한다.

로컬 Qwen ASR worker:

```bash
uv run casrt project transcribe PROJECT_ID \
  --adapter local-qwen-asr \
  --model-id Qwen/Qwen3-ASR-1.7B
```

동작:

- `casrt`가 내부적으로 `python -m custom_asmr_srt_stack.qwen_asr_worker` subprocess를 시작한다.
- worker와 JSON Lines로 통신한다.
- worker는 모델을 lazy load하고 같은 CLI/WebUI 프로세스 안에서 재사용한다.
- ASMR 품질 경로에서는 MIX 전사를 우선하고 L/R은 channel attribution 근거로 사용한다.
- `CASRT_VAD_COMMAND`가 설정되어 있으면 고정 VAD command의 interval을 사용한다.
- `CASRT_VAD_COMMAND`가 설정되어 있지 않으면 MIX energy 기반 speech chunking으로 발화 단위 전사를 시도한다.
- energy chunking은 `CASRT_QWEN_ENERGY_*` env로만 내부 튜닝한다. `CASRT_QWEN_ENERGY_MAX_CHUNK_MS`는 긴 interval을 일정 길이 이하로 자르는 실험 옵션이며, 2026-06-30 01/04/07 front120 평가에서는 기본 승격하지 않는다.
- L/R energy 차이가 충분할 때만 channel을 L 또는 R로 확정하고, 애매하면 MIX로 남긴다.
- worker import, model load, inference, response contract 오류는 실패로 표시한다.

Qwen ASR 파이프라인의 세부 값과 평가 결과는 `docs/local-asr-pipeline.md`에 기록한다.

고정 VAD command contract:

- stdin: `{ audio_file, audio_info }`
- stdout: `{ intervals: [{ start_ms, end_ms }] }`
- interval은 정렬되어야 하고 서로 겹치면 안 된다.
- interval이 음수이거나 audio duration을 넘으면 실패한다.
- 이 옵션은 WebUI/CLI 모델 선택 UI에 노출하지 않는다.
- ASMR Whisper ONNX VAD 후보는 전용 venv의 절대 실행 파일로 `casrt vad whisper-asmr-onnx --model /path/to/model.onnx --metadata /path/to/model_metadata.json --force-cpu --num-threads 1`를 실행한다.
- 이 command는 외부 model repository의 `inference.py`를 실행하지 않고, 16kHz mono 변환, 30초 chunk, ONNX Runtime, hysteresis postprocess만 수행한다.
- 모델 디렉터리는 `model.onnx`와 `model_metadata.json` 두 파일만 허용한다. wrapper는 metadata/shape/provider 계약이 맞지 않으면 실패한다.
- VAD subprocess는 timeout과 최소 env로 실행하며 HF/API/W&B token을 상속하지 않는다.
- `--energy-rescue-min-ms`는 energy interval을 유지하고 ONNX-only gap만 추가하는 내부 실험 옵션이다. 2026-06-28 gold 결과에서 text가 악화되어 기본값으로 쓰지 않는다.

### 평가

```bash
uv run casrt eval-transcript reference.srt candidate.json --json -o eval.json
```

동작:

- reference와 candidate는 SRT 또는 `master.json`을 받을 수 있다.
- speech text strict CER와 practical CER를 계산한다.
- segment index 기준 mean start/end/boundary error를 계산한다.
- segment 수나 split이 다른 후보를 평가하기 위해 time-overlap 기반 `timing_time_aligned`를 계산한다.
- forced alignment 재평가를 위해 boundary sample 수, max boundary error, 250ms/500ms 이내 boundary ratio를 계산한다.
- channel attribution 튜닝을 위해 index 기반 `channel`과 time-overlap 기반 `channel_time_aligned`의 L/R/MIX confusion, candidate MIX 유지 비율, L/R channel accuracy를 계산한다.
- `needs_review` 비율을 계산한다.
- 평가는 모델 기본값 승격이나 threshold 변경 전에 실행한다.
- `--max-practical-cer`, `--min-time-aligned-500ms-ratio`, `--min-channel-time-aligned-accuracy`, `--max-channel-time-aligned-mix-ratio`를 지정하면 품질 gate로 동작한다. gate 실패 시 report는 stdout/file에 남기고 exit code를 실패로 반환한다.

여러 샘플을 한 번에 평가할 때는 gold set manifest를 사용한다.

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

동작:

- `reference`와 `candidate` 경로는 manifest 파일 위치 기준 상대 경로 또는 absolute path를 받는다.
- 각 case는 기존 `eval-transcript`와 동일한 리포트를 보존한다.
- summary CER는 case 평균이 아니라 전체 edit distance / 전체 reference characters로 계산한다.
- summary timing/channel/review는 전체 paired/boundary/comparable/candidate segment 수 기준으로 가중 집계한다.
- case `id`는 중복될 수 없다.
- `eval-manifest`의 품질 gate는 summary metric 기준으로 판단한다.

### 선택 segment 재전사

```bash
uv run casrt project retranscribe PROJECT_ID seg_000001 \
  --adapter openai-compatible \
  --endpoint-url http://127.0.0.1:8000/v1 \
  --model-id gemma-4-e4b
```

동작:

- 해당 segment의 channel audio에서 segment 시간 범위만 잘라 모델에 보낸다.
- 모델 결과 timing을 원본 timeline으로 offset한다.
- 기존 segment를 새 segment들로 교체한다.
- segment id를 다시 안정적으로 부여한다.
- 갱신된 `master.json`을 저장한다.

### 내보내기

project master JSON:

```bash
uv run casrt project export-master PROJECT_ID -o master.json
```

번역 도구용 clean JSON:

```bash
uv run casrt project export-translation PROJECT_ID -o translation.json
```

원문 SRT:

```bash
uv run casrt project export-srt PROJECT_ID -o source.srt
```

번역 SRT:

```bash
uv run casrt project export-srt PROJECT_ID --translated translated.json -o translated.srt
```

## 공통 옵션

자동화용 JSON 출력:

```text
--json
```

project 저장 root 지정:

```text
--project-root PATH
```

지정하지 않으면 기존 WebUI와 동일하게 현재 작업 디렉터리의 `.casrt/projects`를 사용한다.

## 실패 동작

CLI는 실패 시 non-zero exit code로 종료한다.

예상 가능한 프로젝트, 파일, 모델 응답 오류는 traceback 대신 stderr에 한 줄 `error: ...` 메시지로 출력한다.

조용히 보정하지 않는 오류:

- invalid JSON
- missing/duplicate translated id
- invalid timestamp
- project not found
- invalid eval manifest
- unanalyzed project transcribe
- missing model endpoint fields
- unsupported adapter
- audio decode failure
- aligner output id mismatch
- model response contract violation

## 출력 원칙

기본 출력은 짧고 사람이 읽기 쉬워야 한다.

예:

```text
project 5f... created from audio: voice.wav
project 5f... analyzed: channels=L,R,MIX duration_ms=123456
project 5f... transcribed: segments=42 review=3
```

스크립트 연동이 필요한 명령은 `--json`을 사용한다.

## 구현 원칙

- WebUI API handler를 호출하지 않고 core 함수들을 직접 호출한다.
- CLI 전용 business logic을 중복 구현하지 않는다.
- 테스트는 subprocess가 아니라 CLI `main(argv)` 또는 command 함수의 관찰 가능한 파일 출력/저장 결과를 검증한다.
- 네트워크 모델 호출 테스트는 fake transcriber로 격리한다.
