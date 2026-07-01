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
- 수정된 `master.json` 저장으로 segment text/start/end/channel/review flag 반영
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

HF-native Qwen ASR worker도 endpoint URL을 입력하지 않는다. 실제 benchmark는 repo id가 아니라 exact revision의 local snapshot directory를 model id로 사용한다.

```bash
uv run casrt model validate \
  --adapter local-qwen-hf-asr \
  --model-id /path/to/Qwen3-ASR-1.7B-hf
```

로컬 Cohere ASR worker도 endpoint URL을 입력하지 않는다. 실제 benchmark는 repo id가 아니라 exact revision의 local snapshot directory를 model id로 사용한다.

```bash
uv run casrt model validate \
  --adapter local-cohere-asr \
  --model-id /path/to/cohere-transcribe-03-2026/snapshots/<commit>
```

로컬 Granite ASR worker도 endpoint URL을 입력하지 않는다. 실제 benchmark는 repo id가 아니라 exact revision의 local snapshot directory를 model id로 사용한다.

```bash
uv run casrt model validate \
  --adapter local-granite-asr \
  --model-id /path/to/granite-speech-4.1-2b/snapshots/<commit>
```

보안 검토가 필요한 local snapshot은 benchmark 전에 digest report를 남긴다.

```bash
uv run casrt model digest /path/to/model/snapshots/<commit> \
  -o model-digest.json \
  --json
```

`model digest`는 local directory만 읽고 각 파일 SHA-256과 snapshot 전체 SHA-256을 기록한다. download/evaluation 결과 문서에는 exact revision path와 digest report path를 함께 적는다. 장기 보관할 model snapshot은 `/tmp`가 아니라 gitignored `.casrt/models/<model>-<revision>` 아래에 둔다. digest report는 `.casrt/model-digests/<model>-<revision>-digest.json`에 둔다. `/tmp`는 다운로드 staging이나 삭제되어도 되는 실험 출력 전용이다.

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
- 로컬 ASR adapter인 `local-transformers`, `local-qwen-asr`, `local-qwen-hf-asr`, `local-cohere-asr`는 L/R이 있어도 MIX-first로 전사한다.
- 로컬 ASR adapter는 silence/energy 기반 chunk interval별로 MIX 오디오를 잘라 모델에 보낸다.
- `local-transformers` adapter는 worker 모델의 audio limit을 고려해 chunk를 30초 이하 subchunk로 다시 자른다.
- 로컬 ASR adapter가 반환한 MIX segment는 L/R energy 기반 channel attribution을 적용한다.
- 모델이 반환한 chunk-relative timing을 원본 timeline timing으로 offset한다.
- 결과를 시간순으로 정렬하고 stable segment id를 다시 부여한다.
- `master.json`을 project에 저장한다.
- `CASRT_ALIGNER_COMMAND`가 설정되어 있으면 고정 aligner hook을 실행한다.
- `CASRT_ALIGNER_ENV_MODE=offline`이면 aligner subprocess env는 CUDA/cache/path와 `CASRT_ALIGNER_`, `CASRT_QWEN_ALIGNER_` prefix만 보존하고 API key/token/secret류 env, `CASRT_ALIGNER_COMMAND`, `PYTHONPATH`를 제거하며 `PYTHONNOUSERSITE=1`을 강제한다.
- `python -m custom_asmr_srt_stack.qwen_aligner_worker --model-id LOCAL_SNAPSHOT`은 Qwen3-ForcedAligner를 generic aligner command로 실행한다.
- Qwen aligner worker는 `CASRT_ALIGNER_ENV_MODE=offline`, `CASRT_QWEN_ALIGNER_REQUIRE_LOCAL_MODEL_PATH=1`, `CASRT_QWEN_ALIGNER_LOCAL_FILES_ONLY=1`, `CASRT_QWEN_ALIGNER_DISABLE_NETWORK=1` 조건이 모두 없으면 실패하며, `--model-id`는 existing local directory만 허용한다.
- Qwen aligner worker는 `qwen-asr==0.0.6`, dist-info `RECORD` SHA-256 `56454a099599cb3c86fd96347baa86269cc62e0d9eced004eeb2faa26b3a8a7c`, RECORD에 기록된 각 설치 파일 hash, `qwen_asr` import origin을 강제한다.
- Qwen aligner worker는 segment text를 바꾸지 않고 speech segment 내부 start/end만 재정렬한다. `CASRT_QWEN_ALIGNER_MIN_ALIGNED_DURATION_MS`보다 짧은 span, 또는 원 segment duration 대비 `CASRT_QWEN_ALIGNER_MIN_COVERAGE_RATIO` 미만으로 줄어든 span은 원래 segment timing으로 유지한다. 기본값은 80ms와 0.5이며 WebUI 옵션으로 노출하지 않는다.
- Qwen aligner worker 실패 응답은 요약 오류만 노출하고 traceback은 stdout/stderr/API 경로로 내보내지 않는다.
- `CASRT_LOCAL_WORKER_ENV_MODE=offline`이면 local worker subprocess env는 `CASRT_LOCAL_WORKER_ENV_MODE`를 보존하고, `PYTHONPATH`와 secret/proxy류 env를 넘기지 않으며, `PYTHONNOUSERSITE=1`을 강제한다.
- `python -m custom_asmr_srt_stack.qwen_hf_asr_worker`는 HF-native `Qwen/Qwen3-ASR-1.7B-hf`를 실행하는 worker다. `CASRT_LOCAL_WORKER_ENV_MODE=offline`, `CASRT_QWEN_HF_ASR_REQUIRE_LOCAL_MODEL_PATH=1`, `CASRT_QWEN_HF_ASR_LOCAL_FILES_ONLY=1`, `CASRT_QWEN_HF_ASR_DISABLE_NETWORK=1`이 모두 없으면 실패한다. `AutoProcessor`와 `AutoModelForMultimodalLM`은 local path, `local_files_only=True`, `trust_remote_code=False`, `use_safetensors=True`로만 로드한다.
- Qwen HF ASR worker는 timestamp를 생성하지 않으므로 chunk 전체를 speech segment로 반환하고 `needs_review=true`를 붙인다. timing은 후속 VAD/alignment 평가에서 다룬다.
- Qwen HF ASR worker 실패 응답은 요약 오류만 노출하고 traceback은 stdout/stderr/API 경로로 내보내지 않는다.

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
- 보안 검토가 필요한 benchmark 실행은 `CASRT_LOCAL_WORKER_ENV_MODE=offline`, `CASRT_QWEN_ASR_REQUIRE_LOCAL_MODEL_PATH=1`, `CASRT_QWEN_ASR_LOCAL_FILES_ONLY=1`, `CASRT_QWEN_ASR_DISABLE_NETWORK=1`을 사용하고 `--model-id`에는 repo id가 아니라 고정 local snapshot directory를 넣는다.
- `CASRT_QWEN_ASR_ALIGNER_MODEL_ID`가 설정되면 Qwen3-ForcedAligner timestamp를 사용한다. `CASRT_QWEN_ASR_MIN_ALIGNED_DURATION_MS`보다 짧은 aligned span은 clip bounds로 되돌리며, 이 값은 WebUI/CLI 옵션으로 노출하지 않는다.
- `CASRT_VAD_COMMAND`가 설정되어 있으면 고정 VAD command의 interval을 사용한다.
- `CASRT_VAD_COMMAND`가 설정되어 있지 않으면 MIX energy 기반 speech chunking으로 발화 단위 전사를 시도한다.
- energy chunking은 `CASRT_QWEN_ENERGY_*` env로만 내부 튜닝한다. `CASRT_QWEN_ENERGY_MAX_CHUNK_MS`는 긴 interval을 일정 길이 이하로 자르는 실험 옵션이며, 2026-06-30 01/04/07 front120 평가에서는 기본 승격하지 않는다.
- L/R energy 차이가 충분할 때만 channel을 L 또는 R로 확정하고, 애매하면 MIX로 남긴다.
- worker import, model load, inference, response contract 오류는 실패로 표시한다.

Qwen ASR 파이프라인의 세부 값과 평가 결과는 `docs/local-asr-pipeline.md`에 기록한다.

로컬 Cohere ASR worker:

```bash
uv run casrt project transcribe PROJECT_ID \
  --adapter local-cohere-asr \
  --model-id /path/to/cohere-transcribe-03-2026/snapshots/<commit>
```

동작:

- `casrt`가 내부적으로 `python -m custom_asmr_srt_stack.cohere_asr_worker` subprocess를 시작한다.
- worker는 `CohereAsrForConditionalGeneration`와 `CohereAsrProcessor`를 명시적으로 사용하고 `trust_remote_code=False`, `local_files_only=True`, `use_safetensors=True`로 로드한다.
- `--model-id`는 safetensors weight가 있는 existing local snapshot directory여야 한다. repo id나 cache miss fallback은 실패한다.
- Cohere는 timestamp를 반환하지 않으므로 chunk bounds를 segment timing으로 사용하고, 기존 MIX-first energy chunking과 L/R channel attribution을 적용한다.
- 실제 download/evaluation은 exact revision pin과 `casrt model digest` report 기록 전까지 실행하지 않는다.

로컬 Granite ASR worker:

```bash
uv run casrt project transcribe PROJECT_ID \
  --adapter local-granite-asr \
  --model-id .casrt/models/granite-speech-4.1-2b-de575db64086f84fdc79da4932d1076e965bc546
```

동작:

- `casrt`가 내부적으로 `python -m custom_asmr_srt_stack.granite_asr_worker` subprocess를 시작한다.
- worker는 `AutoModelForSpeechSeq2Seq`와 `AutoProcessor`를 사용하고 `trust_remote_code=False`, `local_files_only=True`, `use_safetensors=True`로 로드한다.
- Granite `AutoProcessor`의 `GraniteSpeechFeatureExtractor`는 `torchaudio`를 요구하므로 `local` extra에 `torchaudio`를 포함한다.
- `--model-id`는 safetensors weight가 있는 existing local snapshot directory여야 한다. repo id나 cache miss fallback은 실패한다.
- Granite는 timestamp를 반환하지 않으므로 chunk bounds를 segment timing으로 사용하고, 기존 MIX-first energy chunking과 L/R channel attribution을 적용한다.
- 기본 prompt는 `<|audio|>transcribe the speech with proper punctuation and capitalization.`이고 `CASRT_GRANITE_ASR_PROMPT`로만 내부 override할 수 있다.
- 현재 local snapshot은 `.casrt/models/granite-speech-4.1-2b-de575db64086f84fdc79da4932d1076e965bc546`이고 digest report는 `.casrt/model-digests/granite-speech-4.1-2b-de575db64086f84fdc79da4932d1076e965bc546-digest.json`이다. snapshot SHA-256은 `67c7d69184b53bae7a2bec077fbc88d8695a72f043fd70831f4e4830dc4752ca`다.
- 2026-07-01 실데이터 01/04/07 front120 pseudo-gold 평가에서 practical CER 24.7%, time-aligned 500ms 23.7%, candidate MIX 54.4%, candidate review ratio 100%, review effort 100%로 product gate를 실패했다. 기본 ASMR 경로로 승격하지 않는다.

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

### 기준본 고정

```bash
uv run casrt freeze-reference reviewed.srt -o refs/front120.master.json --json
uv run casrt freeze-reference reviewed.master.json -o refs/front120.master.json --json
```

동작:

- 입력은 SRT 또는 `master.json`을 받는다.
- segment를 `(start_ms, end_ms, id)` 순으로 정렬한다.
- id를 `seg_000001`부터 다시 부여한다.
- `needs_review`는 모두 `false`로 저장한다.
- stdout JSON에는 `reference_type=human-reviewed`를 포함한다.
- 이 명령은 검수 상태를 추정하지 않는다. 사람이 검수한 파일만 입력으로 넣는 것이 계약이다.
- pseudo-gold나 모델 산출물을 `freeze-reference`로 고정할 수는 있지만, manifest에는 `reference_type=pseudo-gold`로 기록해야 하며 모델 승격 근거로 쓰지 않는다.

### Gold/Eval Case Slicing

```bash
uv run casrt slice-case input.wav input.srt \
  --start-ms 0 \
  --end-ms 120000 \
  --audio-output cases/front120.wav \
  --transcript-output cases/front120.master.json \
  --json
```

동작:

- 입력 transcript는 SRT 또는 `master.json`을 받는다.
- audio는 WAV 또는 ffmpeg로 decode 가능한 audio를 받으며 output audio는 normalized WAV다.
- audio와 transcript를 같은 `[start_ms, end_ms)` 구간으로 자르고 transcript timestamp를 0 기준으로 rebase한다.
- output master `audio.duration_ms`는 `end_ms - start_ms`다.
- 구간과 겹치지 않는 segment는 제외한다.
- 구간 경계에 걸쳐 잘린 segment는 text가 일부만 남을 수 있으므로 `needs_review=true`로 표시한다.
- segment id는 `seg_000001`부터 다시 부여한다.
- 이 명령은 human-reviewed gold를 만들기 위한 case 준비 도구다. 검수 완료 판정은 하지 않으며, 사람이 확인한 뒤 `freeze-reference`와 `reference_type=human-reviewed` manifest를 사용한다.

여러 case를 한 번에 준비할 때는 `prepare-review-cases`를 사용한다.

```json
{
  "format": "custom-asmr-case-slice-plan-v1",
  "reference_type": "pseudo-gold",
  "reference_notes": "stable-ts draft; requires human review",
  "cases": [
    {
      "id": "01-front60",
      "audio": "../uploads/01.wav",
      "reference": "../outputs/01-full.srt",
      "candidate": "../outputs/01-candidate.srt",
      "candidate_id": "stable-ts-attributed",
      "start_ms": 0,
      "end_ms": 60000
    }
  ]
}
```

```bash
uv run casrt prepare-review-cases plan.json -o cases --json
```

동작:

- plan path는 plan 파일 위치 기준 상대 경로 또는 absolute path를 받는다.
- output directory가 존재하고 비어 있지 않으면 실패한다.
- 각 case는 `audio/<id>.wav`와 `references/<id>.master.json`으로 잘린다.
- `case-index.json`은 `custom-asmr-review-case-set-v1`이고 원본 경로, slice range, output 경로, segment/review count를 보존한다.
- `audio-map.json`은 `custom-asmr-review-audio-map-v1`이며 `review-pack`에 바로 넣을 수 있다.
- 모든 case가 `candidate`를 가지면 `candidates/<id>.master.json`과 `eval-manifest.json`을 함께 만든다.
- candidate가 있는 case와 없는 case를 섞으면 실패한다. 검수 준비와 후보 평가를 같은 plan에서 섞지 않는다.
- `prepare-review-cases`도 검수 완료 판정을 하지 않는다. 사람이 확인한 뒤 `freeze-reference`와 `reference_type=human-reviewed` manifest를 사용한다.
- WebUI는 생성된 `case-index.json`을 Review path로 열어 case reference를 직접 편집할 수 있다. 이 편집은 reference master JSON과 case-index count만 갱신하며, `검수 완료`는 현재 `needs_review` segment를 false로 바꾸고 다음 검수 segment로 이동한 뒤 저장한다. `case 목록`/`다음 case` 이동 전에도 저장을 flush한다. Human-reviewed 승격 판정은 CLI gate에 남긴다.

준비된 case set의 파일 누락, stale count, 남은 reference review flag는 `review-case-status`로 확인한다.

```bash
uv run casrt review-case-status cases/case-index.json --json -o cases/status.json
uv run casrt review-case-status cases/case-index.json --fail-on-issues --fail-on-review
```

동작:

- 입력은 `custom-asmr-review-case-set-v1` `case-index.json`이다.
- `audio`, `reference`, optional `candidate` 경로는 `case-index.json` 위치 기준으로 해석한다.
- output format은 `custom-asmr-review-case-status-v1`이다.
- 각 case의 파일 존재 여부, reference/candidate segment count, reference/candidate review count를 실제 파일에서 다시 계산한다.
- `case-index.json`에 기록된 `segments`/`review_count`와 실제 reference가 다르면 `issues`에 남긴다.
- `reference_review_case_count`와 `reference_review_clear_case_count`를 함께 남겨 검수 flag가 남은 case 진행률을 볼 수 있게 한다. Clear count는 reference가 실제로 읽혔고 `needs_review` flag가 없는 case만 세며, human-reviewed 판정은 아니다.
- 기본 exit code는 report 생성을 우선해 성공이다. `--fail-on-issues`는 missing file, parse failure, stale count가 있을 때 report 출력/저장 후 실패한다.
- `--fail-on-review`는 reference에 `needs_review=true`가 남아 있으면 report 출력/저장 후 실패한다.
- 이 명령도 human-reviewed 여부를 추정하지 않는다. `reference_type`은 index에 기록된 값을 집계할 뿐이며, 모델 승격 평가는 여전히 `eval-manifest --require-reference-type human-reviewed`가 담당한다.

편집한 단일 case reference를 저장하고 `case-index.json` count를 갱신할 때는 `save-review-case-reference`를 사용한다.

```bash
uv run casrt save-review-case-reference cases/case-index.json case-id edited.master.json --json
```

동작:

- 입력은 `custom-asmr-review-case-set-v1` `case-index.json`, case id, SRT 또는 master JSON transcript다.
- 해당 case의 reference file을 입력 transcript로 교체한다.
- `case-index.json` item의 `segments`와 `review_count`를 새 reference 기준으로 갱신한다.
- output format은 `custom-asmr-review-case-reference-save-v1`이다.
- 기존 reference file이 없거나 case id가 없으면 실패하며, 새 reference path를 조용히 만들지 않는다.
- 이 명령은 `reference_type`을 변경하거나 human-reviewed 승격을 판정하지 않는다.

사람이 검수한 reference들을 batch로 고정할 때는 `freeze-case-references`를 사용한다.

```bash
uv run casrt freeze-case-references cases/case-index.json \
  --reference-type human-reviewed \
  --reference-notes "manual review pass 2026-06-30" \
  --fail-on-review \
  -o cases-frozen \
  --json
```

동작:

- 입력은 `custom-asmr-review-case-set-v1` `case-index.json`이다.
- 각 reference를 시간순으로 정렬하고 `seg_000001`부터 id를 재부여하며 `needs_review=false`로 저장한다.
- output은 새 case set directory이며 `references/*.master.json`, `case-index.json`, `audio-map.json`을 만든다.
- 입력 case set이 모든 case에 candidate를 가지고 있으면 `eval-manifest.json`도 만든다.
- output `case-index.json`의 audio/candidate path는 원본 case set 파일을 absolute path로 가리킨다. 큰 audio/candidate 파일을 다시 복사하지 않기 위한 결정이다.
- audio/reference/candidate source file이 없거나 candidate가 있는 case와 없는 case가 섞이면 output directory를 만들기 전에 실패한다.
- `--fail-on-review`를 지정하면 reference에 `needs_review=true`가 남아 있을 때 output directory를 만들기 전에 실패한다.
- 이 명령도 human-reviewed 여부를 추정하지 않는다. `--reference-type human-reviewed`는 사람이 실제 검수를 끝낸 reference에만 사용한다.

candidate가 포함된 준비 case set에서 평가 manifest를 다시 만들 때는 `build-eval-manifest`를 사용한다.

```bash
uv run casrt build-eval-manifest cases/case-index.json \
  --reference-type human-reviewed \
  --reference-notes "manual review pass 2026-06-30" \
  --fail-on-review \
  -o cases/eval-manifest.human-reviewed.json \
  --json
```

동작:

- 입력은 `custom-asmr-review-case-set-v1` `case-index.json`이다.
- 모든 case에 `candidate`가 있어야 한다. candidate가 없는 검수 준비 set에서는 실패한다.
- `review-case-status`와 같은 방식으로 파일 존재 여부와 stale count를 확인하고, 문제가 있으면 manifest를 쓰지 않고 실패한다.
- `--fail-on-review`는 reference에 `needs_review=true`가 남아 있으면 manifest를 쓰지 않고 실패한다.
- output file은 `custom-asmr-eval-manifest-v1`이다.
- `--reference-type`이 있으면 root `reference_type`을 override한다. 사람이 검수한 기준본을 모델 승격에 쓰려면 이 값을 `human-reviewed`로 명시한다.
- `--reference-type`이 없으면 `case-index.json`의 root/item reference type을 보존한다.
- 이 명령은 평가를 실행하지 않는다. 생성한 manifest는 `eval-manifest --require-reference-type human-reviewed`와 품질 gate로 별도 평가한다.

### 기존 Transcript Alignment

```bash
CASRT_ALIGNER_COMMAND='.casrt/qwen-asr-venv/bin/python -m custom_asmr_srt_stack.qwen_aligner_worker --model-id /path/to/Qwen3-ForcedAligner-0.6B/snapshot' \
  uv run casrt align-transcript audio.wav candidate.master.json -o candidate.aligned.master.json --json
CASRT_ALIGNER_COMMAND='.casrt/qwen-asr-venv/bin/python -m custom_asmr_srt_stack.qwen_aligner_worker --model-id /path/to/Qwen3-ForcedAligner-0.6B/snapshot' \
  uv run casrt align-transcript audio.wav candidate.master.json -o candidate.aligned.master.json --diagnostics-output alignment-diagnostics.json --json
```

동작:

- 입력 transcript는 SRT 또는 `master.json`을 받는다.
- audio path와 transcript를 `CASRT_ALIGNER_COMMAND`에 넘겨 segment timing을 갱신한다.
- output은 `master.json`으로 저장한다.
- text, channel, kind는 aligner contract상 변경하지 않는다.
- `--diagnostics-output`은 `custom-asmr-alignment-diagnostics-v1` JSON을 쓴다. 각 segment의 original/aligned start/end, start/end/duration delta, review flag 변화를 담으며 WebUI 옵션으로 노출하지 않는다.
- 이 명령은 기존 후보를 평가 harness에 넣기 위한 도구이며 WebUI 옵션을 늘리지 않는다.

### 기존 Transcript Channel Attribution

```bash
uv run casrt attribute-channels audio.wav candidate.master.json -o candidate.attributed.master.json --json
uv run casrt attribute-channels audio.wav candidate.master.json -o candidate.attributed.master.json --diagnostics-output channel-diagnostics.json --json
```

동작:

- 입력 transcript는 SRT 또는 `master.json`을 받는다.
- audio는 stereo WAV 또는 ffmpeg로 decode 가능한 stereo audio를 받는다.
- audio를 L/R/MIX mono WAV로 정규화한 뒤, `MIX` speech segment에만 L/R RMS 기반 channel attribution을 적용한다.
- L/R 확정 기준 기본값은 8dB 차이와 quieter side -40dBFS 이하 gate다. `--threshold-db`와 `--quiet-channel-max-dbfs`는 benchmark 재현용 CLI 옵션이며 WebUI에는 노출하지 않는다.
- `--diagnostics-output`은 `custom-asmr-channel-diagnostics-v1` JSON을 쓴다. 각 segment의 original/attributed channel, L/R dBFS, delta, quieter side dBFS, decision reason을 담으며 WebUI 옵션으로 노출하지 않는다.
- `L`, `R`로 이미 라벨링된 segment, speech가 아닌 segment, 작은 L/R 차이 segment는 변경하지 않는다.
- mono audio나 L/R을 만들 수 없는 audio는 실패한다. 잘못된 channel 후처리를 조용히 통과시키지 않는다.
- stdout JSON에는 `segments`, `changed_segments`, `threshold_db`, `quiet_channel_max_dbfs`, `output`, optional `diagnostics_output`을 포함한다.

여러 case에서 channel attribution threshold를 비교할 때는 `sweep-channel-attribution`을 사용한다.

```bash
uv run casrt sweep-channel-attribution eval-manifest.json \
  --audio-map audio-map.json \
  --threshold-db 6 \
  --threshold-db 8 \
  --threshold-db 10 \
  --product-gate \
  -o channel-sweep \
  --json
```

동작:

- 입력 manifest는 `custom-asmr-eval-manifest-v1`이고 candidate transcript는 channel attribution 전 draft여야 한다.
- `audio-map`은 `custom-asmr-review-audio-map-v1` 또는 `{ "case_id": "audio.wav" }` object를 받는다.
- audio/reference/candidate source file이 없으면 output directory를 만들기 전에 실패한다.
- 각 threshold/quiet-side setting별로 attributed candidate master, generated eval manifest, eval report를 만든다.
- output root에는 `custom-asmr-channel-attribution-sweep-v1` `index.json`과 `comparison.json`을 쓴다.
- `--threshold-db`와 `--quiet-channel-max-dbfs`는 반복 가능하다. 지정하지 않으면 제품 기본값 8dB와 -40dBFS를 사용한다.
- `--product-gate` 또는 개별 gate 인자를 지정하면 `comparison.json`의 각 item에 `gate_passed`와 `gate_failures`를 붙이고, 같은 `quality_gate` metadata를 `index.json`에도 보존한다. Sweep 자체는 gate 실패 때문에 실패 exit code를 반환하지 않는다.
- 이 명령은 threshold를 자동 승격하지 않는다. 사람이 `comparison.json`과 product gate를 보고 기본값 변경 여부를 결정한다.
- WebUI 옵션으로 노출하지 않는 benchmark/운영 도구다.

### 평가

```bash
uv run casrt eval-transcript reference.srt candidate.json --json -o eval.json
```

동작:

- reference와 candidate는 SRT 또는 `master.json`을 받을 수 있다.
- speech text strict CER, practical CER, Japanese relaxed CER를 계산한다.
- segment index 기준 mean start/end/boundary error를 계산한다.
- segment 수나 split이 다른 후보를 평가하기 위해 time-overlap 기반 `timing_time_aligned`를 계산한다.
- forced alignment 재평가를 위해 boundary sample 수, max boundary error, 250ms/500ms 이내 boundary ratio를 계산한다.
- channel attribution 튜닝을 위해 index 기반 `channel`과 time-overlap 기반 `channel_time_aligned`의 L/R/MIX confusion, candidate MIX 유지 비율, L/R channel accuracy를 계산한다.
- candidate `needs_review` 비율을 계산한다. 이 값은 모델/heuristic 승격 gate에서 0이어야 한다.
- `review_effort`는 practical text mismatch, channel mismatch, 500ms 초과 timing mismatch, missing reference, extra candidate를 세고, 같은 reference segment의 중복 수정 필요는 한 번만 `segments_needing_edit`에 반영한다.
- 단일 case report의 `review_effort.items`는 사람이 고쳐야 할 segment와 reasons(`text`, `channel`, `timing`, `missing_reference`, `extra_candidate`)를 담는다. manifest summary에는 큰 리포트 팽창을 피하기 위해 items를 집계하지 않는다.
- `text_japanese_relaxed`는 practical normalization에 더해 장음류 문자 `ー〜～`를 제거한다. ASMR 발화 길이/표기 차이를 관찰하기 위한 보조 metric이며 품질 gate와 `review_effort`에는 사용하지 않는다.
- 평가는 모델 기본값 승격이나 threshold 변경 전에 실행한다.
- `--product-gate`는 문서화된 로컬 ASMR product gate를 적용한다. 기본값은 practical CER 0.10 이하, time-aligned 500ms 0.90 이상, L/R channel accuracy 0.85 이상, candidate MIX ratio 0.50 이하, review effort 0.15 이하, candidate `needs_review` 0이다. `eval-manifest`와 `compare-evals`에서는 `reference_type=human-reviewed`도 요구한다.
- `--max-practical-cer`, `--min-time-aligned-500ms-ratio`, `--min-channel-time-aligned-accuracy`, `--max-channel-time-aligned-mix-ratio`, `--max-segments-needing-edit-ratio`, `--max-candidate-review-ratio`를 지정하면 품질 gate로 동작한다. `--product-gate`와 함께 지정한 개별 threshold는 product 기본값보다 우선한다. gate 실패 시 report는 stdout/file에 남기고 exit code를 실패로 반환한다.

여러 샘플을 한 번에 평가할 때는 gold set manifest를 사용한다.

```json
{
  "format": "custom-asmr-eval-manifest-v1",
  "reference_type": "human-reviewed",
  "reference_notes": "Corrected in editor pass 2026-06-30",
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
- summary timing/channel/review/review_effort는 전체 paired/boundary/comparable/candidate/reference segment 수 기준으로 가중 집계한다.
- case `id`는 중복될 수 없다.
- root 또는 case의 `reference_type`과 `reference_notes`를 report에 보존한다.
- 제품 의사결정에서는 `reference_type=human-reviewed`만 모델 승격 근거로 쓰고, `pseudo-gold`는 regression/상대 비교로만 사용한다.
- `eval-manifest`의 품질 gate는 summary metric 기준으로 판단한다.
- `--require-reference-type human-reviewed`를 지정하면 모든 case의 effective `reference_type`이 일치해야 하며, 실패해도 report는 stdout/file에 먼저 남긴다.

### Review Effort Export

```bash
uv run casrt review-effort eval-suite.json --json -o review-effort.json
```

동작:

- 입력은 `eval-transcript` 단일 report 또는 `eval-manifest` suite report다.
- output format은 `custom-asmr-review-effort-v1`이다.
- manifest suite report에서는 각 item에 `case_id`, `case_candidate_id`, `reference_type`을 붙인다.
- paired item에는 `duration_ms`, `start_delta_ms`, `end_delta_ms`를 계산해 사람이 timing 수정 우선순위를 판단할 수 있게 한다.
- 각 item에는 `priority_score`와 `priority_rank`를 붙이고, output `items`는 `priority_score` 내림차순으로 정렬한다. 구조적 missing/extra, text, timing, channel 순서의 큰 실패를 사람이 먼저 검수하게 하기 위한 고정 queue다.
- `reason_counts`는 `text`, `channel`, `timing`, `missing_reference`, `extra_candidate`별 수정 후보 수를 세어 다음 실험/검수 작업의 우선순위를 정한다.
- 이 명령은 transcript를 수정하지 않고, WebUI 옵션을 늘리지 않는다. 품질 반복을 위한 리뷰 큐 JSON만 생성한다.

### Eval Report Comparison

```bash
uv run casrt compare-evals qwen-report.json stable-report.json quiet8-report.json --json -o comparison.json
```

동작:

- 입력은 `eval-transcript` 단일 report 또는 `eval-manifest` suite report JSON이다.
- output format은 `custom-asmr-eval-comparison-v1`이다.
- 각 report의 practical CER, optional Japanese relaxed CER, time-aligned 500ms ratio, channel time-aligned accuracy, candidate MIX ratio, candidate `needs_review` 비율, `review_effort` 수정 비율을 한 줄 summary로 뽑는다.
- ranking은 `segments_needing_edit_ratio`, practical CER, time-aligned 500ms ratio desc, channel time-aligned accuracy desc 순서다.
- `--product-gate` 또는 개별 gate 인자를 지정하면 각 item에 `gate_passed`와 `gate_failures`를 표시한다. `compare-evals` 자체는 gate 실패 때문에 실패 exit code를 반환하지 않는다.
- 이 명령은 모델/heuristic 승격을 자동 결정하지 않는다. 사람이 다음 실험 후보를 고르는 비교표만 만든다.

### Review Pack

```bash
uv run casrt review-pack review-effort.json \
  --audio-map audio-map.json \
  -o review-pack \
  --json
```

동작:

- 입력은 `custom-asmr-review-effort-v1` report다.
- `--audio`는 단일 audio file report에 사용하고, `--audio-map`은 manifest case별 audio file에 사용한다. 둘 중 하나만 허용한다. 여러 `case_id`가 있는 report에 `--audio`를 쓰면 실패한다.
- audio map은 `custom-asmr-review-audio-map-v1` 또는 `{ "case_id": "audio.wav" }` object를 받는다.
- output directory는 없거나 비어 있어야 한다. 기존 clip과 새 index가 섞이는 것을 막기 위해 non-empty directory에는 쓰지 않는다.
- 각 item의 `start_ms/end_ms`에 기본 500ms context를 붙이고 audio duration 안으로 clamp해서 `clips/*.wav`를 만든다.
- `index.json` format은 `custom-asmr-review-pack-v1`이다. 각 item은 원래 review reason/text/timing, priority score/rank, `clip_file`, `clip_start_ms`, `clip_end_ms`, `clip_context_ms`를 보존한다.
- Pack 생성은 human-reviewed gold 제작을 쉽게 하기 위한 CLI 도구이며 WebUI 옵션을 늘리지 않는다.
- WebUI는 생성된 pack directory 또는 `index.json` path를 열어 priority 순서와 `clips/*.wav`를 재생할 수 있다. WebUI는 pack을 생성하거나 context/threshold를 바꾸지 않는다.

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

수정한 project master JSON 저장:

```bash
uv run casrt project save-master PROJECT_ID edited.master.json
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
