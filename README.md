# custom-asmr-srt-stack

일본 동인 음성을 JSON 중심 전사/자막 데이터로 다루기 위한 로컬 WebUI 스택입니다.

현재 구현 범위:

- `master.json` 데이터 계약 검증
- SRT -> JSON 변환
- JSON -> SRT 변환
- 외부 번역 도구용 `translation.json` export
- 외부 번역 결과 `translated.json` import 후 SRT export
- 오디오 project 저장
- ffmpeg 기반 오디오 16-bit PCM WAV 정규화
- WAV L/R/MIX 채널 분리
- 분석된 chunk 단위 전사
- OpenAI-compatible / Gemini 모델 endpoint adapter
- local Transformers subprocess worker adapter
- local Qwen ASR subprocess worker adapter
- 고정 aligner command hook
- 선택 segment 재전사
- 단일 transcript 및 gold set manifest 평가
- 로컬 WebUI 서버

번역 기능은 제공하지 않습니다.

## 요구 사항

- uv
- Node.js: `web/app.js` 구문 검사용
- ffmpeg: WAV가 아니거나 채널 분리에 맞지 않는 WAV를 16-bit PCM WAV로 정규화할 때 필요

로컬 Transformers worker를 사용하려면 추가 의존성을 설치합니다.

```bash
uv sync --extra local
```

로컬 Qwen ASR worker는 `qwen-asr`가 Transformers 버전을 강하게 고정하므로 별도 uv venv로 격리합니다.

```bash
uv venv .casrt/qwen-asr-venv --python 3.12
uv pip install --python .casrt/qwen-asr-venv/bin/python -e .
uv pip install --python .casrt/qwen-asr-venv/bin/python qwen-asr==0.0.6
```

실험 실행 전 `qwen-asr` package fingerprint도 기록합니다. 현재 허용 fingerprint는 version `0.0.6`, dist-info `RECORD` SHA-256 `56454a099599cb3c86fd96347baa86269cc62e0d9eced004eeb2faa26b3a8a7c`입니다.

실행 시 `CASRT_QWEN_ASR_WORKER_COMMAND`로 Qwen venv의 Python을 지정합니다.

```bash
CASRT_QWEN_ASR_WORKER_COMMAND='.casrt/qwen-asr-venv/bin/python -m custom_asmr_srt_stack.qwen_asr_worker' \
  uv run casrt serve
```

Gemma 4 E4B는 full checkpoint를 다운로드하더라도 로딩은 4-bit runtime quantization을 켜는 구성을 권장합니다. VRAM 여유가 있으면 품질 비교용으로 `CASRT_TRANSFORMERS_QUANTIZATION=8bit`도 사용할 수 있습니다.

```bash
CASRT_TRANSFORMERS_QUANTIZATION=4bit uv run casrt serve
```

필요하면 worker generation 상한을 환경변수로 조정할 수 있습니다.

```bash
CASRT_TRANSFORMERS_MAX_NEW_TOKENS=256 uv run casrt serve
```

## 실행

서버 실행:

```bash
uv run casrt serve
```

브라우저에서 엽니다.

```text
http://127.0.0.1:5173
```

WebUI 기본 흐름:

```text
오디오/SRT/JSON 열기
모델 설정 입력
전사 시작
segment 확인/수정
선택 segment 재전사
translation.json 내보내기
translated.json 가져오기
SRT 내보내기
```

모델 설정은 UI에서 직접 입력합니다.

```text
Adapter: openai-compatible, gemini, local-transformers, local-qwen-asr, local-cohere-asr
Endpoint URL
Model ID
API Key
```

고품질 일본 ASMR 경로는 로컬 처리만 사용합니다. `local-transformers`, `local-qwen-asr`, `local-cohere-asr`는 Endpoint URL과 API Key를 사용하지 않습니다. OpenAI-compatible / Gemini adapter는 기존 호환성과 로컬 HTTP 서버 연결을 위해 남아 있지만 제품 기본 품질 경로는 아닙니다.

고정 VAD command를 사용하려면 서버 실행 전에 `CASRT_VAD_COMMAND`를 설정합니다. 이 명령은 stdin으로 `{ audio_file, audio_info }` JSON을 받고 stdout으로 `{ intervals: [{ start_ms, end_ms }] }` JSON을 반환해야 합니다. 설정하지 않으면 로컬 ASR adapter는 내장 energy splitter를 사용합니다.

```bash
CASRT_VAD_COMMAND='python3 path/to/vad.py' \
  uv run casrt serve
```

ASMR-trained Whisper ONNX VAD 후보는 내장 CLI command로도 실행할 수 있습니다. 이 모델은 ASR이 아니라 speech interval만 반환하며 WebUI 옵션으로 노출하지 않습니다.

```bash
CASRT_VAD_COMMAND='/tmp/casrt-vad-venv/bin/casrt vad whisper-asmr-onnx --model /path/to/model.onnx --metadata /path/to/model_metadata.json --force-cpu --num-threads 1' \
  uv run casrt serve
```

이 command는 전용 venv에 `.[vad-onnx]`만 설치해서 사용합니다. 모델 디렉터리는 `model.onnx`와 `model_metadata.json` 두 파일만 포함해야 하며, subprocess는 timeout과 최소 환경 변수로 실행됩니다.

내장 energy splitter는 `CASRT_QWEN_ENERGY_*` env로 내부 튜닝할 수 있지만 WebUI 옵션으로 노출하지 않습니다. `CASRT_QWEN_ENERGY_MAX_CHUNK_MS`는 긴 interval을 자르는 실험 옵션이며 현재 실데이터 평가에서는 기본값으로 켜지 않습니다.

Qwen3-ForcedAligner는 `CASRT_QWEN_ASR_ALIGNER_MODEL_ID`로 내부 실험할 수 있습니다. `CASRT_QWEN_ASR_MIN_ALIGNED_DURATION_MS`보다 짧은 timestamp span은 clip bounds로 되돌리며, 이 값도 WebUI 옵션으로 노출하지 않습니다.

`local-cohere-asr`는 Cohere Transcribe 03-2026을 위한 로컬 후보입니다. repo id가 아니라 exact revision의 local snapshot directory를 `--model-id`로 받아야 하며, worker는 `trust_remote_code=False`, `local_files_only=True`, `use_safetensors=True`로만 로드합니다. 실제 모델 다운로드/평가는 revision pin과 `casrt model digest` report 기록 전까지 실행하지 않습니다.

```bash
uv run casrt model digest /path/to/model/snapshots/<commit> \
  -o model-digest.json \
  --json
```

실데이터 benchmark처럼 보안 검토가 필요한 로컬 Qwen 실행은 repo id 대신 Hugging Face cache의 고정 snapshot directory를 `--model-id`에 넣고 다음 env를 켭니다.

```bash
CASRT_LOCAL_WORKER_ENV_MODE=offline \
CASRT_QWEN_ASR_REQUIRE_LOCAL_MODEL_PATH=1 \
CASRT_QWEN_ASR_LOCAL_FILES_ONLY=1 \
CASRT_QWEN_ASR_DISABLE_NETWORK=1 \
  uv run casrt project transcribe PROJECT_ID \
    --adapter local-qwen-asr \
    --model-id /path/to/Qwen3-ASR-1.7B/snapshots/<commit>
```

고정 aligner command를 사용하려면 서버 실행 전에 `CASRT_ALIGNER_COMMAND`를 설정합니다. 이 명령은 stdin으로 `{ audio_file, master }` JSON을 받고 stdout으로 `{ segments: [{ id, start_ms, end_ms }] }` JSON을 반환해야 합니다.

```bash
CASRT_ALIGNER_COMMAND='python3 path/to/aligner.py' \
  uv run casrt serve
```

Qwen3-ForcedAligner를 기존 master 텍스트 재정렬용으로 쓰려면 qwen-asr 전용 venv에서 내장 worker를 실행합니다.

```bash
CASRT_ALIGNER_ENV_MODE=offline \
CASRT_QWEN_ALIGNER_REQUIRE_LOCAL_MODEL_PATH=1 \
CASRT_QWEN_ALIGNER_LOCAL_FILES_ONLY=1 \
CASRT_QWEN_ALIGNER_DISABLE_NETWORK=1 \
CASRT_ALIGNER_COMMAND='.casrt/qwen-asr-venv/bin/python -m custom_asmr_srt_stack.qwen_aligner_worker --model-id /path/to/Qwen3-ForcedAligner-0.6B/snapshot' \
  uv run casrt serve
```

이 worker는 `offline + local path-only + local_files_only + network disabled` 조건이 모두 없으면 시작하지 않습니다. 또한 `qwen-asr` RECORD hash, RECORD에 기록된 각 설치 파일 hash, `qwen_asr` import origin을 검증한 뒤에만 import합니다. transcript text를 바꾸지 않고 segment 내부 start/end만 재정렬합니다. 모델 실행 전에는 local snapshot digest, `qwen-asr` package fingerprint, 보안 리뷰 결과를 실험 기록에 남깁니다.

포트를 바꾸려면:

```bash
uv run casrt serve --port 5174
```

## CLI

### Project workflow

오디오 project 생성:

```bash
uv run casrt project create-audio input.wav
```

SRT project 생성:

```bash
uv run casrt project create-srt input.srt
```

project 상태 확인:

```bash
uv run casrt project show PROJECT_ID
```

오디오 분석:

```bash
uv run casrt project analyze PROJECT_ID
```

모델 설정 검증:

```bash
uv run casrt model validate \
  --adapter openai-compatible \
  --endpoint-url http://127.0.0.1:8000/v1 \
  --model-id gemma-4-e4b
```

로컬 Transformers worker 설정 검증:

```bash
uv run casrt model validate \
  --adapter local-transformers \
  --model-id google/gemma-4-E4B-it
```

전체 project 전사:

```bash
uv run casrt project transcribe PROJECT_ID \
  --adapter openai-compatible \
  --endpoint-url http://127.0.0.1:8000/v1 \
  --model-id gemma-4-e4b
```

로컬 Transformers worker로 전사:

```bash
CASRT_TRANSFORMERS_QUANTIZATION=4bit \
  uv run casrt project transcribe PROJECT_ID \
  --adapter local-transformers \
  --model-id google/gemma-4-E4B-it
```

로컬 Qwen ASR worker로 전사:

```bash
uv run casrt project transcribe PROJECT_ID \
  --adapter local-qwen-asr \
  --model-id Qwen/Qwen3-ASR-1.7B
```

전사는 `project analyze`가 저장한 L/R 또는 MIX 채널을 chunk 단위로 잘라 모델에 보낸 뒤, 결과 타임스탬프를 원본 timeline으로 되돌려 저장합니다. `local-transformers`, `local-qwen-asr`, `local-cohere-asr`는 로컬 ASMR 경로로서 L/R이 있어도 MIX-first로 전사하고, L/R은 channel attribution에만 사용합니다.

로컬 ASR 경로는 MIX-first 전사, energy-based speech chunking, L/R energy 기반 channel attribution을 사용합니다. 세부 값과 실험 결과는 [docs/local-asr-pipeline.md](docs/local-asr-pipeline.md)에 기록합니다.

선택 segment 재전사:

```bash
uv run casrt project retranscribe PROJECT_ID seg_000001 \
  --adapter openai-compatible \
  --endpoint-url http://127.0.0.1:8000/v1 \
  --model-id gemma-4-e4b
```

project 산출물 내보내기:

```bash
uv run casrt project export-master PROJECT_ID -o master.json
uv run casrt project export-translation PROJECT_ID -o translation.json
uv run casrt project export-srt PROJECT_ID -o source.srt
uv run casrt project export-srt PROJECT_ID --translated translated.json -o translated.srt
```

자동화용 JSON 출력과 저장 위치 지정:

```bash
uv run casrt project show PROJECT_ID --json
uv run casrt project create-audio --project-root ./projects input.wav
```

### Standalone conversion

SRT를 내부 기준 JSON으로 변환:

```bash
uv run casrt srt-to-json input.srt -o master.json
```

SRT cue 텍스트가 `[L]`, `[R]`, `[LR]`, `[MIX]`로 시작하면 변환 시 channel metadata로 읽고 본문 텍스트에서는 제거합니다. `[LR]`은 내부 데이터 모델에서 `MIX`로 저장합니다.

번역 도구용 clean JSON 생성:

```bash
uv run casrt export-translation-json master.json -o translation.json
```

JSON에서 SRT 생성:

```bash
uv run casrt json-to-srt master.json -o export.srt
```

외부 번역 결과를 병합해서 SRT 생성:

```bash
uv run casrt json-to-srt master.json --translated translated.json -o export.srt
```

검수 완료 SRT 또는 master JSON을 평가 기준본으로 고정:

```bash
uv run casrt freeze-reference reviewed.srt -o refs/front120.master.json --json
```

`freeze-reference`는 segment를 시간순으로 정렬하고 `seg_000001` 형식 id를 다시 부여하며 `needs_review=false`로 저장합니다. 이 산출물은 사람이 검수한 reference manifest에서만 모델 승격 근거로 사용합니다.

기존 SRT 또는 master JSON을 고정 aligner로 재정렬:

```bash
CASRT_ALIGNER_COMMAND='.casrt/qwen-asr-venv/bin/python -m custom_asmr_srt_stack.qwen_aligner_worker --model-id /path/to/Qwen3-ForcedAligner-0.6B/snapshot' \
  uv run casrt align-transcript audio.wav candidate.master.json -o candidate.aligned.master.json --json
```

전사 결과 평가:

```bash
uv run casrt eval-transcript reference.srt candidate.json --json -o eval.json
```

품질 gate를 같이 걸면 기준 미달 시 report를 출력/저장한 뒤 실패 exit code를 반환합니다.

```bash
uv run casrt eval-transcript reference.srt candidate.json \
  --max-practical-cer 0.10 \
  --min-time-aligned-500ms-ratio 0.90 \
  --min-channel-time-aligned-accuracy 0.85 \
  --max-channel-time-aligned-mix-ratio 0.50 \
  --max-segments-needing-edit-ratio 0.15
```

gold set manifest 평가:

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

모델 승격용 평가에서는 reference authority도 gate로 건다.

```bash
uv run casrt eval-manifest gold.json \
  --require-reference-type human-reviewed \
  --max-practical-cer 0.10 \
  --min-time-aligned-500ms-ratio 0.90 \
  --max-segments-needing-edit-ratio 0.15
```

현재 평가는 strict CER, practical CER, segment index 기준 timing 오차, time-overlap 기준 timing ratio, L/R/MIX channel confusion, L/R channel accuracy, review 비율, segment 단위 `review_effort`를 계산합니다.

## 테스트

```bash
uv run python -m unittest discover -s tests
node --check web/app.js
```

## 제품 결정

제품 범위와 데이터 계약은 [docs/product-decisions.md](docs/product-decisions.md)에 기록합니다.
CLI 계약은 [docs/cli-product-decisions.md](docs/cli-product-decisions.md)에 기록합니다.
로컬 ASR 파이프라인과 평가 계획은 [docs/local-asr-pipeline.md](docs/local-asr-pipeline.md)에 기록합니다.
