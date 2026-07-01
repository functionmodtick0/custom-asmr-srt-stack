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
- local Qwen HF ASR subprocess worker adapter
- local Cohere ASR subprocess worker adapter
- local Granite ASR subprocess worker adapter
- 고정 aligner command hook
- 선택 segment 재전사
- 단일 transcript 및 gold set manifest 평가
- 로컬 WebUI 서버

번역 기능은 제공하지 않습니다.

SRT import는 선두의 `[L]`, `[R]`, `[LR]`, `[MIX]`, `[L:SPEAKER_00]`, `[R:SPEAKER_00]`, `[SPEAKER_00]` 같은 channel/speaker metadata label을 본문 텍스트에서 제거합니다. 채널은 `segment.channel`에만 저장합니다.

## 요구 사항

- uv
- Node.js: `web/app.js` 구문 검사용
- ffmpeg: WAV가 아니거나 채널 분리에 맞지 않는 WAV를 16-bit PCM WAV로 정규화할 때 필요

로컬 Transformers/ASR worker를 사용하려면 추가 의존성을 설치합니다. 이 extra에는 Granite processor가 요구하는 `torchaudio`도 포함됩니다.

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
segment text/start/end/channel/review flag 확인/수정
선택 segment 재전사
translation.json 내보내기
translated.json 가져오기
SRT 내보내기
```

검수 큐를 볼 때는 CLI로 만든 `review-pack` directory, `index.json`, review case directory, `case-index.json` 경로를 WebUI의 Review path 입력에 넣고 엽니다. Review pack은 priority 순서, reason, reference/candidate text와 `clips/*.wav`를 보여줍니다. Review case set은 case를 클릭하면 audio와 reference master를 편집 화면에 붙이고, 수정 내용을 reference file과 `case-index.json` count에 자동 저장합니다.

모델 설정은 UI에서 직접 입력합니다.
오디오를 먼저 연 뒤 SRT 또는 `master.json`을 열면, 아직 transcript가 없는 현재 오디오 project에 해당 transcript를 연결합니다.

```text
Adapter: openai-compatible, gemini, local-transformers, local-qwen-asr, local-qwen-hf-asr, local-cohere-asr, local-granite-asr
Endpoint URL
Model ID
API Key
```

고품질 일본 ASMR 경로는 로컬 처리만 사용합니다. `local-transformers`, `local-qwen-asr`, `local-qwen-hf-asr`, `local-cohere-asr`, `local-granite-asr`는 Endpoint URL과 API Key를 사용하지 않습니다. OpenAI-compatible / Gemini adapter는 기존 호환성과 로컬 HTTP 서버 연결을 위해 남아 있지만 제품 기본 품질 경로는 아닙니다.

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

Qwen3-ForcedAligner는 `CASRT_QWEN_ASR_ALIGNER_MODEL_ID`로 내부 실험할 수 있습니다. `CASRT_QWEN_ASR_MIN_ALIGNED_DURATION_MS`보다 짧은 timestamp span은 clip bounds로 되돌리며, 이 값도 WebUI 옵션으로 노출하지 않습니다. Generic Qwen aligner worker는 `CASRT_QWEN_ALIGNER_MIN_ALIGNED_DURATION_MS=80`과 `CASRT_QWEN_ALIGNER_MIN_COVERAGE_RATIO=0.5` 기본 guard로 비현실적으로 짧거나 원 segment 절반 미만으로 잘린 span을 원래 timing으로 유지합니다.

`local-cohere-asr`는 Cohere Transcribe 03-2026을 위한 로컬 후보입니다. repo id가 아니라 exact revision의 local snapshot directory를 `--model-id`로 받아야 하며, worker는 `trust_remote_code=False`, `local_files_only=True`, `use_safetensors=True`로만 로드합니다. 실제 모델 다운로드/평가는 revision pin과 `casrt model digest` report 기록 전까지 실행하지 않습니다.

`local-granite-asr`는 `ibm-granite/granite-speech-4.1-2b`를 위한 로컬 후보입니다. 현재 Transformers 5.12.1이 `granite_speech`를 native 지원하므로 remote model code 없이 실행할 수 있습니다. repo id가 아니라 exact revision의 local snapshot directory를 `--model-id`로 받아야 하며, worker는 `trust_remote_code=False`, `local_files_only=True`, `use_safetensors=True`로만 로드합니다. 2026-07-01 01/04/07 front120 pseudo-gold 평가에서는 practical CER 24.7%, time-aligned 500ms 23.7%, review effort 100%라 기본 승격하지 않습니다.

```bash
uv run casrt model digest /path/to/model/snapshots/<commit> \
  -o model-digest.json \
  --json
```

큰 local snapshot은 `/tmp`가 아니라 gitignored `.casrt/models/`에 보관합니다. digest report는 `.casrt/model-digests/`에 둡니다. 현재 재사용 가능한 로컬 캐시는 다음과 같습니다.

```text
.casrt/models/qwen3-asr-1.7b-hf-057a3b044fcd31c433e7971ab40d68d20e7eae6d
.casrt/model-digests/qwen3-asr-1.7b-hf-057a3b044fcd31c433e7971ab40d68d20e7eae6d-digest.json
.casrt/models/granite-speech-4.1-2b-de575db64086f84fdc79da4932d1076e965bc546
.casrt/model-digests/granite-speech-4.1-2b-de575db64086f84fdc79da4932d1076e965bc546-digest.json
```

외부 런타임/다운로드 도구 실행처럼 보안 검토가 필요한 로컬 Qwen benchmark는 repo id 대신 Hugging Face cache의 고정 snapshot directory를 `--model-id`에 넣고 다음 env를 켭니다.

```bash
CASRT_LOCAL_WORKER_ENV_MODE=offline \
CASRT_QWEN_ASR_REQUIRE_LOCAL_MODEL_PATH=1 \
CASRT_QWEN_ASR_LOCAL_FILES_ONLY=1 \
CASRT_QWEN_ASR_DISABLE_NETWORK=1 \
  uv run casrt project transcribe PROJECT_ID \
    --adapter local-qwen-asr \
    --model-id /path/to/Qwen3-ASR-1.7B/snapshots/<commit>
```

HF-native Qwen3-ASR worker를 사용할 때도 endpoint URL은 쓰지 않고 local snapshot directory만 넘깁니다.

```bash
CASRT_LOCAL_WORKER_ENV_MODE=offline \
CASRT_QWEN_HF_ASR_REQUIRE_LOCAL_MODEL_PATH=1 \
CASRT_QWEN_HF_ASR_LOCAL_FILES_ONLY=1 \
CASRT_QWEN_HF_ASR_DISABLE_NETWORK=1 \
  uv run casrt project transcribe PROJECT_ID \
    --adapter local-qwen-hf-asr \
    --model-id /path/to/Qwen3-ASR-1.7B-hf
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

전사는 `project analyze`가 저장한 L/R 또는 MIX 채널을 chunk 단위로 잘라 모델에 보낸 뒤, 결과 타임스탬프를 원본 timeline으로 되돌려 저장합니다. `local-transformers`, `local-qwen-asr`, `local-qwen-hf-asr`, `local-cohere-asr`는 로컬 ASMR 경로로서 L/R이 있어도 MIX-first로 전사하고, L/R은 channel attribution에만 사용합니다.

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

수정한 master JSON을 project에 다시 저장:

```bash
uv run casrt project save-master PROJECT_ID edited.master.json
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

긴 원본 audio와 SRT/master에서 평가 case를 자르기:

```bash
uv run casrt slice-case input.wav input.srt \
  --start-ms 0 \
  --end-ms 120000 \
  --audio-output cases/front120.wav \
  --transcript-output cases/front120.master.json \
  --json
```

`slice-case`는 audio와 transcript를 같은 구간으로 자르고 transcript timestamp를 0 기준으로 되돌립니다. 경계에서 잘린 segment는 사람이 확인하도록 `needs_review=true`로 표시합니다.

여러 검수 case를 한 번에 준비:

```bash
uv run casrt prepare-review-cases plan.json -o cases --json
```

`plan.json`은 `custom-asmr-case-slice-plan-v1` 형식이며 각 case의 `id`, `audio`, `reference`, `start_ms`, `end_ms`를 담습니다. 모든 case에 `candidate`가 있으면 `eval-manifest.json`도 함께 생성합니다. 출력에는 `audio-map.json`, `case-index.json`, `audio/*.wav`, `references/*.master.json`이 포함됩니다.

준비된 case set 상태 확인:

```bash
uv run casrt review-case-status cases/case-index.json --json -o cases/status.json
```

`review-case-status`는 `case-index.json` 기준으로 audio/reference/candidate 파일 존재 여부, 실제 segment 수, 남은 `needs_review` 수를 다시 계산합니다. `--fail-on-issues`는 파일 누락이나 stale count가 있을 때, `--fail-on-review`는 reference에 검수 flag가 남았을 때 report를 출력/저장한 뒤 실패 exit code를 반환합니다.

편집한 단일 case reference를 저장하고 `case-index.json` count를 갱신:

```bash
uv run casrt save-review-case-reference cases/case-index.json case-id edited.master.json --json
```

`save-review-case-reference`는 master JSON 또는 SRT 입력을 받아 해당 case의 reference 파일을 교체하고 `segments`/`review_count`를 다시 기록합니다. 검수 완료 여부나 `reference_type`은 바꾸지 않습니다.

검수한 case reference를 한 번에 고정:

```bash
uv run casrt freeze-case-references cases/case-index.json \
  --reference-type human-reviewed \
  --reference-notes "manual review pass" \
  --fail-on-review \
  -o cases-frozen \
  --json
```

`freeze-case-references`는 reference id를 안정화하고 `needs_review=false`로 저장한 새 case set을 만듭니다. 사람이 실제로 검수한 경우에만 `--reference-type human-reviewed`를 사용합니다. `--fail-on-review`를 같이 쓰면 검수 flag가 남아 있을 때 output을 만들기 전에 실패합니다. 입력 case set에 candidate가 있으면 `eval-manifest.json`도 함께 생성합니다.

candidate가 포함된 case set에서 평가 manifest 재생성:

```bash
uv run casrt build-eval-manifest cases/case-index.json \
  --reference-type human-reviewed \
  --fail-on-review \
  -o cases/eval-manifest.human-reviewed.json \
  --json
```

`build-eval-manifest`는 파일 누락이나 stale count가 있으면 manifest를 쓰지 않고 실패합니다. 사람이 검수한 기준본을 모델 승격에 쓸 때는 `--reference-type human-reviewed`와 `eval-manifest --require-reference-type human-reviewed`를 함께 사용합니다.

기존 SRT 또는 master JSON을 고정 aligner로 재정렬:

```bash
CASRT_ALIGNER_COMMAND='.casrt/qwen-asr-venv/bin/python -m custom_asmr_srt_stack.qwen_aligner_worker --model-id /path/to/Qwen3-ForcedAligner-0.6B/snapshot' \
  uv run casrt align-transcript audio.wav candidate.master.json -o candidate.aligned.master.json --json
```

튜닝/검수용으로는 `--diagnostics-output alignment-diagnostics.json`을 추가해 segment별 original/aligned timing delta를 저장할 수 있습니다.

기존 SRT 또는 master JSON에 L/R energy channel attribution 적용:

```bash
uv run casrt attribute-channels audio.wav candidate.master.json -o candidate.attributed.master.json --json
```

기본값은 L/R 차이 8dB 이상 및 quieter side -40dBFS 이하입니다. 2026-06-30 01/04/07 front120 pseudo-gold에서 stable-ts MIX-only 후보에 적용했을 때 practical CER 16.1%, time-aligned 500ms 56.7%, channel time-aligned accuracy 68.8%, MIX ratio 40.3%, review effort 64/74였습니다.

튜닝/검수용으로는 `--diagnostics-output channel-diagnostics.json`을 추가해 segment별 L/R dBFS와 판정 이유를 저장할 수 있습니다.

여러 threshold를 같은 eval manifest에서 비교:

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

`sweep-channel-attribution`은 setting별 attributed candidates, eval reports, `comparison.json`을 생성합니다. `--product-gate` 또는 개별 gate 인자를 함께 넣으면 `comparison.json`과 `index.json`에 gate 결과를 주석으로 남깁니다. 이 명령은 WebUI 옵션을 늘리지 않는 CLI-only benchmark 도구이며 threshold를 자동 승격하지 않습니다.

전사 결과 평가:

```bash
uv run casrt eval-transcript reference.srt candidate.json --json -o eval.json
```

품질 gate를 같이 걸면 기준 미달 시 report를 출력/저장한 뒤 실패 exit code를 반환합니다.

```bash
uv run casrt eval-transcript reference.srt candidate.json --product-gate
```

개별 threshold를 명시하면 product gate 기본값보다 우선합니다.

```bash
uv run casrt eval-transcript reference.srt candidate.json \
  --product-gate \
  --max-practical-cer 0.10 \
  --min-time-aligned-500ms-ratio 0.90 \
  --min-channel-time-aligned-accuracy 0.85 \
  --max-channel-time-aligned-mix-ratio 0.50 \
  --max-segments-needing-edit-ratio 0.15 \
  --max-candidate-review-ratio 0.00
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

모델 승격용 평가에서는 `--product-gate`가 reference authority와 품질 threshold를 함께 건다.

```bash
uv run casrt eval-manifest gold.json --product-gate
```

현재 평가는 strict CER, practical CER, Japanese relaxed CER, segment index 기준 timing 오차, time-overlap 기준 timing ratio, L/R/MIX channel confusion, L/R channel accuracy, candidate `needs_review` 비율, segment 단위 `review_effort`를 계산합니다. Japanese relaxed CER는 practical CER에서 장음류 문자 `ー〜～`를 추가로 제거한 보조 metric이며 품질 gate에는 사용하지 않습니다. 모델 승격 gate에서는 candidate `needs_review`가 남아 있지 않아야 합니다.

여러 평가 report를 한 번에 비교:

```bash
uv run casrt compare-evals qwen-report.json stable-report.json quiet8-report.json --json -o comparison.json
```

`compare-evals`는 `review_effort` 수정 비율, practical CER, timing/channel 지표를 후보별로 뽑고 사람이 다음 실험 후보를 고르기 쉽도록 정렬합니다.
품질 gate 인자를 함께 넣으면 실패 exit code 대신 후보별 `gate_passed`와 `gate_failures`를 표시합니다. `--product-gate`는 practical CER, timing, channel, MIX ratio, review effort, candidate `needs_review`, human-reviewed reference 조건을 한 번에 표시합니다.

평가 report에서 사람이 바로 볼 수정 큐 JSON도 만들 수 있습니다.

```bash
uv run casrt review-effort eval-suite.json --json -o review-effort.json
```

`custom-asmr-review-effort-v1`에는 case id, 후보 id, reference type, 수정 reason, reference/candidate text와 timing delta가 들어갑니다. Items는 `priority_score` 내림차순으로 정렬되고 `priority_rank`가 붙어, missing/extra/text/timing/channel 실패를 큰 것부터 검수할 수 있습니다.

수정 큐에서 검수용 audio clip pack도 만들 수 있습니다.

```bash
uv run casrt review-pack review-effort.json --audio-map audio-map.json -o review-pack --json
```

`review-pack/index.json`과 `review-pack/clips/*.wav`가 생성되며, 사람이 human-reviewed gold를 만들 때 다음 수정 후보를 바로 들을 수 있습니다. `review-effort`의 priority 순서와 score/rank는 pack index에도 보존됩니다.

생성된 pack은 WebUI에서도 열 수 있습니다.

```text
Review path: /path/to/review-pack
Review path: /path/to/review-cases
```

WebUI는 review pack을 새 project로 저장하지 않고, priority item을 클릭할 때 해당 clip만 재생하는 검수 큐 보기 모드로 다룹니다. Review case set은 사람이 reference를 고치는 편집 모드로 열며, 목록에서 전체 `needs_review` flag 수와 flag가 남은 case를 표시합니다. `검수 완료`로 현재 `needs_review` segment를 처리하고 다음 검수 segment로 이동할 수 있습니다. `case 목록`과 `다음 case`로 검수 case 사이를 이동할 수 있습니다. 모델/VAD/threshold 옵션은 추가하지 않습니다.

## 테스트

```bash
uv run python -m unittest discover -s tests
node --check web/app.js
```

## 제품 결정

제품 범위와 데이터 계약은 [docs/product-decisions.md](docs/product-decisions.md)에 기록합니다.
CLI 계약은 [docs/cli-product-decisions.md](docs/cli-product-decisions.md)에 기록합니다.
로컬 ASR 파이프라인과 평가 계획은 [docs/local-asr-pipeline.md](docs/local-asr-pipeline.md)에 기록합니다.
