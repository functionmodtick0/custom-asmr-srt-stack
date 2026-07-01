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

2026-07-01 all8 front120 batch CLI 평가에서 official Qwen3-ASR 1.7B local snapshot은 practical CER 59.7%, time-aligned 500ms 16.0%, review effort 100%로 실패했습니다. Granite base보다 text CER는 낮지만 product gate에는 한참 못 미칩니다. 2026-07-02 energy VAD t54/pad800/max30s sweep 후보는 coverage recall을 99.5%까지 올렸지만 실제 Qwen ASR에서는 practical CER 60.2%, time-aligned 500ms 15.2%로 baseline보다 악화되어 기본값으로 승격하지 않습니다. 같은 all8 set에서 `neosophie/Qwen3-ASR-1.7B-JA` local snapshot은 practical CER 59.4%, time-aligned 500ms 16.0%, review effort 100%로 Qwen official보다 text만 아주 조금 낫지만 product gate를 통과하지 못했습니다. Qwen3-ForcedAligner context 500/2000ms 실험도 Qwen official all8의 time-aligned 500ms를 각각 11.1%/6.9%로 낮춰 기본값으로 승격하지 않습니다. 같은 aligner를 reference-copy oracle에 적용했을 때도 time-aligned 500ms가 95.1%에서 51.2%로 떨어졌으므로, 현재 Qwen3-ForcedAligner는 기본 alignment 계층으로 승격하지 않습니다. 현 최선 alignment 정책은 no-op baseline이며, reference-copy oracle 비교에서는 time-aligned 500ms 95.1%로 alignment gate를 통과합니다. `th2_quietnone` channel attribution 후보는 candidate energy audit에서 energy-labeled 64개를 모두 맞춰 energy-proxy channel gate를 통과하지만, reference L/R label은 energy와 30개 mismatch/18개 uncertain이므로 human review 전에는 reference blocker로 남습니다. 따라서 현재 상태는 VAD/chunk/alignment/channel이 모두 실패하고 ASR text만 남은 것이 아니라, VAD는 gate pass, alignment는 no-op pass, channel attribution은 energy-proxy pass, reference human review가 ASR-only blocker로 남고 text ASR은 product quality blocker로 남은 상태입니다.

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

검수 큐를 볼 때는 CLI로 만든 `review-pack` directory, `index.json`, review case directory, `case-index.json` 경로를 WebUI의 Review path 입력에 넣고 엽니다. Review pack은 priority 순서, reason, reference/candidate text와 `clips/*.wav`를 보여줍니다. `case_summaries`와 `next_case_id`가 있는 pack은 clip을 고르기 전에도 `case 열기`로 다음 검수 case를 바로 엽니다. Review case set은 case를 클릭하면 audio와 reference master를 편집 화면에 붙이고, 수정 내용을 reference file과 `case-index.json` count에 자동 저장합니다.

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

VAD/chunking 후보의 reference speech coverage는 CLI에서만 비교합니다.

```bash
uv run casrt vad coverage audio.wav reference.master.json --json -o vad-coverage.json
uv run casrt vad coverage audio.wav reference.master.json --intervals vad-intervals.json --json
uv run casrt vad coverage-cases cases/case-index.json --json -o vad-coverage-suite.json
uv run casrt vad coverage-cases cases/case-index.json --energy-threshold-dbfs -54 --energy-pad-ms 800 --energy-max-chunk-ms 30000 --json
uv run casrt vad coverage-cases cases/case-index.json --vad-command "$CASRT_VAD_COMMAND" --json
uv run casrt vad compare-coverage energy-vad-coverage.json onnx-vad-coverage.json --json -o vad-coverage-comparison.json
uv run casrt vad compare-coverage energy-vad-coverage.json full-audio-vad-coverage.json --max-detected-interval-ms 60000 --fail-on-gate --json
uv run casrt vad compare-coverage energy-a.json energy-b.json \
  --max-detected-interval-ms 30000 \
  --max-missed-reference-ms 5000 \
  --min-reference-recall 0.995 \
  --min-detected-precision 0.99 \
  --fail-on-gate \
  --json
```

Report는 reference recall, detected precision, missed reference duration, extra detected duration, missed/extra interval 목록, detected chunk 최대/평균 길이를 포함합니다. `coverage-cases`는 prepared review case set 전체를 같은 VAD source로 집계해 case별 report와 duration-weighted summary를 함께 저장합니다. Built-in energy 옵션을 CLI에서 주면 `source_settings`에 기록됩니다. `compare-coverage`는 여러 coverage report를 missed reference duration, extra detected duration 순으로 정렬하고, `--max-detected-interval-ms`, `--max-missed-reference-ms`, `--min-reference-recall`, `--min-detected-precision`을 gate failure로 표시합니다. `--fail-on-gate`를 함께 주면 비교 JSON을 출력/저장한 뒤 gate 실패 후보가 있을 때 exit 1로 끝납니다. 이 값은 ASR text 품질이 아니라 VAD/chunking 경계 후보 비교용입니다. 현재 파이프라인은 VAD/chunk/channel/alignment 품질 검증이 끝난 상태가 아니며, ASR 모델 교체와 분리해 이 batch coverage를 먼저 확인합니다.

Qwen3-ForcedAligner는 `CASRT_QWEN_ASR_ALIGNER_MODEL_ID`로 내부 실험할 수 있습니다. `CASRT_QWEN_ASR_MIN_ALIGNED_DURATION_MS`보다 짧은 timestamp span은 clip bounds로 되돌리며, 이 값도 WebUI 옵션으로 노출하지 않습니다. Generic Qwen aligner worker는 `CASRT_QWEN_ALIGNER_MIN_ALIGNED_DURATION_MS=80`과 `CASRT_QWEN_ALIGNER_MIN_COVERAGE_RATIO=0.5` 기본 guard로 비현실적으로 짧거나 원 segment 절반 미만으로 잘린 span을 원래 timing으로 유지합니다. `CASRT_QWEN_ALIGNER_CONTEXT_MS`는 기본 0ms이며, 실험 시 segment 앞뒤 context를 붙여 기존 segment 밖 boundary 보정을 허용합니다.

`local-cohere-asr`는 Cohere Transcribe 03-2026을 위한 로컬 후보입니다. repo id가 아니라 exact revision의 local snapshot directory를 `--model-id`로 받아야 하며, worker는 `trust_remote_code=False`, `local_files_only=True`, `use_safetensors=True`로만 로드합니다. 실제 모델 다운로드/평가는 revision pin과 `casrt model digest` report 기록 전까지 실행하지 않습니다.

`local-granite-asr`는 `ibm-granite/granite-speech-4.1-2b`를 위한 로컬 후보입니다. 현재 Transformers 5.12.1이 `granite_speech`와 `granite_speech_plus`를 native 지원하므로 remote model code 없이 실행할 수 있습니다. repo id가 아니라 exact revision의 local snapshot directory를 `--model-id`로 받아야 하며, worker는 `trust_remote_code=False`, `local_files_only=True`, `use_safetensors=True`로만 로드합니다. Granite Plus timestamp prompt 실험은 `CASRT_GRANITE_ASR_PARSE_TIMESTAMPS=1`로 `[T:N]` centisecond tag를 segment timing으로 변환합니다. 이 값은 내부 실험 env이고 WebUI 옵션으로 노출하지 않습니다. 2026-07-01 01/04/07 front120 pseudo-gold 평가에서는 Granite base + filter가 practical CER 23.6%, time-aligned 500ms 21.8%, review effort 100%라 기본 승격하지 않습니다. 같은 후보에 Qwen3-ForcedAligner를 붙이면 time-aligned 500ms는 32.7%로 개선되지만 text/review gate는 그대로 실패합니다. Granite Plus timestamp prompt는 practical CER 84.1%, time-aligned 500ms 22.2%라 더 나빠 기본 승격하지 않습니다. 2026-07-01 all8 front120 batch CLI 평가에서도 Granite base는 practical CER 63.8%, time-aligned 500ms 16.3%, review effort 100%로 실패했습니다.

로컬 일본어 ASR worker는 정리 후 일본어 문자가 하나도 없는 segment를 hallucination으로 보고 버립니다. 예를 들어 `!`나 영어-only fragment는 `master.json`에 넣지 않습니다.

```bash
uv run casrt model digest /path/to/model/snapshots/<commit> \
  -o model-digest.json \
  --json
```

큰 local snapshot은 `/tmp`가 아니라 gitignored `.casrt/models/`에 보관합니다. digest report는 `.casrt/model-digests/`에 둡니다. `/tmp`는 재부팅이나 정리 작업으로 사라질 수 있으므로 다운로드 staging이나 삭제되어도 되는 실험 출력에만 사용합니다. 현재 재사용 가능한 로컬 캐시는 다음과 같습니다.

```text
.casrt/models/qwen3-asr-1.7b-hf-057a3b044fcd31c433e7971ab40d68d20e7eae6d
.casrt/model-digests/qwen3-asr-1.7b-hf-057a3b044fcd31c433e7971ab40d68d20e7eae6d-digest.json
.casrt/models/granite-speech-4.1-2b-de575db64086f84fdc79da4932d1076e965bc546
.casrt/model-digests/granite-speech-4.1-2b-de575db64086f84fdc79da4932d1076e965bc546-digest.json
.casrt/models/granite-speech-4.1-2b-plus-1454e6e1e33845ca9280ff65f52cf1141ba6e6e2
.casrt/model-digests/granite-speech-4.1-2b-plus-1454e6e1e33845ca9280ff65f52cf1141ba6e6e2-digest.json
.casrt/model-digests/qwen3-forced-aligner-0.6b-c7cbfc2048c462b0d63a45797104fc9db3ad62b7-digest.json
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

현재 durable human-review 시작점은 `.casrt/experiments/all8-front120-review-cases/case-index.json`입니다. `/home/brain-offloaded/vscode/asmr/whisperx-webui/data/uploads` 01~08 WAV와 2025-12-22 stable-ts draft SRT에서 만든 120초 pseudo-gold set이며, `review-case-status` 기준 `case_count=8`, `reference_review_count=15`, `reference_review_duration_ms=163066`, `case_issue_count=0`입니다. WebUI Review path로 이 `case-index.json`을 열고 reference를 검수한 뒤 `freeze-case-references --reference-type human-reviewed`로 승격합니다.

준비된 case set 상태 확인:

```bash
uv run casrt review-case-status cases/case-index.json --json -o cases/status.json
uv run casrt review-case-status cases/case-index.json \
  --include-reference-audits \
  --reference-channel-threshold-db 2 \
  --reference-channel-quiet-max-dbfs none \
  --json \
  -o cases/status-with-audits.json
```

`review-case-status`는 `case-index.json` 기준으로 audio/reference/candidate 파일 존재 여부, 실제 segment 수, 남은 `needs_review` 수와 duration을 다시 계산합니다. Report에는 `next_review_case_id`, candidate가 없는 `cases_missing_candidate`, candidate `needs_review`가 남은 `cases_with_candidate_review`, case별 `first_review_segment`도 포함되어 CLI/WebUI가 같은 다음 검수/후보 준비 위치를 표시할 수 있습니다. `--include-reference-audits`를 주면 구조 audit과 reference channel audit summary를 status에 붙입니다. `--fail-on-issues`는 파일 누락이나 stale count가 있을 때, `--fail-on-review`는 reference에 검수 flag가 남았을 때, `--fail-on-missing-candidates`는 candidate path가 없는 case가 있을 때, `--fail-on-candidate-review`는 candidate에 검수 flag가 남았을 때, `--fail-on-reference-audit`과 `--fail-on-reference-channel-audit`은 각 audit queue가 남았을 때 report를 출력/저장한 뒤 실패 exit code를 반환합니다.

```bash
uv run casrt audit-review-case-references cases/case-index.json \
  --json \
  -o cases/reference-audit.json \
  --review-effort-output cases/reference-audit-review-effort.json
uv run casrt audit-review-case-references cases/case-index.json \
  --fail-on-audit \
  --json \
  -o cases/reference-audit.json
```

`audit-review-case-references`는 prepared reference set의 overlap, same-channel overlap, exact-boundary overlap, long segment, near-full speech coverage, 남은 review flag를 text 없이 segment id/time/channel 중심으로 진단합니다. 기본 overlap 기준은 100ms 이상, long segment 기준은 31,000ms 이상이라 SRT 경계의 20ms 안팎 jitter와 30초 근처 절단 오차는 product blocker로 보지 않습니다. Cross-channel exact-boundary overlap은 ASMR의 동시 L/R 발화일 수 있어 raw metric으로만 남기고, same-channel exact-boundary duplicate만 구조 blocker와 review queue에 넣습니다. 1ms strict 진단이 필요하면 `--overlap-min-ms 1`을 명시합니다. `--review-effort-output`을 주면 기존 `review-pack`에 넣을 수 있는 구조 검수 queue도 만듭니다. `--fail-on-audit`은 queue item이 남아 있을 때 report를 출력/저장한 뒤 실패합니다. Pseudo-gold를 human-reviewed로 올리기 전 구조 검수 우선순위를 정하는 CLI-only 도구이며 reference를 수정하지 않습니다.

```bash
uv run casrt audit-review-case-channels cases/case-index.json \
  --threshold-db 2 \
  --quiet-channel-max-dbfs none \
  --json \
  -o cases/reference-channel-audit.json \
  --review-effort-output cases/reference-channel-audit-review-effort.json
```

`audit-review-case-channels`는 prepared reference의 L/R label을 stereo energy와 비교합니다. Output `custom-asmr-reference-channel-audit-suite-v1`은 segment id/time/channel, L/R dBFS, energy channel, match/mismatch/uncertain status만 저장하고 transcript text는 저장하지 않습니다. `--review-effort-output`은 mismatch/uncertain segment를 기존 `review-pack`에 넣을 수 있는 queue로 만듭니다. 긴 segment는 원본 `start_ms/end_ms`를 보존하면서 최대 5초 `review_clip_start_ms/review_clip_end_ms` 증거 구간을 같이 저장해 사람이 channel evidence만 빠르게 들을 수 있게 합니다. 2026-07-02 all8 pseudo-gold `threshold=2`, quiet gate off 기준 match ratio는 53.1%라, 현재 channel attribution 실패는 모델만의 문제가 아니라 reference channel 검수도 필요한 상태입니다.

```bash
uv run casrt audit-candidate-channels cases/eval-manifest.json \
  --audio-map cases/audio-map.json \
  --threshold-db 2 \
  --quiet-channel-max-dbfs none \
  -o cases/candidate-channel-audit.json
```

`audit-candidate-channels`는 candidate L/R/MIX label을 stereo energy와 비교하되 reference label을 사용하지 않습니다. Output `custom-asmr-candidate-channel-audit-suite-v1`은 energy-labeled 구간의 match, MIX로 남은 missed attribution, wrong-side, over-attribution counts/ratios를 저장하고 transcript text는 저장하지 않습니다. 이 report는 channel heuristic을 pseudo-gold reference label 문제와 분리해 보는 CLI-only 진단 입력이며, `pipeline-readiness --candidate-channel-audit`에 넣으면 `channel_attribution` stage를 energy proxy 기준으로 판정합니다.

```bash
uv run casrt review-pack cases/reference-audit-review-effort.json \
  --source-case-index cases/case-index.json \
  -o cases/reference-audit-review-pack \
  --json
```

준비된 case set의 남은 reference 검수 구간만 audio clip queue로 만들기:

```bash
uv run casrt review-case-pack cases/case-index.json -o cases/review-case-pack --json
```

`review-case-pack`은 각 reference master의 `needs_review=true` segment를 기존 `custom-asmr-review-pack-v1` 형식으로 잘라냅니다. 생성된 `index.json`은 WebUI Review path에서 열 수 있고, 사람이 pseudo-gold reference를 human-reviewed로 올리기 전에 남은 구간만 빠르게 들을 때 사용합니다.

편집한 단일 case reference를 저장하고 `case-index.json` count를 갱신:

```bash
uv run casrt save-review-case-reference cases/case-index.json case-id edited.master.json --json
```

`save-review-case-reference`는 master JSON 또는 SRT 입력을 받아 해당 case의 reference 파일을 교체하고 `segments`/`review_count`를 다시 기록합니다. 검수 완료 여부나 `reference_type`은 바꾸지 않습니다.

이미 준비된 case set에 case-local candidate transcript를 붙이기:

```bash
uv run casrt transcribe-review-case-candidates cases/case-index.json \
  -o candidate-outputs \
  --adapter local-granite-asr \
  --model-id /path/to/granite-speech-snapshot \
  --json
uv run casrt build-candidate-attach-plan cases/case-index.json candidate-outputs \
  -o attach-candidates.json \
  --candidate-id granite-base-filtered \
  --json
uv run casrt attach-review-case-candidates cases/case-index.json attach-candidates.json --json
```

`transcribe-review-case-candidates`는 prepared case audio를 기존 project 분석/전사 workflow로 처리해 `<case-id>.master.json` 후보 파일을 만듭니다. 기본 local 품질 경로에서는 `local-*` adapter와 로컬 snapshot/model id를 사용합니다. `build-candidate-attach-plan`은 candidate directory에서 `<case-id>.master.json`, `<case-id>.json`, `<case-id>.srt`를 찾아 `custom-asmr-case-candidate-attach-plan-v1` plan을 만듭니다. 모든 case가 정확히 하나의 파일에 매칭되어야 하며, 누락/모호한 중복은 output을 만들기 전에 실패합니다. Candidate 입력은 해당 case audio 기준의 SRT 또는 master JSON이어야 하며, `attach-review-case-candidates`는 `candidates/*.master.json`을 쓰고 `case-index.json`을 갱신합니다. 기존 candidate를 덮어쓸 때만 `--replace`를 사용합니다.

검수한 case reference를 한 번에 고정:

```bash
uv run casrt freeze-case-references cases/case-index.json \
  --reference-type human-reviewed \
  --reference-notes "manual review pass" \
  --fail-on-review \
  --fail-on-reference-audit \
  --fail-on-reference-channel-audit \
  --reference-channel-threshold-db 2 \
  --reference-channel-quiet-max-dbfs none \
  -o cases-frozen \
  --json
```

`freeze-case-references`는 reference id를 안정화하고 `needs_review=false`로 저장한 새 case set을 만듭니다. 사람이 실제로 검수한 경우에만 `--reference-type human-reviewed`를 사용합니다. `--fail-on-review`를 같이 쓰면 검수 flag가 남아 있을 때 output을 만들기 전에 실패하고, `--fail-on-reference-audit`은 기본 100ms 이상 same-channel overlap, same-channel exact-boundary duplicate, 31초 이상 long segment 같은 구조 검수 항목이 남아 있을 때 output을 만들기 전에 실패합니다. `--fail-on-reference-channel-audit`은 reference L/R label이 stereo energy mismatch/uncertain queue를 남길 때 output을 만들기 전에 실패합니다. 이 gate는 energy를 정답으로 승격하지 않고, human-reviewed 표시 전에 channel label 검수 누락을 막는 CLI-only 보호 장치입니다. 입력 case set에 candidate가 있으면 `eval-manifest.json`도 함께 생성합니다.

candidate가 포함된 case set에서 평가 manifest 재생성:

```bash
uv run casrt build-eval-manifest cases/case-index.json \
  --reference-type human-reviewed \
  --fail-on-review \
  --fail-on-reference-audit \
  --fail-on-reference-channel-audit \
  --reference-channel-threshold-db 2 \
  --reference-channel-quiet-max-dbfs none \
  -o cases/eval-manifest.human-reviewed.json \
  --json
```

`build-eval-manifest`는 파일 누락이나 stale count가 있으면 manifest를 쓰지 않고 실패합니다. 사람이 검수한 기준본을 모델 승격에 쓸 때는 `--reference-type human-reviewed`, `--fail-on-review`, `--fail-on-reference-audit`, `--fail-on-reference-channel-audit`, `eval-manifest --require-reference-type human-reviewed`를 함께 사용합니다.

기존 SRT 또는 master JSON을 고정 aligner로 재정렬:

```bash
CASRT_ALIGNER_COMMAND='.casrt/qwen-asr-venv/bin/python -m custom_asmr_srt_stack.qwen_aligner_worker --model-id /path/to/Qwen3-ForcedAligner-0.6B/snapshot' \
  uv run casrt align-transcript audio.wav candidate.master.json -o candidate.aligned.master.json --json
```

튜닝/검수용으로는 `--diagnostics-output alignment-diagnostics.json`을 추가해 segment별 original/aligned timing delta와 boundary delta 분포를 저장할 수 있습니다.

prepared case set의 모든 candidate를 같은 aligner로 일괄 재정렬:

```bash
CASRT_ALIGNER_COMMAND='.casrt/qwen-asr-venv/bin/python -m custom_asmr_srt_stack.qwen_aligner_worker --model-id /path/to/Qwen3-ForcedAligner-0.6B/snapshot' \
  uv run casrt align-review-case-candidates cases/case-index.json \
  -o aligned-candidates \
  --json
uv run casrt eval-manifest aligned-candidates/eval-manifest.json --product-gate
```

`align-review-case-candidates`는 원본 case set과 candidate를 수정하지 않고 `candidates/*.master.json`, `diagnostics/*.alignment-diagnostics.json`, `attach-plan.json`, `eval-manifest.json`을 새 output directory에 씁니다. Batch report에는 changed segment 수, max/mean boundary delta, 250ms/500ms 이내 boundary 비율이 포함됩니다. 이 명령도 새 WebUI 옵션을 만들지 않는 CLI-only benchmark 경로입니다.

기존 SRT 또는 master JSON에 L/R energy channel attribution 적용:

```bash
uv run casrt attribute-channels audio.wav candidate.master.json -o candidate.attributed.master.json --json
```

기본값은 L/R 차이 8dB 이상 및 quieter side -40dBFS 이하입니다. `--quiet-channel-max-dbfs none`은 quieter-side gate를 끄는 CLI-only 실험 옵션입니다. 2026-06-30 01/04/07 front120 pseudo-gold에서 stable-ts MIX-only 후보에 적용했을 때 practical CER 16.1%, time-aligned 500ms 56.7%, channel time-aligned accuracy 68.8%, MIX ratio 40.3%, review effort 64/74였습니다.

튜닝/검수용으로는 `--diagnostics-output channel-diagnostics.json`을 추가해 segment별 L/R dBFS, 판정 이유, reason/channel count summary를 저장할 수 있습니다.

여러 threshold를 같은 eval manifest에서 비교:

```bash
uv run casrt sweep-channel-attribution eval-manifest.json \
  --audio-map audio-map.json \
  --threshold-db 6 \
  --threshold-db 8 \
  --threshold-db 10 \
  --quiet-channel-max-dbfs none \
  --reset-speech-channels-to-mix \
  --product-gate \
  -o channel-sweep \
  --json
```

`sweep-channel-attribution`은 setting별 attributed candidates, eval reports, `comparison.json`을 생성합니다. 이미 L/R이 붙은 candidate를 threshold별로 공정하게 다시 채점할 때는 `--reset-speech-channels-to-mix`로 speech channel을 sweep copy 안에서만 MIX로 되돌립니다. `index.json`의 setting item은 `reason_counts`와 attributed channel counts를 보존해 threshold가 낮아서 wrong L/R을 만드는지, 높아서 MIX를 과하게 남기는지 확인하게 합니다. `--product-gate` 또는 개별 gate 인자를 함께 넣으면 `comparison.json`과 `index.json`에 gate 결과를 주석으로 남깁니다. 이 명령은 WebUI 옵션을 늘리지 않는 CLI-only benchmark 도구이며 threshold를 자동 승격하지 않습니다.

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

현재 평가는 strict CER, practical CER, Japanese relaxed CER, segment index 기준 timing 오차, time-overlap 기준 timing ratio, L/R/MIX channel confusion, L/R channel accuracy, candidate `needs_review` 비율, segment 단위 `review_effort`와 text/channel/timing/missing/extra breakdown ratio, ASMR artifact 보조 지표를 계산합니다. ASMR artifact는 non-Japanese text, 15 chars/sec 초과 high text density, 12자 이상 repeated text pattern을 세며 product gate에는 사용하지 않습니다. Japanese relaxed CER도 practical CER에서 장음류 문자 `ー〜～`를 추가로 제거한 보조 metric이며 품질 gate에는 사용하지 않습니다. 모델 승격 gate에서는 candidate `needs_review`가 남아 있지 않아야 합니다.

여러 평가 report를 한 번에 비교:

```bash
uv run casrt compare-evals qwen-report.json stable-report.json quiet8-report.json --json -o comparison.json
```

`compare-evals`는 `review_effort` 수정 비율, practical CER, timing/channel 지표, breakdown ratio, dominant review-effort reason, ASMR artifact ratio를 후보별로 뽑고 사람이 다음 실험 후보를 고르기 쉽도록 정렬합니다.
품질 gate 인자를 함께 넣으면 실패 exit code 대신 후보별 `gate_passed`와 `gate_failures`를 표시합니다. `--product-gate`는 practical CER, timing, channel, MIX ratio, review effort, candidate `needs_review`, human-reviewed reference 조건을 한 번에 표시합니다.

```bash
uv run casrt compare-review-effort qwen-report.json neosophie-report.json granite-report.json \
  --json \
  -o review-effort-comparison.json
```

`compare-review-effort`는 여러 eval report의 실패 item을 reference segment 기준으로 묶어 후보별 pass/fail과 text/channel/timing/missing/extra reason을 비교합니다. 후보끼리 서로 보완되는지, 아니면 같은 구간에서 함께 실패하는지 확인하는 CLI-only 진단 도구입니다.

여러 수정 큐를 하나로 합쳐 WebUI에서 한 번에 볼 수도 있습니다.

```bash
uv run casrt merge-review-effort reference-audit-review-effort.json reference-channel-audit-review-effort.json \
  --json \
  -o combined-review-effort.json
```

`merge-review-effort`는 `custom-asmr-review-effort-v1` queue를 입력 순서대로 검증하고, 같은 case/reference/candidate/time range issue는 reason과 evidence를 합쳐 하나의 item으로 만듭니다. Output에는 `case_summaries`, `case_count`, `next_case_id`도 들어가므로 WebUI/CLI가 어떤 case부터 검수할지 바로 표시할 수 있습니다. 같은 `source_case_index`를 가진 입력은 값을 보존하므로, 병합 결과를 바로 `review-pack`에 넣어도 case별 audio를 다시 지정할 필요가 없습니다. 서로 다른 `source_case_index`가 섞이면 출력 파일을 쓰기 전에 실패합니다. 이 명령은 reference/candidate transcript를 수정하지 않습니다.

현재 파이프라인이 ASR text 모델만 남은 단계인지 확인:

```bash
uv run casrt pipeline-readiness \
  --reference-audit cases/reference-audit.json \
  --reference-channel-audit cases/reference-channel-audit.json \
  --vad-comparison cases/vad-coverage-comparison.json \
  --eval-comparison cases/eval-comparison.json \
  --alignment-comparison cases/alignment-comparison.json \
  --channel-comparison cases/channel-sweep/comparison.json \
  --candidate-channel-audit cases/candidate-channel-audit.json \
  --product-gate \
  --fail-unless-asr-only-ready \
  --json \
  -o cases/pipeline-readiness.json
```

`pipeline-readiness`는 reference audit, optional reference channel audit, VAD coverage comparison, eval comparison을 읽어 `custom-asmr-pipeline-readiness-v1`을 만듭니다. `asr_only_ready`는 reference, VAD/chunking, alignment, channel attribution stage가 모두 pass일 때만 true입니다. VAD comparison에 `quality_gate`가 있으면 통과 후보를 VAD stage pass로 보고, gate가 없으면 missed reference speech가 남은 후보를 fail로 봅니다. `--reference-channel-audit`을 주면 reference L/R label mismatch/uncertain count도 reference stage blocker로 봅니다. `--alignment-comparison`이나 `--channel-comparison`을 주면 해당 stage만 별도 eval comparison에서 읽어, aligner oracle이나 reference-copy channel sweep을 ASR text 후보 평가와 분리할 수 있습니다. `--candidate-channel-audit`을 주면 `channel_attribution` stage는 reference label 대신 candidate energy-proxy audit로 판정하며 `--channel-comparison`보다 우선합니다. 기본 판정은 남은 edit ratio가 0보다 크면 fail인 엄격 모드이고, `--product-gate` 또는 개별 gate 인자를 주면 alignment/channel/text stage는 문서화된 threshold 기준으로 pass/fail을 계산합니다. `--product-gate`의 human-reviewed reference 조건은 reference stage에 붙어 pseudo-gold 기준본을 ASR-only ready로 보지 않습니다. Text ASR은 별도 `text_asr` stage라서, “텍스트 모델만 남았는지”와 “제품 품질이 끝났는지”를 분리해 봅니다. `--fail-unless-asr-only-ready`는 report를 출력/저장한 뒤 아직 ASR-only 단계가 아니면 실패합니다.

평가 report에서 사람이 바로 볼 수정 큐 JSON도 만들 수 있습니다.

```bash
uv run casrt review-effort eval-suite.json --json -o review-effort.json
```

`custom-asmr-review-effort-v1`에는 case id, 후보 id, reference type, 수정 reason, reference/candidate text와 timing delta가 들어갑니다. Items는 `priority_score` 내림차순으로 정렬되고 `priority_rank`가 붙어, missing/extra/text/timing/channel 실패를 큰 것부터 검수할 수 있습니다.

수정 큐에서 검수용 audio clip pack도 만들 수 있습니다.

```bash
uv run casrt review-pack review-effort.json \
  --source-case-index cases/case-index.json \
  -o review-pack \
  --json
```

`review-pack/index.json`과 `review-pack/clips/*.wav`가 생성되며, 사람이 human-reviewed gold를 만들 때 다음 수정 후보를 바로 들을 수 있습니다. `review-effort`의 priority 순서와 score/rank, root `case_summaries`/`case_count`/`next_case_id`는 pack index에도 보존됩니다. Item에 `review_clip_start_ms/review_clip_end_ms`가 있으면 clip WAV는 그 focus 구간과 context만 잘라 만들고, 원래 `start_ms/end_ms`는 source case 편집 위치로 유지합니다. `--source-case-index`를 주면 case-index의 `items[].audio`에서 case별 audio path를 자동으로 가져오고, WebUI에서 후보 실패 clip을 보다가 `case 열기`로 원 reference segment 편집 화면에 바로 들어갈 수 있습니다. `next_case_id`가 있으면 clip을 선택하지 않아도 `case 열기`가 해당 case의 첫 queue item으로 이동합니다. `review-effort` 안에 `source_case_index`가 이미 들어 있으면 이 옵션도 생략할 수 있습니다.

생성된 pack은 WebUI에서도 열 수 있습니다.

```text
Review path: /path/to/review-pack
Review path: /path/to/review-cases
```

WebUI는 review pack을 새 project로 저장하지 않고, priority item을 클릭할 때 해당 clip만 재생하는 검수 큐 보기 모드로 다룹니다. `review-effort`에서 만든 후보 수정 pack과 `review-case-pack`에서 만든 reference 검수 pack은 같은 loader를 사용합니다. Candidate가 없는 reference-only pack은 segment id를 표시하고 빈 `CAND` 줄은 숨깁니다. Reference overlap audit은 두 번째 segment를 `REF2`로 표시하고, reference channel energy audit은 `ENERGY` verdict와 L/R dBFS/delta evidence를 표시합니다. Pack item 또는 pack root에 source case 정보가 있으면 `case 열기`로 해당 case editor와 reference segment를 바로 열고, reference audit/channel audit item은 같은 구조 또는 energy evidence를 status에 유지합니다. Reference overlap audit에서 source case를 열면 `REF2` segment row도 함께 표시해 두 segment를 빠르게 비교할 수 있습니다. Review case set은 사람이 reference를 고치는 편집 모드로 열며, 목록에서 전체 `needs_review` flag 수, 남은 review duration, flag가 남은 case, 각 case의 첫 미검수 segment 시간/텍스트를 표시합니다. `검수 완료`로 현재 `needs_review` segment를 처리하고 다음 검수 segment로 이동할 수 있습니다. `case 목록`과 `다음 case`로 검수 case 사이를 이동할 수 있습니다. 모델/VAD/threshold 옵션은 추가하지 않습니다.

## 테스트

```bash
uv run python -m unittest discover -s tests
node --check web/app.js
```

## 제품 결정

제품 범위와 데이터 계약은 [docs/product-decisions.md](docs/product-decisions.md)에 기록합니다.
CLI 계약은 [docs/cli-product-decisions.md](docs/cli-product-decisions.md)에 기록합니다.
로컬 ASR 파이프라인과 평가 계획은 [docs/local-asr-pipeline.md](docs/local-asr-pipeline.md)에 기록합니다.
