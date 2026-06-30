# 제품 결정 기록

작성일: 2026-06-27

## 목표

일본 동인 음성을 편집 가능한 텍스트/자막 데이터로 만드는 로컬 WebUI 도구를 만든다.

제품의 핵심 약속은 다음이다.

```text
오디오 드롭 -> JSON 생성 -> 필요한 부분만 검토 -> JSON/SRT 내보내기
```

## 비목표

- 번역하지 않는다.
- 범용 자막 편집기를 만들지 않는다.
- 연구용 모델 튜닝 옵션을 기본 화면에 노출하지 않는다.
- WhisperX를 핵심 파이프라인으로 쓰지 않는다.
- 외부 프론티어 API를 고품질 기본 경로로 사용하지 않는다.
- 같은 목적에 사용할 수 있는 최신 로컬 ASR/aligner 모델이 있다면 구형/소형 ASR 모델을 추천하지 않는다.

번역은 명시적으로 범위 밖이다. 번역은 외부 번역 도구가 담당한다.

## 기본 산출물

JSON을 기본 제품 포맷으로 사용한다.

SRT는 호환성, 가져오기, 내보내기를 위해 지원하지만 내부 기준 데이터는 아니다.

```text
master.json          내부 기준 데이터
translation.json     외부 번역 도구용 clean JSON
translated.json      외부 번역 도구에서 돌아온 번역 결과
export.srt           생성된 자막 출력
```

## 기본 워크플로우

1. 사용자가 WebUI에 오디오 파일을 드롭한다.
2. 시스템이 오디오를 좌측, 우측, 믹스 트랙으로 분리한다.
3. 사용자가 지정한 메인 모델이 오디오를 전사한다.
4. 고정된 alignment 계층이 segment 단위 타이밍을 만든다.
5. 시스템이 `master.json`을 저장한다.
6. 사용자는 필요한 경우 segment 텍스트를 검토하고 수정한다.
7. 사용자는 외부 번역 도구에 넣을 `translation.json`을 내보낸다.
8. 사용자는 외부 번역 도구가 만든 `translated.json`을 가져온다.
9. 시스템은 원본 타이밍과 번역 텍스트를 합쳐 SRT/JSON을 내보낸다.

## 모델 정책

고품질 제품 경로는 로컬 처리만 사용한다.

외부 프론티어 API는 제품 방향이 아니다. OpenAI-compatible / Gemini 어댑터는 기존 호환성과 로컬 HTTP 서버 연동을 위해 남길 수 있지만, 제품 기본값과 품질 목표는 로컬 모델 파일 또는 로컬 런타임이다.

메인 전사 모델은 사용자가 직접 지정한다.

UI에는 직접 연결 필드를 노출한다.

```text
Endpoint URL
Model ID
API Key
연결 테스트
```

제품은 `best`, `private`, `hybrid` 같은 모드 프리셋을 제공하지 않는다.

초기 어댑터 대상은 다음이다.

- OpenAI-compatible multimodal endpoint
- Gemini API endpoint
- Local Transformers worker
- Local Qwen ASR worker

vLLM, SGLang 등으로 서빙되는 로컬 모델은 가능하면 OpenAI-compatible 어댑터로 연결할 수 있다. 다만 고품질 일본 ASMR 경로는 HTTP API가 필수가 아니며, 애플리케이션이 직접 로컬 worker subprocess를 띄울 수 있어야 한다.

전사 endpoint는 오디오 입력을 실제로 받아야 한다. vision-only 또는 text-only multimodal endpoint는 모델 이름이 최신이어도 ASR 경로로 사용할 수 없다.

로컬 Transformers 모델은 HTTP 서버를 필수로 요구하지 않는다. 애플리케이션은 `local-transformers` adapter를 통해 별도 Python subprocess worker를 띄우고 JSON Lines로 통신한다.

```text
WebUI/CLI -> local-transformers adapter -> transformers worker subprocess -> Gemma/Qwen/etc.
```

worker는 첫 전사 요청 때 lazy start하며, 같은 애플리케이션 프로세스 안에서 모델을 메모리에 유지한다. worker가 죽거나 응답 계약을 어기면 fallback으로 조용히 넘기지 않고 오류를 표시한다.

Gemma 4 E2B/E4B 계열처럼 audio clip 길이 제한이 있는 모델을 고려해, `local-transformers` adapter는 silence/energy 기반 chunk를 내부적으로 30초 이하 subchunk로 나눠 보낸다. worker가 세부 timestamp를 안정적으로 만들 수 없으면 clip 전체를 하나의 speech segment로 반환하고 `needs_review`를 표시한다. 필요한 경우 고정 alignment 계층이 후속 timing을 정리한다.

로컬 ASR adapter인 `local-transformers`와 `local-qwen-asr`는 모두 MIX-first로 전사한다. L/R 단독 전사는 조용한 ASMR에서 bleed와 low-SNR 문제를 키우므로, L/R은 텍스트 입력이 아니라 channel attribution 근거로 사용한다.

`google/gemma-4-E4B-it`의 full HF `model.safetensors`는 약 16GB라 16GB VRAM 환경에서 full precision 로딩을 기본값으로 쓰기 부적절하다. Gemma 4 E4B 로컬 실행은 `CASRT_TRANSFORMERS_QUANTIZATION=4bit` runtime quantization을 권장한다. VRAM 여유가 있으면 `8bit`도 품질 비교 대상으로 사용한다.

Gemma 4 audio path는 audio tower까지 pre-quantize된 bnb checkpoint에서 깨질 수 있다. 실제 smoke에서 `unsloth/gemma-4-E4B-it-unsloth-bnb-4bit`는 audio tower의 `bitsandbytes` `AssertionError`로 실패했다. 따라서 worker의 runtime 4-bit quantization은 `lm_head`와 `model.audio_tower`를 quantization에서 제외한다.

로컬 Transformers worker는 chunk 전사마다 generation을 기본 256 new tokens로 제한한다. 30초 이하 ASMR chunk에서 1024 tokens는 반복 생성과 장시간 block을 만들 수 있으므로 정상 경로가 아니다.

## 일본 ASMR 고품질 로컬 파이프라인

일본 ASMR/동인음성의 기본 파이프라인은 다음 순서를 목표로 한다.

```text
원본 오디오
-> WAV 정규화
-> L/R/MIX 생성
-> VAD/silence 기반 chunking
-> MIX-first ASR
-> L/R channel attribution
-> forced alignment
-> master.json
```

ASR 텍스트는 기본적으로 `MIX`에서 만든다. 실험 결과, 조용한 ASMR 파일에서 L/R 단독 전사는 channel bleed, low SNR, whisper 때문에 더 쉽게 깨졌다. L/R은 텍스트 인식의 주 경로가 아니라 발화 위치와 겹침 판단을 위한 근거로 사용한다.

우선 구현 대상은 다음이다.

- `Qwen/Qwen3-ASR-1.7B`: 주력 로컬 ASR 후보
- `neosophie/Qwen3-ASR-1.7B-JA`: Qwen3-ASR-1.7B 기반 일본어 fine-tune 후보. 2026-06-28 01/04/07 front120 확장 gold에서 practical CER 29.6%, time-aligned 500ms ratio 29.5%, channel time-aligned accuracy 73.1%라 기본 승격하지 않는다.
- `Qwen/Qwen3-ASR-1.7B-hf`: native Transformers 후보. 현재 설치된 Transformers에서는 `qwen3_asr` 아키텍처가 remote code 없이 로딩되지 않는다. 공식 Transformers main `5.13.0.dev0`에서는 support를 확인했지만 weight 다운로드가 장시간 진행돼 아직 점수화하지 못했다.
- `mistralai/Voxtral-Mini-4B-Realtime-2602`: remote model code 없이 native Transformers에서 실행 가능한 최신 로컬 후보. 2026-06-28 01/04/07 front120 확장 gold에서 practical CER 40.0%, time-aligned 500ms ratio 28.7%, channel time-aligned accuracy 63.6%라 기본 승격하지 않는다.
- `mistralai/Voxtral Mini Transcribe 2.0`: Mistral API batch transcription 제품으로 확인됐다. open-weight 로컬 checkpoint가 확인되지 않았고 외부 API는 제품 방향이 아니므로 기본 경로에서 제외한다.
- `Atotti/llm-jp-4-8b-speech-asr`: 일본어 ASR 특화 8B 후보. 현재 official Transformers에서 `LlamaForSpeechLM`을 제공하지 않고 model card의 third-party runtime package가 필요하므로 사용자 명시 승인 전에는 자동 검증하지 않는다.
- `AutoArk-AI/ARK-ASR-3B`, `CohereLabs/cohere-transcribe-03-2026`: 최신 로컬 후보. model metadata상 custom code가 필요하고 Cohere는 gated 모델이라, 사용자 명시 승인 또는 first-party package 경로가 확인된 뒤 실제 benchmark 대상으로 삼는다.
- stable-ts/Whisper계 산출물은 2026-06-28 01/04/07 front120 확장 gold에서 practical CER 16.1%, time-aligned 500ms ratio 56.7%였다. Qwen/Neosophie/Voxtral보다 text는 낫지만 기준을 만족하지 못하고 MIX-only라 channel 제품 요구사항도 충족하지 않는다. 제품 기본 경로가 아니라 비교 baseline으로만 유지한다.
- `google/gemma-4-E4B-it`: 공식 오디오 입력을 지원하는 최신 로컬 multimodal 후보. 2026-06-28 smoke 전사는 성공했지만 01/04/07 front120 gold에서 4-bit practical CER 42.3%, 8-bit practical CER 46.1%라 기본 승격하지 않는다.
- `zhifeixie/Mega-ASR`: Qwen3-ASR-1.7B 기반 robust ASR 후보. 2026-06-28 01/04/07 front120 gold에서 routed practical CER 30.9%, base-only threshold 1.1 practical CER 30.8%, forced LoRA practical CER 77.6%라 기본 승격하지 않는다.
- `TransWithAI/Whisper-Vad-EncDec-ASMR-onnx`: ASR 모델이 아니라 VAD 후보. 공개 discussion 기준 일본어 ASMR 약 500시간으로 학습된 Whisper encoder 기반 VAD다. `casrt vad whisper-asmr-onnx` command로 붙이며, WebUI 옵션으로 노출하지 않고 `CASRT_VAD_COMMAND` 뒤에 붙일 내부 후보로 둔다. 실행은 `gpt-5.4 xhigh` 정적 보안 검토의 `PASS_WITH_CONSTRAINTS` 조건을 따른다. 2026-06-28 01/04/07 front120 gold에서 단독 chunker default practical CER 30.2%, tuned practical CER 33.4%, energy-rescue hybrid practical CER 31.0%라 energy baseline 29.6%보다 나빠 기본 교체하지 않는다.
- `Qwen/Qwen3-ASR-0.6B`: 빠른 비교/저사양 후보
- `Qwen/Qwen3-ForcedAligner-0.6B`: 고정 forced alignment 후보

보안 검토가 필요한 로컬 Qwen benchmark는 repo id를 직접 실행하지 않고 고정 snapshot directory를 model id로 사용한다. 이때 `CASRT_LOCAL_WORKER_ENV_MODE=offline`, `CASRT_QWEN_ASR_REQUIRE_LOCAL_MODEL_PATH=1`, `CASRT_QWEN_ASR_LOCAL_FILES_ONLY=1`, `CASRT_QWEN_ASR_DISABLE_NETWORK=1`을 켜서 worker env를 scrub하고, cache miss/network fallback/custom remote code를 실패로 만든다.

Gemma 4 E4B 같은 general multimodal 모델은 실험 대상으로 유지하되, 제품의 일본 ASMR 품질 기준 모델 승격 여부는 동일한 gold set 평가 결과로만 결정한다. 현재 Gemma 4 E4B는 기준 미달이다.

평가 없이 모델을 기본값으로 승격하지 않는다. 최소 평가 기준은 다음이다.

- character error rate
- segment boundary error와 threshold ratio
- channel attribution accuracy
- L/R/MIX channel confusion과 candidate MIX 유지 비율
- `needs_review` segment 비율
- 사람이 실제로 고쳐야 하는 구간 수

구현 세부 값, 실험 결과, 다음 작업 계획은 [local-asr-pipeline.md](local-asr-pipeline.md)에 기록한다.

2026년 1월 이후 공개 후기/툴링 조사 기준, ASMR/동인음성에서는 단일 최신 ASR 모델보다 scene detection, ASMR-friendly VAD, forced alignment, hallucination filtering을 조합하는 파이프라인의 영향이 크다. WhisperJAV는 ASMR/VR/whisper 콘텐츠에 `fidelity` pipeline과 `aggressive` sensitivity를 추천하고, ChronosJAV는 Qwen ASR/anime-whisper/Kotoba처럼 timestamp가 없는 모델의 text generation과 timestamp alignment를 분리한다. 이 구조는 제품 기본 방향인 `MIX-first ASR -> alignment -> channel attribution`과 맞는다.

## Alignment 정책

Alignment는 애플리케이션이 고정으로 제공하며, 기본 UI에서 사용자가 선택하지 않는다.

현재 의도한 방향은 다음이다.

```text
메인 모델 전사 결과 + 오디오 -> Qwen3-ForcedAligner -> segment timing
```

aligner가 내부적으로 word/character timing을 만들 수 있더라도, 기본 제품 데이터에는 segment 단위 timing만 저장한다. word/character alignment는 외부 번역 워크플로우를 애매하게 만들기 때문에 번역용 JSON에 포함하지 않는다.

상세 alignment 데이터가 디버깅에 필요하면 제품 JSON과 분리해서 저장한다.

```text
debug_alignment.json
```

## 채널 정책

좌우 채널 정보는 텍스트가 아니라 메타데이터다.

`[L]`, `[R]` 같은 라벨을 번역 대상 텍스트에 섞지 않는다.

`master.json`은 채널을 필드로 저장한다.

```json
{
  "id": "seg_000001",
  "start_ms": 12340,
  "end_ms": 16780,
  "channel": "L",
  "kind": "speech",
  "text": "ねえ、聞こえてる？"
}
```

허용 채널 값은 다음이다.

```text
L
R
MIX
```

좌우 segment가 시간상 겹치면 둘 다 저장한다. 사용자가 볼 데이터 모델에 별도의 `overlap` 개념을 추가하지 않는다. 나중에 실제 동작 요구가 생길 때만 추가한다.

## Segment Kind 정책

`kind`는 메타데이터다.

초기 값은 다음이다.

```text
speech
breath
sfx
silence
```

내보내기 동작은 다음을 기본으로 한다.

- `translation.json`은 기본적으로 `speech` 텍스트만 내보낸다.
- SRT는 원문 또는 번역 텍스트만 사용한다.
- 숨소리/효과음 포함 여부는 나중에 export 옵션으로 둘 수 있지만, 기본 UI를 복잡하게 만들지 않는다.

## JSON 계약

### Master JSON

`master.json`은 내부 기준 데이터다.

```json
{
  "format": "custom-asmr-master-v1",
  "source_language": "ja",
  "audio": {
    "source_file": "original.wav",
    "duration_ms": 1234567
  },
  "segments": [
    {
      "id": "seg_000001",
      "start_ms": 12340,
      "end_ms": 16780,
      "channel": "L",
      "kind": "speech",
      "text": "ねえ、聞こえてる？",
      "needs_review": false
    }
  ]
}
```

segment 필수 동작은 다음이다.

- `id`는 export/import 전체에서 안정적으로 유지한다.
- `start_ms`와 `end_ms`는 segment 단위 timing이다.
- `text`에는 검토/내보내기 대상 원문 텍스트만 넣는다.
- 채널 라벨, 모델 메타데이터, 검토 플래그를 `text`에 섞지 않는다.

### Translation Export JSON

`translation.json`은 외부 번역 도구에 넣기 위한 clean JSON이다.

```json
{
  "format": "custom-asmr-translation-v1",
  "source_language": "ja",
  "items": [
    {
      "id": "seg_000001",
      "text": "ねえ、聞こえてる？"
    }
  ]
}
```

번역 도구가 번역하거나 보존해야 할 필드만 포함한다.

포함하지 않는 것:

- channel label
- timestamp
- word/character alignment
- model metadata
- review flag

### Translated JSON

`translated.json`은 외부 번역 후 가져오는 결과다.

```json
{
  "format": "custom-asmr-translated-v1",
  "source_language": "ja",
  "target_language": "ko",
  "items": [
    {
      "id": "seg_000001",
      "text": "저기, 들려?"
    }
  ]
}
```

가져오기 동작은 다음이다.

- `id`로 `master.json` segment와 매칭한다.
- timing과 metadata는 `master.json`의 값을 사용한다.
- 번역 SRT/JSON export에는 translated `text`를 사용한다.
- 필요한 id가 없거나 중복되면 조용히 넘어가지 말고 실패를 표시한다.

## SRT 변환

SRT import/export를 지원한다.

### SRT -> JSON

SRT import는 `master.json`과 호환되는 구조를 만든다.

기대 동작은 다음이다.

- 안정적인 segment id를 생성한다.
- SRT timestamp를 `start_ms`, `end_ms`로 변환한다.
- 자막 텍스트를 `text`에 저장한다.
- 채널 metadata가 없으면 `channel: "MIX"`를 사용한다.
- 기본값으로 `kind: "speech"`를 사용한다.

### JSON -> SRT

SRT export는 `master.json`의 segment timing과 원문 또는 번역 JSON의 text를 사용한다.

기본 동작은 다음이다.

- 텍스트만 출력한다.
- 채널 라벨을 넣지 않는다.
- 시간순을 유지한다.
- timing이 유효하지 않으면 조용히 보정하지 말고 실패를 표시한다.

## WebUI 범위

기본 WebUI는 최소화한다.

기본 컨트롤은 다음이다.

```text
오디오 드롭
전사 시작
모델 설정 모달 열기
선택 segment 재생
선택 segment 텍스트 수정
선택 segment 재전사
JSON 내보내기
translated JSON 가져오기
SRT 내보내기
```

기본 UI에 노출하지 않을 항목은 다음이다.

```text
VAD threshold
chunk length
alignment model
temperature
prompt template
dedupe threshold
timestamp merge rules
diarization options
```

고급 내부 설정은 나중에 config 파일이나 개발자 설정으로 둘 수 있지만, 기본 워크플로우에는 넣지 않는다.

## 편집 범위

MVP 편집 기능은 다음이다.

- segment를 클릭하면 해당 구간을 재생한다.
- segment 텍스트를 수정한다.
- 선택 segment를 재전사한다.
- JSON을 export/import한다.
- SRT를 export한다.

split/merge는 MVP 필수 요구사항으로 만들지 않는다. 실제 번역/자막 검토 과정에서 필요성이 확인될 때만 추가한다.

## Review Flag

MVP에서는 복잡한 검토 플래그 시스템을 만들지 않는다.

명백한 문제가 있을 때만 단일 `needs_review` boolean을 사용한다.

- timing이 유효하지 않음
- speech segment인데 text가 비어 있음
- segment가 비정상적으로 김
- 해당 segment의 transcription/alignment 실패

많은 confidence 값이나 heuristic 분류는 사용자가 파일을 더 빨리 끝내는 데 직접 도움이 될 때만 추가한다.

## 실패 동작

처리는 chunk 단위로 안전해야 한다.

기대 동작은 다음이다.

- 부분 결과를 보존한다.
- 실패한 chunk만 다시 시도할 수 있다.
- 부분 `master.json` 결과도 열 수 있다.
- invalid JSON, missing id, duplicate id, invalid timestamp는 실패를 표시한다.
- 텍스트나 timing 데이터를 조용히 버리지 않는다.

## 구현된 정책

현재 구현에서 확정된 정책은 다음이다.

- chunk interval 기본 최대 길이는 180초다.
- 오디오는 분석 전에 WAV로 정규화한다.
- SRT import는 cue text 선두의 `[L]`, `[R]`, `[LR]`, `[MIX]`를 channel metadata로 읽고 본문 텍스트에서 제거한다.
- `[LR]`은 현재 channel model에서 `MIX`로 저장한다.
- WAV가 아닌 입력은 ffmpeg로 16-bit PCM WAV로 변환한다.
- stereo WAV는 `L`, `R`, `MIX` 세 channel 파일로 분리한다.
- mono WAV는 `MIX`만 만든다.
- 모델 adapter는 `openai-compatible`, `gemini`, `local-transformers`, `local-qwen-asr`다.
- 모델 설정은 UI에서 사용자가 직접 입력한다.
- `local-qwen-asr`는 stereo 입력에서도 `MIX`를 먼저 전사한다.
- `local-qwen-asr`는 `CASRT_VAD_COMMAND`가 있으면 고정 VAD command interval을 사용하고, 없으면 MIX energy 기반 speech chunking을 사용한다.
- Qwen 내장 energy 기본값은 threshold `-48.0 dBFS`, window `100ms`, min silence `500ms`, min speech `200ms`, pad `200ms`다.
- Qwen 내장 energy 값은 `CASRT_QWEN_ENERGY_*` env로만 튜닝하고 WebUI 옵션으로 노출하지 않는다.
- `CASRT_QWEN_ENERGY_MAX_CHUNK_MS`는 긴 energy interval을 고정 길이 이하로 자르는 실험 옵션이다. 2026-06-30 01/04/07 front120에서 max10000은 practical CER를 29.5% -> 29.3%로만 낮추고 channel accuracy를 73.1% -> 72.0%로 떨어뜨렸으므로 기본값으로 켜지 않는다.
- `local-qwen-asr`는 L/R energy 차이로 channel attribution을 수행한다.
- VAD는 UI에서 선택하지 않는다.
- VAD command는 stdin으로 `{ audio_file, audio_info }` JSON을 받고 stdout으로 `{ intervals: [{ start_ms, end_ms }] }` JSON을 반환한다.
- VAD interval이 정렬되지 않았거나 겹치거나 audio duration을 넘으면 실패한다.
- alignment는 UI에서 선택하지 않는다.
- `CASRT_ALIGNER_COMMAND`가 설정된 경우, 앱은 고정 aligner command를 실행한다.
- aligner command는 stdin으로 `{ audio_file, master }` JSON을 받고 stdout으로 `{ segments: [{ id, start_ms, end_ms }] }` JSON을 반환한다.
- aligner output이 id를 누락하거나 중복하면 실패한다.
- `needs_review`는 WebUI segment row에 표시한다.

## 열린 결정

- breath/SFX를 기본 export에 포함할지, 요청 시에만 포함할지.
- 실제 파일 테스트 후 split/merge가 필요한지.
- debug alignment output을 자동 보존할지, developer mode에서만 남길지.
- Qwen3-ForcedAligner를 기본 timing 보정 경로로 승격할지.
- Silero/TEN VAD 중 어떤 로컬 VAD 구현을 기본 command로 고정할지.
