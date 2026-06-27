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
- 같은 목적에 사용할 수 있는 최신 프론티어 멀티모달 모델이 있다면 구형/소형 ASR 모델을 추천하지 않는다.

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

메인 전사 모델은 사용자가 100% 직접 지정한다.

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

vLLM, SGLang 등으로 서빙되는 로컬 모델은 가능하면 OpenAI-compatible 어댑터로 연결한다.

전사 endpoint는 오디오 입력을 실제로 받아야 한다. vision-only 또는 text-only multimodal endpoint는 모델 이름이 최신이어도 ASR 경로로 사용할 수 없다.

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
- WAV가 아닌 입력은 ffmpeg로 16-bit PCM WAV로 변환한다.
- stereo WAV는 `L`, `R`, `MIX` 세 channel 파일로 분리한다.
- mono WAV는 `MIX`만 만든다.
- 모델 endpoint adapter는 `openai-compatible`, `gemini` 두 가지다.
- 모델 설정은 UI에서 사용자가 직접 입력한다.
- alignment는 UI에서 선택하지 않는다.
- `CASRT_ALIGNER_COMMAND`가 설정된 경우, 앱은 고정 aligner command를 실행한다.
- aligner command는 stdin으로 `{ audio_file, master }` JSON을 받고 stdout으로 `{ segments: [{ id, start_ms, end_ms }] }` JSON을 반환한다.
- aligner output이 id를 누락하거나 중복하면 실패한다.
- `needs_review`는 WebUI segment row에 표시한다.

## 열린 결정

- breath/SFX를 기본 export에 포함할지, 요청 시에만 포함할지.
- 실제 파일 테스트 후 split/merge가 필요한지.
- debug alignment output을 자동 보존할지, developer mode에서만 남길지.
