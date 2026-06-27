# custom-asmr-srt-stack

일본 동인 음성을 JSON 중심 전사/자막 데이터로 다루기 위한 로컬 WebUI 스택입니다.

현재 구현 범위:

- `master.json` 데이터 계약 검증
- SRT -> JSON 변환
- JSON -> SRT 변환
- 외부 번역 도구용 `translation.json` export
- 외부 번역 결과 `translated.json` import 후 SRT export
- 오디오 project 저장
- ffmpeg 기반 오디오 WAV 정규화
- WAV L/R/MIX 채널 분리
- OpenAI-compatible / Gemini 모델 endpoint adapter
- 고정 aligner command hook
- 로컬 WebUI 서버

번역 기능은 제공하지 않습니다.

## 요구 사항

- Python 3.11+
- Node.js: `web/app.js` 구문 검사용
- ffmpeg: WAV가 아닌 오디오를 전사 전에 WAV로 정규화할 때 필요

## 실행

설치 없이 실행:

```bash
PYTHONPATH=src python3 -m custom_asmr_srt_stack.cli serve
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
translation.json 내보내기
translated.json 가져오기
SRT 내보내기
```

모델 설정은 UI에서 직접 입력합니다.

```text
Adapter: openai-compatible 또는 gemini
Endpoint URL
Model ID
API Key
```

고정 aligner command를 사용하려면 서버 실행 전에 `CASRT_ALIGNER_COMMAND`를 설정합니다. 이 명령은 stdin으로 `{ audio_file, master }` JSON을 받고 stdout으로 `{ segments: [{ id, start_ms, end_ms }] }` JSON을 반환해야 합니다.

```bash
CASRT_ALIGNER_COMMAND='python3 path/to/aligner.py' \
  PYTHONPATH=src python3 -m custom_asmr_srt_stack.cli serve
```

포트를 바꾸려면:

```bash
PYTHONPATH=src python3 -m custom_asmr_srt_stack.cli serve --port 5174
```

## CLI

SRT를 내부 기준 JSON으로 변환:

```bash
PYTHONPATH=src python3 -m custom_asmr_srt_stack.cli srt-to-json input.srt -o master.json
```

번역 도구용 clean JSON 생성:

```bash
PYTHONPATH=src python3 -m custom_asmr_srt_stack.cli export-translation-json master.json -o translation.json
```

JSON에서 SRT 생성:

```bash
PYTHONPATH=src python3 -m custom_asmr_srt_stack.cli json-to-srt master.json -o export.srt
```

외부 번역 결과를 병합해서 SRT 생성:

```bash
PYTHONPATH=src python3 -m custom_asmr_srt_stack.cli json-to-srt master.json --translated translated.json -o export.srt
```

## 테스트

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
node --check web/app.js
```

## 제품 결정

제품 범위와 데이터 계약은 [docs/product-decisions.md](docs/product-decisions.md)에 기록합니다.
