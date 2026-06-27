# custom-asmr-srt-stack

일본 동인 음성을 JSON 중심 전사/자막 데이터로 다루기 위한 로컬 WebUI 스택입니다.

현재 구현 범위:

- `master.json` 데이터 계약 검증
- SRT -> JSON 변환
- JSON -> SRT 변환
- 외부 번역 도구용 `translation.json` export
- 외부 번역 결과 `translated.json` import 후 SRT export
- 로컬 WebUI 서버

번역 기능은 제공하지 않습니다.

## 실행

설치 없이 실행:

```bash
PYTHONPATH=src python3 -m custom_asmr_srt_stack.cli serve
```

브라우저에서 엽니다.

```text
http://127.0.0.1:5173
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
