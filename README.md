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
uv pip install --python .casrt/qwen-asr-venv/bin/python qwen-asr
```

실행 시 `CASRT_QWEN_ASR_WORKER_COMMAND`로 Qwen venv의 Python을 지정합니다.

```bash
CASRT_QWEN_ASR_WORKER_COMMAND='.casrt/qwen-asr-venv/bin/python -m custom_asmr_srt_stack.qwen_asr_worker' \
  uv run casrt serve
```

Gemma 4 E4B는 full checkpoint를 다운로드하더라도 로딩은 4-bit runtime quantization을 켜는 구성을 권장합니다.

```bash
CASRT_TRANSFORMERS_QUANTIZATION=4bit uv run casrt serve
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
Adapter: openai-compatible, gemini, local-transformers, local-qwen-asr
Endpoint URL
Model ID
API Key
```

고품질 일본 ASMR 경로는 로컬 처리만 사용합니다. `local-transformers`와 `local-qwen-asr`는 Endpoint URL과 API Key를 사용하지 않습니다. OpenAI-compatible / Gemini adapter는 기존 호환성과 로컬 HTTP 서버 연결을 위해 남아 있지만 제품 기본 품질 경로는 아닙니다.

고정 aligner command를 사용하려면 서버 실행 전에 `CASRT_ALIGNER_COMMAND`를 설정합니다. 이 명령은 stdin으로 `{ audio_file, master }` JSON을 받고 stdout으로 `{ segments: [{ id, start_ms, end_ms }] }` JSON을 반환해야 합니다.

```bash
CASRT_ALIGNER_COMMAND='python3 path/to/aligner.py' \
  uv run casrt serve
```

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

전사는 `project analyze`가 저장한 L/R 또는 MIX 채널을 chunk 단위로 잘라 모델에 보낸 뒤, 결과 타임스탬프를 원본 timeline으로 되돌려 저장합니다.

`local-qwen-asr`는 일본 ASMR 고품질 경로로, MIX-first 전사, energy-based speech chunking, L/R energy 기반 channel attribution을 사용합니다. 세부 값과 실험 결과는 [docs/local-asr-pipeline.md](docs/local-asr-pipeline.md)에 기록합니다.

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

전사 결과 평가:

```bash
uv run casrt eval-transcript reference.srt candidate.json --json -o eval.json
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

현재 평가는 strict CER, practical CER, segment index 기준 timing 오차, L/R channel accuracy, review 비율을 계산합니다.

## 테스트

```bash
uv run python -m unittest discover -s tests
node --check web/app.js
```

## 제품 결정

제품 범위와 데이터 계약은 [docs/product-decisions.md](docs/product-decisions.md)에 기록합니다.
CLI 계약은 [docs/cli-product-decisions.md](docs/cli-product-decisions.md)에 기록합니다.
로컬 ASR 파이프라인과 평가 계획은 [docs/local-asr-pipeline.md](docs/local-asr-pipeline.md)에 기록합니다.
