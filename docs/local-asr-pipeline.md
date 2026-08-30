# 로컬 ASMR ASR 파이프라인

작성일: 2026-06-27

## 목적

일본 ASMR/동인음성 전사는 외부 API 없이 로컬에서 처리한다.

이 문서는 현재 구현된 고품질 로컬 경로, 실험 결과, 다음 작업 계획을 기록한다. 제품 범위와 데이터 계약은 `docs/product-decisions.md`가 기준이고, 이 문서는 ASR 파이프라인의 세부 구현/평가 기록이다.

## 현재 기본 방향

현재 고품질 경로는 다음 순서를 따른다.

```text
원본 오디오
-> 16-bit PCM WAV 정규화
-> L/R/MIX 채널 생성
-> 고정 20초 비중첩 chunking
-> 사용자가 고른 단일 로컬 모델로 stereo L/R 각각 전사, mono MIX 전사
-> 두 채널 결과 보존 및 시간순 정렬
-> no-op alignment
-> master.json 저장
```

핵심 결정:

- Stereo ASR 텍스트는 L/R에서 각각 만들고 둘 다 보존한다.
- Mono 입력만 MIX를 사용한다.
- Energy VAD, channel attribution, forced alignment는 명시적 개발 CLI 실험이며 production 경로가 아니다.
- `CASRT_VAD_COMMAND`, `CASRT_QWEN_ENERGY_*`, `CASRT_LOCAL_ASR_CHANNEL_MODE`, `CASRT_ALIGNER_COMMAND`는 server/project 전사를 바꾸지 않는다.
- Production local worker subprocess는 Qwen forced-aligner와 Granite timestamp parsing 실험 환경변수를 상속하지 않는다.
- `text`에는 `[L]`, `[R]` 같은 라벨을 넣지 않는다.
- 모든 channel 정보는 `segment.channel`에만 저장한다.
- 번역은 이 파이프라인에서 하지 않는다.

## Human-Reviewed Reference Workflow

제품 품질 판단은 사람이 검수한 기준본만 사용한다. 현재 01/04/07 front120 pseudo-gold는 stable-ts 산출물에서 만든 상대 비교용 기준이고, 모델 승격 근거가 아니다.

검수 흐름:

```text
후보 transcript 생성
-> 사람이 SRT 또는 master JSON에서 text, timing, channel을 검수
-> casrt freeze-reference로 reference master JSON 고정
-> gold manifest에 reference_type=human-reviewed 기록
-> eval-manifest 품질 gate로 모델/heuristic 변경 판단
```

명령:

```bash
uv run casrt freeze-reference reviewed.srt -o refs/front120.master.json --json
```

`freeze-reference`는 시간순 정렬, stable id 재부여, `needs_review=false` 저장만 수행한다. 검수 여부를 자동 판정하지 않으므로 사람이 검수하지 않은 pseudo-gold는 `reference_type=pseudo-gold`로만 기록한다.

## 로컬 Qwen ASR 런타임

기본 adapter:

```text
local-qwen-asr
```

권장 모델:

```text
Qwen/Qwen3-ASR-1.7B
```

외부 runtime이나 downloaded tooling을 실행하는 실험/benchmark는 repo id 대신 고정 snapshot directory를 model id로 넘긴다. 우리 저장소의 일반 wrapper, 테스트, 문서 변경은 subagent 보안 검토 대상이 아니며 behavior test와 자체 리뷰로 검증한다.

큰 local snapshot은 `/tmp`에만 두지 않는다. `/tmp`는 재부팅이나 정리 작업으로 사라질 수 있으므로 다운로드 staging이나 삭제되어도 되는 실험 출력에만 사용한다. 재다운로드를 피하기 위해 gitignored `.casrt/models/` 또는 persistent Hugging Face snapshot cache를 쓰고, digest report는 `.casrt/model-digests/`에 둔다. benchmark에는 그 exact directory와 digest report를 기록한다. 과거 문서의 `/tmp/casrt-quality...` 경로는 실행 provenance일 뿐 장기 보존 원본이 아니다.

```text
Qwen3-ASR-1.7B snapshot: /home/brain-offloaded/.cache/huggingface/hub/models--Qwen--Qwen3-ASR-1.7B/snapshots/7278e1e70fe206f11671096ffdd38061171dd6e5
Qwen3-ForcedAligner-0.6B snapshot: /home/brain-offloaded/.cache/huggingface/hub/models--Qwen--Qwen3-ForcedAligner-0.6B/snapshots/c7cbfc2048c462b0d63a45797104fc9db3ad62b7
```

Qwen runtime은 `qwen-asr`가 `transformers==4.57.6`을 강하게 고정하므로 root venv와 분리한다.

```bash
uv venv .casrt/qwen-asr-venv --python 3.12
uv pip install --python .casrt/qwen-asr-venv/bin/python -e .
uv pip install --python .casrt/qwen-asr-venv/bin/python qwen-asr==0.0.6
```

허용 package fingerprint:

```text
qwen-asr version: 0.0.6
qwen_asr-0.0.6.dist-info/RECORD SHA-256: 56454a099599cb3c86fd96347baa86269cc62e0d9eced004eeb2faa26b3a8a7c
```

실행 예:

```bash
CASRT_QWEN_ASR_WORKER_COMMAND='.casrt/qwen-asr-venv/bin/python -m custom_asmr_srt_stack.qwen_asr_worker' \
CASRT_QWEN_ASR_DEVICE_MAP=cuda:0 \
CASRT_QWEN_ASR_DTYPE=bfloat16 \
uv run casrt project transcribe PROJECT_ID \
  --adapter local-qwen-asr \
  --model-id Qwen/Qwen3-ASR-1.7B
```

Qwen worker는 JSON Lines subprocess protocol을 사용한다. worker import, model load, inference, response contract 오류는 fallback 없이 실패로 표시한다.

## 로컬 Qwen HF ASR 런타임

HF-native Qwen3-ASR는 qwen-asr package 경로와 분리된 adapter를 사용한다.

```text
local-qwen-hf-asr
```

실행 조건:

```text
CASRT_LOCAL_WORKER_ENV_MODE=offline
CASRT_QWEN_HF_ASR_REQUIRE_LOCAL_MODEL_PATH=1
CASRT_QWEN_HF_ASR_LOCAL_FILES_ONLY=1
CASRT_QWEN_HF_ASR_DISABLE_NETWORK=1
```

worker는 `AutoProcessor`와 `AutoModelForMultimodalLM`을 사용하고, model load는 local snapshot path + `local_files_only=True` + `trust_remote_code=False` + `use_safetensors=True`로 고정한다. Qwen HF ASR는 transcript text만 반환하므로 chunk 전체 timing과 `needs_review=true`를 반환하고, timing 품질은 후속 VAD/alignment 평가에서 본다.

`CASRT_QWEN_HF_ASR_NUM_BEAMS`는 Qwen HF generation beam count를 지정하는 CLI/runtime 실험 env다. 양의 정수만 허용하고 기본값은 `1`(greedy)이며 WebUI 옵션으로 노출하지 않는다. Model card가 beam 5 결과를 보고한 후보는 `CASRT_QWEN_HF_ASR_NUM_BEAMS=5`를 명시하고, base와 fine-tune 모두 같은 값으로 재평가해야 model weight 효과와 decoding 효과를 분리할 수 있다. `CASRT_QWEN_HF_ASR_MAX_NEW_TOKENS` 기본값은 계속 `256`이다.

Qwen HF snapshot은 load 전에 native snapshot preflight를 통과해야 한다. Root allowlist는 Qwen tokenizer/processor JSON/text files, `chat_template.jinja`, provenance files, `model.safetensors` 또는 sharded safetensors/index뿐이다. Symlink, nested runtime files, Python code, pickle/bin/pt/pth/ckpt, ONNX/GGUF/shared library/archive 등 allowlist 밖 파일은 실패한다. `config.json`은 `model_type=qwen3_asr`, architecture `Qwen3ASRForConditionalGeneration`이어야 하고 `auto_map`, `custom_code`, `remote_code`, `trust_remote_code`, internal attention override 같은 dynamic setting을 허용하지 않는다. External `chat_template.jinja`가 있으면 `CASRT_QWEN_HF_ASR_EXPECTED_CHAT_TEMPLATE_SHA256`에 64자리 expected SHA-256을 반드시 넘기고 mismatch에서 실패한다.

2026-06-30 최신 후보 상태:

- 모델: `Qwen/Qwen3-ASR-1.7B-hf`
- revision: `057a3b044fcd31c433e7971ab40d68d20e7eae6d`
- local dir: `.casrt/models/qwen3-asr-1.7b-hf-057a3b044fcd31c433e7971ab40d68d20e7eae6d` (moved to persistent cache on 2026-07-01)
- digest report: `.casrt/model-digests/qwen3-asr-1.7b-hf-057a3b044fcd31c433e7971ab40d68d20e7eae6d-digest.json`
- snapshot SHA-256: `9c5e214252ebc2be3d989c83bddc1dc7c8981389e8c27fb99f27516a1dfa556c`
- `model.safetensors` SHA-256: `2db53c7d81bd9b8cbc6a074e89be2c968a0d373fb4ee68bb1b1e14f7042dfee1`, size 4,076,193,080 bytes.
- root Transformers 5.12.1 smoke: `qwen3_asr` architecture unknown, fail closed.
- Transformers main venv: `5.13.0.dev0`, commit `45b004d7bb505a258542d1965b0f9e0d8b03b89d`.
- 5s smoke: `やば、見つかっちゃった。`
- 01/04/07 front120 pseudo-gold benchmark: practical CER 29.4%, time-aligned 500ms ratio 27.3%, channel time-aligned accuracy 68.2%, review effort 75/75 segments. Output dir: `/tmp/casrt-quality.Q5OdDf/qwen-hf-asr-transformers-main`, report: `/tmp/casrt-quality.Q5OdDf/qwen-hf-asr-transformers-main-3case-report.json`, review pack: `/tmp/casrt-quality.Q5OdDf/review-pack-qwen-hf-asr-transformers-main`.
- 결정: HF-native Qwen3-ASR는 local adapter로 유지하지만 기본 ASMR 경로로 승격하지 않는다. Timestamp 없는 full-chunk output 때문에 alignment/review burden이 크고, text도 기존 Qwen/Neosophie 계열을 이기지 못했다.

2026-08-30 latest local ASR 후보 분류:

- 바로 adapter가 있는 후보: `Qwen/Qwen3-ASR-1.7B`와 `neosophie/Qwen3-ASR-1.7B-JA`는 기존 Qwen worker/HF Qwen worker 평가 경로가 있고, `Junlaii/Bro-ASR-1.7B`는 같은 native `qwen3_asr` HF adapter를 사용한다. `ibm-granite/granite-speech-4.1-2b`와 `CohereLabs/cohere-transcribe-03-2026`은 전용 local worker가 있다. Qwen model card는 Japanese ASR와 forced alignment 지원을 명시하고, Neosophie는 Qwen3-ASR 기반 일본어 fine-tune이 Whisper/ReazonSpeech/Granite와의 자체 비교에서 낮은 CER를 보였다고 설명한다. Bro는 noisy Japanese 17,619 unseen segment에서 base보다 micro CER를 0.533 percentage point 낮췄다고 보고하며, exact revision은 `6e8f651ad6137457a2a6e813b810349a7e272f4b`다. Granite 4.1 2B model card는 Japanese ASR와 keyword-biased ASR용 synthetic data를 포함했다고 설명한다. Cohere Transcribe 03-2026 model card는 2B audio-in/text-out ASR와 14개 언어 지원을 명시한다.
- 검토 후 adapter가 필요한 후보: `Audio8/ARK-ASR-3B`는 Japanese를 지원하고 3B audio-capable ASR지만 model card가 `trust_remote_code=True`와 custom `arkasr` remote code를 요구하므로 현 local-only 기본 경로에는 바로 넣지 않는다. `Atotti/llm-jp-4-8b-speech-asr`는 Japanese speech-language ASR지만 별도 `speech_llm_ja` package/class 경로가 필요하다.
- 2026-08-30 Bro/Qwen HF external runtime 정적 보안 재검토 verdict는 `PASS_WITH_CONSTRAINTS`다. 허용 범위는 official Transformers exact commit `45b004d7bb505a258542d1965b0f9e0d8b03b89d` 전용 persistent venv, Bro exact revision local snapshot, safetensors/JSON/text allowlist, pinned chat template hash, preflight/digest, offline/local-only/network-blocked worker다. Model repo code, unsafe weights, alternate runtimes, editable external install은 금지한다. 실행 전 dependency freeze, `direct_url.json` commit, import origin, snapshot digest를 persistent artifact로 기록한다.
- 2026-08-30 Bro 실행 준비 검증: exact revision snapshot을 `.casrt/models/bro-asr-1.7b-6e8f651ad6137457a2a6e813b810349a7e272f4b`에 저장했다. 허용된 7개 root file만 남긴 뒤 Qwen HF snapshot preflight가 통과했다. Digest report는 `.casrt/model-digests/bro-asr-1.7b-6e8f651ad6137457a2a6e813b810349a7e272f4b-digest.json`, snapshot SHA-256은 `12d96810a8d5146f7da52d50c8c9576ea76d84cc56d65aca95e95e5c377fd565`, `model.safetensors` SHA-256은 upstream LFS digest와 일치하는 `b01396acde403a8eba9246cbb929db7333d28b1a9860cc14b34827a9348882d7`, `chat_template.jinja` SHA-256은 official Qwen HF와 같은 `a103579066642f0e035873f12de6046a1bbc909a7a2833732fda9fd6bc7c528d`다. Runtime provenance는 `.casrt/runtime-provenance/qwen-hf-transformers-45b004d7.json`과 대응 freeze file에 보존했다.
- 2026-08-30 Bro beam 5 real-audio smoke: `.casrt/experiments/upload-real-crop/01-front10.wav`를 실제 GPU에서 로드/추론했다. 첫 실행은 검증용 최소 venv에 Transformers audio decoder dependency인 `librosa`가 없어 model load 후 fail-fast했다. Repo `local` extra와 `uv.lock`에는 이미 `librosa==0.11.0`이 포함되어 있으므로 별도 제품 dependency 변경 없이, 전용 venv에 hash-generated requirements로 같은 버전과 wheel dependency만 추가했다. 설치 전후 freeze와 hashed requirements는 `.casrt/runtime-provenance/`에 보존했고 `uv pip check`는 77 packages compatible로 통과했으며 Transformers/Torch/NumPy/SoundFile 버전은 바뀌지 않았다. 재실행 결과 energy VAD 3개 구간에 `やば、見つかっちゃった。`, `ね、ね、魔女ちゃん。`, `こいつ強い？えっと。`를 반환했다. Reference 첫 문장 내용과 일치하는 정상 smoke지만 10초 한 건은 승격 근거가 아니므로, 다음 단계는 official Qwen HF base와 동일 beam 5, max-new-tokens 256, all8 front120 비교다.
- 2026-08-30 Bro vs official Qwen HF all8 beam 5: 두 모델 모두 exact local snapshot, official Transformers commit `45b004d7bb505a258542d1965b0f9e0d8b03b89d`, max-new-tokens 256, default energy VAD, MIX-first/channel attribution, offline worker를 사용했다. Bro candidates/report/comparison은 `.casrt/experiments/all8-front120-bro-asr-beam5-candidates`, `.casrt/experiments/all8-front120-bro-asr-beam5-eval-report.json`, `.casrt/experiments/all8-front120-qwen-hf-base-vs-bro-beam5-comparison.json`; base counterparts는 `all8-front120-qwen-hf-base-beam5-*`다. Bro는 193 candidate segments, practical CER `58.84%`, Japanese-relaxed CER `57.91%`, time-aligned 500ms `16.05%`, channel time-aligned accuracy `53.33%`, MIX ratio `62.96%`, review effort `100%`였다. Base는 167 segments, practical CER `59.87%`, Japanese-relaxed CER `58.89%`, timing `16.25%`, channel accuracy `55.17%`, MIX ratio `63.75%`, review effort `100%`였다. Bro는 practical CER를 `1.03` percentage points 낮췄고 case delta report `.casrt/experiments/all8-front120-qwen-hf-base-vs-bro-beam5-case-deltas.json`에서 8개 중 7개 case를 개선했지만 timing/channel/review는 개선하지 못했다. Bro review queue/pack은 각각 `.casrt/experiments/all8-front120-bro-asr-beam5-review-effort.json`, `.casrt/experiments/all8-front120-bro-asr-beam5-review-pack`이며 82/82 reference segments가 edit 대상이다.
- 위 all8 비교의 판단: Bro는 현재 같은 HF runtime/decoding에서 official base보다 소폭 나은 text 후보지만 기본 승격하지 않는다. Candidate practical characters는 Bro `2,876`, base `2,753`으로 reference `5,993`보다 크게 적고, pseudo-gold reference는 동시 L/R text를 별도 segment로 포함하는 반면 MIX-first ASR은 한 chunk에서 한 transcript만 만든다. 따라서 현재 high CER에는 모델 누락뿐 아니라 overlap/channel transcript 계약 mismatch도 포함된다. 다음 pipeline 반복은 모델만 교체하지 않고 human-reviewed reference와 overlap-aware stereo transcription 후보로 이 원인을 분리해야 한다.
- 2026-08-30 stereo transcription benchmark plan (historical, superseded): 당시 `CASRT_LOCAL_ASR_CHANNEL_MODE=stereo` 실험 mode로 MIX와 L/R을 비교했다. 후속 continuous20 결과와 runtime 단순화 결정으로 환경변수 mode는 production에서 제거됐고, 현재 stereo L/R은 고정 production 계약이다.
- 2026-08-30 Bro all8 stereo 결과: candidates `.casrt/experiments/all8-front120-bro-asr-beam5-stereo-candidates`, report `.casrt/experiments/all8-front120-bro-asr-beam5-stereo-eval-report.json`, MIX comparison `.casrt/experiments/all8-front120-bro-asr-beam5-mix-vs-stereo-comparison.json`. Stereo는 candidate segments `350`, practical characters `5,109`로 MIX `193`/`2,876`보다 text 양을 늘렸지만 practical CER는 `58.84% -> 64.33%`, Japanese-relaxed CER는 `57.91% -> 64.10%`로 악화했다. Timing 500ms는 `16.05% -> 15.85%`, time-aligned channel accuracy는 `53.33% -> 51.22%`, review effort는 양쪽 모두 `100%`다. Case 01/02/04는 stereo가 개선했지만 03/05/06/07/08은 악화했고 08은 practical CER `49.56% -> 89.36%`로 bleed duplicate 영향이 컸다. 전 구간 L/R 전사는 기본 승격하지 않고 CLI-only diagnostics로 유지한다.
- 2026-08-30 stereo energy gate 상한: stereo candidate를 threshold 2dB, quiet gate off로 audit한 `.casrt/experiments/all8-front120-bro-asr-beam5-stereo-candidate-channel-audit-th2.json`에서 350 segments 중 energy match `137`, wrong-side `150`, uncertain/over-attribution `63`이었다. `status=match`만 남긴 post-filter candidates/report는 `.casrt/experiments/all8-front120-bro-asr-beam5-stereo-th2-match-candidates`, `.casrt/experiments/all8-front120-bro-asr-beam5-stereo-th2-match-eval-report.json`; practical CER `66.96%`, timing 500ms `14.81%`, channel accuracy `53.09%`, review effort `100%`로 MIX와 raw stereo보다 모두 나빴다. Stereo가 개선된 case와 energy match ratio도 일관되게 상관하지 않았다. 결론: RMS 우세는 source direction evidence이지 transcript correctness/omission evidence가 아니므로 단순 energy drop filter를 제품 경로에 추가하지 않는다.
- 2026-08-30 pipeline status 판단: 구현과 automatic proxy 기준으로는 VAD/chunking, no-op alignment, candidate channel energy proxy가 pass하지만, 이번 model/channel 실험은 pseudo-gold의 긴 L/R overlap segment와 실제 후보의 짧은 MIX/LR segment 계약 차이 때문에 text/timing/channel 수치를 동시에 크게 흔든다. 따라서 ASR만 남은 상태로 보지 않는다. 다음 승격 근거는 현재 8-case reference의 남은 15 structure review items와 48 unresolved channel items를 사람이 확정한 human-reviewed manifest여야 하며, 그 전에는 Bro/MIX를 current text candidate로만 유지한다.
- 2026-08-30 channel-aware CER plan: 기존 `text_practical`은 모든 speech segment를 document order로 이어 붙이므로 겹치는 L/R candidate의 boundary 차이가 global interleave 순서를 바꿀 때 text 자체보다 큰 벌점을 줄 수 있다. `text_practical_channel_aware`를 L/R/MIX별 practical edit distance의 micro-average로 추가해 raw stereo와 energy-filtered stereo를 다시 진단한다. Wrong channel/MIX text는 deletion+insertion으로 벌점이 남고 metric은 1.0을 넘을 수 있다. 이 지표는 stereo 원인 분석 전용이며 product gate는 바꾸지 않는다.
- 2026-08-30 channel-aware CER all8 결과: 기존 manifest를 재평가했고 comparison은 `.casrt/experiments/all8-front120-bro-asr-beam5-channel-aware-comparison.json`이다. Bro MIX는 global practical CER `58.84%`, channel-aware `117.12%`; raw stereo는 global `64.33%`, channel-aware `48.21%` (`L=51.42%`, `R=45.08%`); 2dB energy-match-only stereo는 global `66.96%`, channel-aware `71.22%`였다. Raw stereo는 8개 모든 case에서 MIX보다 channel-aware CER가 낮았고, global CER 악화의 상당 부분이 L/R boundary 차이로 인한 interleave 순서 벌점임을 확인했다. 그러나 MIX는 구조적으로 불리하고 reference도 pseudo-gold라 수치 자체는 승격 근거가 아니다. 후속 결정은 fixed continuous20 L/R을 production으로 채택하고 단순 energy filter는 탈락시키는 것이다.
- 2026-08-30 fixed residual demix smoke: FFmpeg `L-0.5R`, `R-0.5L` PCM16 residual을 01/04/08 front120에 적용해 Bro stereo beam 5로 평가했다. Artifacts는 `.casrt/experiments/bro-demix-k05-3case*`; original stereo subset report는 `.casrt/experiments/all8-front120-bro-asr-beam5-stereo-01-04-08-eval-report.json`. Residual은 candidate segments `145 -> 154`, global practical CER `63.04% -> 61.77%`로 좋아졌지만 channel-aware CER `41.67% -> 42.21%`로 악화했고, case 01만 `27.10% -> 26.47%`, 04는 `29.11% -> 29.32%`, 08은 `76.09% -> 78.43%`였다. 고정 감산은 승격하지 않고 추가 coefficient brute-force도 중단한다.
- 2026-08-30 stereo aligner input bug: generic Qwen aligner worker가 L/R segment에도 full stereo clip을 전달해 aligner decoder의 channel 처리에 의존하고 있었다. Worker가 입력을 L/R/MIX mono로 한 번 분리하고 segment channel과 같은 clip을 사용하도록 수정한다. Mono audio의 L/R label은 유일한 MIX waveform을 사용한다. 기존 MIX/reference-copy aligner 실패 결론은 유지하지만, stereo candidate alignment는 수정 경로에서 다시 평가해야 한다.
- 2026-08-30 channel-aware stereo aligner result: output `.casrt/experiments/all8-front120-bro-asr-beam5-stereo-qwen-aligner-channel-aware`, comparison `comparison.json`. 350 segments 중 268개 timing이 바뀌었고 mean absolute boundary delta `311.9ms`, 자체 boundary 500ms 이내 변화율 `81.4%`였다. Unrestricted reference timing 500ms는 `15.85% -> 8.02%`, 같은 채널 전용 timing 500ms도 `15.79% -> 8.00%`로 악화했다. Channel-aware mean boundary error는 `6.88s -> 7.37s`, reference match ratio는 `92.68% -> 91.46%`로 나빠졌다. Channel accuracy `51.22% -> 55.56%`와 channel edit ratio `48.78% -> 43.90%` 개선은 바뀐 time pairing 효과이며 timing 실패를 상쇄하지 못한다. 결정: stereo input bug 수정은 유지하지만 Qwen3-ForcedAligner는 MIX와 stereo 모두 기본 승격하지 않고 no-op alignment를 유지한다.
- 2026-08-30 canonical human-review working set: Bro stereo 후보를 평가용 case copy가 아니라 권위 있는 `.casrt/experiments/all8-front120-review-cases/case-index.json`의 8개 case에 candidate draft로 연결했다. 연결 전 index는 `case-index.pre-bro-stereo-attach.json`, reference 전후 digest는 `reference-sha256.pre-bro-stereo-attach.txt`와 `reference-sha256.post-bro-stereo-attach.txt`에 보존했으며 두 digest file은 동일하다. Status는 `candidate_case_count=8`, candidate segment/review count `350`, missing file/issue `0`, reference review count `15`, unresolved channel count `48`이다. Structure/channel effort `63`개는 `.casrt/experiments/all8-front120-bro-stereo-human-reference-review-effort.json`에서 중복 제거된 `55`개 issue로 병합했고, canonical source를 여는 pack은 `.casrt/experiments/all8-front120-bro-stereo-human-reference-review-pack-canonical`이다. Pack은 55개 clip, 실제 clip duration 합계 `356010ms`, `next_case_id=01-front120-existing-srt`를 기록한다. Candidate 연결은 reference text/timing/channel 또는 `pseudo-gold` authority를 바꾸지 않는다. HTTP smoke에서 case index의 candidate path/id만 반환하고 candidate master 본문을 누락하는 정상 경로 버그를 발견해, server가 attached candidate를 검증/로드하고 WebUI reference row가 시간이 겹치는 같은-channel/MIX candidate channel/time/text를 읽기 전용으로 표시하도록 수정했다. 실제 canonical first case Playwright smoke는 `10 ref / 50 cand`, 표시 candidate context line `47`, desktop/mobile horizontal overflow `0`, line overflow `0`, page error `0`이었다. Screenshot은 pack directory의 `ui-desktop.png`, `ui-mobile.png`에 보존했다. 사람 검수와 `human-reviewed` freeze 후 같은 기준으로 VAD, MIX/stereo channel policy, no-op/new aligner, text model을 다시 평가하기 전에는 non-text stage를 최종 완료로 간주하지 않는다.
- 2026-08-30 reference authority evidence audit: 기존 canonical pack은 구조/channel issue `55`개만 포함해 전체 reference `82`개 중 자동 flag가 없는 나머지 text/timing을 사람이 들었다는 증거를 만들 수 없었다. `needs_review=false`를 human review로 해석하지 않도록 master segment에 backward-compatible `content_reviewed` boolean을 추가했으며, 누락은 `false`다. 기존 WebUI `검수 완료`가 내용 검수 증거를 저장하고 text/time 재편집 또는 clipped slice가 이를 무효화한다. Channel-only 완료는 별도 `channel_reviewed`만 저장한다. `review-case-status` output `.casrt/experiments/all8-front120-review-cases/status-content-review-evidence.json`은 content reviewed `0`, unreviewed `82`, unreviewed segment-duration sum `1676656ms`, 8/8 pending case, file/index issue `0`을 확인했다. 전체 pack `.casrt/experiments/all8-front120-full-content-human-reference-review-pack`은 clip `82`, reasons `{reference-content-unreviewed:82, reference-needs-review:15}`, actual clip-duration sum `1745030ms`, max clip `31007ms`, first case `01-front120-existing-srt`를 기록한다. Content-only clip filename도 `reference-content-unreviewed` provenance를 사용하고, 자동 flag가 함께 있으면 `reference-needs-review`를 primary filename reason으로 쓴다. `freeze-case-references`와 `build-eval-manifest`는 effective `human-reviewed` authority를 쓰기 전에 모든 segment의 명시적 content review evidence를 fail-fast로 요구한다. 판단: 이 82개 전체 content-review pack을 사람이 처리하기 전에는 reference, VAD/chunking, alignment, channel, ASR 어느 단계도 최종 품질로 승격하지 않는다.
- 2026-08-30 unified content/channel review queue: reference audit output `.casrt/experiments/all8-front120-review-cases/reference-audit-content-authority.json`과 effort output `...-review-effort.json`은 content reviewed `0`, unreviewed `82`, 자동 flag `15`, product same-channel overlap/long segment `0`을 기록하고 review item을 중복 없이 `82`개로 만든다. 이를 human-review-aware channel effort 48개와 합친 `.casrt/experiments/all8-front120-combined-content-channel-review-effort.json`은 input `130 -> 82` items, reasons `{reference-content-unreviewed:82, reference-needs-review:15, reference-channel-energy-mismatch:30, reference-channel-energy-uncertain:18}`다. Channel focus 5초를 그대로 두면 content text/timing을 검수할 수 없으므로, merge는 channel-only item에만 focus를 유지하고 content/structure reason이 있으면 full-segment scope를 우선한다. 최종 pack `.casrt/experiments/all8-front120-combined-content-channel-human-reference-review-pack`은 clip `82`, channel evidence item `48`, focus item `0`, clip duration sum `1745030ms`, max `31007ms`, case `8`, next case `01-front120-existing-srt`다. 판단: 두 pack을 오갈 필요 없이 이 경로를 canonical human review UI로 사용한다.
- 2026-08-30 content-authority readiness rerun: output `.casrt/experiments/all8-front120-pipeline-readiness-content-authority-vad-downstream.json`은 reference reason을 review flags `15`, explicit content evidence missing `82`, pseudo-gold authority, unresolved channel mismatch `30`/uncertain `18`로 모두 표시한다. ASR-only blockers는 `[reference,vad_chunking]`, product blockers는 `[reference,vad_chunking,text_asr]`로 유지된다. 판단: 이전 readiness의 `reference` 실패를 단순 pseudo-gold label이 아니라 실제 82개 검수 진행률로 관측할 수 있게 됐으며, 사람이 처리하기 전 ASR-only 단계라고 주장할 수 없다.
- 2026-08-30 canonical WebUI/seek verification: localhost `8766`에서 combined pack과 canonical case API를 실제로 열었다. 긴 WAV API가 HTTP Range를 무시해 source segment click이 요청한 `98.513s` 대신 0초에서 재생되는 정상 경로 버그를 발견했다. Server에 single-byte Range `206`/`Content-Range`/`Accept-Ranges`와 invalid range `416`을 추가하고, WebUI는 `loadeddata` 이후 최신 playback request만 실행한다. Curl smoke `bytes=1000-1999`는 `206`, `Content-Length=1000`, source size `23040044`를 반환했다. Playwright report와 screenshots는 combined pack의 `ui-smoke-report.json`, `ui-desktop.png`, `ui-mobile.png`다. Desktop `1440x1000`과 mobile `390x844` 모두 pack header `82 review clips · 8 cases · listen 29:05.030`, first source case `10 ref · 50 cand`, selected `seg_000009`, candidate context line `47`/selected line `7`, active `검수 완료`, playback `98.6s`, horizontal overflow `0`, page/console error `0`이었다.
- 2026-08-30 resumable canonical review queue: static 82-item pack이 사람이 처리한 item도 재로드 시 계속 노출하는 진행 상태 결손을 수정했다. Loader는 source case/reference segment를 한 index당 한 번 읽고 reason별 content/review-flag/channel requirement와 현재 evidence를 결합한다. Canonical combined pack을 실제 loader로 읽은 결과는 total `82`, pending `82`, resolved `0`; 첫 item requirements는 `[content, review-flag, channel]`이고 unresolved다. WebUI는 완료 item을 숨기고 priority slot을 건너뛰며 남은 clip duration만 표시한다. Case 저장이 성공할 때 in-memory pack state도 갱신하므로 완료 후 목록 복귀나 다음 issue 이동에 재생성이 필요 없고, 이후 evidence-invalidating edit을 저장하면 다시 pending이 된다. Source segment 누락은 fail-visible이다. 판단: 사람 검수는 같은 canonical pack 경로에서 중단/재개할 수 있으며 별도 progress 파일을 관리하지 않는다.
- 2026-08-30 overlap-context VAD split challenger plan: t54/pad800/max30s는 coverage와 일부 global/timing metric을 개선했지만 raw stereo보다 channel-aware CER가 악화했다. 기본 VAD/제품 preset은 유지하고, 내부 실험용으로 각 VAD/core 경계 양쪽에 bounded audio context를 붙인 뒤 output segment midpoint가 해당 core 안에 있는 결과만 채택하는 owner-filtered splitter를 구현한다. Core는 gap/overlap 없이 검출 interval을 정확히 덮고 audio bounds만 file 범위 안에서 VAD interval 바깥으로 확장한다. 음수 context와 max chunk 없는 context 설정은 fail-fast한다. 먼저 기존 raw/t54 결과가 엇갈린 `01/04/08` front120에서 exact Bro snapshot, beam 5, stereo, t54/pad800/max30s를 고정해 no-context 대 context 후보를 비교한다. Global practical CER, channel-aware practical CER, unrestricted/same-channel timing 500ms 중 하나라도 baseline을 악화하면 중단하고, 모두 non-regression일 때만 all8로 확장한다. Artifacts와 최종 채택/기각 판단은 `.casrt/experiments/`에 보존한다.
- 2026-08-30 overlap-context v1 invalid-hypothesis result: 첫 구현은 context를 원 energy interval 안으로 제한했고 `01/04/08` Bro stereo context1000 결과가 no-context와 segment/text/timing/channel metric까지 완전히 동일했다. Artifacts는 `.casrt/experiments/bro-stereo-t54-pad800-max30s-context1000-01-04-08-{candidates,projects,eval-manifest.json,eval-report.json,comparison.json}`과 대응 no-context subset manifest/report다. Aggregate는 두 후보 모두 candidate `121`, practical CER `60.99%`, channel-aware CER `44.67%`, unrestricted timing 500ms `21.67%`, same-channel timing 500ms `18.33%`였다. 원인을 전수 확인한 결과 all8 L/R t54 energy interval은 `320`개, 최대 `27,000ms`, 30초 hard split `0`개여서 context가 실제 audio bounds를 전혀 바꾸지 않았다. 판단: max30 hard-cut 원인 가설은 기각한다. 구현은 VAD interval 자체의 시작/끝 바깥으로 context audio를 확장하되 accept core는 원 interval에 유지하도록 수정하고 같은 3-case를 새 output에서 재실행한다.
- 2026-08-30 overlap-context VAD-boundary result/rejection: corrected context1000 candidate는 `.casrt/experiments/bro-stereo-t54-pad800-max30s-vad-context1000-01-04-08-{candidates,projects,eval-manifest.json,eval-report.json,comparison.json}`에 보존했다. Segment count는 no-context와 같은 `121`이지만 practical candidate characters가 `2,411 -> 3,106`으로 늘었고 practical CER `60.99% -> 77.48%`, channel-aware CER `44.67% -> 67.80%`, unrestricted timing 500ms `21.67% -> 8.33%`, same-channel timing 500ms `18.33% -> 8.33%`로 네 승격 metric이 모두 악화했다. Channel accuracy만 `50.00% -> 53.33%`로 올랐지만 text/timing 실패를 상쇄하지 못한다. 원인은 Bro/Qwen timestamp segment가 owner core 안에 있어도 text 자체는 앞뒤 context 발화를 함께 포함할 수 있어 midpoint filtering이 중복 text를 자르지 못하는 것이다. 계획의 first-stage non-regression gate에서 탈락했으므로 context 크기 추가 sweep과 all8 추론을 중단하고, `CASRT_QWEN_ENERGY_CHUNK_CONTEXT_MS`, accept/core path, 관련 제품 계약을 코드에서 제거했다. 판단: token/word-level timestamp ownership이 없는 현재 local ASR output에는 hard VAD bounds를 유지한다.
- 2026-08-30 stereo VAD coverage parity plan: Bro stereo t54 candidate의 ASR segment `320`개가 L/R energy interval `320`개와 정확히 같아 VAD clip이 text context와 timing을 직접 소유하지만, 기존 `vad coverage(-cases)`는 원 stereo interleaved RMS 한 벌만 측정해 local ASR runtime과 일치하지 않는다. CLI에 `--channel-mode {mix,stereo}`를 추가하고 기본 `mix`는 runtime과 같은 L/R 평균 mono MIX waveform을 사용한다. Stereo는 WAV를 L/R mono로 분리하고 L unit은 reference L/MIX, R unit은 R/MIX speech를 비교한다. Suite root `case_count`는 source case 수를 유지하고 summary `coverage_unit_count`로 실제 waveform 평가 수를 별도 기록하며 case item은 stable `case-id:MIX/L/R`, `case_id`, `channel`을 가진다. Mono, stereo+single fixed `--intervals`, L/R 없는 input은 fail-fast하고 external VAD command도 실제 runtime처럼 channel별로 실행한다. 과거 raw-stereo coverage report는 historical provenance로만 유지한다. 먼저 all8 t54/pad800/max30s에서 parity report를 만들고, 그 위에서 min-silence `500/800/1200/2000ms` interval/coverage grid를 계산해 01/04/08 Bro downstream 후보 하나만 선택한다. 옵션은 diagnostic CLI에만 두고 WebUI에는 노출하지 않는다.
- 2026-08-30 stereo VAD runtime-parity grid: `vad coverage-cases --channel-mode stereo` 구현은 default mix에서 runtime mono MIX waveform을, stereo에서 L/R mono와 same-channel+MIX reference를 사용한다. Behavior tests는 source case/coverage unit 분리, MIX reference 양쪽 포함, mono/stereo+fixed-interval fail-fast를 검증한다. All8 t54/pad800/max30s outputs는 `.casrt/experiments/all8-front120-stereo-energy-t54-sil{500,800,1200,2000}-pad800-max30s-vad-coverage-runtime-parity.json`이다. `sil500`: intervals `320`, mean/max `5,390/27,000ms`, miss `168,621ms`, extra `217,488ms`, recall/precision `89.94/87.39%`. `sil800`: `162`, `10,649/30,000ms`, miss `168,221ms`, extra `217,488ms`, `89.96/87.39%`. `sil1200`: `99`, `17,519/30,000ms`, miss `158,921ms`, extra `217,488ms`, `90.52/87.46%`. `sil2000`: `71`, `24,466/30,000ms`, miss `157,710ms`, extra `218,977ms`, `90.59/87.39%`. 과거 combined-RMS t54 recall `99.51%`는 stereo runtime VAD 근거로 사용할 수 없다. 01/04/08 subset interval/miss는 각각 sil500 `121/1,387ms`, sil800 `62/1,387ms`, sil1200 `32/1,387ms`, sil2000 `25/1,176ms`; pseudo-reference segment `30`과 granularity가 가장 가깝고 all8 recall/precision도 sil500보다 나은 `sil1200` 하나만 exact Bro downstream으로 진행한다. Coverage만으로 승격하지 않는다.
- 2026-08-30 Bro stereo t54/sil1200 3-case downstream gate: exact Bro snapshot, beam 5, stereo, pad800/max30s를 고정하고 min-silence만 `500 -> 1200ms`로 바꾼 output/report/comparison은 `.casrt/experiments/bro-stereo-t54-sil1200-pad800-max30s-01-04-08-{candidates,projects,eval-manifest.json,eval-report.json,comparison.json}`다. Candidate segments `121 -> 32`; practical CER `60.99% -> 60.87%`, channel-aware CER `44.67% -> 42.12%`, unrestricted timing 500ms `21.67% -> 26.67%`, same-channel timing 500ms `18.33% -> 25.00%`로 선언한 네 non-regression gate를 모두 통과했다. 반면 channel time-aligned accuracy는 `50.00% -> 46.67%`, review effort는 계속 `100%`다. 판단: 기본/승격 후보가 아니라 all8 확장 자격만 얻었으며, all8에서 text/channel-aware/timing 개선 유지와 channel accuracy 손실을 함께 판정한다.
- 2026-08-30 Bro stereo t54/sil1200 all8 result: output/report/comparison은 `.casrt/experiments/all8-front120-bro-stereo-t54-sil1200-pad800-max30s-{candidates,projects,eval-manifest.json,eval-report.json,comparison.json}`다. Sil500 대비 segments `320 -> 99`, practical CER `63.36% -> 61.04%`, channel-aware CER `49.31% -> 47.44%`, unrestricted timing 500ms `17.68% -> 23.17%`, same-channel timing 500ms `16.03% -> 23.08%`로 3-case 개선이 all8에서도 유지됐다. Channel time-aligned accuracy는 `52.44% -> 48.78%`, review effort는 `100%`라 product/default VAD 승격은 금지한다. Raw stereo 대비로도 practical `64.33% -> 61.04%`, channel-aware `48.21% -> 47.44%`, timing `15.85% -> 23.17%`, same-channel timing `15.79% -> 23.08%`가 개선되고 channel accuracy만 `51.22% -> 48.78%`로 악화한다. Candidate energy audit `.casrt/experiments/all8-front120-bro-stereo-t54-sil1200-pad800-max30s-candidate-channel-audit-th2.json`은 sil500의 match/wrong-side `124/138`(`47.33/52.67%`)에서 `39/35`(`52.70/47.30%`)로 자기 waveform 일관성을 개선하고 uncertain을 `58/320 -> 25/99`로 줄였다. 판단: unresolved pseudo-reference channel label 때문에 pipeline 승격은 보류하지만, 사람 검수용 read-only candidate draft는 raw 350 segments보다 text/timing/밀도가 나은 sil1200 99 segments로 교체한다. Reference master/digest/authority는 바꾸지 않는다.
- 2026-08-30 canonical candidate draft replacement: attach plan `.casrt/experiments/all8-front120-bro-stereo-t54-sil1200-pad800-max30s-attach-plan.json`으로 canonical `.casrt/experiments/all8-front120-review-cases/case-index.json`의 candidate 8/8을 `bro-asr-1.7b-beam5-stereo-t54-sil1200-pad800-max30s`로 `--replace`했다. Candidate count/review count는 `99/99`, missing file/case issue `0`; status는 `.casrt/experiments/all8-front120-review-cases/status-after-sil1200-candidate-attach.json`이다. 교체 전후 reference SHA-256 8개는 각각 `dc86107a`, `629977f1`, `f82129f8`, `a02764fb`, `a335b369`, `4a02b295`, `1fd549c8`, `3a1e64b7` prefix로 모두 동일하다. Server loader smoke는 candidate segments `99`, first case `9`, content unreviewed `82`, canonical pack pending/resolved `82/0`을 확인했다. 판단: WebUI 사람이 보는 candidate context만 `350 -> 99`로 줄었고 reference, review evidence, pack progress, pseudo-gold authority는 바뀌지 않았다.
- 2026-08-30 runtime-parity/current-candidate readiness: VAD comparison `.casrt/experiments/all8-front120-stereo-energy-sil500-vs-sil1200-vad-quality-gated-runtime-parity.json`은 max30s/recall90% gate에서 sil1200을 coverage winner로 고르고 sil500을 recall `89.94%`로 탈락시킨다. 같은 Bro stereo downstream과 실제 sil1200 candidate energy audit을 사용한 readiness `.casrt/experiments/all8-front120-pipeline-readiness-runtime-parity-sil1200-real-candidate-channel.json`은 ASR-only blockers `[reference,vad_chunking,channel_attribution]`, product blockers에 `text_asr`를 추가한다. Reference는 flags `15`, content evidence missing `82`, pseudo-gold, channel mismatch/uncertain `30/18`; VAD는 practical/channel-aware CER와 두 timing metric 개선에도 channel accuracy `52.44% -> 48.78%` 회귀 하나로 fail; channel은 energy-labeled match `52.70% < 85%`, wrong-side `47.30%`, uncertain over-attribution `25/99`로 fail이다. Alignment는 reference-copy no-op proxy pass, text best는 Bro MIX practical CER `58.84%`/review effort `100%`로 fail이다. 판단: review UX와 candidate draft는 개선됐지만 파이프라인은 ASR text-only 단계가 아니다.
- 2026-08-30 padded energy interval overlap bug/repair plan: `speech_intervals_by_energy`는 raw speech ranges를 min-silence로 merge한 뒤 각 range를 pad하지만 padded 결과를 다시 merge하지 않는다. Sil1200/pad800 all8 L/R 실데이터에서 overlap `23`쌍, 합계 `4,000ms`가 발생해 같은 audio를 인접 ASR call에 중복 전달했다. Case/channel pair count는 01R `1`, 02L `4`, 03R `1`, 04L `1`, 06L/R `4/1`, 07L/R `5/2`, 08L/R `2/2`다. 이는 튜닝 변동이 아니라 non-overlap chunk 계약 위반이므로 padding 후 overlap/접촉 interval을 병합하고 index를 다시 부여한다. Default pad200/min-silence500처럼 overlap이 없는 경로는 그대로여야 하며, empty/fallback 동작도 유지한다. 수정 후 runtime-parity grid와 sil1200 01/04/08 Bro downstream을 새 artifact에서 재실행하고, gate 통과 시에만 all8/canonical draft를 다시 교체한다. 기존 sil1200 결과는 overlapped-chunk provenance로 보존하고 최종 VAD 근거로 쓰지 않는다.
- 2026-08-30 non-overlap stereo VAD grid: core repair 후 full 342-test suite가 통과했다. 새 reports는 기존 artifact를 덮지 않은 `.casrt/experiments/all8-front120-stereo-energy-t54-sil{500,800,1200,2000}-pad800-max30s-vad-coverage-runtime-parity-nonoverlap.json`이다. Coverage union miss/extra/recall/precision은 예상대로 이전과 같지만 실제 chunk count/mean은 sil500 `320 -> 80`/`21,559ms`, sil800 `162 -> 79`/`21,837ms`, sil1200 `99 -> 78`/`22,236ms`, sil2000 `71`/`24,466ms`다; 모두 max `30,000ms`이며 padded chunk overlap `0`이다. 01/04/08 subset은 sil500/800/1200이 모두 `26 chunks`, miss `1,387ms`, extra `117,168ms`로 동일하고 sil2000만 `25`, miss `1,176ms`, extra `117,757ms`다. 판단: 이전 sil1200 개선은 min-silence 단독 효과로 해석할 수 없고 padding-overlap이 주 원인이었다. 동일 설정 fixed sil1200 26-chunk subset을 새로 전사해 buggy sil1200 32-chunk 결과와 raw/sil500 provenance를 다시 비교한다.
- 2026-08-30 fixed non-overlap sil1200 3-case rejection: outputs는 `.casrt/experiments/bro-stereo-t54-sil1200-pad800-max30s-nonoverlap-01-04-08-{candidates,projects,eval-manifest.json,eval-report.json,comparison.json}`다. Buggy sil1200 대비 segments `32 -> 26`, practical CER는 `60.87%` 동일하지만 channel-aware CER `42.12% -> 42.33%`, unrestricted timing 500ms `26.67% -> 20.00%`, same-channel timing `25.00% -> 20.00%`로 악화했다. 원래 sil500 subset 대비 practical/channel-aware/same-channel timing은 개선하지만 unrestricted timing `21.67% -> 20.00%`, channel accuracy `50.00% -> 46.67%`가 악화해 사전 non-regression gate를 실패한다. 판단: fixed sil1200 all8과 추가 min-silence sweep을 중단한다. Overlap repair는 chunk contract bug fix로 유지하지만, overlap이 있던 sil1200 99-segment output은 canonical draft에서 제거하고 default pad200/min-silence500으로 생성돼 overlap이 없는 raw Bro stereo 350-segment draft로 복구한다.
- 2026-08-30 canonical raw draft restore/final corrected readiness: restore plan `.casrt/experiments/all8-front120-bro-stereo-raw-restore-attach-plan.json`으로 canonical candidate를 `bro-asr-1.7b-beam5-stereo` 350 segments로 되돌렸다. Status `.casrt/experiments/all8-front120-review-cases/status-after-raw-candidate-restore.json`은 candidate cases `8`, candidate review `350`, content unreviewed `82`, missing/file issue `0`; reference digest 8개와 pack pending/resolved `82/0`은 그대로다. Default runtime-parity stereo coverage `.casrt/experiments/all8-front120-stereo-energy-default-vad-coverage-runtime-parity-nonoverlap.json`은 intervals `350`, mean/max `4,213/25,000ms`, miss `379,591ms`, extra `178,258ms`, recall/precision `77.35/87.91%`다. Comparison `.casrt/experiments/all8-front120-stereo-default-vs-t54-sil1200-vad-quality-gated-runtime-parity-nonoverlap.json`은 fixed sil1200 coverage를 고르지만 corrected all8 downstream report가 없으므로, final readiness `.casrt/experiments/all8-front120-pipeline-readiness-nonoverlap-runtime-parity-raw-candidate.json`은 VAD를 `downstream validation reports were not provided`로 fail한다. ASR-only blockers는 `[reference,vad_chunking,channel_attribution]`; raw candidate channel match `47.74% < 85%`, text Bro MIX practical CER `58.84%`, alignment no-op reference-copy proxy만 pass다. 판단: 현재 파이프라인은 명백히 text-ASR-only 단계가 아니다.
- 2026-08-30 continuous 30-second chunk challenger plan: all8 pseudo-reference speech union이 audio의 `99.61%`이고 runtime-parity default stereo energy VAD recall은 `77.35%`라, near-continuous ASMR에서 silence rejection 자체가 부적합할 수 있다. 과거 full-audio/no-VAD report는 120초 단일 interval의 coverage만 측정해 max-duration gate에서 탈락했으며, 30초 비중첩 연속 청크의 실제 ASR downstream은 평가하지 않았다. Experiment-only CASRT VAD command로 각 channel waveform을 `[0,30000)`, `[30000,60000)`처럼 끝까지 빠짐없이 나누고 WebUI나 제품 기본 옵션은 추가하지 않는다. 먼저 `01/04/08` front120에서 exact Bro snapshot, beam 5, stereo, no alignment를 raw canonical subset과 비교한다. Practical CER `<=63.04%`, channel-aware CER `<=41.67%`, time-aligned 500ms `>=20.00%`, channel time-aligned accuracy `>=46.67%`를 모두 만족하고 적어도 하나를 개선할 때만 all8로 확장한다. Plan/repro provenance는 `.casrt/experiments/bro-stereo-continuous30-01-04-08-plan.json`과 같은 prefix에 보존한다.
- 2026-08-30 continuous 30-second chunk result/20-second follow-up plan: coverage는 6개 L/R unit, 24 chunks, max/mean `30,000ms`, reference recall `100%`, precision `83.57%`였다. Exact Bro downstream은 candidate segments `145 -> 24`, practical CER `63.04% -> 60.25%`, channel accuracy `46.67% -> 46.67%`로 개선/유지했지만 channel-aware CER `41.67% -> 41.84%`, timing 500ms `20.00% -> 18.33%`로 악화해 gate를 실패했다. Artifacts는 `.casrt/experiments/bro-stereo-continuous30-01-04-08-{vad-coverage.json,candidates,projects,eval-manifest.json,eval-report.json,comparison.json,gate.json}`이다. 판단: 30초 후보는 all8/default/canonical로 승격하지 않는다. 이 subset의 reference speech duration/segment가 약 20초이므로 text 개선 신호를 유지하면서 boundary 오차를 줄일 수 있는 비중첩 20초 후보 하나만 동일 raw gate로 평가하고, 실패하면 continuous fixed-window sweep을 종료한다.
- 2026-08-30 continuous 20-second subset gate pass/all8 plan: coverage는 6개 L/R unit, 36 chunks, max/mean `20,000ms`, recall `100%`, precision `83.57%`였다. Exact Bro downstream은 candidate segments `145 -> 36`, practical CER `63.04% -> 56.23%`, timing 500ms `20.00% -> 21.67%`로 개선했고 channel-aware CER `41.67% -> 41.67%`, channel accuracy `46.67% -> 46.67%`는 동일해 네 사전 gate를 모두 통과했다. Review effort는 여전히 `100%`다. Artifacts는 `.casrt/experiments/bro-stereo-continuous20-01-04-08-{vad-coverage.json,candidates,projects,eval-manifest.json,eval-report.json,comparison.json,gate.json}`이다. 계획대로 exact runtime/settings를 all8로 한 번 확장하고 raw stereo all8 대비 practical CER, channel-aware CER, unrestricted/channel-aware timing 500ms, channel accuracy를 모두 비회귀해야만 내부 VAD challenger로 채택한다. Human-reviewed reference 전에는 WebUI/default/canonical을 교체하지 않는다.
- 2026-08-30 continuous 20-second all8 result/rejection: all8 coverage는 16개 L/R unit, 96 chunks, max/mean `20,000ms`, recall `100%`, precision `87.28%`; exact Bro inference는 empty-text chunk 하나를 제외한 95 segments를 만들었다. Raw stereo 350 segments 대비 practical CER `64.33% -> 57.70%`, channel-aware CER `48.21% -> 46.72%`, unrestricted timing 500ms `15.85% -> 21.95%`, channel-aware timing `15.79% -> 21.95%`로 개선했지만 channel accuracy가 `51.22% -> 48.78%`로 하락해 all8 non-regression gate를 실패했다. Review effort/candidate review는 `100%`, repeated-text artifact는 `1/95`; candidate energy match/wrong-side/uncertain은 `34/35/26`, match ratio `49.28% < 85%`다. Artifacts는 `.casrt/experiments/all8-front120-{stereo-continuous20-vad-coverage.json,stereo-default-vs-continuous20-vad-quality-gated.json,bro-stereo-continuous20-candidates,bro-stereo-continuous20-projects,bro-stereo-continuous20-eval-manifest.json,bro-stereo-continuous20-eval-report.json,bro-stereo-continuous20-comparison.json,bro-stereo-continuous20-candidate-channel-audit-th2.json,pipeline-readiness-continuous20-challenger.json}`이며 non-mutating attach plan도 보존했다. Readiness는 VAD fail reason을 channel accuracy regression으로 명시하고 blockers `[reference,vad_chunking,channel_attribution]`을 유지한다. 판단: continuous20은 human-reviewed 재평가용 최강 VAD/text/timing challenger로 보존하지만 product/default/WebUI/canonical로 승격하지 않고 fixed-window sweep을 종료한다.
- 2026-08-30 equal-boundary time-aligned matcher repair plan: continuous20 all8 `channel_time_aligned.confusion`이 reference L/R `40/42`개를 candidate L `82`개로 전부 매칭했다. 원인은 `time_aligned_segment_pairs`가 overlap 최대값만 비교하고 동률이면 정렬상 첫 후보를 유지해, 동일한 20초 bounds의 L/R 후보에서 항상 L을 고르는 평가 편향이다. Channel label로 tie-break하면 channel metric을 정답에 맞추는 순환 평가가 되므로 금지한다. Observable contract는 positive overlap 최대화, boundary absolute error 최소화, 두 값도 같을 때 practical-normalized text edit ratio 최소화, 그래도 같으면 stable input order 유지다. 동일 bounds L/R 후보가 반대 순서여도 각 reference text에 맞는 후보를 고르는 behavior test를 추가하고 full suite를 통과시킨 뒤 raw/continuous20 all8 report, comparison, gate, readiness를 새 artifact suffix로 재생성한다. 기존 reports는 evaluator-v1 provenance로 보존하고 덮어쓰지 않는다.
- 2026-08-30 equal-boundary matcher repair/continuous20 VAD pass: 새 behavior test는 기존 channel accuracy `0.5` 편향을 재현한 뒤 수정 경로에서 `1.0`을 확인했고 full `343` tests가 통과했다. Raw/continuous20 v2 reports와 comparison은 `.casrt/experiments/all8-front120-bro-asr-beam5-stereo-eval-report-text-tiebreak-v2.json`, `.casrt/experiments/all8-front120-bro-stereo-continuous20-eval-report-text-tiebreak-v2.json`, `.casrt/experiments/all8-front120-bro-stereo-continuous20-comparison-text-tiebreak-v2.json`이다. Text/channel-aware/timing 값은 그대로이고 raw channel accuracy는 `51.22% -> 52.44%`, continuous20은 편향된 `48.78% -> 54.88%`로 정정됐다. Continuous20은 raw 대비 practical CER `64.33% -> 57.70%`, channel-aware CER `48.21% -> 46.72%`, unrestricted/channel-aware timing `15.85/15.79% -> 21.95/21.95%`, channel accuracy `52.44% -> 54.88%`를 모두 개선하며 downstream regression `0`이다. Readiness `.casrt/experiments/all8-front120-pipeline-readiness-continuous20-text-tiebreak-v2.json`에서 `vad_chunking=pass`, ASR-only blockers는 `[reference,channel_attribution]`, product blockers는 `[reference,channel_attribution,text_asr]`로 줄었다. 판단: continuous20을 현 automated evidence상 최선 VAD/chunking challenger로 승격하지만, pseudo-gold reference와 energy channel match `49.28%` 때문에 제품/WebUI/default/canonical 변경은 human-reviewed channel 재평가 전까지 보류한다. Evaluator-v1의 continuous20 channel regression 결론은 폐기하고 provenance로만 보존한다.
- 2026-08-30 continuous20 bounded de-bleed sweep plan: continuous20은 같은 20초 bounds의 L/R을 모두 전사하므로 candidate energy audit에서 match/wrong-side/uncertain `34/35/26`이 나온다. Opposite-channel segment와 time coverage `>=50%`, practical text matching coverage `>=70%` 또는 `85%`, 최소 4자를 만족하면서 own-channel RMS가 other side보다 `2/4/6dB` 이상 낮은 segment만 제거하는 6개 experiment-only 설정을 평가한다. 기존 raw-stereo oriented sweep 로직을 continuous20 manifest/audio에 맞춘 durable repro로 실행하되 production/CLI/WebUI 코드는 추가하지 않는다. Corrected continuous20 v2 대비 practical CER, channel-aware CER, unrestricted/channel-aware timing 500ms, channel accuracy, MIX/review/candidate-review/artifact metric을 모두 비회귀하고 candidate energy match ratio를 strict 개선하는 후보만 내부 channel challenger 자격을 얻는다. 하나라도 실패하면 자동 de-bleed를 폐기하며, 통과해도 pseudo-gold channel authority 때문에 제품/default/canonical 승격은 human review 뒤로 둔다.
- 2026-08-30 continuous20 bounded de-bleed sweep rejection: durable repro/output은 `.casrt/experiments/all8-front120-bro-stereo-continuous20-dedupe-{repro,sweep}`이고 corrected comparison은 sweep root `comparison-text-tiebreak-v2.json`, gate summary는 `gate-text-tiebreak-v2.json`이다. 6개 설정은 95개 중 `9/10/15/17/26/28` segments를 제거했다. 가장 높은 energy match인 `energy2-text70`은 `49.28% -> 82.93%`로 85% gate에도 못 미치면서 practical CER `57.70% -> 59.34%`, channel-aware CER `46.72% -> 58.48%`, channel-aware timing `21.95% -> 17.16%`, channel accuracy `54.88% -> 51.22%`로 악화했다. 가장 적게 제거하고 practical CER가 가장 낮은 filtered 후보 `energy6-text85`도 practical `58.33%`, channel-aware `48.77%`, channel-aware timing `21.52%`, channel accuracy `52.44%`로 baseline을 지배하지 못했다. 모든 설정의 review effort/candidate review는 계속 `100%`이고 artifact ratio도 개선하지 못했다. 판단: energy proxy를 맞추기 위한 자동 segment 삭제는 실제 transcript coverage/channel metric을 해치므로 전부 기각한다. Continuous20 원본을 내부 최선으로 유지하고, reference L/R mismatch `30`/uncertain `18` 및 candidate energy ambiguity를 사람이 판정하기 전 channel attribution을 자동 승격하지 않는다.
- 2026-08-30 no-training product constraint: 사용자는 project-owned fine-tuning을 하지 않기로 확정했다. 이 파이프라인은 공개된 pretrained 또는 third-party fine-tuned local checkpoint의 exact snapshot 평가, decoding, VAD/chunking, channel handling, alignment, deterministic post-processing만 개선 대상으로 삼는다. Training dataset/loop, LoRA/adapter 생성, 사용자별 checkpoint는 비목표다. Product gate `CER <=10%`, review effort `<=15%`는 임의로 낮추지 않되, 공개 모델 ceiling이 이를 만족하지 못하면 fine-tuning을 후속 계획으로 기록하지 않고 실제 human-reviewed metric과 필요한 검수량을 명시한다. 현재 human-reviewed authority가 없어 ceiling 자체도 아직 확정할 수 없다.
- 2026-08-30 human-free runtime constraint: 사용자는 사람을 정상 전사 단계에서 배제하기로 확정했다. Production runtime은 사용자가 선택한 메인 모델 하나와 고정 VAD/channel/alignment pipeline만 실행하며 multi-model consensus, MBR/ROVER, 추가 audio-conditioned verifier를 숨은 정상 비용이나 fallback으로 호출하지 않는다. 여러 모델 후보와 audio verifier는 개발 단계에서 main model/pipeline을 비교하는 benchmark에만 사용하고 WebUI 옵션으로 노출하지 않는다. 서로 다른 architecture의 confidence는 calibration 없이 직접 비교하지 않는다. Runtime은 불확실한 항목도 단일 main-model 결과와 machine uncertainty를 저장하고 완료하며 Review UI 승인을 기다리지 않는다. Human-reviewed gold가 전혀 없으면 correctness/CER는 증명할 수 없으므로 benchmark proxy agreement 개선과 실제 정답 개선을 구분한다.
- 2026-08-30 community-first validation constraint: 공식 benchmark/leaderboard는 일본 동인 음성 실사용 품질의 모델 선택 근거에서 제외한다. 후보 discovery/ranking은 최근 실제 사용자 커뮤니티의 ASMR/whisper/Japanese 파일 사용 후기, exact model/quant/runtime, 공유된 output/failure case, 독립 재현 수를 우선한다. 공식 자료는 modality/language/runtime/license/revision 사실 확인에만 쓴다. 후보는 사용자의 실제 파일에서 단일 모델로 실행하고 Codex가 일본어 출력 붕괴, 반복, 문맥 단절, 누락 징후, L/R duplicate, timing 구조를 spot-audit한다. 현재 Codex는 audio를 직접 청취하지 못하므로 이 audit은 제한된 개발 판단이며 실제 발화 correctness나 CER를 증명하지 않는다.
- 2026-08-30 production runtime simplicity reset 구현 완료: `workflow.py`는 stereo L/R, mono MIX, non-overlap `20,000ms`, 결과 무삭제/시간순 병합, no-op alignment의 단일 계약을 사용한다. `CASRT_VAD_COMMAND`, `CASRT_QWEN_ENERGY_*`, `CASRT_LOCAL_ASR_CHANNEL_MODE`, `CASRT_ALIGNER_COMMAND`는 server/project transcription에서 읽지 않는다. External VAD/alignment와 energy sweep은 명시적 개발 CLI command에만 남겼다. Behavior tests는 stereo `L/R`, mono `MIX`, `20s+tail`, 이전 metadata 재분할, 환경변수 무시, stable sort를 검증한다.
- 2026-08-30 fixed production real-data E2E: durable source `.casrt/experiments/all8-front120-review-cases/audio/01-front120-existing-srt.wav`(120초 stereo)를 새 project로 생성했다. 분석은 L/R/MIX와 정확한 20초 비중첩 6개 interval을 만들었다. Exact Bro snapshot `.casrt/models/bro-asr-1.7b-6e8f651ad6137457a2a6e813b810349a7e272f4b`, pinned Transformers `45b004d7`, beam 5, offline/network-disabled worker로 fixed production 전사를 실행해 L 6개/R 6개, 총 12개 segment를 0..120초에 정렬했다. 일본어 문장 붕괴나 무한 반복은 없었지만 모든 L/R 쌍이 거의 같은 내용이라 bleed duplicate가 명확했고, `見。`, `う`처럼 20초 hard boundary에서 잘린 tail도 관찰됐다. 자동 dedupe나 overlap context를 다시 넣지 않고 현재 단순 계약의 알려진 한계로 기록한다. Codex는 audio를 직접 듣지 않았으므로 실제 발화 correctness 판정은 아니다. Project/master/translation/SRT 산출물은 `.casrt/experiments/product-fixed-e2e-20260830/`에 보존한다. 최초 SRT export에서 L/R label이 빠져 SRT->JSON channel이 MIX로 소실되는 실제 계약 버그를 발견했고, L/R cue에 `[L]`/`[R]`을 쓰도록 수정했다. 수정 후 master->SRT->master의 timing/channel/kind/text와 SRT bytes가 완전 왕복했고, `translation.json` 12개 item에는 channel label이 없음을 확인했다.
- 같은 E2E master를 실제 `casrt serve` HTTP API에 import했고 primary `/`에는 review control이 없고 `/diagnostics.html`에만 있음을 확인했다. HTTP `export-translation-json`은 CLI와 같은 label-free 12 items, HTTP `json-to-srt`는 CLI SRT와 byte-identical output을 반환했다. Label-free translation JSON을 text 변경 없는 identity translated JSON으로 바꿔 다시 가져왔을 때도 CLI와 HTTP translated SRT가 원본 timing/channel SRT와 byte-identical했다. 번역 자체는 실행하지 않았다. CLI와 WebUI는 같은 data contract를 사용한다.
- 2026-08-30 community-first candidate refresh: 최근 ASMR/JAV 도메인 툴 WhisperJAV v1.9는 `jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame`를 base가 놓친 line을 복구하는 새 후보로 포함하면서 junk insertion 증가를 함께 경고한다. 같은 툴은 whisper/ASMR sensitivity를 aggressive로 두고 최신 ChronosJAV timing 기본을 VAD frame으로 바꿔 forced aligner load를 제거했다. 일반 local ASR 사용자 후기에서도 Qwen3-ASR은 비음성 hallucination이 적고 신뢰도가 높다는 반복 보고가 있지만, timestamp alignment는 좋지 않다는 별도 firsthand 보고가 있다. Cohere Transcribe는 일본어 clip에서 Whisper v3 Turbo보다 못했다는 보고와 noise에서 반복 후 많은 speech를 건너뛰었다는 보고가 있어, 이미 로컬에서 부진했던 Cohere를 다시 선두 후보로 올리지 않는다. 이 판단은 official leaderboard가 아니라 실제 사용자 실패 양상과 도메인 툴 채택을 기준으로 한다.
- JA Anime-Galgame external snapshot review/download: exact native HF revision `5a6a789ceb2f22d2b8606743b13a8159af218362`을 `gpt-5.4 xhigh`가 실행 없이 정적 검토했고 verdict는 `PASS_WITH_CONSTRAINTS`다. 명시된 13개 `chat_template.jinja`/config/tokenizer/safetensors shard/index만 받고 `ctc_aligner.pt`, `ctc_aligner_jav_vocalisation_v2.pt`, conversion report, code/pickle/bin은 제외했다. Local path `.casrt/models/qwen3-asr-1.7b-ja-anime-galgame-hf-5a6a789ceb2f22d2b8606743b13a8159af218362`; snapshot digest report `.casrt/model-digests/qwen3-asr-1.7b-ja-anime-galgame-hf-5a6a789ceb2f22d2b8606743b13a8159af218362-digest.json`; snapshot SHA-256 `c8acca75127d93f9aadeb5274654b5e114d465ce1d6f1c90ed0a33cff24658e1`; raw Jinja SHA-256 `a103579066642f0e035873f12de6046a1bbc909a7a2833732fda9fd6bc7c528d`. Existing worker preflight, six-shard index, offline/local-only/network-disabled guards를 통과했다.
- JA Anime-Galgame 01/04/08 fixed production spot-audit: 동일 pinned Transformers `45b004d7`, beam 5, max tokens 256, L/R 각각 20초 비중첩 조건으로 세 120초 case를 실행했다. Durable outputs는 `.casrt/experiments/product-ja-anime-e2e-20260830/`. 각 case는 L 6/R 6의 12 segments이고 candidate text characters는 01 `951`, 04 `935`, 08 `896`으로 동일 Bro outputs `907`, `882`, `867`보다 많았다. 04/08에서 Bro의 `ベロ中`, `よらしい大気`, `人間一種` 같은 붕괴를 `ベロチュー`, `いやらしい唾液`, `人間椅子`로 고치고 expressive vocalisation/punctuation을 더 보존했다. 01에서는 Bro가 맞춘 `退治`, `小悪党`을 `介助`, `孤児棟`으로 오인하는 tradeoff가 있었다. 08 마지막 L segment를 max tokens 512로 재실행해도 256과 동일했으므로 후반 누락은 generation cap이 아니다. 모든 case에 near-duplicate L/R과 일부 hard-boundary/후반 누락 신호가 남는다. Codex가 audio를 직접 듣지 않았으므로 absolute correctness 주장이 아니라 output/reference consistency spot-audit다. 2/3 case의 큰 질적 개선과 domain fit이 01의 일부 한자 오인보다 중요하다고 판정해 이 exact snapshot을 현재 제품 권장 모델로 확정한다. 추가 측정은 사용자 책임으로 넘기지 않는다. Runtime은 사용자 자유 입력을 위해 자동 선택하지 않는다.
- 2026-08-30 stereo bleed dedupe sweep: raw stereo 350 segments 중 반대 채널과 time/text가 유사한 segment가 많아, duplicate evidence와 own-channel energy 열세를 모두 요구하는 bounded post-filter를 평가했다. 첫 scratch `.casrt/experiments/all8-front120-bro-stereo-bleed-dedupe-sweep`은 짧은 쪽을 coverage denominator로 써 짧은 fragment 하나가 긴 segment의 고유 tail까지 삭제할 수 있는 계약 결함이 있어 채택하지 않는다. Corrected oriented sweep은 제거될 segment 자체의 time coverage `>=50%`, practical text matching coverage `>=70%/85%`, 최소 4자, own channel energy 열세 `2/4/6dB`를 요구한다. Artifacts는 `.casrt/experiments/all8-front120-bro-stereo-bleed-dedupe-oriented-sweep`; 가장 낮은 global CER인 `2dB/text85`는 94개를 제거해 global practical CER `64.33% -> 58.02%`를 개선했지만 channel-aware CER `48.21% -> 59.40%`, same-channel timing 500ms `15.79% -> 13.82%`로 악화했다. Strict sweep `.casrt/experiments/all8-front120-bro-stereo-bleed-dedupe-strict-sweep`의 `8dB/text95`는 47개를 제거해 global CER `61.97%`, channel-aware `53.36%`; `10dB/text95`는 30개 제거, global `63.89%`, channel-aware `51.21%`; `12dB/text95`는 19개 제거, global `64.41%`, channel-aware `50.01%`였다. 모든 후보의 review effort는 `100%`이고 raw stereo를 global/channel-aware 양쪽에서 동시에 이긴 후보가 없다. 결정: bleed duplicate 진단은 유효하지만 pseudo-gold의 중복 L/R authority가 불확실하므로 자동 삭제를 제품 경로나 production CLI에 추가하지 않는다. Human-reviewed manifest 재실행용 exact scripts는 `.casrt/experiments/all8-front120-bro-stereo-bleed-dedupe-repro/run-oriented-sweep.py`와 `run-strict-sweep.py`에 보존한다.
- 2026-08-30 Bro stereo VAD t54/pad800/max30s all8: exact Bro snapshot, pinned HF runtime, beam 5, max tokens 256, offline/network-disabled worker, stereo mode를 유지하고 VAD만 `threshold=-54dBFS`, `pad=800ms`, `max chunk=30000ms`로 바꿔 실제 GPU 전사를 완료했다. Candidates/projects/report/comparison은 `.casrt/experiments/all8-front120-bro-asr-beam5-stereo-vad-t54-pad800-max30s-candidates`, `...-projects`, `...-eval-report.json`, `...-comparison.json`; case delta는 `...-case-deltas.json`, energy audit은 `...-candidate-channel-audit-th2.json`이다. Candidate는 8/8 case, segments `350 -> 320`, practical characters `5,109 -> 5,361`; global practical CER `64.33% -> 63.36%`, time-aligned 500ms `15.85% -> 17.68%`, time-aligned channel accuracy `51.22% -> 52.44%`로 개선했다. 반면 channel-aware practical CER는 `48.21% -> 49.31%`, same-channel timing 500ms는 `15.79% -> 16.03%`로 text/channel 종합은 악화와 미세 개선이 섞였고 review effort는 계속 `100%`다. Case별 global CER는 6/8 개선했지만 channel-aware CER는 3/8만 개선했다. Candidate energy audit 2dB/quiet-off에서 match/wrong-side/uncertain은 raw `137/150/63`에서 tuned `124/138/58`, energy-labeled match ratio `47.74% -> 47.33%`, wrong-side ratio `52.26% -> 52.67%`로 bleed를 개선하지 못했다. 결정: fragmentation/global/timing challenger로 보존하지만 raw stereo를 지배하지 못하므로 MIX/default VAD나 canonical human-review candidate를 교체하지 않는다. Human-reviewed L/R manifest에서 raw/tuned 둘을 다시 비교한다.
- 2026-08-30 Bro-aware model/readiness audit: persistent `.casrt/models` snapshot은 Bro, official Qwen HF, Neosophie Qwen JA, Granite base/plus 다섯 개이며 모두 all8 또는 bounded real-data 평가가 있어 다운로드 없는 최신 미평가 후보는 없었다. Latest comparison `.casrt/experiments/all8-front120-local-model-comparison-bro-aware.json`에서 Bro MIX beam5가 practical CER `58.84%`로 6개 후보 중 text 1위지만 review effort `100%`다. Coverage-only readiness가 t54 VAD를 pass로 표시하던 계약을 수정해 product preset이 같은-scope baseline/candidate eval의 downstream no-regression을 요구하도록 했다. Actual output `.casrt/experiments/all8-front120-pipeline-readiness-current-best-bro-aware-vad-downstream.json`은 reference와 VAD/chunking을 ASR-only blocker, reference/VAD/text ASR을 product blocker로 판정한다. VAD는 coverage recall `99.51%`지만 동일 Qwen MIX downstream에서 practical CER `59.74% -> 60.19%`, timing 500ms `16.05% -> 15.24%`, MIX ratio `62.96% -> 70.73%`가 악화해 fail이다. Alignment no-op과 isolated energy channel attribution은 각각 proxy pass로 유지한다.
- 2026-08-30 JA Anime-Galgame 비파괴 timing 재실험: 현재 권장 snapshot의 fixed L/R 20초 결과 01/04/08, 총 36 segments를 대상으로 text/channel/id/kind/segment 수를 그대로 보존하고 각 owner 20초 구간의 바깥 무음만 energy interval로 줄이는 edge snap을 비교했다. Artifacts와 재현 script는 `.casrt/experiments/vad-edge-snap-ja-anime-20260830/`에 있다. Baseline same-channel time-aligned 500ms ratio/mean boundary error는 `21.67% / 4412.9ms`였다. Energy `-54dBFS`, silence `800ms`, pad `400ms`는 6/36 segments를 바꿨지만 `15.00% / 4448.8ms`로 악화했고, 가장 공격적인 `-48dBFS`, silence `300ms`, pad `100ms`는 mean error를 `4148.9ms`로 낮추는 대신 500ms ratio를 `7.14%`로 크게 낮췄다. Qwen3-ForcedAligner는 33/36 segments를 평균 절대 `631.1ms` 이동하고 same-channel 500ms ratio/mean error를 `8.62% / 4613.1ms`, reference match ratio를 `96.67%`로 악화했다. 가장 덜 해로운 VAD 뒤 aligner 조합도 `8.62% / 4603.5ms`, reference match `96.67%`로 기준선을 넘지 못했다. 모든 후보의 practical CER `56.89%`, channel-aware CER `41.80%`, review effort `100%`는 text 불변 조건대로 동일하다. Reference는 stable-ts 기반 pseudo-gold라 absolute accuracy 근거는 아니지만, 어느 후보도 같은 scope의 no-op 기준선을 지배하지 못했다. 한 20초 transcript가 여러 실제 발화를 합친 상태에서는 VAD/aligner가 whole-segment outer bounds만 움직여 문장별 text ownership을 만들 수 없는 것이 원인이다. Production은 fixed 20초/no-op을 유지하며 이 energy edge snap과 Qwen aligner를 숨은 후처리로 추가하지 않는다.
- 판단: 모델 후보 선정은 공식 benchmark/leaderboard 주장으로 승격하지 않는다. 최근 relevant community firsthand evidence로 후보를 좁힌 뒤 local snapshot digest, offline worker 실행, 사용자 실제 동인 음성 output과 Codex spot-audit로 개발 기본 후보를 정한다. Human-reviewed gold가 없으면 결과를 absolute accuracy가 아닌 best-effort 실사용 선택으로 기록한다.

Sources checked on 2026-08-30: WhisperJAV v1.9 domain release `https://sourceforge.net/projects/whisperjav.mirror/files/v1.9.0/`, WhisperJAV current README `https://github.com/meizhong986/WhisperJAV`, JA Anime-Galgame native HF snapshot `https://huggingface.co/jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf`, Cohere Japanese/noise firsthand thread `https://www.reddit.com/r/LocalLLaMA/comments/1s48jtu/cohere_transcribe_released/`, Cohere vs Whisper Japanese firsthand report `https://www.reddit.com/r/MacWhisper/comments/1s49zhh/new_cohere_model_can_it_be_added_looks_good/`, current local ASR/timestamp firsthand thread `https://www.reddit.com/r/LocalLLaMA/comments/1ups1c8/are_there_any_local_asr_models_that_surpass/`, current Qwen reliability firsthand thread `https://www.reddit.com/r/LocalLLaMA/comments/1u9ggke/whats_the_best_open_speech_to_text_today/`, Qwen3-ASR model card `https://huggingface.co/Qwen/Qwen3-ASR-1.7B`, Neosophie Qwen3-ASR-JA blog `https://neosophie.com/en/blog/20260427-qwen-finetuned-model`, Granite Speech 4.1 2B model card `https://huggingface.co/ibm-granite/granite-speech-4.1-2b`, Cohere Transcribe 03-2026 model card `https://huggingface.co/CohereLabs/cohere-transcribe-03-2026`, ARK-ASR-3B model card `https://huggingface.co/Audio8/ARK-ASR-3B`, LLM-jp 4 8B speech ASR model card `https://huggingface.co/Atotti/llm-jp-4-8b-speech-asr`, Bro-ASR-1.7B model card `https://huggingface.co/Junlaii/Bro-ASR-1.7B`.

외부 runtime benchmark의 local-only 실행 조건:

```text
CASRT_LOCAL_WORKER_ENV_MODE=offline
CASRT_TRANSFORMERS_REQUIRE_LOCAL_MODEL_PATH=1
CASRT_TRANSFORMERS_LOCAL_FILES_ONLY=1
CASRT_TRANSFORMERS_DISABLE_NETWORK=1
CASRT_COHERE_ASR_DISABLE_NETWORK=1
CASRT_QWEN_ASR_REQUIRE_LOCAL_MODEL_PATH=1
CASRT_QWEN_ASR_LOCAL_FILES_ONLY=1
CASRT_QWEN_ASR_DISABLE_NETWORK=1
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
HF_DATASETS_OFFLINE=1
WANDB_MODE=disabled
```

- `CASRT_LOCAL_WORKER_ENV_MODE=offline`은 local worker subprocess 환경에서 token/proxy류 env를 제거하고 offline flags를 주입한다.
- `CASRT_QWEN_ASR_REQUIRE_LOCAL_MODEL_PATH=1`은 `model_id`와 `CASRT_QWEN_ASR_ALIGNER_MODEL_ID`가 존재하는 local directory가 아니면 실패시킨다.
- `CASRT_QWEN_ASR_LOCAL_FILES_ONLY=1`은 Transformers model load kwargs에 `local_files_only=True`, `trust_remote_code=False`를 붙인다.
- `CASRT_QWEN_ASR_DISABLE_NETWORK=1`은 Qwen worker process 내부 Python socket 생성을 차단한다.
- 실험 산출물은 고정 `/tmp/casrt-quality`를 재사용하지 않고, 매 실행마다 새 `0700` directory 아래에 둔다. 원본 fixture tree에는 쓰지 않는다.

## 오디오 전처리

로컬 ASR worker는 추론 직전에 조용한 PCM16 WAV를 bounded gain으로 보정한다.

현재 값:

```text
target RMS: -24.0 dBFS
max peak: -3.0 dBFS
max gain: 4.0x
```

이 보정은 ASMR의 저음량/속삭임 구간에서 모델 입력을 안정화하기 위한 것이다. 클리핑을 숨기는 fallback이 아니라, 피크 제한이 있는 좁은 범위의 전처리다.

## Speech Chunking

Production의 모든 adapter는 동일한 고정 20초 비중첩 chunk를 사용한다. 마지막 tail은 남은 길이만 사용하고 침묵 구간도 임의로 삭제하지 않는다. 이전 버전의 긴 분석 chunk metadata도 전사 시점에 20초 이하로 다시 나눈다.

현재 production 값:

```text
max_chunk_ms: 20000
overlap_ms: 0
silence_drop: false
```

Energy/VAD coverage와 외부 VAD command는 `casrt vad ...` 개발 명령에서만 비교한다. `CASRT_VAD_COMMAND`와 `CASRT_QWEN_ENERGY_*`를 설정해도 production 전사는 바뀌지 않는다.

```bash
uv run casrt vad coverage audio.wav reference.master.json \
  --vad-command 'python3 path/to/vad.py' --json
```

VAD command contract:

- stdin: `{ audio_file, audio_info }`
- stdout: `{ intervals: [{ start_ms, end_ms }] }`
- interval은 정렬되어야 하고 서로 겹치면 안 된다.
- interval이 audio duration을 넘거나 malformed이면 fallback하지 않고 실패한다.
- Production WebUI/CLI 옵션으로 노출하지 않는다.

개발용 energy coverage는 다음 env/helper와 명시적 CLI 옵션으로 재현할 수 있다.

```text
CASRT_QWEN_ENERGY_THRESHOLD_DBFS
CASRT_QWEN_ENERGY_WINDOW_MS
CASRT_QWEN_ENERGY_MIN_SILENCE_MS
CASRT_QWEN_ENERGY_MIN_SPEECH_MS
CASRT_QWEN_ENERGY_PAD_MS
CASRT_QWEN_ENERGY_MAX_CHUNK_MS
```

`CASRT_QWEN_ENERGY_MAX_CHUNK_MS`는 긴 energy interval을 고정 길이 이하로 자르는 내부 실험 옵션이다. 기본값은 unset이며 WebUI/CLI 모델 선택 옵션으로 노출하지 않는다.

고정 production 경로의 이유:

- 실제 all8 실험에서 continuous 20초 L/R 후보가 energy VAD 후보보다 누락 신호와 종합 proxy가 안정적이었다.
- 불확실한 VAD가 조용한 속삭임을 삭제하는 것보다 모든 구간을 단일 모델에 넘기는 쪽을 택한다.
- Chunk 길이는 WebUI 옵션으로 노출하지 않는다.

## ASR Text Cleanup

로컬 일본어 ASR worker는 모델 출력 텍스트를 segment 저장 전에 정리한다.

현재 동작:

- common prefix(`Transcription:`, `Transcript:`, `文字起こし:`, `書き起こし:`)를 제거한다.
- 일본어 문자 사이의 불필요한 공백과 punctuation 주변 공백을 압축한다.
- 앞뒤의 비일본어 noise를 제거한다.
- 정리 후 일본어 문자가 하나도 없으면 hallucination segment로 보고 버린다.

이 필터는 punctuation-only/English-only 같은 비일본어-only 출력에 한정한다. 일본어 문자가 포함된 segment는 여기서 버리지 않고 평가와 human review에서 다룬다.

## Channel Attribution

Production은 channel attribution을 하지 않는다. Stereo 입력은 모델 호출 시점부터 L/R이 정해져 있으며 두 결과를 그대로 저장한다. Mono 입력만 MIX다.

기존 transcript의 진단/후처리를 위한 `attribute-channels` CLI는 개발 도구로 유지한다.

현재 값:

```text
L/R 확정 기준: 8.0 dB 이상 차이
quieter side gate: -40.0 dBFS 이하
```

동작:

- L이 R보다 8dB 이상 크고 R이 -40dBFS 이하이면 `channel: "L"`
- R이 L보다 8dB 이상 크고 L이 -40dBFS 이하이면 `channel: "R"`
- 차이가 작거나 양쪽이 모두 충분히 active이면 `channel: "MIX"`

이 기준은 개발용 기존 transcript 후처리에만 적용되며 production 전사를 바꾸지 않는다.

기존 transcript 후처리:

```bash
uv run casrt attribute-channels audio.wav candidate.master.json -o candidate.attributed.master.json --json
```

이 명령은 `MIX` speech segment만 relabel하며, 이미 `L`/`R`인 segment와 speech가 아닌 segment는 바꾸지 않는다. mono audio나 L/R을 만들 수 없는 audio는 실패한다. `--threshold-db`와 `--quiet-channel-max-dbfs`는 benchmark 재현용 CLI 옵션이고 WebUI에는 노출하지 않는다.

## ForcedAligner 상태

Production 기본값은 no-op이다. 아래 Qwen3-ForcedAligner와 generic aligner command는 `align-transcript`, `align-review-case-candidates` 같은 명시적 개발 CLI 실험에서만 사용한다. Server와 `project transcribe`는 `CASRT_ALIGNER_COMMAND`를 읽지 않는다.

후보:

```text
Qwen/Qwen3-ForcedAligner-0.6B
```

실행은 다음 env로 켠다.

```bash
CASRT_QWEN_ASR_ALIGNER_MODEL_ID=Qwen/Qwen3-ForcedAligner-0.6B
CASRT_QWEN_ASR_MIN_ALIGNED_DURATION_MS=80
```

`CASRT_QWEN_ASR_MIN_ALIGNED_DURATION_MS`보다 짧은 aligned span은 비현실적인 timestamp contract 위반으로 보고 해당 clip bounds로 되돌린다. 2026-06-30 forced aligner 3-case 실험에서 1ms segment가 관측되어 추가했다. 기본값은 80ms이며 WebUI/CLI 옵션으로 노출하지 않는다.

기존 master 텍스트를 재정렬하는 generic aligner command도 제공한다.

```bash
CASRT_ALIGNER_ENV_MODE=offline \
CASRT_QWEN_ALIGNER_REQUIRE_LOCAL_MODEL_PATH=1 \
CASRT_QWEN_ALIGNER_LOCAL_FILES_ONLY=1 \
CASRT_QWEN_ALIGNER_DISABLE_NETWORK=1 \
CASRT_ALIGNER_COMMAND='.casrt/qwen-asr-venv/bin/python -m custom_asmr_srt_stack.qwen_aligner_worker --model-id /path/to/Qwen3-ForcedAligner-0.6B/snapshot'
```

이 command는 `{ audio_file, master }`를 받아 speech segment별 clip을 만들고 `Qwen3ForcedAligner.align(audio, text, language)`로 segment start/end를 갱신한다. text, channel, kind는 변경하지 않는다. 기본 `CASRT_QWEN_ALIGNER_CONTEXT_MS=0`에서는 기존 segment 내부만 재정렬하고, context를 지정한 실험에서는 segment 앞뒤 audio를 함께 잘라 aligner가 기존 segment 밖 boundary로 이동할 수 있게 한다. 실행은 local snapshot path, offline env scrub, network-disabled Python socket guard, `local_files_only=True`, `trust_remote_code=False` 조건에서만 허용한다. worker는 `CASRT_ALIGNER_ENV_MODE=offline`, `CASRT_QWEN_ALIGNER_REQUIRE_LOCAL_MODEL_PATH=1`, `CASRT_QWEN_ALIGNER_LOCAL_FILES_ONLY=1`, `CASRT_QWEN_ALIGNER_DISABLE_NETWORK=1`이 모두 없으면 실패한다. `qwen-asr` package version, RECORD hash, RECORD에 기록된 각 설치 파일 hash, `qwen_asr` import origin도 고정값과 다르면 실패한다.

Generic Qwen aligner worker는 두 가지 bounded fallback을 가진다. `CASRT_QWEN_ALIGNER_MIN_ALIGNED_DURATION_MS=80`보다 짧은 span은 비현실적인 timestamp로 보고 원래 segment timing을 유지한다. `CASRT_QWEN_ALIGNER_MIN_COVERAGE_RATIO=0.5`보다 원 segment coverage가 낮은 span도 과도한 trim으로 보고 원래 timing을 유지한다. Context 실험에서도 coverage denominator는 padded clip 길이가 아니라 원 segment duration이다. 이 값들은 UI/CLI 옵션으로 노출하지 않고 env 계약으로만 남긴다.

2026-06-30 정적 보안 재검토 결과:

- reviewer: `gpt-5.4 xhigh` subagent
- scope: `qwen_aligner_worker.py`, `alignment.py` offline env, tests, docs
- verdict: `PASS`
- 허용 조건: offline env, local path-only model id, `local_files_only=True`, `trust_remote_code=False`, Python socket network guard, `qwen-asr==0.0.6` RECORD hash, per-file RECORD hash, import origin 검증
- caution: Python socket guard는 OS-level egress control이 아니며, 실패 요약에는 local path 같은 운영 정보가 남을 수 있다.

2026-06-30 coverage guard 변경 정적 보안 재검토:

- reviewer: `gpt-5.4 xhigh` subagent
- scope: commit `9723036`, `CASRT_QWEN_ALIGNER_MIN_COVERAGE_RATIO` guard
- verdict: `PASS`
- 판단: 실행 경계, local path-only, `local_files_only`, `trust_remote_code=False`, socket guard, package/hash/import-origin 검증, no-traceback contract를 약화하지 않는다. env 값이 잘못되면 fail-open이 아니라 `ValueError`로 실패한다.

2026-06-30 실제 로딩/추론 smoke:

- command: `.casrt/qwen-asr-venv/bin/python -m custom_asmr_srt_stack.qwen_aligner_worker --model-id /home/brain-offloaded/.cache/huggingface/hub/models--Qwen--Qwen3-ForcedAligner-0.6B/snapshots/c7cbfc2048c462b0d63a45797104fc9db3ad62b7`
- env: `CASRT_ALIGNER_ENV_MODE=offline`, `CASRT_QWEN_ALIGNER_REQUIRE_LOCAL_MODEL_PATH=1`, `CASRT_QWEN_ALIGNER_LOCAL_FILES_ONLY=1`, `CASRT_QWEN_ALIGNER_DISABLE_NETWORK=1`
- model snapshot digest report: `/tmp/casrt-quality.Q5OdDf/qwen3-forced-aligner-snapshot-digest.json`
- model snapshot SHA-256: `5b0efb9cbc06d49988d4593c5d8bc52947ff0dfc20731e230dddb1fe0f8f2573`
- input: `/tmp/casrt-quality.Q5OdDf/01-front120.wav` + first segment from `ref-01-front120.master.json`
- result: `seg_000001` moved from `980-3800ms` to `1460-2660ms`
- output: `/tmp/casrt-quality.Q5OdDf/qwen-aligner-smoke-output.json`

기존 transcript를 benchmark용으로 재정렬할 때는 `align-transcript`를 사용한다.

```bash
CASRT_ALIGNER_COMMAND='.casrt/qwen-asr-venv/bin/python -m custom_asmr_srt_stack.qwen_aligner_worker --model-id /path/to/Qwen3-ForcedAligner-0.6B/snapshot' \
  uv run casrt align-transcript audio.wav candidate.master.json -o candidate.aligned.master.json --json
```

2026-06-30 `align-transcript` CLI smoke:

- input: `/tmp/casrt-quality.Q5OdDf/qwen-aligner-smoke-input.master.json`
- output: `/tmp/casrt-quality.Q5OdDf/qwen-aligner-smoke-cli-output.master.json`
- result: `seg_000001` moved from `980-3800ms` to `1460-2660ms`, matching direct worker smoke
- note: sandboxed run used `UV_CACHE_DIR=/tmp/casrt-uv-cache` to avoid default uv cache write errors.
- 2026-06-30 `--diagnostics-output` no-op real-data smoke: input `/tmp/casrt-quality.Q5OdDf/stable-ts-cli-attributed-quiet8/01-front120.master.json`, audio `/tmp/casrt-quality.Q5OdDf/01-front120.wav`, output `/tmp/casrt-quality.Q5OdDf/alignment-diagnostics-smoke/01-front120.aligned.master.json`, diagnostics `/tmp/casrt-quality.Q5OdDf/alignment-diagnostics-smoke/01-front120.alignment-diagnostics.json`. Result: `segments=25`, `changed_segments=0`, `review_flag_changes=0`, `max_boundary_delta_ms=0`; output master is byte-identical to input.

현재 판단:

- 기본 경로로 승격하지 않는다.
- 10초 실데이터 crop에서 일부 timestamp가 초반으로 잘 맞지 않고, token duration이 0인 항목이 있었다.
- 2026-06-30 01/04/07 front120에서는 text를 바꾸지 않고 time-aligned 500ms와 channel time-aligned accuracy를 개선했지만 practical CER가 여전히 높고 1ms span이 관측됐다.
- ForcedAligner는 duration guard 적용 후 다시 평가한다.

## 평가 Harness

CLI:

```bash
uv run casrt eval-transcript reference.srt candidate.json --json -o eval.json
```

여러 샘플은 gold set manifest로 묶어서 평가한다.

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

현재 측정값:

- speech text strict CER
- speech text practical CER
- speech text Japanese relaxed CER
- segment index 기준 mean start/end/boundary error
- time-overlap 기준 `timing_time_aligned` mean start/end/boundary error
- boundary sample 수, max/mean boundary delta, 250ms/500ms 이내 boundary ratio
- L/R/MIX channel confusion
- candidate MIX 유지 비율
- index 기준 `channel` 및 time-overlap 기준 `channel_time_aligned` L/R channel accuracy
- candidate `needs_review` 비율
- segment 단위 `review_effort`: practical text mismatch, channel mismatch, 500ms 초과 timing mismatch, missing reference, extra candidate
- `review_effort` breakdown ratios: 같은 denominator(`reference_segments + extra_candidate_segments`) 기준 text/channel/timing/missing/extra 수정 비율. 이 값은 오디오->텍스트 모델 문제와 VAD/chunking/alignment/channel attribution 문제를 분리해 다음 개선 단위를 정하는 데 쓴다.
- `asr_artifacts`: candidate speech segment 기준 non-Japanese text, 15 chars/sec 초과 high text density, 12자 이상 repeated text pattern count/ratio. 이 값은 ASMR식 hallucination/repetition/chunking 실패를 CER와 분리해 보기 위한 보조 metric이며 product gate에는 쓰지 않는다.
- case별 `review_effort.items`: human review와 heuristic 개선이 어느 segment를 봐야 하는지 알 수 있도록 reasons와 reference/candidate text/channel/timing을 보존한다.
- `casrt vad coverage audio.wav reference.master.json --json -o vad-coverage.json`: built-in energy intervals, `--intervals` JSON, 또는 `--vad-command` CASRT VAD command를 reference speech union과 비교해 VAD/chunking coverage를 계산한다. Built-in energy sweep은 `--energy-*` CLI options로 지정하고 적용값을 `source_settings`에 기록한다. Reference recall, detected precision, missed reference duration, extra detected duration, missed/extra interval 목록, detected max/mean interval을 분리해 ASR text 품질과 VAD/chunking boundary/chunk granularity 문제를 따로 본다. Coverage recall/precision은 union interval 기준이고, detected max/mean interval은 max-chunk split이 보이도록 union merge 전 detected chunk 기준이다.
- `casrt vad coverage-cases cases/case-index.json --json -o vad-coverage-suite.json`: prepared review case set 전체를 같은 VAD source로 평가해 case별 `custom-asmr-vad-coverage-v1` report와 duration-weighted suite summary를 만든다. Summary는 detected max interval과 mean interval도 포함한다. Batch coverage는 오디오->텍스트 모델 후보 평가와 별개인 파이프라인 gate이며, ASMR 파이프라인은 이 VAD/chunk/channel/alignment 검증을 통과하기 전까지 “텍스트 모델만 남은 상태”로 보지 않는다.
- `casrt vad compare-coverage report-a.json report-b.json --json -o vad-coverage-comparison.json`: single/suite VAD coverage report를 missed reference duration, extra detected duration 순으로 정렬한다. `--max-detected-interval-ms`, `--max-missed-reference-ms`, `--min-reference-recall`, `--min-detected-precision`을 주면 detected chunk가 너무 길거나, reference speech를 많이 놓치거나, 과검출이 큰 후보를 gate failure로 표시한다. `--fail-on-gate`는 비교 JSON을 출력/저장한 뒤 gate 실패 후보가 있으면 exit 1로 끝내 자동 실험에서 다음 ASR 평가 단계로 넘어가지 않게 한다. Coverage는 후보 필터일 뿐이며, full-audio/no-VAD처럼 recall을 쉽게 올리는 후보는 실제 ASR chunk length, timing, text CER, channel attribution gate를 이어서 통과해야 한다.
- `casrt review-effort eval-suite.json --json -o review-effort.json`: suite/single report에서 `custom-asmr-review-effort-v1` 수정 큐를 추출한다. manifest case context와 timing delta를 보존하므로 다음 human review 또는 heuristic 개선 순서를 정하는 기본 산출물이다.
- `casrt review-pack review-effort.json --source-case-index cases/case-index.json -o review-pack --json`: 수정 큐 item별 audio clip과 `custom-asmr-review-pack-v1` index를 만든다. `--source-case-index`를 지정하면 `case-index.json`의 `items[].audio`를 case별 audio source로 사용하고, pack root와 item에 source `case-index.json`을 보존해 WebUI `case 열기`가 후보 실패 clip에서 원 reference segment 편집 화면으로 이동할 수 있다. review-effort 안에 `source_case_index`가 이미 있으면 이 옵션도 생략할 수 있다. Pack root `item_count`/`reason_counts`는 실제 packed item 기준으로 계산하고, input queue의 `case_summaries`, `case_count`, `next_case_id`는 pack index에도 보존해 WebUI header가 case 수와 다음 검수 case를 보여줄 수 있게 한다. Root `duration_summary`는 원래 item duration, focus 적용 후 effective duration, 실제 clip duration 합계/max를 기록해 검수 비용을 비교한다. pseudo-gold 비교에서 발견한 실패 구간을 사람이 빠르게 듣고 human-reviewed reference로 승격하기 위한 표준 산출물이다.

strict CER는 공백만 제거한다.

practical CER는 현재 다음을 정규화한다.

- Unicode NFKC
- 공백 제거
- punctuation/symbol 제거

practical CER는 자막 실용 비교용이다. 원문 보존 품질은 strict CER를 같이 본다.

Japanese relaxed CER는 practical CER에 더해 장음류 문자 `ー〜～`를 제거한다. ASMR 속삭임의 발화 길이 표기 차이를 분리해 보기 위한 보조 metric이며, 모델 승격 gate와 `review_effort`는 계속 practical CER를 사용한다.

2026-06-30 일본어 ASMR relaxed normalization 후보 실험:

- 후보: current practical에 더해 장음 부호 `ー〜～` 제거, 소형 kana를 대형 kana로 치환, 또는 둘 다 적용.
- 01/04/07 front120 pseudo-gold 기준 stable-ts CLI attributed quiet8: current 16.1%, 장음 제거 15.5%, 소형 kana 치환 16.1%, 둘 다 15.4%.
- Qwen HF ASR Transformers main: current 29.4%, 장음 제거 27.5%, 소형 kana 치환 29.4%, 둘 다 27.5%.
- 결정: 장음 제거는 ASMR 발화 길이 차이 노이즈를 줄이지만 실제 표기 차이까지 숨길 수 있다. 기존 품질 gate와 historical report의 의미를 바꾸지 않기 위해 current practical CER 기본값은 유지한다. `text_japanese_relaxed`를 별도 metric으로 추가하고 기본 gate로 쓰지는 않는다.
- 2026-06-30 구현 smoke: `/tmp/casrt-quality.Q5OdDf/stable-ts-cli-attributed-quiet8-3case-gold.json`를 새 report `/tmp/casrt-quality.Q5OdDf/stable-ts-cli-attributed-quiet8-3case-report-relaxed.json`로 재평가했다. Summary는 practical CER 16.1%, Japanese relaxed CER 15.5%다. 기존 report와 새 report를 같이 넣은 compare output `/tmp/casrt-quality.Q5OdDf/eval-comparison-old-new-relaxed.json`에서 old report의 `japanese_relaxed_cer`는 `null`, new report는 `0.1547`로 표시되어 기존 report 비교 호환성을 확인했다.

manifest summary는 case별 평균이 아니라 전체 edit distance/reference characters와 전체 paired/boundary/comparable segment 수 기준으로 가중 집계한다. 짧은 clip과 긴 clip이 같은 비중을 갖지 않게 하기 위한 결정이다. 품질 threshold 판단은 segment split 차이에 덜 취약한 `timing_time_aligned`와 `channel_time_aligned`를 우선 사용한다.

2026-07-01 결정: ASMR pipeline 병목은 아직 오디오->텍스트 모델 하나로 축소하지 않는다. `review_effort` summary와 `compare-evals` item에 text/channel/timing/missing/extra breakdown ratio를 추가해, 새 모델 실험 전에 VAD/chunking/alignment/channel attribution 중 어느 단계가 가장 큰 수정 비용을 만드는지 같은 지표로 비교한다. 2026-07-02부터 `compare-evals` item은 `dominant_review_effort_reason`, `dominant_review_effort_ratio`, `review_effort_reason_ranking`도 포함해 다음 개선 축을 숫자 비교 없이 바로 확인하게 한다.

2026-07-02 결정: 후보 간 보완 가능성을 segment 단위로 보기 위해 `compare-review-effort`를 추가했다. Output `custom-asmr-review-effort-comparison-v1`은 여러 eval report의 `review_effort.items`를 reference segment 기준으로 묶고, candidate status 순서를 입력 report 순서로 유지한다. 이 명령은 transcript를 수정하지 않고 WebUI 옵션을 늘리지 않는 CLI-only 진단 도구다.

2026-07-02 결정: `merge-review-effort` output root에 `case_summaries`, `case_count`, `next_case_id`를 추가하고, `review-pack`이 이 값을 pack index에 보존하게 했다. `case_summaries`는 case별 item count, reason counts, review duration sum, first/last issue time, top priority score/rank를 담고 `top_priority_rank` 순서로 정렬한다. WebUI review-pack header는 이 값이 있으면 clip 수와 함께 case 수와 다음 검수 case를 표시한다. Pack root에 `next_case_id`가 있고 아직 clip을 선택하지 않았다면 기존 `case 열기` 버튼은 해당 case의 첫 source item을 연다. 이 변경은 item priority order, transcript, reference, candidate, audio를 수정하지 않는다.

2026-07-02 결정: `review-pack` root `item_count`/`reason_counts`를 실제 packed item 기준으로 계산하고, `review-case-pack`도 root `item_count`, `reason_counts`, `case_count`, `next_case_id`, `case_summaries`, `duration_summary`를 내보내게 했다. 판단: 일반 후보 실패 pack과 reference-only pack이 WebUI에서 같은 header/첫 case navigation 계약을 쓰면 human-reviewed reference 제작의 반복 클릭과 JSON 확인 비용이 줄어든다. 이 변경은 clip queue metadata만 보강하며 reference/candidate/audio/model을 수정하지 않고 WebUI 옵션도 늘리지 않는다.

2026-07-02 결정: all8 pseudo-gold가 모든 현재 local 후보를 전 구간 실패로 보이게 하므로, reference 구조 자체를 먼저 검수할 수 있게 `audit-review-case-references`를 추가했다. Output `custom-asmr-reference-audit-suite-v1`은 transcript text를 저장하지 않고 segment id/time/channel 중심으로 overlap, same-channel/cross-channel overlap, exact-boundary overlap total과 same-channel/cross-channel split, long segment, near-full speech coverage, review flag를 기록한다. Product default는 overlap 100ms 이상, long segment 31,000ms 이상이다. all8 strict 1ms audit에서 same-channel overlap 42쌍이 모두 3-20ms였으므로, SRT 경계 jitter를 ASR-only blocker로 과대평가하지 않기 위해 100ms 미만 overlap은 기본 gate에서 제외한다. all8 product audit의 exact-boundary overlap 2쌍은 모두 L/R cross-channel이라 ASMR의 동시 좌우 발화일 수 있으므로 raw metric으로만 남기고, same-channel exact-boundary duplicate만 구조 blocker와 review queue에 넣는다. 30,007ms segment 1개도 30초 chunk 경계 근처 절단 오차로 보아 기본 long threshold를 31초로 둔다. Strict 진단은 `--overlap-min-ms 1` 또는 `--long-segment-ms 30000`으로 명시한다. `--review-effort-output`은 same-channel overlap, same-channel exact-boundary duplicate, long segment, reference review flag를 기존 `review-pack`에 넣을 수 있는 `custom-asmr-review-effort-v1` queue로 만든다. `--fail-on-audit`은 queue item이 남아 있을 때 report를 남긴 뒤 실패해 batch script에서 human-reviewed 승격 전 구조 검수를 강제한다. 이 명령은 reference를 수정하지 않고 WebUI 옵션을 늘리지 않는 CLI-only 진단 도구다.

2026-07-02 결정: channel attribution 실패가 heuristic 문제인지 reference label 문제인지 분리하기 위해 `audit-review-case-channels`를 추가했다. Output `custom-asmr-reference-channel-audit-suite-v1`은 prepared review case reference의 L/R segment를 stereo energy와 비교해 segment id/time/channel, L/R dBFS, energy channel, match/mismatch/uncertain status를 기록한다. Transcript text는 저장하지 않는다. `--review-effort-output`은 mismatch/uncertain segment를 기존 `review-pack`에 넣을 수 있는 `custom-asmr-review-effort-v1` queue로 만든다. 이 명령은 energy channel을 정답으로 승격하거나 reference를 수정하지 않고, human-reviewed 전 L/R label 검수 우선순위를 정하는 CLI-only 진단 도구다.

2026-08-30 결정: energy proxy와 다른 reference L/R이 실제 청취상 맞을 수 있으므로 segment master contract에 optional `channel_reviewed` boolean을 추가했다. 기본값은 `false`이며 사람만 명시적으로 `true`로 저장한다. Channel audit은 raw mismatch/uncertain 수치를 계속 보존하면서 `channel_reviewed=true` 예외를 뺀 `unresolved_mismatch_count`, `unresolved_uncertain_count`, `unresolved_count`를 별도로 기록한다. Review queue, `--fail-on-reference-channel-audit`, `pipeline-readiness` reference blocker는 unresolved count만 사용하고, old audit report는 raw count를 unresolved로 간주한다. Segment channel 또는 timing evidence가 바뀌거나 case slice에서 boundary가 잘리면 기존 channel review 판정은 무효화한다. 판단: 사람이 energy 제안을 거절하고 기존 라벨을 유지할 수 없는 이전 계약은 energy proxy를 사실상 정답으로 강제했으므로 human-reviewed reference 제작을 완료할 수 없는 정상 경로 버그였다.

2026-07-02 결정: `freeze-case-references`와 `build-eval-manifest`에 `--fail-on-reference-channel-audit`를 추가했다. Human-reviewed 승격 또는 모델 promotion manifest 생성 전에 reference L/R label의 unresolved stereo energy mismatch/uncertain queue가 남아 있으면 output을 쓰기 전에 실패한다. `--reference-channel-threshold-db`와 `--reference-channel-quiet-max-dbfs`는 이 gate의 audit 기준이며, all8 현재 검수 기준은 threshold 2dB, quiet gate off다. 이 gate는 energy channel을 정답으로 승격하지 않고, 사람이 channel label을 검수하지 않은 pseudo-gold가 human-reviewed reference로 넘어가는 것을 막는 CLI-only 보호 장치다.

2026-07-02 결정: channel attribution heuristic 자체를 pseudo-gold reference L/R label과 분리해 보기 위해 `audit-candidate-channels`를 추가했다. Output `custom-asmr-candidate-channel-audit-suite-v1`은 eval manifest candidate의 speech segment를 stereo energy와 비교하고, transcript text와 reference label 없이 segment id/time, candidate channel, energy channel, L/R dBFS, status를 저장한다. Status는 energy-labeled L/R 기준 `match`, `missed_attribution`, `wrong_side`, energy-uncertain 기준 `mix_match`, `over_attribution`으로 나눈다. 이 report는 energy channel을 human-reviewed truth로 승격하지 않고, reference label 검수 전 channel attribution heuristic을 진단하는 CLI-only proxy다.

2026-07-02 결정: ASMR 파이프라인이 “오디오->텍스트 모델만 남은 단계”인지 자동 판정하기 위해 `pipeline-readiness`를 추가했다. Output `custom-asmr-pipeline-readiness-v1`은 reference audit, optional reference channel audit, VAD coverage comparison, eval comparison을 읽어 `reference`, `vad_chunking`, `alignment`, `channel_attribution`, `text_asr` stage별 상태를 남긴다. `asr_only_ready`는 reference/VAD/alignment/channel 네 stage가 모두 pass일 때만 true이고, text ASR은 별도 product quality stage다. VAD comparison에 `quality_gate`가 있으면 gate를 통과한 chosen candidate를 VAD pass로 보고, gate가 없으면 missed reference speech가 남은 chosen candidate를 fail로 본다. `--reference-channel-audit`은 reference L/R label energy mismatch/uncertain count를 reference stage blocker와 metrics에 포함한다. `--alignment-comparison`은 alignment stage만, `--channel-comparison`은 channel attribution stage만 별도 eval comparison에서 읽어 aligner oracle/channel sweep과 ASR text 평가를 분리한다. `--candidate-channel-audit`은 channel attribution stage만 candidate stereo-energy proxy report에서 읽고 `--channel-comparison`보다 우선한다. 기본 eval-derived stage 판정은 edit-free strict mode이고, `--product-gate` 또는 개별 gate 인자를 지정하면 documented product threshold 기준으로 alignment/channel/text stage를 판정한다. `--product-gate`의 human-reviewed reference 조건은 reference stage에서 판정해 pseudo-gold reference를 ASR-only ready로 보지 않는다. `--fail-unless-asr-only-ready`는 report를 남긴 뒤 아직 ASR-only 단계가 아니면 실패한다. 이 명령은 기존 report를 읽기만 하며 WebUI 옵션을 늘리지 않는 CLI-only 상태 요약 도구다.

## 10초 실데이터 실험

입력:

```text
.casrt/experiments/upload-real-crop/01-front10.wav
```

참조:

```text
やばっ!見つかっちゃったぁ…。
ね、魔女ちゃん、こいつ強い?えっと…。
```

결과 요약:

| 경로 | segment 수 | CER | mean boundary error |
| --- | ---: | ---: | ---: |
| Qwen3-ASR 1.7B, 10초 통째 | 1 | 20.6% | 3600.0ms |
| Qwen3-ASR 1.7B + ForcedAligner | 1 | 20.6% | 3300.0ms |
| Qwen3-ASR 1.7B + energy chunking | 2 | 23.5% | 302.5ms |
| Qwen3-ASR 1.7B + energy chunking + channel attribution | 2 | 23.5% | 302.5ms |

energy chunking + channel attribution 출력:

```text
00:00.600-00:04.200  L
やば、見つかっちゃった。

00:04.700-00:09.700  MIX
ねね魔女ちゃんこいつ強い？えっと。
```

해석:

- timing은 energy chunking으로 크게 개선됐다.
- strict CER는 punctuation/表記差 때문에 높게 나온다.
- 첫 segment는 L/R energy 차이가 충분해서 L로 확정됐다.
- 둘째 segment는 L/R 차이가 6dB 미만이라 MIX로 남았다.

## 120초 품질 루프

일자: 2026-06-28

입력:

```text
data/uploads/01.淫魔＆魔女との遭遇.wav 앞 120초
```

reference:

```text
data/outputs/eval-csv-srt-01-full.srt에서 120초 crop
```

현재 실사용 후보 기준:

- practical CER: 5~10% 이하
- time-aligned 500ms boundary ratio: 90% 이상
- 명확한 L/R 구간 channel accuracy: 85~90% 이상
- unresolved candidate `needs_review`: 0%

자동 gate 예시:

```bash
uv run casrt eval-transcript ref.master.json candidate.master.json --product-gate
```

모델 승격용 manifest 평가는 reference authority도 gate로 강제한다.

```bash
uv run casrt eval-manifest gold.json --product-gate
```

결과:

| 후보 | segments | practical CER | time-aligned 500ms ratio | channel time-aligned accuracy | 판단 |
| --- | ---: | ---: | ---: | ---: | --- |
| Qwen3-ASR 1.7B, energy 800/400 | 7 | 29.3% | 10.0% | 100.0% on 2 comparable | 불합격: 과도한 병합 |
| Qwen3-ASR 1.7B, energy 500/200 | 26 | 21.7% | 25.0% | 66.7% | 불합격 |
| Qwen3-ASR 1.7B, energy 500/100 | 26 | 22.8% | 27.1% | 66.7% | 불합격 |
| Qwen3-ASR 1.7B, energy 500/200 + context | 26 | 46.5% | 25.0% | 66.7% | 불합격: context hallucination |
| Qwen3-ASR 1.7B, energy 500/200 + ForcedAligner | 26 | 21.7% | 31.2% | 83.3% | 불합격: timing/text 부족 |
| stable-ts baseline | 25 | 7.8% | 56.5% | n/a | text 합격, timing 불합격 |
| neosophie/Qwen3-ASR-1.7B-JA, energy 500/200 | 26 | 20.4% | 25.0% | 66.7% | 불합격 |
| neosophie/Qwen3-ASR-1.7B-JA, full 120s | 1 | 28.6% | 4.0% | n/a | 불합격: 과도한 병합 |
| neosophie/Qwen3-ASR-1.7B-JA, energy 1500/200 | 1 | 27.3% | 4.0% | n/a | 불합격: 과도한 병합 |

확장 pseudo-gold 결과:

주의: 현재 01/04/07 front120 reference는 `/home/brain-offloaded/vscode/asmr/whisperx-webui/data/outputs/eval-csv-srt-*-full.srt`에서 만든 pseudo-gold다. 해당 CSV/SRT의 `source_backend`는 stable-ts이며, 사람 검수 ground truth가 아니다. 따라서 아래 수치는 실제 정확도라기보다 stable-ts 계열 pseudo-reference와의 일치도다. stable-ts CSV channel을 candidate로 다시 넣으면 practical CER 0%, timing 100%, channel 93.1%가 나오므로 이것은 benchmark leakage로 간주하고 모델 승격 근거로 쓰지 않는다.

| 후보 | cases | reference segments | candidate segments | practical CER | time-aligned 500ms ratio | channel time-aligned accuracy | 판단 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Qwen/Qwen3-ASR-1.7B, energy 500/200, 01/04/07 front120 | 3 | 74 | 81 | 29.5% | 29.5% | 73.1% | 불합격 |
| Qwen/Qwen3-ASR-1.7B + Qwen3-ForcedAligner, energy 500/200, 01/04/07 front120 | 3 | 74 | 81 | 29.5% | 36.6% | 75.0% | 불합격: timing/channel 개선, text 미달, 1ms span 관측 |
| Qwen/Qwen3-ASR-1.7B + Qwen3-ForcedAligner guard80, energy 500/200, 01/04/07 front120 | 3 | 74 | 81 | 29.5% | 36.1% | 75.0% | 불합격: 80ms 미만 span 제거, text 미달 |
| Qwen/Qwen3-ASR-1.7B, energy 500/200 + max10000, 01/04/07 front120 | 3 | 74 | 83 | 29.3% | 30.1% | 72.0% | 불합격: 미세 개선, channel 악화 |
| Qwen/Qwen3-ASR-1.7B, energy 500/200 + max6000, 01/04/07 front120 | 3 | 74 | 96 | 30.3% | 30.1% | 72.0% | 불합격: text 악화 |
| neosophie/Qwen3-ASR-1.7B-JA, 01/04/07 front120 | 3 | 74 | 81 | 29.6% | 29.5% | 73.1% | 불합격 |
| neosophie/Qwen3-ASR-1.7B-JA + ASMR ONNX VAD default, 01/04/07 front120 | 3 | 74 | 47 | 30.2% | 17.6% | 73.7% | 불합격: timing 악화 |
| neosophie/Qwen3-ASR-1.7B-JA + ASMR ONNX VAD t035-pad400-sil400, 01/04/07 front120 | 3 | 74 | 16 | 33.4% | 7.0% | 76.9% | 불합격: 과도한 병합 |
| neosophie/Qwen3-ASR-1.7B-JA + ASMR ONNX VAD hybrid rescue500, 01/04/07 front120 | 3 | 74 | 109 | 31.0% | 31.1% | 73.1% | 불합격: text 악화 |
| mistralai/Voxtral-Mini-4B-Realtime-2602, 01/04/07 front120 | 3 | 74 | 44 | 40.0% | 28.7% | 63.6% | 불합격 |
| google/gemma-4-E4B-it, 4-bit local-transformers, MIX-first, 01/04/07 front120 | 3 | 74 | 81 | 42.3% | 29.5% | 73.1% | 불합격 |
| google/gemma-4-E4B-it, 8-bit local-transformers, MIX-first, 01/04/07 front120 | 3 | 74 | 81 | 46.1% | 29.5% | 73.1% | 불합격 |
| zhifeixie/Mega-ASR, routed, MIX-first, 01/04/07 front120 | 3 | 74 | 68 | 30.9% | 28.4% | 69.6% | 불합격 |
| zhifeixie/Mega-ASR, base-only threshold 1.1, MIX-first, 01/04/07 front120 | 3 | 74 | 64 | 30.8% | 27.3% | 68.2% | 불합격 |
| zhifeixie/Mega-ASR, forced LoRA, MIX-first, 01/04/07 front120 | 3 | 74 | 81 | 77.6% | 29.5% | 73.1% | 불합격: LoRA가 ASMR에서 악화 |
| stable-ts baseline, 01/04/07 front120 | 3 | 74 | 60 | 16.1% | 56.7% | n/a | 불합격: text/timing 부족, MIX-only |

2026-06-30 `review_effort` 재평가:

- Qwen/Qwen3-ASR-1.7B + Qwen3-ForcedAligner, 01/04/07 front120: `review_effort.segments_needing_edit=77`, ratio 100.0%. pseudo-gold 기준에서도 전 구간 수정 대상이라 기본 승격 불가다. Report: `/tmp/casrt-quality.Q5OdDf/qwen17-align-review-effort-report.json`.
- stable-ts CSV channel leakage candidate: `review_effort.segments_needing_edit=2`, ratio 2.7%. 낮은 값은 같은 stable-ts 계열 reference와 candidate를 비교한 누수 결과이므로 모델 품질 근거가 아니고, reference authority gate 필요성을 확인하는 값이다. Report: `/tmp/casrt-quality.Q5OdDf/stable-ts-csv-channel-review-effort-report.json`.
- stable-ts CSV channel + secured Qwen3-ForcedAligner re-alignment: practical CER 0.0%, time-aligned 500ms ratio 62.8%, channel time-aligned accuracy 86.2%, `review_effort.segments_needing_edit=48`, ratio 64.9%. 같은 stable-ts 계열 pseudo-reference 기준으로는 timing/review_effort가 크게 악화되므로 기본 승격하지 않는다. Report: `/tmp/casrt-quality.Q5OdDf/stable-ts-csv-channel-qwen-aligner-3case-report.json`.
- 2026-06-30부터 `review_effort.items`는 `casrt review-effort`로 별도 JSON 추출한다. 이 산출물은 모델 승격 근거가 아니라, 사람이 볼 다음 수정 후보와 heuristic 실패 패턴을 고르는 작업 큐다.
- 2026-06-30 `casrt review-effort` 실제 추출:
  - Qwen/Qwen3-ASR-1.7B + Qwen3-ForcedAligner: `/tmp/casrt-quality.Q5OdDf/qwen17-align-review-effort-items.json`, `item_count=77`, `reason_counts={text: 70, timing: 65, channel: 36, missing_reference: 2, extra_candidate: 3}`. Text 오류가 대부분이므로 alignment만으로 구제할 수 없는 후보로 본다.
  - stable-ts CSV channel + Qwen3-ForcedAligner: `/tmp/casrt-quality.Q5OdDf/stable-ts-csv-channel-qwen-aligner-review-effort-items.json`, `item_count=48`, `reason_counts={timing: 47, text: 4, channel: 4}`. Text는 보존되지만 Qwen aligner가 segment span을 자주 줄여 pseudo-reference 기준 timing review가 크게 늘어난다.
- 2026-06-30 existing artifact 재계산에서 stable-ts CSV channel + Qwen aligner에 coverage fallback을 적용하면 threshold 0.5 기준 `review_effort` 48 -> 33, time-aligned 500ms 62.8% -> 75.7%로 개선된다. threshold 0.9는 `review_effort` 6, time-aligned 96.6%까지 올라가지만 원 timing을 대부분 보존하는 값이라 기본 guard로 쓰지 않는다. 제품 기본값은 과도한 trim만 막는 0.5다.
- 2026-06-30 실제 Qwen aligner coverage05 재실행:
  - candidate dir: `/tmp/casrt-quality.Q5OdDf/stable-ts-csv-channel-qwen-aligner-coverage05`
  - manifest: `/tmp/casrt-quality.Q5OdDf/stable-ts-csv-channel-qwen-aligner-coverage05-3case-gold.json`
  - report: `/tmp/casrt-quality.Q5OdDf/stable-ts-csv-channel-qwen-aligner-coverage05-3case-report.json`
  - review queue: `/tmp/casrt-quality.Q5OdDf/stable-ts-csv-channel-qwen-aligner-coverage05-review-effort-items.json`
  - result: practical CER 0.0%, time-aligned 500ms ratio 75.7%, channel time-aligned accuracy 89.7%, `review_effort.segments_needing_edit=33`, ratio 44.6%, `reason_counts={timing: 32, text: 3, channel: 3}`.
  - 판단: coverage guard는 실제 worker 실행에서도 over-trim을 줄였지만 stable-ts 원본 pseudo-reference의 `review_effort=2`, 500ms 99.3%보다 여전히 나쁘다. 따라서 stable-ts 계열 후보에는 Qwen aligner를 기본 적용하지 않는다.
- 2026-06-30 `casrt review-pack` 실제 생성:
  - audio map: `/tmp/casrt-quality.Q5OdDf/review-audio-map.json`
  - Qwen/Qwen3-ASR-1.7B + Qwen3-ForcedAligner pack: `/tmp/casrt-quality.Q5OdDf/review-pack-qwen17-align`, clips 77개.
  - stable-ts CSV channel + Qwen3-ForcedAligner pack: `/tmp/casrt-quality.Q5OdDf/review-pack-stable-ts-qwen-aligner`, clips 48개.
  - stable-ts CSV channel + Qwen3-ForcedAligner coverage05 pack: `/tmp/casrt-quality.Q5OdDf/review-pack-stable-ts-qwen-aligner-coverage05`, clips 33개.
  - stable-ts CLI attributed 6dB pack: `/tmp/casrt-quality.Q5OdDf/review-pack-stable-ts-cli-attributed`, clips 66개.
  - stable-ts CLI attributed 8dB quiet-side pack: `/tmp/casrt-quality.Q5OdDf/review-pack-stable-ts-cli-attributed-quiet8`, clips 64개.
  - stable-ts CLI attributed 10dB pack: `/tmp/casrt-quality.Q5OdDf/review-pack-stable-ts-cli-attributed-th10`, clips 61개.
  - 여섯 pack 모두 `custom-asmr-review-pack-v1` index와 `clips/*.wav` 생성을 확인했다.
  - priority queue pack: input `/tmp/casrt-quality.Q5OdDf/stable-ts-cli-attributed-quiet8-review-effort-priority.json`, output `/tmp/casrt-quality.Q5OdDf/review-pack-stable-ts-cli-attributed-quiet8-priority`, clips 64개. `priority_rank=1`이 `clips/000001__01-front120__text__seg_000003__seg_000004.wav`, `priority_rank=64`가 `clips/000064__04-front120__channel__seg_000002__seg_000002.wav`로 들어가 review-effort 우선순서와 score/rank가 pack index에 보존됨을 확인했다.
- 2026-06-30 WebUI review-pack viewer smoke:
  - server: `uv run casrt serve --port 5174`
  - load API input: `/tmp/casrt-quality.Q5OdDf/review-pack-stable-ts-cli-attributed-quiet8-priority`
  - result: `clip_count=64`, first item `priority_rank=1`, `clip_url` 생성 확인.
  - first clip GET: `200 audio/wav 1962284 bytes` for `clips/000001__01-front120__text__seg_000003__seg_000004.wav`.
  - 판단: pack 생성은 CLI가 담당하고 WebUI는 local path로 priority 검수 큐를 열어 clip을 재생하는 viewer 역할만 한다. 추가 threshold/model 옵션은 노출하지 않는다.
  - 2026-07-02 behavior update: reference overlap audit item은 `REF2` segment id/channel을 표시하고, channel energy audit item은 `ENERGY` verdict와 L/R dBFS/delta evidence를 표시한다. `case 열기`로 source case editor에 들어간 뒤에도 reference audit overlap/long-segment evidence와 channel audit energy evidence를 status에 유지하고, overlap item은 `REF2` segment row를 보조 표시한다. 판단: human-reviewed reference 제작 중 pack 화면에서 본 검수 단서가 편집 화면 진입 시 사라지지 않게 하되, WebUI 옵션은 늘리지 않는다.
  - 2026-07-02 behavior update: review-pack root `duration_summary`가 있으면 WebUI header에 `listen` duration과 `focus effective/source` duration을 표시한다. Item `review_clip_*`가 원 segment bounds와 다르면 row meta에도 `focus 0:01.200 - 0:01.700` 같은 focus time range를 표시한다. Example test fixture header: `2 review clips · 2 cases · listen 0:06.000 · focus 0:05.000/0:10.000 · next front-b`. 판단: focus-window 검수 부담 감소를 JSON 파일을 열지 않고 확인할 수 있게 하며, WebUI option은 추가하지 않는다.
  - 2026-07-02 behavior update: channel audit item에서 source case를 열면 `ENERGY L/R 적용` 버튼으로 현재 source reference segment의 channel을 suggested energy channel로 저장할 수 있다. 이 동작은 사람이 명시적으로 누르는 편집 shortcut이며, `needs_review`는 그대로 둬 `검수 완료`와 분리한다. 판단: energy evidence를 정답으로 자동 승격하지 않으면서 channel label correction 클릭 수를 줄인다.
  - 2026-08-30 behavior update: channel audit target에서는 기존 `검수 완료` control을 `Channel 검수 완료`로 바꿔 표시하고, 사람이 energy 제안을 쓰지 않고 현재 L/R을 유지해도 `channel_reviewed=true`를 source reference에 저장한다. 새 WebUI option/button은 추가하지 않는다. `ENERGY L/R 적용`, 수동 channel 변경, start/end 변경은 기존 channel 검수 판정을 `false`로 무효화하므로 변경 뒤에는 다시 명시적으로 완료해야 한다.
  - 2026-08-30 behavior update: review pack에서 source editor를 연 동안 기존 `다음 case` control은 `다음 issue`로 표시한다. Click하면 현재 reference save를 flush하고 pack priority의 다음 source item을 같은 editor에 바로 연다. 마지막 source item에서는 disabled되고, `pack 목록`으로 돌아가면 실제로 열었던 item이 selected 상태로 복원된다. 새 button/option을 추가하지 않으면서 issue마다 `pack 목록 -> 다음 clip -> case 열기`를 반복하던 경로를 제거한다.
  - 2026-07-02 behavior update: focus range가 있는 review-pack item에서 source case를 열면 status hint에 focus range를 유지하고, 해당 source segment를 재생할 때 original segment bounds 대신 focus range만 재생한다. Segment `start_ms/end_ms`는 그대로 보존한다. 판단: pack에서 짧게 들은 evidence를 source editor에서도 다시 긴 segment 전체로 되돌리지 않고 검수할 수 있다.
  - 2026-07-02 behavior update: review-pack에서 source case를 열 때 원 pack context와 selected index를 보존하고, source editor의 `pack 목록` button으로 원래 review pack queue에 돌아갈 수 있게 했다. 판단: channel 적용/검수 완료 후 사람이 pack path를 다시 입력하거나 reload하지 않고 다음 priority item으로 이어갈 수 있다.
  - 2026-07-02 behavior update: review-pack viewer에서 기존 `다음 case` button을 `다음 clip`으로 재사용해 priority 순서의 다음 clip을 바로 재생한다. 선택이 없으면 첫 clip부터 시작하고 마지막 item에서는 disabled 된다. 판단: review-pack 생성 option을 늘리지 않고 반복 청취/navigation 클릭 수를 줄인다.
- 2026-06-30 WebUI review case set smoke:
  - server: `uv run casrt serve --port 5174`
  - load API input: `/tmp/casrt-quality.Q5OdDf/all8-front120-review-cases`
  - result: `kind=review-case-set`, `case_count=8`, first case `id=01-front120-existing-srt`, `segments=10`, `review_count=2`.
  - first case audio GET: `200 audio/wav 23040044 bytes`.
  - 동작: case list는 전체 content pending 수/duration과 `needs_review` flag 수, 각 case의 첫 미검수 segment 시간/텍스트 preview를 표시한다. Case click은 reference master를 기존 segment editor에 붙이고, edit/save는 reference master JSON과 `case-index.json` count를 갱신한다. 일반 `검수 완료`는 현재 segment의 `needs_review=false`, `content_reviewed=true`를 저장하고 다음 내용 미검수 segment로 이동한다. Text/time 재편집은 content evidence를 무효화한다. Channel-only audit target의 `Channel 검수 완료`는 `channel_reviewed=true`만 기록한다. `case 목록`/`다음 case` 이동은 pending save를 flush한다.
  - 2026-07-01 durable all8 loader duration smoke: input `.casrt/experiments/all8-front120-review-cases`, result `kind=review-case-set`, `case_count=8`, `review_count=15`, `review_duration_ms=163066`, first case `review_duration_ms=42974`.
  - 판단: human-reviewed gold 제작을 돕는 편집 경로이며, 검수 완료 판정은 하지 않는다.
- 2026-06-30 real SRT import audit:
  - `/home/brain-offloaded/vscode/asmr/whisperx-webui/data/outputs/02.敗北確定えっちバトル-c28f819996c9400cb05ec6ccbea1849f.srt` 같은 실제 산출물은 `[L:SPEAKER_01]`, `[R:SPEAKER_00]` prefix를 사용한다.
  - SRT import는 compound channel/speaker label을 본문 text에서 제거하고 channel metadata만 보존하도록 확장했다. `[SPEAKER_00]` 단독 prefix도 번역 대상 text에서 제거한다.
  - smoke output: `/tmp/casrt-quality.Q5OdDf/real-02-compound-label-smoke.master.json`; `rg "SPEAKER|\\[L|\\[R|\\[MIX|\\[LR"` 결과 없음, 첫 segment `channel=L`, 둘째 segment `channel=R`.
  - 목적: 외부 번역용 JSON에 speaker/channel label이 섞이지 않게 유지한다.

case별 practical CER:

| 후보 | case | practical CER | time-aligned 500ms ratio | channel time-aligned accuracy |
| --- | --- | ---: | ---: | ---: |
| Qwen3-ASR 1.7B energy 500/200 | 01-front120 | 20.4% | 25.0% | 66.7% |
| Qwen3-ASR 1.7B energy 500/200 | 04-front120 | 21.2% | 37.0% | 60.0% |
| Qwen3-ASR 1.7B energy 500/200 | 07-front120 | 52.7% | 26.9% | 80.0% |
| Qwen3-ASR 1.7B + ForcedAligner | 01-front120 | 21.7% | 31.2% | 83.3% |
| Qwen3-ASR 1.7B + ForcedAligner | 04-front120 | 20.4% | 43.2% | 50.0% |
| Qwen3-ASR 1.7B + ForcedAligner | 07-front120 | 51.8% | 36.0% | 78.6% |
| Qwen3-ASR 1.7B + ForcedAligner guard80 | 01-front120 | 21.7% | 31.2% | 80.0% |
| Qwen3-ASR 1.7B + ForcedAligner guard80 | 04-front120 | 20.4% | 43.2% | 50.0% |
| Qwen3-ASR 1.7B + ForcedAligner guard80 | 07-front120 | 51.8% | 34.6% | 80.0% |
| Qwen3-ASR 1.7B max10000 | 01-front120 | 21.3% | 27.1% | 60.0% |
| Qwen3-ASR 1.7B max10000 | 04-front120 | 20.4% | 37.0% | 60.0% |
| Qwen3-ASR 1.7B max10000 | 07-front120 | 51.8% | 26.9% | 80.0% |
| Qwen3-ASR 1.7B max6000 | 01-front120 | 21.9% | 20.8% | 60.0% |
| Qwen3-ASR 1.7B max6000 | 04-front120 | 22.7% | 41.3% | 60.0% |
| Qwen3-ASR 1.7B max6000 | 07-front120 | 51.5% | 28.8% | 80.0% |
| Neosophie | 01-front120 | 20.4% | 25.0% | 66.7% |
| Neosophie | 04-front120 | 21.3% | 37.0% | 60.0% |
| Neosophie | 07-front120 | 53.0% | 26.9% | 80.0% |
| Neosophie + ASMR ONNX VAD default | 01-front120 | 20.4% | 17.4% | 75.0% |
| Neosophie + ASMR ONNX VAD default | 04-front120 | 22.0% | 17.4% | 66.7% |
| Neosophie + ASMR ONNX VAD default | 07-front120 | 54.3% | 18.2% | 75.0% |
| Neosophie + ASMR ONNX VAD t035-pad400-sil400 | 01-front120 | 28.2% | 8.3% | n/a |
| Neosophie + ASMR ONNX VAD t035-pad400-sil400 | 04-front120 | 20.4% | 4.3% | n/a |
| Neosophie + ASMR ONNX VAD t035-pad400-sil400 | 07-front120 | 57.6% | 8.3% | 76.9% |
| Neosophie + ASMR ONNX VAD hybrid rescue500 | 01-front120 | 22.1% | 28.0% | 57.1% |
| Neosophie + ASMR ONNX VAD hybrid rescue500 | 04-front120 | 21.8% | 39.1% | 75.0% |
| Neosophie + ASMR ONNX VAD hybrid rescue500 | 07-front120 | 55.2% | 26.9% | 80.0% |
| Voxtral Mini Realtime | 01-front120 | 22.4% | 22.7% | 66.7% |
| Voxtral Mini Realtime | 04-front120 | 20.8% | 37.0% | 60.0% |
| Voxtral Mini Realtime | 07-front120 | 89.0% | 0.0% | n/a |
| Gemma 4 E4B 4-bit | 01-front120 | 30.9% | 25.0% | 66.7% |
| Gemma 4 E4B 4-bit | 04-front120 | 30.9% | 37.0% | 60.0% |
| Gemma 4 E4B 4-bit | 07-front120 | 72.6% | 26.9% | 80.0% |
| Gemma 4 E4B 8-bit | 01-front120 | 27.7% | 25.0% | 66.7% |
| Gemma 4 E4B 8-bit | 04-front120 | 31.4% | 37.0% | 60.0% |
| Gemma 4 E4B 8-bit | 07-front120 | 90.2% | 26.9% | 80.0% |
| Mega-ASR routed | 01-front120 | 23.0% | 22.7% | 66.7% |
| Mega-ASR routed | 04-front120 | 20.1% | 37.0% | 60.0% |
| Mega-ASR routed | 07-front120 | 55.8% | 25.0% | 75.0% |
| Mega-ASR base-only threshold 1.1 | 01-front120 | 23.5% | 22.7% | 66.7% |
| Mega-ASR base-only threshold 1.1 | 04-front120 | 20.1% | 37.0% | 60.0% |
| Mega-ASR base-only threshold 1.1 | 07-front120 | 54.6% | 21.4% | 72.7% |
| Mega-ASR forced LoRA | 01-front120 | 66.2% | 25.0% | 66.7% |
| Mega-ASR forced LoRA | 04-front120 | 81.3% | 37.0% | 60.0% |
| Mega-ASR forced LoRA | 07-front120 | 88.4% | 26.9% | 80.0% |
| stable-ts baseline | 01-front120 | 7.8% | 56.5% | n/a |
| stable-ts baseline | 04-front120 | 7.3% | 60.9% | n/a |
| stable-ts baseline | 07-front120 | 39.0% | 52.4% | n/a |

stable-ts에 L/R energy attribution만 붙인 channel 진단:

| attribution setting | practical CER | time-aligned 500ms ratio | channel time-aligned accuracy | comparable segments | candidate MIX ratio |
| ---: | ---: | ---: | ---: | ---: | ---: |
| threshold-only 3dB | 16.1% | 56.7% | 68.2% | 22 | 17.9% |
| threshold-only 6dB | 16.1% | 56.7% | 65.0% | 20 | 23.9% |
| threshold-only 10dB | 16.1% | 56.7% | 76.9% | 13 | 58.2% |
| 8dB + quiet <= -40dBFS | 16.1% | 56.7% | 68.8% | 16 | 40.3% |

2026-06-30 `casrt attribute-channels` 재현:

- threshold-only 6dB historical run: input dir `/tmp/casrt-quality.Q5OdDf/stable-ts-cli-mix`, output dir `/tmp/casrt-quality.Q5OdDf/stable-ts-cli-attributed`, report `/tmp/casrt-quality.Q5OdDf/stable-ts-cli-attributed-3case-report.json`. Result: practical CER 16.1%, time-aligned 500ms 56.7%, channel time-aligned accuracy 65.0%, candidate MIX ratio 23.9%, review effort 66/74, channel edits 41.
- 10dB sweep: output dir `/tmp/casrt-quality.Q5OdDf/stable-ts-cli-attributed-th10`, report `/tmp/casrt-quality.Q5OdDf/stable-ts-cli-attributed-th10-3case-report.json`. Result: practical CER 16.1%, time-aligned 500ms 56.7%, channel time-aligned accuracy 76.9%, candidate MIX ratio 58.2%, review effort 61/74, channel edits 28.
- 8dB + quiet-side -40dBFS default: output dir `/tmp/casrt-quality.Q5OdDf/stable-ts-cli-attributed-quiet8`, report `/tmp/casrt-quality.Q5OdDf/stable-ts-cli-attributed-quiet8-3case-report.json`, review queue `/tmp/casrt-quality.Q5OdDf/stable-ts-cli-attributed-quiet8-review-effort-items.json`, review pack `/tmp/casrt-quality.Q5OdDf/review-pack-stable-ts-cli-attributed-quiet8`. Result: practical CER 16.1%, time-aligned 500ms 56.7%, channel time-aligned accuracy 68.8%, candidate MIX ratio 40.3%, review effort 64/74, channel edits 36.
- 개발용 `attribute-channels` CLI의 기본값은 8dB + quiet-side -40dBFS gate다. Production 전사는 이 gate를 사용하지 않는다. 기존 6dB threshold-only보다 review effort가 66 -> 64로 줄었고 MIX ratio 40.3%로 50% gate 안에 남는다. 10dB threshold-only는 wrong L/R를 더 줄이지만 MIX ratio가 50% gate를 넘으므로 개발 기본값으로 승격하지 않는다.
- 2026-06-30 `--diagnostics-output` smoke: output dir `/tmp/casrt-quality.Q5OdDf/stable-ts-cli-attributed-quiet8-diagnostics`. 01/04/07 front120 MIX master에 대해 diagnostics JSON을 생성했고 attributed master는 기존 quiet8 output과 byte-identical이다. Reason counts는 `below_threshold=23`, `left_dominant=14`, `right_dominant=19`, `quieter_side_active=4`다.
- 2026-07-02 all8 channel summary smoke: command `uv run casrt sweep-channel-attribution .casrt/experiments/all8-front120-candidate-attach-smoke/eval-manifest.json --audio-map .casrt/experiments/all8-front120-candidate-attach-smoke/audio-map.json --threshold-db 8 --reset-speech-channels-to-mix -o .casrt/experiments/all8-front120-channel-summary-smoke --json`. This used the reference-copy attach smoke set as a diagnostics contract check, not as a model-quality benchmark. Result: `case_count=8`, `setting_id=th8_quietm40`, `speech_segments=82`, `changed_segments=14`, `changed_segment_ratio=17.1%`, reason counts `{below_threshold:67, left_dominant:11, quieter_side_active:1, right_dominant:3}`, attributed channel counts `{L:11, MIX:68, R:3}`. Follow-up comparison showed `channel_time_aligned_accuracy=53.8%`, `channel_time_aligned_mix_ratio=84.1%`, `segments_needing_edit_ratio=91.5%`. 판단: all8 pseudo-gold/reference-copy reset smoke에서도 기본 channel attribution은 대부분 MIX를 남기므로 channel 단계도 완료된 상태가 아니다. Summary fields let future threshold sweeps distinguish low-confidence MIX retention from L/R assignment errors without opening per-segment diagnostics first.
- 2026-07-02 all8 channel quiet-none sweeps: command `uv run casrt sweep-channel-attribution .casrt/experiments/all8-front120-candidate-attach-smoke/eval-manifest.json --audio-map .casrt/experiments/all8-front120-candidate-attach-smoke/audio-map.json --threshold-db 6 --threshold-db 8 --threshold-db 10 --quiet-channel-max-dbfs none --reset-speech-channels-to-mix -o .casrt/experiments/all8-front120-channel-quiet-none-sweep --json`; follow-up low-threshold command used thresholds `2/3/4/5` and output `.casrt/experiments/all8-front120-channel-quiet-none-low-threshold-sweep`. Best review-effort candidate was `th2_quietnone`: channel accuracy `50.0%`, MIX ratio `19.5%`, channel edit ratio `59.8%`, segments needing edit `63.4%`. `th3_quietnone` had MIX ratio `45.1%` but channel accuracy `46.7%`; `th4_quietnone` had MIX ratio `50.0%` and channel accuracy `48.8%`. 판단: disabling the quiet-side gate reduces MIX retention and can lower edit count on pseudo-gold/reference-copy diagnostics, but L/R accuracy remains around chance and far below the 85% product gate. Do not promote quiet-none or lower threshold as default; use it only as CLI-only evidence that channel blocker is not solved by threshold/quiet-side tuning alone.
- 2026-06-30 `compare-evals` smoke: outputs `/tmp/casrt-quality.Q5OdDf/eval-comparison-current.json` and gated `/tmp/casrt-quality.Q5OdDf/eval-comparison-current-gated.json`. Compared Qwen HF ASR, stable-ts quiet8, stable-ts 10dB, stable-ts + Qwen aligner reports. Ranking by review effort put 10dB first (`61/74`, ratio 82.4%), quiet8 second (`64/74`, ratio 86.5%), Qwen aligner third (`69/74`, ratio 93.2%), Qwen HF ASR fourth (`75/75`, ratio 100%). With product gates, all candidates fail; 10dB additionally fails MIX ratio gate at 58.2%, so default remains quiet8.
- 2026-06-30 Japanese relaxed CER 포함 재비교: output `/tmp/casrt-quality.Q5OdDf/eval-comparison-current-relaxed-gated.json`. Ranking은 기존과 동일하게 10dB, quiet8, Qwen aligner, Qwen HF 순서다. Japanese relaxed CER는 stable-ts 계열 15.5%, Qwen HF 27.5%로 practical CER보다 낮지만, 모든 후보가 practical CER, time-aligned 500ms, channel accuracy, review effort gate를 실패한다. 10dB는 MIX ratio 58.2%도 실패하므로 default는 계속 quiet8이다.
- 2026-06-30 unresolved candidate review gate 포함 재비교: output `/tmp/casrt-quality.Q5OdDf/eval-comparison-current-relaxed-review-gated.json`. `--max-candidate-review-ratio 0.00`을 추가해도 ranking은 10dB, quiet8, Qwen aligner, Qwen HF 순서로 유지된다. stable-ts 계열은 `candidate_review_ratio=0.0`이지만 기존 product gate를 실패한다. Qwen HF ASR는 `candidate_review_ratio=1.0`이라 timestamp/alignment 미확정 후보로도 실패한다.
- 2026-06-30 `--product-gate` preset smoke: output `/tmp/casrt-quality.Q5OdDf/eval-comparison-current-product-gate.json`. Ranking은 기존과 동일하며 모든 후보가 `reference_type 'pseudo-gold' != 'human-reviewed'`를 포함해 실패한다. stable-ts 계열은 candidate review gate는 통과하지만 practical CER, timing, channel/review-effort gate를 실패하고, Qwen HF는 candidate review ratio 100%도 함께 실패한다.
- 2026-06-30 priority review queue smoke: input `/tmp/casrt-quality.Q5OdDf/stable-ts-cli-attributed-quiet8-3case-report-relaxed.json`, output `/tmp/casrt-quality.Q5OdDf/stable-ts-cli-attributed-quiet8-review-effort-priority.json`. Result: `sort=priority_score_desc`, `item_count=64`, `reason_counts={text:46, channel:36, timing:41, missing_reference:7}`. Top item은 `01-front120` `seg_000003` vs `seg_000004`, reasons `text/channel/timing`, score `6590.22`로 사람이 먼저 들을 큰 복합 실패를 큐 상단에 올렸다.
- 2026-06-30 `sweep-channel-attribution` smoke: input manifest `/tmp/casrt-quality.Q5OdDf/stable-ts-cli-mix-3case-gold.json`, audio map `/tmp/casrt-quality.Q5OdDf/review-audio-map.json`, output `/tmp/casrt-quality.Q5OdDf/channel-sweep-smoke-guard`. Settings: 8dB/-40dBFS changed 33/60 segments, review effort 64/74, channel time-aligned accuracy 68.8%, MIX ratio 40.3%; 10dB/-40dBFS changed 20/60, review effort 60/74, channel time-aligned accuracy 75.0%, MIX ratio 62.7%. 10dB+quiet lowers edit count but violates the 50% MIX ratio gate, so default remains 8dB+quiet-side -40dBFS.
- 2026-07-01 `sweep-channel-attribution --product-gate` smoke: input manifest `/tmp/casrt-quality.Q5OdDf/stable-ts-cli-mix-3case-gold.json`, audio map `/tmp/casrt-quality.Q5OdDf/review-audio-map.json`, output `/tmp/casrt-channel-sweep-gate.OcMn59/sweep`. `comparison.json`과 `index.json`에 `quality_gate.preset=local-asmr-v1`가 저장된다. 두 setting 모두 `reference_type=pseudo-gold`와 text/timing/edit gate 때문에 실패한다. th8/-40은 channel accuracy 68.8%지만 MIX ratio 40.3%로 MIX gate는 통과한다. th10/-40은 channel accuracy 75.0%와 edit ratio 81.1%로 상대 개선이 있으나 MIX ratio 62.7%가 gate를 깨므로 기본값은 계속 8dB+quiet-side -40dBFS다.
- 2026-07-01 `sweep-channel-attribution --reset-speech-channels-to-mix` all8 Qwen official smoke: input manifest `.casrt/experiments/all8-front120-qwen-official-eval-cases/eval-manifest.json`, audio map `.casrt/experiments/all8-front120-qwen-official-eval-cases/audio-map.json`, output `.casrt/experiments/all8-front120-qwen-official-channel-sweep-reset`. 이 과정에서 relative manifest input일 때 generated setting manifest의 reference path가 output directory 아래로 잘못 해석되는 버그를 고쳐, setting directory 기준 상대경로로 저장하게 했다. Results: th6 changed 102/192, channel accuracy 51.5%, MIX ratio 59.3%, channel edit ratio 78.0%; th8 changed 89/192, channel accuracy 53.3%, MIX ratio 63.0%, channel edit ratio 79.3%; th10 changed 71/192, channel accuracy 51.9%, MIX ratio 66.7%, channel edit ratio 81.7%. 판단: all8 Qwen 후보에서는 reset sweep로도 기본 8dB/-40dBFS를 바꿀 근거가 없다.
- Qwen3-ForcedAligner를 6dB `stable-ts-cli-attributed` 후보에 적용한 실험은 output dir `/tmp/casrt-quality.Q5OdDf/stable-ts-cli-attributed-qwen-aligner`, report `/tmp/casrt-quality.Q5OdDf/stable-ts-cli-attributed-qwen-aligner-3case-report.json`에 있다. Result: practical CER 16.1%, time-aligned 500ms 47.0%, channel time-aligned accuracy 65.0%, candidate MIX ratio 25.8%, review effort 69/74, timing edits 51. 원 stable-ts CLI attributed의 timing/review effort보다 나빠 기본 경로로 쓰지 않는다.

window 단위 dominant fraction attribution도 01/04/07 front120 stable-ts baseline에서 실험했다. 100ms window, active threshold -60dBFS, margin 1~10dB, dominant fraction 35~75% sweep 기준 최고 channel time-aligned accuracy는 71.4%였고, segment 전체 RMS 10dB 방식의 76.9%보다 낮았다. 따라서 window 방식은 기본 구현으로 승격하지 않는다.

결정:

- Qwen 내장 energy 기본값은 `min_silence_ms=500`, `pad_ms=200`으로 낮춘다.
- `CASRT_QWEN_ENERGY_MAX_CHUNK_MS`는 추가했지만 기본값으로 켜지 않는다. `max10000`은 official Qwen 3-case에서 practical CER 29.5% -> 29.3%, time-aligned 500ms 29.5% -> 30.1%로 미세 개선했지만 channel accuracy가 73.1% -> 72.0%로 떨어졌다. `max6000`은 practical CER 30.3%로 악화됐다.
- `CASRT_QWEN_ASR_CONTEXT`에 긴 glossary를 그대로 넣는 방식은 기본값으로 쓰지 않는다. 짧은 구간에서 glossary 전체를 출력하는 hallucination이 발생했다.
- Qwen3-ForcedAligner는 official Qwen 3-case에서 text를 바꾸지 않고 time-aligned 500ms 29.5% -> 36.6%, channel time-aligned 73.1% -> 75.0%로 개선했다. `CASRT_QWEN_ASR_MIN_ALIGNED_DURATION_MS=80` guard 적용 후 80ms 미만 span은 제거됐고 time-aligned 500ms는 36.1%, channel time-aligned는 75.0%다. practical CER 29.5%가 여전히 기준 미달이므로 text 병목은 별도 모델/전처리/후처리로 풀어야 한다.
- 2026-06-30 audit에서 01/04/07 front120 reference가 stable-ts 기반 pseudo-gold임을 확인했다. `eval-manifest`는 `reference_type`과 `reference_notes`를 report에 보존한다. 제품 기본 모델 승격은 `reference_type=human-reviewed` gold에서 다시 판단해야 한다.
- 현재 Qwen3-ASR 1.7B 경로만으로는 품질 기준을 만족하지 못한다.
- 2026-07-01 official Qwen3-ASR all8 batch CLI benchmark: candidates `.casrt/experiments/all8-front120-qwen-official-candidates`, projects `.casrt/experiments/all8-front120-qwen-official-projects`, attach plan `.casrt/experiments/all8-front120-qwen-official-attach-plan.json`, eval case copy `.casrt/experiments/all8-front120-qwen-official-eval-cases`, report `.casrt/experiments/all8-front120-qwen-official-eval-report.json`, product gate report `.casrt/experiments/all8-front120-qwen-official-product-gate-report.json`. Command used `CASRT_LOCAL_WORKER_ENV_MODE=offline`, qwen-asr venv worker, local snapshot `/home/brain-offloaded/.cache/huggingface/hub/models--Qwen--Qwen3-ASR-1.7B/snapshots/7278e1e70fe206f11671096ffdd38061171dd6e5`, local path-only, `local_files_only`, and Python network block. Summary: reference segments 82, candidate segments 192, practical CER 59.7%, Japanese relaxed CER 58.7%, time-aligned 500ms ratio 16.0%, channel time-aligned accuracy 53.3%, candidate MIX ratio 63.0%, candidate review ratio 100%, review effort 82 segments / 100%. Product gate failed on pseudo-gold reference type, practical CER, timing, channel accuracy, MIX ratio, review effort, and candidate review ratio.
- 2026-07-02 official Qwen3-ASR all8 with energy t54/pad800/max30s: coverage candidate from `.casrt/experiments/all8-front120-review-cases/all8-energy-sweep-vad-coverage-comparison.json` was tested through actual ASR. Candidates `.casrt/experiments/all8-front120-qwen-official-energy-t54-pad800-max30s-candidates`, projects `.casrt/experiments/all8-front120-qwen-official-energy-t54-pad800-max30s-projects`, eval cases `.casrt/experiments/all8-front120-qwen-official-energy-t54-pad800-max30s-eval-cases`, report `.casrt/experiments/all8-front120-qwen-official-energy-t54-pad800-max30s-eval-report.json`, comparison `.casrt/experiments/all8-front120-qwen-official-energy-sweep-comparison.json`. Same local Qwen snapshot and offline worker constraints as the official baseline, plus energy settings now reproduced with `casrt vad coverage-cases --energy-threshold-dbfs -54 --energy-pad-ms 800 --energy-max-chunk-ms 30000` and stored in coverage `source_settings`. Result: candidate segments 144, practical CER 60.2%, Japanese relaxed CER 59.2%, time-aligned 500ms ratio 15.2%, channel time-aligned accuracy 58.3%, candidate MIX ratio 70.7%, candidate review ratio 100%, review effort 82 segments / 100%. Compared with baseline Qwen, text CER and timing are slightly worse, MIX ratio and channel edit ratio are worse, and every promotion gate still fails. 판단: better VAD coverage alone does not improve Qwen ASR quality; t54/pad800/max30s is not promoted.
- 2026-07-02 Neosophie Qwen3-ASR-JA all8 batch CLI benchmark: candidates `.casrt/experiments/all8-front120-neosophie-qwen-ja-candidates`, projects `.casrt/experiments/all8-front120-neosophie-qwen-ja-projects`, attach plan `.casrt/experiments/all8-front120-neosophie-qwen-ja-attach-plan.json`, eval cases `.casrt/experiments/all8-front120-neosophie-qwen-ja-eval-cases`, report `.casrt/experiments/all8-front120-neosophie-qwen-ja-eval-report.json`, product gate report `.casrt/experiments/all8-front120-neosophie-qwen-ja-product-gate-report.json`, comparison `.casrt/experiments/all8-front120-local-model-comparison-with-neosophie.json`. Command used the qwen-asr venv worker with `CASRT_LOCAL_WORKER_ENV_MODE=offline`, local path-only, `local_files_only`, and Python network block. Snapshot is `.casrt/models/neosophie-qwen3-asr-1.7b-ja-987bda160f2dabfa6757550bcff7cdda2ba0648c`; digest report `.casrt/model-digests/neosophie-qwen3-asr-1.7b-ja-987bda160f2dabfa6757550bcff7cdda2ba0648c-digest.json` has snapshot SHA-256 `9d3ad302e1265a2272bde5d548bdd159dd183d4d2823e7d1ff256955ae8272f9`. Result: reference segments 82, candidate segments 192, practical CER 59.4%, Japanese relaxed CER 58.4%, time-aligned 500ms ratio 16.0%, channel time-aligned accuracy 53.3%, candidate MIX ratio 63.0%, candidate review ratio 100%, review effort 82 segments / 100%. It is the best all8 local candidate by text CER among the current Qwen/Granite comparison, but the margin over Qwen official is small and every promotion gate still fails. 판단: model substitution alone has not solved ASMR quality; VAD/chunking, alignment, channel attribution, postprocessing, and human-reviewed gold remain active work.
- 2026-07-01 official Qwen3-ASR all8 + Qwen3-ForcedAligner batch benchmark: output `.casrt/experiments/all8-front120-qwen-official-qwen-aligner`, report `.casrt/experiments/all8-front120-qwen-official-qwen-aligner/eval-report.json`, diagnostics `diagnostics/*.alignment-diagnostics.json`. `align-review-case-candidates` ran with offline `CASRT_ALIGNER_COMMAND` and local Qwen3-ForcedAligner snapshot `c7cbfc2048c462b0d63a45797104fc9db3ad62b7`; elapsed about 55s for 8 case subprocesses. Result: changed segments 146/192, practical CER unchanged 59.7%, time-aligned 500ms ratio 12.3% (worse than base 16.0%), channel time-aligned accuracy 54.3%, candidate MIX ratio 56.8%, candidate review ratio 100%, review effort 82 segments / 100%. Product gate failed on pseudo-gold reference type, practical CER, timing, channel accuracy, MIX ratio, review effort, and candidate review ratio. 판단: forced alignment on top of weak text/chunk segmentation is not a default improvement; do not promote for all8.
- 2026-07-02 all8 alignment diagnostics summary smoke: no-op `CASRT_ALIGNER_COMMAND` ran on `.casrt/experiments/all8-front120-candidate-attach-smoke/case-index.json`, output `.casrt/experiments/all8-front120-alignment-diagnostics-noop-smoke`. Result: `candidate_count=8`, `segments=82`, `changed_segments=0`, `boundary_count=164`, `max_boundary_delta_ms=0`, `mean_abs_boundary_delta_ms=0.0`, `within_250ms_boundary_ratio=1.0`, `within_500ms_boundary_ratio=1.0`. 판단: batch alignment output now records enough boundary distribution diagnostics to catch aligners that move candidate timing aggressively before comparing eval manifests.
- 2026-07-01 all8 local model comparison: output `.casrt/experiments/all8-front120-local-model-comparison.json`. Ranking is Qwen official, Qwen official + aligner, then Granite base because Qwen has lower practical CER 59.7% vs Granite 63.8% and the aligner does not change text. All candidates fail every promotion-relevant gate and remain below the stable-ts pseudo-gold baseline. Breakdown regeneration after the ratio contract shows Qwen text/timing/channel edit ratios 98.8%/97.6%/79.3%, Qwen+aligner 97.6%/97.6%/75.6%, Granite 97.6%/96.3%/78.0%. 판단: current local Qwen/Granite open models do not solve ASMR quality by themselves, and the bottleneck is not reducible to text recognition only; human-reviewed gold, VAD/chunking/alignment/channel attribution improvements, review-first candidate selection, and possibly a stronger local model/runtime remain necessary.
- 2026-07-01 ASMR artifact metric smoke on all8: reports `.casrt/experiments/all8-front120-qwen-official-eval-report-artifacts.json`, `.casrt/experiments/all8-front120-granite-base-eval-report-artifacts.json`, comparison `.casrt/experiments/all8-front120-local-model-comparison-artifacts.json`. Qwen official summary: candidate segments 192, artifact segments 0, non-Japanese/high-density/repeated all 0. Granite base summary: candidate segments 163, artifact segments 1, high-density 1, repeated 1, non-Japanese 0. 판단: Qwen all8 실패는 단순 repeated hallucination 지표로 설명되지 않고 text mismatch/segmentation/channel 문제가 크다. Granite에는 ASMR-style repetition artifact가 소량 남지만 전체 실패 원인은 역시 CER/review-effort/timing/channel이다. 이 metric은 product gate가 아니라 다음 모델/VAD/chunking 후보 분석용으로 유지한다.
- `neosophie/Qwen3-ASR-1.7B-JA`는 다운로드 재시도 후 점수화했다. 120초 gold 기준 Qwen3-ASR 1.7B보다 약간 낫지만 practical CER 20.4%라 기본 승격하지 않는다.
- 01/04/07 front120 확장 gold에서도 Neosophie/Qwen3-ASR-JA는 practical CER 29.6%라 기본 승격하지 않는다. 2026-07-02 all8 front120 pseudo-gold에서는 practical CER 59.4%, time-aligned 500ms 16.0%, channel time-aligned accuracy 53.3%, review effort 100%라 Qwen official보다 text만 아주 조금 낫고 여전히 기본 승격하지 않는다. 특히 07의 whisper/침대 ASMR 구간에서 텍스트 인식이 크게 무너졌다.
- Neosophie full-window와 1.5초 silence 병합 실험은 text와 timing이 모두 악화됐다. 이 샘플에서는 chunk를 길게 잡는 것이 해결책이 아니다.
- `Qwen/Qwen3-ASR-1.7B-hf`는 Hugging Face metadata상 `automatic-speech-recognition`, `ja` 지원, `transformers` 모델이다. 2026-06-30 root Transformers 5.12.1에서는 `qwen3_asr` 아키텍처를 인식하지 못해 fail closed했고, Transformers main `5.13.0.dev0` commit `45b004d7bb505a258542d1965b0f9e0d8b03b89d` venv에서 5초 smoke와 01/04/07 front120 benchmark를 완료했다. 결과는 practical CER 29.4%, time-aligned 500ms 27.3%, channel time-aligned 68.2%, review effort 75/75라 기본 승격하지 않는다.
- `mistralai/Voxtral-Mini-4B-Realtime-2602`는 remote model code 없이 Transformers `VoxtralRealtimeForConditionalGeneration`으로 로딩됐다. 8.9GB weight는 단일 HF stream이 느려 HTTP range 8조각 병렬 다운로드로 확보했다. 30초 smoke와 01/04 일부 텍스트는 Qwen보다 자연스러웠지만, 07 whisper/침대 ASMR에서 chunked 입력은 대부분 빈 출력이었고 120초 full-window 입력도 앞부분만 출력해 기본 승격하지 않는다.
- `mistralai/Voxtral Mini Transcribe 2.0`는 Mistral API batch transcription 제품으로 확인됐고 open-weight 로컬 checkpoint는 확인하지 못했다. 외부 API는 제품 방향이 아니므로 기본 경로에서 제외한다.
- `google/gemma-4-E4B-it`는 공식 오디오 입력을 지원하고 5초 smoke에서 유의미한 전사를 반환했다. 그러나 01/04/07 front120 확장 gold에서 4-bit practical CER 42.3%, 8-bit practical CER 46.1%로 기준을 크게 벗어났다. 8-bit는 01 smoke와 01 case를 조금 개선했지만 07 whisper/침대 ASMR에서 반복 hallucination이 발생해 전체 지표가 악화됐다. 따라서 기본 승격하지 않는다.
- Gemma E4B 실험 산출물은 `/tmp/casrt-quality/gemma-e4b-4bit-bounded-results`, `/tmp/casrt-quality/gemma-e4b-8bit-bounded-results`, report는 `/tmp/casrt-quality/gemma-e4b-4bit-bounded-3case-report.json`, `/tmp/casrt-quality/gemma-e4b-8bit-bounded-3case-report.json`에 있다.
- `CohereLabs/cohere-transcribe-03-2026`는 2026년 2B local ASR 후보이며 일본어 포함 14개 언어를 지원한다. 공식 card는 Transformers native, safetensors, no timestamps/diarization, VAD 필요를 명시한다. Root Transformers 5.12.1에 Cohere ASR class가 있어 `local-cohere-asr` adapter를 구현한다. 다만 gated/custom_code repo이므로 실제 download/evaluation은 exact revision pin과 `casrt model digest` report 기록 후, local snapshot path + `trust_remote_code=False` + `local_files_only=True` + `use_safetensors=True` 조건에서만 한다.
- Cohere exact revision은 `b1eacc2686a3d08ceaae5f24a88b1d519620bc09`로 확인했다. `model.safetensors` LFS SHA-256은 `987bd3e141c7bfdb5a78f5db11397ee7737308357e6cc0a3f36a4979b158137a`, size는 4,131,862,976 bytes다. 2026-06-30 anonymous download는 gated 403으로 실패했다. 사용자가 HF access를 승인한 뒤 같은 revision을 받아 `casrt model digest`를 기록해야 평가 가능하다.
- 2026-06-30 live HF metadata refresh:
  - `microsoft/VibeVoice-ASR`는 `automatic-speech-recognition`, `transformers`, `safetensors`, `ja` tag가 있고 exact revision은 `d0c9efdb8d614685062c04425d91e01b6f37d944`다. Config architecture는 `VibeVoiceForASRTraining`, model type은 `vibevoice`, BF16 parameter count는 8.67B다. 현재 repo env의 Transformers 5.12.1에는 `VibeVoiceForASRTraining`가 없어 바로 실행하지 않는다.
  - `microsoft/VibeVoice-ASR-HF`는 `audio-text-to-text`, `transformers`, `safetensors`, `ja` tag가 있고 exact revision은 `f22241c2062b3b25272bf117397e03d73381037a`다. HF metadata상 `AutoModel`을 가리키지만 현재 repo env에는 VibeVoice 전용 audio-text-to-text class가 없어 바로 실행하지 않는다.
  - `OpenMOSS-Team/MOSS-Transcribe-preview-2B`는 2026-06-26 공개된 2.4B safetensors ASR 후보이고 exact revision은 `c98175cb20e48bd9be4e95f6c85f2af18899f780`다. 그러나 metadata에 `custom_code`가 있고 language tag가 `en` 중심이라 일본 ASMR 우선순위는 낮다. 실행하려면 외부 model code 검토가 먼저 필요하다.
  - `cstr/MOSS-Transcribe-preview-2B-GGUF`는 2026-06-30 공개 GGUF 변환이고 language tag는 `en`, `zh`다. 일본어 tag가 없고 GGUF runtime은 별도 실행 경로가 필요하므로 현재 로컬 ASMR 우선 후보가 아니다.
  - `XiaomiMiMo/MiMo-V2.5-ASR`는 2026-04-24 revision `98641d537df521ac6df05f74090475694d9510b7`의 ASR 후보지만 language tag가 `zh`, `en`, `yue`이고 일본어 tag가 없다. 일본 ASMR 후보 우선순위에서 제외한다.
- 2026-07-01 live HF metadata refresh:
  - `ibm-granite/granite-speech-4.1-2b`: revision `de575db64086f84fdc79da4932d1076e965bc546`, tags `transformers`, `safetensors`, `granite_speech`, `automatic-speech-recognition`, `ja`, license Apache-2.0. Model card는 2026-04-29 release, Japanese ASR support, native `transformers>=4.52.1`, and Japanese-tailored synthetic data를 명시한다. 현재 repo env Transformers 5.12.1에서 `transformers.models.granite_speech`와 `AutoModelForSpeechSeq2Seq` import가 가능해 `local-granite-asr` adapter를 추가했다. Persistent cache는 `.casrt/models/granite-speech-4.1-2b-de575db64086f84fdc79da4932d1076e965bc546`, digest report는 `.casrt/model-digests/granite-speech-4.1-2b-de575db64086f84fdc79da4932d1076e965bc546-digest.json`, snapshot SHA-256은 `67c7d69184b53bae7a2bec077fbc88d8695a72f043fd70831f4e4830dc4752ca`다. 실제 evaluation은 이 exact local snapshot digest 기준으로 수행한다.
  - `local-granite-asr`는 Granite Plus timestamp prompt도 같은 worker로 실험한다. `CASRT_GRANITE_ASR_PARSE_TIMESTAMPS=1`이면 `[T:N]` centisecond tag를 unwrap해 speech segment timing으로 쓰고 `_` silence marker로 segment를 split/trim한다. Tag가 없으면 기존 chunk-bound segment로 남기고 `needs_review=true`는 유지한다. 이 env는 내부 실험 경로이고 WebUI/CLI 옵션으로 노출하지 않는다.
  - Granite runtime note: `AutoProcessor` 생성 시 `GraniteSpeechFeatureExtractor`가 `torchaudio`를 요구한다. 첫 실제 smoke는 model load 후 `torchaudio` missing으로 실패했고, `local` extra에 `torchaudio`를 추가한 뒤 `torch 2.12.1+cu130` / `torchaudio 2.11.0+cu130` import smoke와 실제 전사를 통과했다.
  - 2026-07-01 Granite 10초 smoke: input `.casrt/experiments/upload-real-crop/01-front10.wav`, project root `.casrt/experiments/granite-smoke/projects`, project `52cf1cc9379a484e97cb866a3ec48399`, command env `CASRT_LOCAL_WORKER_ENV_MODE=offline CASRT_GRANITE_ASR_MAX_NEW_TOKENS=128`. Result: 3 speech segments, first text `やばい、見つかっちゃった`, all `needs_review=true`.
  - 2026-07-01 Granite 01/04/07 front120 pseudo-gold benchmark before non-Japanese hallucination filter: regenerated references/audio under `.casrt/experiments/granite-front120-eval`, manifest `.casrt/experiments/granite-front120-eval/granite-front120-3case-manifest.json`, report `.casrt/experiments/granite-front120-eval/reports/granite-front120-3case-report.json`. Summary: reference segments 60, candidate segments 69, practical CER 24.7%, Japanese relaxed CER 23.8%, time-aligned 500ms ratio 23.7%, candidate MIX ratio 54.4%, candidate review ratio 100%, review effort 67 segments / 100%.
  - 2026-07-01 Granite non-Japanese hallucination filter benchmark: manifest `.casrt/experiments/granite-front120-eval/granite-filtered-front120-3case-manifest.json`, report `.casrt/experiments/granite-front120-eval/reports/granite-filtered-front120-3case-report.json`. Summary: reference segments 60, candidate segments 61, practical CER 23.6%, Japanese relaxed CER 22.5%, time-aligned 500ms ratio 21.8%, candidate MIX ratio 56.4%, candidate review ratio 100%, review effort 62 segments / 100%. The filter removed punctuation-only/English-only hallucinations and slightly improved text/review effort, but `--product-gate` still failed on practical CER, timing, unavailable L/R channel accuracy, MIX ratio, review effort, and unresolved review flags. Granite is useful as a 2026 local candidate but does not beat the current best practical baseline and is not promoted.
  - 2026-07-01 Granite filtered + Qwen3-ForcedAligner benchmark: manifest `.casrt/experiments/granite-front120-eval/granite-filtered-qwen-aligner-front120-3case-manifest.json`, report `.casrt/experiments/granite-front120-eval/reports/granite-filtered-qwen-aligner-front120-3case-report.json`, aligner diagnostics under `.casrt/experiments/granite-front120-eval/reports/*qwen-aligner.diagnostics.json`. Summary: reference segments 60, candidate segments 61, practical CER 23.6%, Japanese relaxed CER 22.5%, time-aligned 500ms ratio 32.7%, candidate MIX ratio 54.5%, candidate review ratio 100%, review effort 62 segments / 100%. Diagnostics changed 18/22, 19/21, and 15/18 segments for cases 01/04/07, with max boundary deltas 1140ms, 2060ms, and 1420ms. The aligner is useful for timing but does not change text/channel uncertainty and is not enough to promote Granite.
  - 2026-07-01 Granite base all8 batch CLI benchmark: candidates `.casrt/experiments/all8-front120-granite-base-candidates`, projects `.casrt/experiments/all8-front120-granite-base-projects`, attach plan `.casrt/experiments/all8-front120-granite-base-attach-plan.json`, eval case copy `.casrt/experiments/all8-front120-granite-base-eval-cases`, report `.casrt/experiments/all8-front120-granite-base-eval-report.json`, product gate report `.casrt/experiments/all8-front120-granite-base-product-gate-report.json`. Command used `CASRT_LOCAL_WORKER_ENV_MODE=offline CASRT_GRANITE_ASR_DISABLE_NETWORK=1` with local snapshot `.casrt/models/granite-speech-4.1-2b-de575db64086f84fdc79da4932d1076e965bc546`. Summary: reference segments 82, candidate segments 163, practical CER 63.8%, Japanese relaxed CER 63.2%, time-aligned 500ms ratio 16.3%, channel time-aligned accuracy 55.2%, candidate MIX ratio 63.8%, candidate review ratio 100%, review effort 82 segments / 100%. Product gate failed on practical CER, timing, channel accuracy, MIX ratio, review effort, and candidate review ratio. 판단: 새 batch CLI real local model path는 동작하지만 Granite base all8 품질은 기존 3-case보다 더 낮아 기본 승격하지 않는다.
  - `ibm-granite/granite-speech-4.1-2b-plus`: revision `1454e6e1e33845ca9280ff65f52cf1141ba6e6e2`, tags `transformers`, `safetensors`, `granite_speech_plus`, multilingual ASR지만 HF card language metadata에 `ja`가 없다. Local snapshot은 `.casrt/models/granite-speech-4.1-2b-plus-1454e6e1e33845ca9280ff65f52cf1141ba6e6e2`, digest report는 `.casrt/model-digests/granite-speech-4.1-2b-plus-1454e6e1e33845ca9280ff65f52cf1141ba6e6e2-digest.json`, snapshot SHA-256은 `1ef78d5809fbf87e6d2b7cad64aab4b7cb35a460683f988db09e83f216644326`다.
  - 2026-07-01 Granite Plus 10초 prompt sweep: input `.casrt/experiments/upload-real-crop/01-front10.wav`, output `.casrt/experiments/granite-plus-smoke/01-front10-prompt-sweep.json`. Card timestamp prompt는 3 segments, raw `_ [T:90]いやば [T:146] ...`, 900~5190ms까지만 유의미했다. Japanese timestamp prompt도 3 segments, 0~3860ms까지만 유의미했다. Plain ASR prompt는 1 segment로 전체 10초를 뭉갰다.
  - 2026-07-01 Granite Plus timestamp 01/04/07 front120 benchmark: manifest `.casrt/experiments/granite-front120-eval/granite-plus-ts-front120-3case-manifest.json`, report `.casrt/experiments/granite-front120-eval/reports/granite-plus-ts-front120-3case-report.json`, product gate report `.casrt/experiments/granite-front120-eval/reports/granite-plus-ts-front120-3case-product-gate-report.json`. Summary: reference segments 60, candidate segments 94, practical CER 84.1%, Japanese relaxed CER 35.3%, time-aligned 500ms ratio 22.2%, candidate MIX ratio 53.7%, candidate review ratio 100%, review effort 68 segments / 98.6%. Timestamp tags produce more segment boundaries, but text hallucination/repetition and over-fragmentation make it worse than Granite base and Qwen aligner. Do not promote.
  - `efwkjn/cohere-asr-ja`: revision `8f1794e22b802731bdbf8ce53ff08f96a5af2bb4`, tags `safetensors`, `cohere_asr`, `custom_code`, `ja`, base model `CohereLabs/cohere-transcribe-03-2026`. Current Transformers 5.12.1 has `cohere_asr`, but metadata includes `custom_code`; execution priority는 official Cohere snapshot과 Granite 이후로 둔다.
  - `AutoArk-AI/ARK-ASR-3B`: revision `1e28271b79edc97635783bea65abc89195a09ed3`, tags include `ja`, `safetensors`, `custom_code`; current Transformers 5.12.1 has no `arkasr`, so still external code/runtime review 대상이다.
- `zhifeixie/Mega-ASR`는 2026-05 공개 Qwen3-ASR-1.7B 기반 robust ASR 후보이며, noisy/reverberant/clipped/band-limited/overlapping 등 어려운 실제 녹음에서 empty output, omission, repetition, hallucination을 줄이는 것을 목표로 한다. ASMR 전용은 아니지만 현재 07 whisper/침대 구간 실패 양상과 맞닿아 있으므로 다음 우선 모델 실험으로 둔다. 공식 runtime은 `xzf-thu/Mega-ASR` repository 코드와 checkpoint 배치를 요구하므로 `/tmp` 격리 환경에서 실행한다.
- Mega-ASR runtime은 실행 전 `gpt-5.4 xhigh` subagent가 정적 보안 검토했다. Verdict는 `PASS_WITH_CONSTRAINTS`다. 허용 범위는 `/tmp` 별도 venv, `/tmp` HF cache, Hugging Face allowlist download, Transformers backend만, `infer.py`/`evaluate_wer.py`만, vLLM/webui/training/wandb 금지, safetensors-only checkpoint 강제다. `adapter_model.bin`, `.pt`, `.pth` 또는 router non-safetensors checkpoint를 읽게 되면 미승인으로 간주한다.
- Mega-ASR 정적 검토에서 확인한 고위험 지점은 `lora_switch.py`의 `adapter_model.bin` fallback `torch.load`와 `router.py`의 non-safetensors `torch.load(weights_only=False)`다. 따라서 실험 전 `mega-asr-merged/adapter_model.safetensors`와 `audio_quality_router/best_acc_model.safetensors` 존재를 확인하고 unsafe fallback 파일은 사용하지 않는다.
- Mega-ASR는 `/tmp/casrt-quality/mega-asr-venv`, `HF_HOME=/tmp/casrt-quality/hf-home-mega-asr`, offline env에서 점수화했다. Checkpoint에는 unsafe pickle fallback 대상 파일이 없고, `adapter_model.safetensors`와 `best_acc_model.safetensors`를 확인했다.
- Mega-ASR 5초 smoke는 `やば、見つかっちゃった。`를 반환했고 router는 `use_lora=False`, degraded probability 0.0559였다. 그러나 01/04/07 front120 routed practical CER는 30.9%라 Neosophie 29.6%보다 약간 낮고 기준에 크게 못 미친다. threshold 1.1로 base-only에 가깝게 만든 경로도 30.8%로 유의미한 개선이 없다. forced LoRA는 77.6%로 크게 악화되어 ASMR 기본 경로로 쓰지 않는다.
- Mega-ASR 산출물은 `/tmp/casrt-quality/mega-asr-results/routed`, `/tmp/casrt-quality/mega-asr-results/base-threshold-1p1`, `/tmp/casrt-quality/mega-asr-results/force-lora`에 있다. Report는 각각 `routed-3case-report.json`, `base-threshold-1p1-3case-report.json`, `force-lora-3case-report.json`이다.
- `Atotti/llm-jp-4-8b-speech-asr`는 일본어 ASR 특화 8B 후보지만 model card상 `speech_llm_ja` 패키지(`git+https://github.com/Atotti/ja-speech-llm.git`)가 필요하다. 현재 설치된 Transformers `5.12.1`와 official main `5.13.0.dev0` 모두 `LlamaForSpeechLM`을 노출하지 않는다. 원격/외부 패키지 코드를 실행해야 하므로 사용자 명시 승인 전에는 자동 검증하지 않는다.
- `AutoArk-AI/ARK-ASR-3B`는 최신 로컬 후보지만 model card metadata에 `custom_code`가 있다. 외부 모델 저장소 코드를 실행하는 `trust_remote_code=True`는 기본 실험 경로로 쓰지 않고, 사용자 명시 승인이나 first-party package 지원이 있을 때만 검증한다.
- stable-ts/Whisper계 baseline은 현재 후보 중 text가 가장 좋지만 3-case practical CER 16.1%로 기준 10%를 넘고, time-aligned 500ms ratio도 56.7%로 기준 90%에 못 미친다. L/R energy attribution을 후처리로 붙여도 channel accuracy가 85%에 도달하지 않는다. 따라서 제품 기본 경로로 승격하지 않고 품질 상한 비교용으로만 유지한다. 다만 human-reviewed gold를 만들 때 우선 검수할 후보는 stable-ts CLI attributed 계열이다. 8dB + quiet-side gate가 기본 channel attribution pack이고, 10dB는 MIX ratio가 높아 기본값은 아니지만 review effort가 64 -> 61로 낮아 사람 검수 시작점으로 비교할 수 있다.
- 2026년 공개 파이프라인 조사에서 WhisperJAV는 ASMR/VR/whisper 콘텐츠에 `fidelity` pipeline과 `aggressive` sensitivity를 권장한다. 또한 ChronosJAV는 Qwen ASR, anime-whisper, Kotoba처럼 timestamp 없는 모델의 text generation과 timestamp alignment를 분리한다. 이 방향은 모델 단독 교체보다 VAD/scene detection/alignment를 분리해서 검증해야 함을 뒷받침한다.
- `TransWithAI/Whisper-Vad-EncDec-ASMR-onnx`는 Whisper encoder 기반 VAD이며 공개 discussion에서 일본어 ASMR 약 500시간으로 학습됐다고 설명된다. ASR 모델이 아니므로 text CER를 직접 개선하지는 않지만, energy splitter보다 ASMR whisper boundary를 더 잘 잡는지 `CASRT_VAD_COMMAND` 후보로 비교한다.
- ASMR Whisper ONNX VAD는 외부 `inference.py`를 실행하지 않고 `casrt vad whisper-asmr-onnx`로 직접 구현한다. 입력은 CASRT VAD command stdin contract를 따르고 출력은 `{ intervals }`만 반환한다. 전처리는 16kHz mono, 30초 chunk, WhisperFeatureExtractor, ONNX Runtime, sigmoid activation, hysteresis postprocess로 제한한다.
- ASMR Whisper ONNX VAD 실행 전 `gpt-5.4 xhigh` subagent가 정적 보안 검토했다. Verdict는 `PASS_WITH_CONSTRAINTS`다. 조건은 전용 venv, `model.onnx`/`model_metadata.json` 두 파일만 있는 전용 모델 디렉터리, SHA-256 기록, CPU-only `--force-cpu --num-threads 1`, 외부 `inference.py` 실행 금지, HF/API/W&B token 제거, VAD subprocess timeout, metadata/shape fail-closed 검증이다. ORT가 custom op/external tensor/unexpected provider를 요구하거나 모델 디렉터리에 extra file이 있으면 중단한다.
- ONNX Runtime session metadata는 input shape를 `['s6', 80, 3000]`, output shape를 `[1, 1500]`, provider를 `CPUExecutionProvider`로 노출했다. `s6`는 symbolic batch dim으로 보고 이 축만 허용하며, feature/time/output shape는 metadata 계약 그대로 고정 검증한다.
- ASMR Whisper ONNX VAD 파일은 `/tmp/casrt-quality/whisper-vad-asmr-onnx-model-v1`에 두 파일만 저장했고 SHA-256은 `/tmp/casrt-quality/whisper-vad-asmr-onnx-model-v1.sha256`에 있다. `model.onnx`는 `cd47513515766d57f740e3094440dbbca9ab87e026b9cf21540d7ad588c0e047`, `model_metadata.json`은 `aeb23b4d032b38e8fe36d6eb350c91f1ae751e0ce11813633ab9533ada4c55b3`다.
- VAD coverage 단독 비교에서 default ONNX VAD는 recall 87.5%, precision 87.2%, interval 47개였고, tuned `threshold=0.35,pad=400,min_silence=400`는 recall 93.6%, precision 83.0%, interval 16개였다. Energy 500/200 baseline은 recall 91.3%, precision 85.0%, interval 69개다. Tuned VAD는 coverage만 보면 좋아 보이지만 실제 ASR에서는 chunk가 과도하게 병합되어 timing과 text가 악화됐다.
- Neosophie/Qwen3-ASR-JA에 ONNX VAD를 붙인 실제 ASR 산출물은 `/tmp/casrt-quality/projects-neosophie-onnx-vad-default`, `/tmp/casrt-quality/projects-neosophie-onnx-vad-t035-pad400`, `/tmp/casrt-quality/projects-neosophie-onnx-vad-hybrid-rescue500`에 있다. Report는 `/tmp/casrt-quality/neosophie-onnx-vad-default-3case-report.json`, `/tmp/casrt-quality/neosophie-onnx-vad-t035-pad400-3case-report.json`, `/tmp/casrt-quality/neosophie-onnx-vad-hybrid-rescue500-3case-report.json`이다. 결론은 ASMR ONNX VAD를 단독 chunker로 기본 교체하지 않는 것이다.
- `--energy-rescue-min-ms 500` hybrid는 coverage recall 95.5%와 time-aligned 500ms 31.1%로 energy baseline보다 timing은 조금 높였지만 practical CER가 31.0%로 악화됐다. 따라서 hybrid도 기본 승격하지 않는다.
- vocal separation은 무조건 적용하지 않는다. WhisperJAV README는 blanket denoise/vocal separation이 Whisper log-Mel feature를 망가뜨릴 수 있다고 경고한다. 반면 WhisperJAV issue에서는 강한 BGM/환경음이 있을 때 UVR/MDX/Demucs류 분리의 필요성이 제기됐다. 따라서 BGM/SFX가 강한 case에서만 별도 실험으로 둔다.
- 다음 개선은 forced alignment 재평가, channel attribution 재평가 순서로 검증한다.

## 다음 작업 계획

완료된 제품 계획(2026-08-30):

1. Primary WebUI에서 review/evaluation loader를 별도 diagnostics 화면으로 격리하고, 정상 화면은 오디오 열기, 모델 설정, 전사, segment 수정/재전사, JSON/SRT 입출력만 남긴다.
2. WebUI와 CLI가 동일한 fixed production workflow를 호출하는지 실제 uploads/outputs 데이터로 end-to-end 검증한다.
3. 최근 일본어 ASMR/동인 음성 firsthand community report에서 로컬 단일 모델 후보를 갱신하고, 이미 보유한 exact local snapshot을 우선 실제 데이터에 실행한다.
4. Codex는 출력 구조, 일본어 붕괴/반복, 명백한 누락 신호, L/R duplicate, timestamp를 spot-check하되 audio-ground-truth 정확도를 주장하지 않는다.
5. 전체 행동 테스트와 문서 일치 검사를 통과한 뒤 작은 commit으로 push한다.

완료 상태: 1~5를 모두 수행했다. 최종 Python behavior suite `341` tests와 `node --check web/app.js`가 통과했고, production source audit에서 experiment VAD/aligner/channel env 참조, evaluation/readiness import, primary review control이 검출되지 않았다.

아래 항목은 과거 개발/평가 계획과 실행 기록이다. 현재 production 계약보다 우선하지 않는다.

1. Gold set 운영
   - gold set manifest CLI는 추가됐다.
   - `casrt slice-case`는 긴 원본 audio와 SRT/master에서 matching WAV/master eval case를 자르고 timestamp를 0 기준으로 rebase한다. 경계에서 잘린 segment는 `needs_review=true`로 표시한다.
   - `casrt prepare-review-cases`는 여러 slice plan을 한 번에 처리해 `audio-map.json`, `case-index.json`, `audio/*.wav`, `references/*.master.json`을 만들고, 모든 case에 candidate가 있으면 `eval-manifest.json`도 만든다.
   - `casrt review-case-status`는 준비된 `case-index.json`에서 audio/reference/candidate 파일 존재 여부, 실제 segment/review count, stale index count, 남은 reference/candidate `needs_review` case, candidate attach 진행률을 다시 계산한다. Report에는 `next_review_case_id`, `cases_missing_candidate`, `cases_with_candidate_review`, case별 `first_review_segment`를 포함해 CLI/WebUI가 같은 다음 검수/후보 준비 위치를 보여준다. `--include-reference-audits`는 structure/channel audit summary를 붙여, freeze/build 전에 같은 status command에서 남은 reference blocker를 볼 수 있게 한다. 모델 승격 전에는 `--fail-on-issues --fail-on-review --fail-on-missing-candidates --fail-on-candidate-review --fail-on-reference-audit --fail-on-reference-channel-audit`로 운영 gate를 걸 수 있지만, human-reviewed 여부 자체는 추정하지 않는다.
   - `casrt review-case-pack`은 준비된 case set의 reference `needs_review=true` 또는 `content_reviewed!=true` segment를 기존 `custom-asmr-review-pack-v1` audio clip queue로 만든다. 이 pack은 WebUI review-pack loader로 들을 수 있고 `case 열기`로 source case editor의 해당 reference segment로 이동할 수 있지만, reference 편집 source of truth는 `case-index.json` review case set이다.
   - 2026-07-01 판단: ASMR 품질 병목은 아직 오디오->텍스트 모델 하나로 축소되지 않았다. Durable review cases와 status/reporting은 준비됐지만, human-reviewed gold 승격, VAD/chunking/alignment/channel attribution 후보 비교, ASMR-specific hallucination/review-effort gate 통과가 남아 있다. ASR 모델 개선은 이 루프에서 후보별 오차가 측정된 뒤 병렬로 진행한다.
   - `casrt save-review-case-reference`는 WebUI 없이도 편집한 단일 SRT/master를 prepared case reference에 저장하고 `case-index.json` count를 갱신한다. Reference authority는 바꾸지 않는다.
   - `casrt transcribe-review-case-candidates`는 prepared case audio를 기존 project workflow로 일괄 전사해 case id별 candidate master JSON을 만든다. VAD/energy chunking, local adapter mix-first, channel attribution, optional aligner hook은 project transcription과 같은 경로를 사용한다.
   - `casrt align-review-case-candidates`는 candidate가 붙은 prepared case set을 받아 기존 candidate timing만 `CASRT_ALIGNER_COMMAND`로 일괄 재정렬하고 aligned candidates, diagnostics, attach plan, eval manifest를 만든다. 원본 case set과 candidate는 수정하지 않으며, forced alignment 후보를 base candidate와 같은 manifest gate로 비교하기 위한 CLI-only 경로다. Diagnostics summary는 boundary count, mean absolute boundary delta, 250ms/500ms 이내 boundary 비율을 포함하므로, aligner가 실제 품질 평가 전에 후보 timing을 과도하게 흔드는지 확인할 수 있다.
   - `casrt build-candidate-attach-plan`은 model/VAD/alignment 후보 산출물이 case id 파일명으로 정리되어 있을 때 attach plan을 자동 생성한다. 이 단계는 후보 transcript를 만들지 않고, 모든 case id가 정확히 하나의 candidate file과 매칭되는지만 검증한다.
   - `casrt attach-review-case-candidates`는 이미 준비된 case set에 case-local candidate SRT/master를 붙여 `candidates/*.master.json`과 index candidate fields를 만든다. Candidate 없는 durable gold set을 사람이 검수한 뒤 후보 평가 manifest로 넘기는 연결 단계이며, reference는 수정하지 않는다.
   - `casrt freeze-case-references`는 사람이 검수한 prepared reference들을 batch로 stable id와 `needs_review=false` 상태로 고정하고 새 case set을 만든다. 실검수 여부를 자동 판정하지 않으므로 pseudo-gold smoke에는 `reference_type=pseudo-gold`를 사용한다. Human-reviewed 승격 전에는 `--fail-on-review`, `--fail-on-reference-audit`, `--fail-on-reference-channel-audit`를 사용해 남은 flag, 구조 검수 queue, channel label energy queue가 output으로 고정되는 것을 막는다.
   - `casrt build-eval-manifest`는 candidate가 있는 `case-index.json`에서 `custom-asmr-eval-manifest-v1`을 다시 만든다. 사람이 reference를 수정한 뒤에는 `--reference-type human-reviewed --fail-on-review --fail-on-reference-audit --fail-on-reference-channel-audit`로 manifest를 만들고, 이어서 `eval-manifest --require-reference-type human-reviewed`로 품질 gate를 실행한다.
   - 2026-06-30 실데이터 smoke: `/home/brain-offloaded/vscode/asmr/whisperx-webui/data/uploads/01.淫魔＆魔女との遭遇.wav`와 `eval-01-full-stable-ts.srt`에서 0~60000ms를 잘라 `/tmp/casrt-quality.Q5OdDf/slice-case-smoke/01-front60.wav`와 `/tmp/casrt-quality.Q5OdDf/slice-case-smoke/01-front60.master.json`을 생성했다. Result: duration 60000ms, segments 14, review_count 1.
   - 2026-06-30 `prepare-review-cases` 실데이터 smoke: plan `/tmp/casrt-quality.Q5OdDf/prepare-review-cases-smoke-plan.json`, output `/tmp/casrt-quality.Q5OdDf/prepare-review-cases-smoke`. 01/04/07 front60 3개 case를 생성했고 result는 `case_count=3`, total `review_count=2`, audio duration은 모두 60000ms다. Case별 segments/review_count는 01: 14/1, 04: 9/0, 07: 11/1이다.
   - 2026-06-30 `review-case-status` 실데이터 smoke: input `/tmp/casrt-quality.Q5OdDf/prepare-review-cases-smoke/case-index.json`, output `/tmp/casrt-quality.Q5OdDf/prepare-review-cases-smoke/status.json`. Result: `case_count=3`, `candidate_case_count=0`, `reference_type_counts={pseudo-gold: 3}`, `missing_file_count=0`, `case_issue_count=0`, `reference_review_count=2`, `cases_needing_review=[01-front60, 07-front60]`.
   - 2026-06-30 `build-eval-manifest` 실데이터 smoke: plan `/tmp/casrt-quality.Q5OdDf/build-eval-manifest-smoke-plan.json`, output dir `/tmp/casrt-quality.Q5OdDf/build-eval-manifest-smoke-cases`, rebuilt manifest `/tmp/casrt-quality.Q5OdDf/build-eval-manifest-smoke-cases/eval-manifest.rebuilt.json`. Result: `case_count=1`, `candidate_case_count=1`, `reference_type=pseudo-gold`, `reference_review_count=1`, `missing_file_count=0`, `case_issue_count=0`. `--reference-type human-reviewed --fail-on-review`는 expected failure로 `review_count=1`을 막았고 output file을 만들지 않았다.
   - 2026-06-30 `freeze-case-references` 실데이터 smoke: input `/tmp/casrt-quality.Q5OdDf/build-eval-manifest-smoke-cases/case-index.json`, output `/tmp/casrt-quality.Q5OdDf/freeze-case-references-smoke`, `reference_type=pseudo-gold`로 실행했다. Result: `case_count=1`, frozen `review_count=0`, generated `audio-map.json`, `case-index.json`, `eval-manifest.json`. `review-case-status --fail-on-review`는 `reference_review_count=0`, `missing_file_count=0`, `case_issue_count=0`으로 통과했고, generated manifest 평가 report는 `/tmp/casrt-quality.Q5OdDf/freeze-case-references-smoke/eval-report.json`에 저장했다.
   - 2026-06-30 8개 실제 원본 front120 review case set 생성:
     - plan: `/tmp/casrt-quality.Q5OdDf/all8-front120-review-cases-plan.json`
     - output: `/tmp/casrt-quality.Q5OdDf/all8-front120-review-cases`
     - source: `/home/brain-offloaded/vscode/asmr/whisperx-webui/data/uploads`의 01~08 wav와 기존 `/data/outputs` SRT 산출물.
     - result: `case_count=8`, `reference_type_counts={pseudo-gold: 8}`, `missing_file_count=0`, `case_issue_count=0`, `reference_review_count=15`.
     - case별 segments/review_count: 01 `10/2`, 02 `10/1`, 03 `11/2`, 04 `12/2`, 05 `10/2`, 06 `11/2`, 07 `10/2`, 08 `8/2`.
     - `review-case-status --fail-on-review`는 report 출력 후 expected failure로 `review_count=15`를 반환했다.
     - 2026-07-01 진행률 field smoke: output `/tmp/casrt-review-progress-status.json`, `reference_review_count=15`, `reference_review_case_count=8`, `reference_review_clear_case_count=0`.
     - 2026-07-01 `save-review-case-reference` 복사본 smoke: copied case set `/tmp/casrt-save-review-case-smoke.c7G3HG/cases`, command saved `01-front120-existing-srt` reference to itself, result `segments=10`, `review_count=2`; follow-up status stayed `reference_review_count=15`, `case_issue_count=0`.
     - 2026-07-01 `freeze-case-references --fail-on-review` 실데이터 expected failure: input all8 case set, `reference_type=human-reviewed`, output `/tmp/casrt-freeze-fail-on-review-smoke`; failed with `reference review_count=15` and did not create output directory.
     - 2026-07-01 durable recreation: plan `.casrt/experiments/all8-front120-review-cases-plan.json`, output `.casrt/experiments/all8-front120-review-cases`, status `.casrt/experiments/all8-front120-review-cases/status.json`. Source SRT는 2025-12-22 hash outputs `01-251e...`, `02-ae45...`, `03-7363...`, `04-84f8...`, `05-0a9d...`, `06-803d...`, `07-1402...`, `08-4be4...`다. Result: `case_count=8`, `candidate_case_count=0`, `missing_candidate_case_count=8`, `reference_type_counts={pseudo-gold: 8}`, `missing_file_count=0`, `case_issue_count=0`, `reference_review_count=15`, `reference_review_case_count=8`, `reference_review_clear_case_count=0`. `review-case-status --fail-on-issues --fail-on-review` failed as expected with `review_count=15`.
     - 2026-07-02 `review-case-status --include-reference-audits` all8 smoke: command `uv run casrt review-case-status .casrt/experiments/all8-front120-review-cases/case-index.json --include-reference-audits --reference-channel-threshold-db 2 --reference-channel-quiet-max-dbfs none -o .casrt/experiments/all8-front120-review-cases/status-with-reference-audits.json`. Result: `reference_review_count=15`, `reference_review_duration_ms=163066`, `reference_audit.item_count=15`, `reference_audit.reason_counts={reference-needs-review:15}`, `reference_channel_audit.item_count=48`, `reference_channel_audit.reason_counts={reference-channel-energy-mismatch:30, reference-channel-energy-uncertain:18}`. Follow-up `review-case-status --fail-on-reference-channel-audit` saved `.casrt/experiments/all8-front120-review-cases/status-fail-reference-channel-audit.json` then failed with `reference_channel_audit_item_count=48`. 판단: human review 중에는 destructive freeze/build 명령을 실행하지 않아도 status 한 번으로 남은 structure/channel blockers를 볼 수 있다.
     - 2026-07-01 `review-case-status` duration field smoke: durable review set output `.casrt/experiments/all8-front120-review-cases/status-with-duration.json`, Qwen official eval case output `.casrt/experiments/all8-front120-qwen-official-eval-cases/status-with-duration.json`. Durable review set result: `reference_review_count=15`, `reference_review_duration_ms=163066`, `candidate_review_count=0`. Qwen official eval case result: `reference_review_count=15`, `reference_review_duration_ms=163066`, `candidate_review_count=192`, `candidate_review_duration_ms=850300`. 판단: human-reviewed gold 제작 잔량은 reference 기준 약 163초이고, Qwen 후보는 미확정 candidate flag가 약 850초라 promotion gate 전에 candidate review 자체도 막힌다.
     - 2026-07-01 `review-case-pack` durable output: `.casrt/experiments/all8-front120-review-case-pack`. Result: `format=custom-asmr-review-pack-v1`, `clip_count=15`, `items=15`, missing clips `0`, clip duration range `522..21987ms`, duration sum `170566ms`. 첫 item은 `01-front120-existing-srt/seg_000009`, 마지막 item은 `08-front120-existing-srt/seg_000008`이다. 이 pack은 위 15개 남은 reference review flag를 사람이 빠르게 듣기 위한 queue다.
     - 2026-07-02 `review-case-pack` root summary metadata smoke: command `uv run casrt review-case-pack .casrt/experiments/all8-front120-review-cases/case-index.json -o .casrt/experiments/all8-front120-review-case-pack-summary-metadata-smoke --json`. Result: `clip_count=15`, `item_count=15`, clip files `15`, `reason_counts={reference-needs-review:15}`, `case_count=8`, `next_case_id=01-front120-existing-srt`, `duration_summary={source_item_duration_ms_sum:163066,effective_item_duration_ms_sum:163066,clip_duration_ms_sum:170566,clip_duration_ms_max:21987,focus_item_count:0}`, first case summary `01-front120-existing-srt` has `item_count=2` and `review_duration_ms=42974`. 판단: reference-only pack도 WebUI header/첫 case 이동에 필요한 root metadata를 일반 review-pack과 같은 계약으로 제공한다.
     - 2026-07-01 `attach-review-case-candidates` real-data smoke: copied case set `.casrt/experiments/all8-front120-candidate-attach-smoke`, plan `.casrt/experiments/all8-front120-candidate-attach-smoke-plan.json`, candidate inputs는 복사본 reference master를 사용한 path/contract smoke다. 첫 실행은 plan 상대경로 오류로 output side effect 없이 실패했고, 수정 후 result `candidate_count=8`, `candidates/*.master.json` files `8`, follow-up status `candidate_case_count=8`, `missing_candidate_case_count=0`, `missing_file_count=0`, `case_issue_count=0`, `reference_review_count=15`. `build-eval-manifest` output `.casrt/experiments/all8-front120-candidate-attach-smoke/eval-manifest.json`, `case_count=8`, `reference_type=pseudo-gold`. 판단: candidate attach와 manifest 경로는 동작하지만, human-reviewed 승격 전 모델 promotion 평가는 아니다.
     - 2026-07-01 `review-case-status --fail-on-missing-candidates` real-data gate smoke: durable review set `.casrt/experiments/all8-front120-review-cases/case-index.json`는 expected failure로 `missing_candidate_count=8`을 반환했고, candidate attach smoke set `.casrt/experiments/all8-front120-candidate-attach-smoke/case-index.json`은 `missing_candidate_case_count=0`으로 통과했다. 판단: candidate 미부착 상태는 이제 eval manifest 준비 전 CLI gate에서 명시적으로 막힌다.
     - 2026-07-01 `review-case-status --fail-on-candidate-review` real-data gate smoke: Qwen official eval case `.casrt/experiments/all8-front120-qwen-official-eval-cases/case-index.json`는 expected failure로 `candidate_review_count=192`를 반환했고, Granite base eval case `.casrt/experiments/all8-front120-granite-base-eval-cases/case-index.json`도 `candidate_review_count=163`으로 실패했다. 판단: candidate가 붙어 있어도 미확정 모델 후보는 promotion gate 전 단계에서 명시적으로 막힌다.
     - 2026-07-01 `build-candidate-attach-plan` real-data smoke: copied candidate files는 `.casrt/experiments/all8-front120-candidate-attach-smoke/candidates`, output plan `.casrt/experiments/all8-front120-built-attach-plan.json`, candidate id `reference-copy-attach-smoke`. Result: `candidate_count=8`, all case ids matched one file. 이 plan은 기존 candidate attach smoke 산출물을 재사용한 path/contract smoke이며, 실제 모델 후보 품질 평가는 아니다.
     - 2026-07-01 `transcribe-review-case-candidates` real-data stub smoke: local OpenAI-compatible stub endpoint로 durable all8 case audio를 처리했다. Output `.casrt/experiments/all8-front120-transcribe-candidates-stub`, project root `.casrt/experiments/all8-front120-transcribe-projects-stub`, result `candidate_count=8`, each case `segments=2`, `review_count=0`. Follow-up `build-candidate-attach-plan` output `.casrt/experiments/all8-front120-transcribe-candidates-stub-attach-plan.json`, result `candidate_count=8`. 판단: 실제 모델 품질 평가는 아니지만, real audio 기준 batch transcription -> candidate files -> attach plan 연결은 동작한다.
     - 2026-07-01 Qwen official all8 review queue/pack: review-effort output `.casrt/experiments/all8-front120-qwen-official-review-effort.json`, review pack `.casrt/experiments/all8-front120-qwen-official-review-pack`. Result: `format=custom-asmr-review-effort-v1`, `item_count=82`, `reason_counts={text:81, channel:65, timing:80, missing_reference:1}`, `sort=priority_score_desc`; pack `format=custom-asmr-review-pack-v1`, `clip_count=82`, clip files `82`, duration range `522..51696ms`, duration sum `1832130ms`. 첫 item은 `02-front120-existing-srt` text/channel/timing 복합 실패, 마지막 item은 `08-front120-existing-srt` text/channel 실패다. 판단: Qwen official all8 후보는 사람이 볼 우선순위 pack까지 준비됐지만 review-effort가 높아 promotion 후보가 아니라 human-reviewed gold 제작과 실패 패턴 분석 입력이다.
     - 2026-07-01 Qwen official all8 linked review pack smoke: input review-effort `.casrt/experiments/all8-front120-qwen-official-review-effort.json`, audio map `.casrt/experiments/all8-front120-qwen-official-eval-cases/audio-map.json`, source case index `.casrt/experiments/all8-front120-qwen-official-eval-cases/case-index.json`, output `.casrt/experiments/all8-front120-qwen-official-review-pack-linked`. Result: `clip_count=82`, pack `source_case_index`와 첫 item `source_case_index`가 같은 case index path를 보존한다. 판단: WebUI review pack에서 후보 실패 clip을 들은 뒤 `case 열기`로 source case reference segment에 들어갈 연결이 준비됐다.
     - 2026-07-02 all8 review-effort comparison smoke: command `uv run casrt compare-review-effort .casrt/experiments/all8-front120-qwen-official-eval-report.json .casrt/experiments/all8-front120-qwen-official-energy-t54-pad800-max30s-eval-report.json .casrt/experiments/all8-front120-neosophie-qwen-ja-eval-report.json .casrt/experiments/all8-front120-granite-base-eval-report.json -o .casrt/experiments/all8-front120-review-effort-comparison-qwen-energy-neosophie-granite.json`. Result: `format=custom-asmr-review-effort-comparison-v1`, `report_count=4`, `reference_issue_count=82`, `extra_candidate_issue_count=0`, `reference_segments_failed_by_all=82`, `reference_segments_with_any_pass=0`. Candidate failure counts were Qwen official `text=81/channel=65/timing=80/missing=1`, Qwen energy `text=82/channel=68/timing=81/missing=0`, Neosophie `text=81/channel=65/timing=80/missing=1`, Granite `text=80/channel=64/timing=79/missing=2`. 판단: current all8 local candidates are not complementary at the reference-segment level; every reference segment still needs edits for every candidate, so the next quality work is not ASR model substitution alone.
     - 2026-07-02 all8 compare-evals dominant reason smoke: command `uv run casrt compare-evals .casrt/experiments/all8-front120-qwen-official-eval-report.json .casrt/experiments/all8-front120-qwen-official-energy-t54-pad800-max30s-eval-report.json .casrt/experiments/all8-front120-neosophie-qwen-ja-eval-report.json .casrt/experiments/all8-front120-granite-base-eval-report.json -o .casrt/experiments/all8-front120-local-model-comparison-dominant-reasons.json --json`. Result: all four candidates have `dominant_review_effort_reason=text`; dominant ratios are Neosophie `98.8%`, Qwen official `98.8%`, Qwen energy `100.0%`, Granite `97.6%`. Timing remains nearly as high: Neosophie/Qwen official `97.6%`, Qwen energy `98.8%`, Granite `96.3%`; channel edit ratios are `78.0%..82.9%`. 판단: dominant reason summary makes the current bottleneck visible without manually comparing five ratios. Current candidates fail first on text, but timing/segmentation and channel attribution remain co-bottlenecks, so the pipeline is still not reduced to a final ASR-only tuning step.
     - 2026-07-02 all8 strict 1ms reference audit smoke: current equivalent command `uv run casrt audit-review-case-references .casrt/experiments/all8-front120-review-cases/case-index.json --overlap-min-ms 1 --json -o .casrt/experiments/all8-front120-review-cases/reference-audit.json --review-effort-output .casrt/experiments/all8-front120-review-cases/reference-audit-review-effort.json`. Result: `format=custom-asmr-reference-audit-suite-v1`, `case_count=8`, `segment_count=82`, `speech_segment_count=82`, `review_count=15`, channel counts `L=40/R=42/MIX=0`, `speech_union_duration_ms=956301`, `speech_coverage_ratio=99.6%`, `overlap_pair_count=112`, `same_channel_overlap_pair_count=42`, `cross_channel_overlap_pair_count=70`, `exact_boundary_overlap_pair_count=2`, `pair_overlap_duration_ms=721158`, `long_segment_count=1`, `max_segment_duration_ms=30007`, `flagged_case_count=8`. Review-effort queue `.casrt/experiments/all8-front120-review-cases/reference-audit-review-effort.json` has `item_count=60`, reason counts `{reference-needs-review:15, reference-same-channel-overlap:42, reference-exact-boundary-overlap:2, reference-long-segment:1}`. Follow-up `review-pack` output `.casrt/experiments/all8-front120-review-cases/reference-audit-review-pack` has `clip_count=60`. Follow-up `audit-review-case-references --fail-on-audit` expected failure saved `.casrt/experiments/all8-front120-review-cases/reference-audit-fail-gate-smoke.json` and failed with `reference_audit_item_count=60`. Follow-up `freeze-case-references --fail-on-reference-audit` expected failure blocked output `.casrt/experiments/all8-front120-review-cases/freeze-audit-gate-smoke` with `reference_audit_item_count=60`. 판단: all8 pseudo-gold는 거의 전 구간 speech union이고 strict 1ms same-channel overlap도 많지만, 이후 분포 확인에서 same-channel overlap 42쌍은 모두 3-20ms였으므로 product default gate에는 과민하다.
     - 2026-07-02 all8 product-structure reference audit: command `uv run casrt audit-review-case-references .casrt/experiments/all8-front120-review-cases/case-index.json --json -o .casrt/experiments/all8-front120-review-cases/reference-audit-product-structure.json --review-effort-output .casrt/experiments/all8-front120-review-cases/reference-audit-product-structure-review-effort.json`, followed by product readiness output `.casrt/experiments/all8-front120-pipeline-readiness-product-structure.json`. Result: default thresholds `overlap_min_ms=100`, `long_segment_ms=31000`; segment count `82`, review flags `15`, overlap pairs `63`, same-channel overlap `0`, cross-channel overlap `63`, exact-boundary total `2`, exact-boundary same-channel `0`, exact-boundary cross-channel `2`, long segment `0`, max segment duration `30007ms`. Reference structure review-effort now has `item_count=15`, reason counts `{reference-needs-review:15}`. Product readiness still fails with blockers `[reference,alignment,channel_attribution]`, but reference reasons now exclude same-channel overlap, exact-boundary, and long segment blockers and keep `reference review flags remain: 15`, `reference_type 'pseudo-gold' != 'human-reviewed'`, channel audit mismatch `30`, uncertain `18`. 판단: product reference audit now separates ASMR-valid cross-channel overlap from actual same-channel duplicate risk, so the remaining reference blocker is human review/channel-label confidence rather than overlap structure.
     - 2026-07-02 all8 reference channel audit: command `uv run casrt audit-review-case-channels .casrt/experiments/all8-front120-review-cases/case-index.json --threshold-db 2 --quiet-channel-max-dbfs none --json -o .casrt/experiments/all8-front120-review-cases/reference-channel-audit-th2-quietnone.json --review-effort-output .casrt/experiments/all8-front120-review-cases/reference-channel-audit-th2-quietnone-review-effort.json`. Result: `format=custom-asmr-reference-channel-audit-suite-v1`, eligible L/R reference segments `82`, energy-labeled `64`, uncertain `18`, match `34`, mismatch `30`, match ratio among energy-labeled segments `53.1%`, energy-labeled ratio `78.0%`, energy channel counts `L=25/R=39/MIX=18`. Follow-up `review-pack` output `.casrt/experiments/all8-front120-review-cases/reference-channel-audit-th2-quietnone-review-pack` has `clip_count=48`, reasons `{reference-channel-energy-mismatch:30, reference-channel-energy-uncertain:18}`. Focus-window rerun added `review_clip_*` evidence fields for channel audit review items; pack `.casrt/experiments/all8-front120-review-cases/reference-channel-audit-th2-quietnone-review-pack-focus-metrics-smoke` reports `duration_summary={source_item_duration_ms_sum:984108,effective_item_duration_ms_sum:227816,clip_duration_ms_sum:272440,clip_duration_ms_max:6000,focus_item_count:48}`. 판단: isolated channel sweep의 L/R accuracy가 chance 수준인 이유는 channel heuristic만이 아니라 pseudo-gold reference channel labels도 energy evidence와 거의 chance 수준으로만 맞는다는 점이다. Channel attribution 기본값 승격 전 reference channel human review가 필요하며, focus-window pack은 그 검수 부담을 줄이는 listening aid일 뿐 reference를 자동 수정하지 않는다.
     - 2026-07-02 all8 reference channel gate smoke: implemented `--fail-on-reference-channel-audit` for `freeze-case-references` and `build-eval-manifest`. Commands `uv run casrt freeze-case-references .casrt/experiments/all8-front120-review-cases/case-index.json --reference-type human-reviewed --fail-on-reference-channel-audit --reference-channel-threshold-db 2 --reference-channel-quiet-max-dbfs none -o .casrt/experiments/all8-front120-review-cases/freeze-channel-audit-gate-smoke --json` and `uv run casrt build-eval-manifest .casrt/experiments/all8-front120-candidate-attach-smoke/case-index.json --reference-type human-reviewed --fail-on-reference-channel-audit --reference-channel-threshold-db 2 --reference-channel-quiet-max-dbfs none -o .casrt/experiments/all8-front120-candidate-attach-smoke/eval-manifest.channel-gated-human-reviewed-smoke.json --json` both failed before writing output with `reference_channel_audit_item_count=48`, reason counts `{reference-channel-energy-mismatch:30, reference-channel-energy-uncertain:18}`. 판단: 현재 all8 pseudo-gold는 사람이 channel label을 검수하기 전에는 human-reviewed freeze나 promotion manifest로 넘어갈 수 없게 되었다.
     - 2026-07-02 all8 candidate channel energy audits: implemented `audit-candidate-channels` so candidate L/R/MIX labels can be checked against stereo energy without using pseudo-gold reference labels. Commands wrote `.casrt/experiments/all8-front120-channel-quiet-none-low-threshold-sweep/th2_quietnone/candidate-channel-energy-audit.json`, `.casrt/experiments/all8-front120-qwen-official-candidate-channel-energy-audit.json`, `.casrt/experiments/all8-front120-neosophie-qwen-ja-candidate-channel-energy-audit.json`, and `.casrt/experiments/all8-front120-reference-copy-candidate-channel-energy-audit.json`. Result: `th2_quietnone` channel sweep candidate has `energy_labeled=64`, `matches=64`, `missed=0`, `wrong_side=0`, `over_attribution=0`; Qwen official and Neosophie candidates each have `energy_labeled=157`, `matches=89`, `missed=68`, `wrong_side=0`, `over_attribution=0`; reference-copy pseudo-gold candidate has `energy_labeled=64`, `matches=34`, `missed=0`, `wrong_side=30`, `over_attribution=18`. 판단: the isolated `th2_quietnone` channel attribution heuristic is internally consistent with stereo energy, while the reference-copy/pseudo-gold L/R labels conflict with the same energy evidence. Channel attribution should therefore be measured by candidate energy audit until reference labels are human-reviewed, while reference channel mismatches remain a reference blocker.
     - 2026-07-02 review-pack source case audio inference smoke: command `uv run casrt review-pack .casrt/experiments/all8-front120-review-cases/reference-channel-audit-th2-quietnone-review-effort.json -o .casrt/experiments/all8-front120-review-cases/reference-channel-audit-th2-quietnone-review-pack-inferred --json`. The review-effort already had `source_case_index=.casrt/experiments/all8-front120-review-cases/case-index.json`, so no `--audio-map` or explicit `--source-case-index` was needed. Result: `format=custom-asmr-review-pack-v1`, `clip_count=48`, generated wav files `48`, first item `05-front120-existing-srt/seg_000008`, and pack/item `source_case_index` preserved. 판단: prepared case 기반 reference/channel review queue는 duplicate `audio-map.json` argument 없이 만들 수 있으므로 WebUI 옵션을 늘리지 않고 CLI pack 생성 마찰을 낮춘다.
     - 2026-07-02 combined product reference review queue: implemented `merge-review-effort` to combine multiple `custom-asmr-review-effort-v1` queues before `review-pack` without modifying transcript/reference/audio. Current product command `uv run casrt merge-review-effort .casrt/experiments/all8-front120-review-cases/reference-audit-product-structure-review-effort.json .casrt/experiments/all8-front120-review-cases/reference-channel-audit-th2-quietnone-review-effort.json -o .casrt/experiments/all8-front120-review-cases/combined-reference-review-effort-product-structure.json`, followed by summary-preserving smoke `uv run casrt review-pack .casrt/experiments/all8-front120-review-cases/combined-reference-review-effort-product-structure.json -o .casrt/experiments/all8-front120-review-cases/combined-reference-review-pack-product-structure-summary-smoke --json`. Result: input items `63` merged to `55`, reasons preserved `{reference-needs-review:15, reference-channel-energy-mismatch:30, reference-channel-energy-uncertain:18}`, embedded `source_case_index` preserved, root `case_count=8`, `next_case_id=01-front120-existing-srt`, first case summary item count `6`, review pack clip count `55`, generated wav files `55`, pack root also preserved `case_summaries`, `case_count`, `next_case_id`. Server loader smoke on the same pack returned `kind=review-pack`, `clip_count=55`, `case_count=8`, `next_case_id=01-front120-existing-srt`, and clip URLs. Focus-window metrics smoke output `.casrt/experiments/all8-front120-review-cases/combined-reference-review-pack-product-structure-focus-metrics-smoke` kept `clip_count=55`; root `duration_summary={source_item_duration_ms_sum:1064178,effective_item_duration_ms_sum:307886,clip_duration_ms_sum:356010,clip_duration_ms_max:19978,focus_item_count:48}`. The max remains `19978ms` because non-channel structure-review items keep full source bounds. Root-summary rerun `.casrt/experiments/all8-front120-review-cases/combined-reference-review-pack-product-structure-root-summary-smoke` has `clip_count=55`, `item_count=55`, clip files `55`, root `reason_counts={reference-needs-review:15,reference-channel-energy-mismatch:30,reference-channel-energy-uncertain:18}`, same `case_count=8`, `next_case_id=01-front120-existing-srt`, and same duration summary. 판단: reference 구조 검수와 reference channel 검수를 하나의 WebUI queue로 열 수 있고, overlap/long jitter 항목을 product review queue에서 제외해 첫 readiness blocker를 더 정확하게 줄인다. Pack header and default `case 열기` can now show/open the next review case and blocker composition without adding WebUI options. 이 병합은 pseudo-gold를 수정하거나 human-reviewed로 승격하지 않는다.
     - 2026-07-02 all8 pipeline readiness smoke: command `uv run casrt pipeline-readiness --reference-audit .casrt/experiments/all8-front120-review-cases/reference-audit.json --vad-comparison .casrt/experiments/all8-front120-review-cases/all8-vad-coverage-comparison-gated.json --eval-comparison .casrt/experiments/all8-front120-local-model-comparison-dominant-reasons.json --fail-unless-asr-only-ready --json -o .casrt/experiments/all8-front120-pipeline-readiness.json`. Result: expected failure with `format=custom-asmr-pipeline-readiness-v1`, `asr_only_ready=false`, `production_ready=false`, `next_stage=reference`, `asr_only_blocking_stages=[reference,vad_chunking,alignment,channel_attribution]`, `quality_blocking_stages=[reference,vad_chunking,alignment,channel_attribution,text_asr]`. Stage reasons: reference has `review_count=15`, same-channel overlaps `42`, exact-boundary overlaps `2`, long segments `1`; chosen gated VAD candidate `all8-energy-vad-coverage` still misses `93627ms` reference speech; best eval candidate timing edit ratio `97.6%`, channel edit ratio `79.3%`, text edit ratio `98.8%`, segments needing edit `100%`. 판단: 현재 all8 기준으로 파이프라인은 ASR text 모델만 남은 상태가 아니며, reference 구조 검수와 VAD/chunking/alignment/channel attribution이 모두 다음 병목으로 남아 있다.
     - 2026-07-02 all8 pipeline readiness with best energy sweep VAD: command `uv run casrt pipeline-readiness --reference-audit .casrt/experiments/all8-front120-review-cases/reference-audit.json --vad-comparison .casrt/experiments/all8-front120-review-cases/all8-energy-sweep-vad-coverage-comparison.json --eval-comparison .casrt/experiments/all8-front120-local-model-comparison-dominant-reasons.json --json -o .casrt/experiments/all8-front120-pipeline-readiness-best-vad-sweep.json`. Result: `asr_only_ready=false`, blockers remain `[reference,vad_chunking,alignment,channel_attribution]`. The chosen VAD becomes `all8-energy-t54-pad800-max30s-vad-coverage` with `reference_recall=99.5%`, `detected_precision=99.7%`, `detected_max_interval_ms=30000`, but still misses `4697ms` reference speech. 판단: 최신 energy sweep은 baseline VAD보다 훨씬 낫지만, reference 구조와 timing/channel 실패가 그대로이고 VAD도 아직 완전한 ASR-only readiness로 보지 않는다.
     - 2026-07-02 all8 VAD quality gate smoke: command `uv run casrt vad compare-coverage .casrt/experiments/all8-front120-review-cases/all8-energy-vad-coverage.json .casrt/experiments/all8-front120-review-cases/all8-energy-t54-pad800-max30s-vad-coverage.json .casrt/experiments/all8-front120-review-cases/all8-energy-pad800-vad-coverage.json .casrt/experiments/all8-front120-review-cases/all8-energy-t54-pad400-max30s-vad-coverage.json --max-detected-interval-ms 30000 --max-missed-reference-ms 5000 --min-reference-recall 0.995 --min-detected-precision 0.99 --fail-on-gate --json -o .casrt/experiments/all8-front120-review-cases/all8-energy-sweep-vad-quality-gated.json`. Result: expected failure because 3 of 4 candidates failed; only `all8-energy-t54-pad800-max30s-vad-coverage` passed with `missed_reference_duration_ms=4697`, `reference_recall=99.5%`, `detected_precision=99.7%`, `detected_max_interval_ms=30000`. Failures: `all8-energy-pad800` too-long chunk/missed/recall, `all8-energy-t54-pad400-max30s` missed/recall, baseline `all8-energy-vad` too-long chunk/missed/recall. 판단: VAD coverage gate can now distinguish the best current energy candidate from weaker candidates before ASR evaluation, without promoting it as full product quality.
     - 2026-07-02 all8 readiness with VAD quality gate: command `uv run casrt pipeline-readiness --reference-audit .casrt/experiments/all8-front120-review-cases/reference-audit.json --vad-comparison .casrt/experiments/all8-front120-review-cases/all8-energy-sweep-vad-quality-gated.json --eval-comparison .casrt/experiments/all8-front120-local-model-comparison-dominant-reasons.json --fail-unless-asr-only-ready --json -o .casrt/experiments/all8-front120-pipeline-readiness-vad-quality-gated.json`. Result: expected failure, but `vad_chunking` is now `pass` using chosen `all8-energy-t54-pad800-max30s-vad-coverage`; remaining `asr_only_blocking_stages=[reference,alignment,channel_attribution]`, `quality_blocking_stages=[reference,alignment,channel_attribution,text_asr]`. 판단: with explicit VAD gates, current work can move the VAD blocker out of readiness, but reference structure, alignment, and channel attribution still prevent ASR-only tuning.
     - 2026-07-02 all8 readiness with channel comparison override: command `uv run casrt pipeline-readiness --reference-audit .casrt/experiments/all8-front120-review-cases/reference-audit.json --vad-comparison .casrt/experiments/all8-front120-review-cases/all8-energy-sweep-vad-quality-gated.json --eval-comparison .casrt/experiments/all8-front120-local-model-comparison-dominant-reasons.json --channel-comparison .casrt/experiments/all8-front120-channel-quiet-none-low-threshold-sweep/comparison.json --fail-unless-asr-only-ready --json -o .casrt/experiments/all8-front120-pipeline-readiness-channel-override.json`. Result: expected failure with `vad_chunking=pass`; `channel_attribution` now uses isolated channel sweep best `th2_quietnone` instead of ASR model comparison, with channel edit ratio `59.8%`, channel accuracy `50.0%`, MIX ratio `19.5%`. Remaining ASR-only blockers are `[reference,alignment,channel_attribution]`; product blockers also include `text_asr`. 판단: channel stage can now be measured separately from ASR text/timing, and the best current isolated channel heuristic still fails by L/R accuracy, not by MIX retention.
     - 2026-07-02 all8 product-gated readiness: command `uv run casrt pipeline-readiness --product-gate --reference-audit .casrt/experiments/all8-front120-review-cases/reference-audit.json --vad-comparison .casrt/experiments/all8-front120-review-cases/all8-energy-sweep-vad-quality-gated.json --eval-comparison .casrt/experiments/all8-front120-local-model-comparison-dominant-reasons.json --channel-comparison .casrt/experiments/all8-front120-channel-quiet-none-low-threshold-sweep/comparison.json --fail-unless-asr-only-ready --json -o .casrt/experiments/all8-front120-pipeline-readiness-product-gated.json`. Result: expected failure with `vad_chunking=pass`, blockers `[reference,alignment,channel_attribution]`, product blockers `[reference,alignment,channel_attribution,text_asr]`. Product-gate reasons: reference still has `review_count=15`, same-channel overlaps `42`, exact-boundary overlaps `2`, long segment `1`, and `reference_type pseudo-gold != human-reviewed`; alignment `time-aligned 500ms ratio 16.0% < 90.0%`; channel `accuracy 50.0% < 85.0%` while MIX ratio `19.5%` passes the 50% cap; text `practical CER 59.4% > 10.0%`, review effort `100% > 15%`, candidate review `100% > 0%`. 판단: strict edit-free mode와 product-gate mode 모두 같은 next blockers를 가리키지만, product-gate mode는 왜 fail인지 제품 threshold 언어로 남기고 pseudo-gold reference를 ASR-only ready로 보지 않는다.
     - 2026-07-02 all8 product-gated readiness with reference channel audit: command `uv run casrt pipeline-readiness --product-gate --reference-audit .casrt/experiments/all8-front120-review-cases/reference-audit.json --reference-channel-audit .casrt/experiments/all8-front120-review-cases/reference-channel-audit-th2-quietnone.json --vad-comparison .casrt/experiments/all8-front120-review-cases/all8-energy-sweep-vad-quality-gated.json --eval-comparison .casrt/experiments/all8-front120-local-model-comparison-dominant-reasons.json --channel-comparison .casrt/experiments/all8-front120-channel-quiet-none-low-threshold-sweep/comparison.json --fail-unless-asr-only-ready --json -o .casrt/experiments/all8-front120-pipeline-readiness-product-gated-with-channel-audit.json`. Result: expected failure with `vad_chunking=pass`; blockers remain `[reference,alignment,channel_attribution]`, product blockers `[reference,alignment,channel_attribution,text_asr]`. Reference stage now also records channel audit metrics: eligible `82`, energy-labeled `64`, match `34`, mismatch `30`, uncertain `18`, match ratio `53.1%`. 판단: readiness now makes the reference-channel blocker explicit, so channel attribution cannot be treated as finished until reference channel labels are human-reviewed or a stronger local channel attribution method beats this audit on human-reviewed gold.
     - 2026-07-02 all8 product-gated readiness with alignment override: implemented `--alignment-comparison` so alignment oracle reports can drive only the `alignment` stage while text still comes from ASR model comparison and channel still comes from isolated channel sweep. Command used `--alignment-comparison .casrt/experiments/all8-front120-reference-copy-qwen-aligner-oracle-only-comparison.json` and output `.casrt/experiments/all8-front120-pipeline-readiness-product-gated-alignment-override.json`. Result: expected failure with `vad_chunking=pass`, blockers `[reference,alignment,channel_attribution]`; alignment metrics now come from the Qwen reference-copy oracle and fail with `time_aligned_500ms_ratio=51.2%`, `timing_edit_segment_ratio=75.6%`, `best_label=all8-front120-reference-copy-qwen-aligner-oracle-eval-report`. 판단: readiness can separate ASR text quality from alignment candidate quality; this report means the Qwen forced aligner candidate fails, not that no-op alignment is worse than Qwen.
     - 2026-07-02 Qwen aligner context experiment on Qwen official all8: implemented `CASRT_QWEN_ALIGNER_CONTEXT_MS` so the secured Qwen3-ForcedAligner worker can receive padded audio and move boundaries outside the original candidate segment while keeping coverage checks against the original segment duration. Commands used `align-review-case-candidates` on `.casrt/experiments/all8-front120-qwen-official-eval-cases/case-index.json` with context `500` and `2000`, then `eval-manifest` and `compare-evals --product-gate`; outputs `.casrt/experiments/all8-front120-qwen-official-qwen-aligner-context500*`, `.casrt/experiments/all8-front120-qwen-official-qwen-aligner-context2000*`, comparison `.casrt/experiments/all8-front120-qwen-official-context-aligner-comparison.json`. Result: baseline Qwen official `practical_cer=59.7%`, `time_aligned_500ms=16.0%`, `channel_time_aligned_accuracy=53.3%`, MIX `63.0%`; context500 kept CER `59.7%` but worsened timing to `11.1%`, channel accuracy `54.8%`, MIX `61.7%`; context2000 kept CER `59.7%` but worsened timing to `6.9%`, channel accuracy `56.7%`, MIX `62.5%`. 판단: context padding implementation is useful for bounded experiments, but neither setting is promoted; Qwen ASR text and segment structure remain too weak for forced alignment to rescue.
     - 2026-07-02 Qwen aligner all8 reference-copy oracle: command `CASRT_ALIGNER_ENV_MODE=offline CASRT_QWEN_ALIGNER_REQUIRE_LOCAL_MODEL_PATH=1 CASRT_QWEN_ALIGNER_LOCAL_FILES_ONLY=1 CASRT_QWEN_ALIGNER_DISABLE_NETWORK=1 CASRT_ALIGNER_COMMAND='.casrt/qwen-asr-venv/bin/python -m custom_asmr_srt_stack.qwen_aligner_worker --model-id /home/brain-offloaded/.cache/huggingface/hub/models--Qwen--Qwen3-ForcedAligner-0.6B/snapshots/c7cbfc2048c462b0d63a45797104fc9db3ad62b7' uv run casrt align-review-case-candidates .casrt/experiments/all8-front120-candidate-attach-smoke/case-index.json -o .casrt/experiments/all8-front120-reference-copy-qwen-aligner-oracle --json`, followed by `eval-manifest --product-gate` on the reference-copy baseline and aligned oracle manifests, then `compare-evals --product-gate -o .casrt/experiments/all8-front120-reference-copy-qwen-aligner-oracle-comparison.json`. Align report: `segments=82`, `changed_segments=69`, `max_boundary_delta_ms=13070`, `mean_abs_boundary_delta_ms=1035.9`, `within_500ms_boundary_ratio=54.9%`. Baseline reference-copy report `.casrt/experiments/all8-front120-reference-copy-baseline-eval-report.json`: practical CER `0.0%`, time-aligned 500ms `95.1%`, channel time-aligned accuracy `87.8%`, timing edit ratio `9.8%`, segments needing edit `12.2%`, candidate review ratio `18.3%` from inherited pseudo-gold review flags. Qwen-aligned oracle report `.casrt/experiments/all8-front120-reference-copy-qwen-aligner-oracle-eval-report.json`: practical CER `0.0%`, time-aligned 500ms `51.2%`, channel time-aligned accuracy `80.5%`, timing edit ratio `75.6%`, segments needing edit `75.6%`, same candidate review ratio `18.3%`. 판단: 완벽한 reference-copy text/segment에서도 Qwen3-ForcedAligner가 ASMR pseudo-gold timing을 크게 악화시키므로, 현재 alignment blocker는 ASR text만의 문제가 아니며 이 aligner를 기본 timing 계층으로 승격하지 않는다. 새 alignment 후보는 이 oracle baseline을 최소 기준으로 넘어야 한다.
     - 2026-07-02 all8 product-structure readiness with best alignment policy: command `uv run casrt pipeline-readiness --product-gate --reference-audit .casrt/experiments/all8-front120-review-cases/reference-audit-product-structure.json --reference-channel-audit .casrt/experiments/all8-front120-review-cases/reference-channel-audit-th2-quietnone.json --vad-comparison .casrt/experiments/all8-front120-review-cases/all8-energy-sweep-vad-quality-gated.json --eval-comparison .casrt/experiments/all8-front120-local-model-comparison-dominant-reasons.json --alignment-comparison .casrt/experiments/all8-front120-reference-copy-qwen-aligner-oracle-comparison.json --channel-comparison .casrt/experiments/all8-front120-channel-quiet-none-low-threshold-sweep/comparison.json --fail-unless-asr-only-ready --json -o .casrt/experiments/all8-front120-pipeline-readiness-product-structure-best-alignment-policy.json`. Result: expected failure with `vad_chunking=pass`, `alignment=pass`, blockers `[reference,channel_attribution]`, product blockers `[reference,channel_attribution,text_asr]`. Alignment chose `all8-front120-reference-copy-baseline-eval-report` with `time_aligned_500ms_ratio=95.1%`, while channel still fails at accuracy `50.0%`. 판단: 현 최선 alignment 정책은 forced aligner가 아니라 no-op baseline이므로, “ASR text만 남았는가”의 자동 blocker는 reference human review/channel labels와 channel attribution이다. Text ASR은 그 다음 product quality blocker로 남는다.
     - 2026-07-02 all8 product-structure readiness with energy-proxy channel stage: command `uv run casrt pipeline-readiness --product-gate --reference-audit .casrt/experiments/all8-front120-review-cases/reference-audit-product-structure.json --reference-channel-audit .casrt/experiments/all8-front120-review-cases/reference-channel-audit-th2-quietnone.json --vad-comparison .casrt/experiments/all8-front120-review-cases/all8-energy-sweep-vad-quality-gated.json --eval-comparison .casrt/experiments/all8-front120-local-model-comparison-dominant-reasons.json --alignment-comparison .casrt/experiments/all8-front120-reference-copy-qwen-aligner-oracle-comparison.json --candidate-channel-audit .casrt/experiments/all8-front120-channel-quiet-none-low-threshold-sweep/th2_quietnone/candidate-channel-energy-audit.json --fail-unless-asr-only-ready --json -o .casrt/experiments/all8-front120-pipeline-readiness-product-structure-energy-channel.json`. Result: expected failure with `vad_chunking=pass`, `alignment=pass`, `channel_attribution=pass`, `asr_only_blocking_stages=[reference]`, `quality_blocking_stages=[reference,text_asr]`. Channel stage records `source=candidate_channel_energy_audit`, `energy_labeled_match_ratio=100.0%`, `energy_labeled_mix_ratio=0.0%`, `wrong_side=0`, `over_attribution=0`, plus a warning that this is stereo energy proxy rather than human-reviewed reference labels. 판단: current automatic pipeline work has reduced non-text ASR-only blockers to reference human review/channel label authority. Text ASR remains poor, but product tuning against text should wait until the reference is human-reviewed or clearly separated from pseudo-gold noise.
     - 2026-07-02 all8 current-best readiness rerun after review-pack focus metrics/WebUI display: command `uv run casrt pipeline-readiness --product-gate --reference-audit .casrt/experiments/all8-front120-review-cases/reference-audit-product-structure.json --reference-channel-audit .casrt/experiments/all8-front120-review-cases/reference-channel-audit-th2-quietnone.json --vad-comparison .casrt/experiments/all8-front120-review-cases/all8-energy-sweep-vad-quality-gated.json --eval-comparison .casrt/experiments/all8-front120-local-model-comparison-dominant-reasons.json --alignment-comparison .casrt/experiments/all8-front120-reference-copy-qwen-aligner-oracle-comparison.json --candidate-channel-audit .casrt/experiments/all8-front120-channel-quiet-none-low-threshold-sweep/th2_quietnone/candidate-channel-energy-audit.json --fail-unless-asr-only-ready --json -o .casrt/experiments/all8-front120-pipeline-readiness-current-best.json`. Result: expected failure with `asr_only_blocking_stages=[reference]`, `quality_blocking_stages=[reference,text_asr]`, `vad_chunking=pass`, `alignment=pass`, `channel_attribution=pass`; reference reasons remain `review flags=15`, `reference_type pseudo-gold`, channel mismatch `30`, uncertain `18`; text reasons remain practical CER `59.4%`, segments needing edit `100%`, candidate review ratio `100%`. 판단: 현재 자동화된 non-text stages는 product gate 기준으로 정리됐고, 다음 ASR 모델 품질 비교의 권위 있는 기준을 만들려면 WebUI human review로 reference/channel labels를 확정해야 한다.
     - 2026-07-02 all8 status after ENERGY apply UI shortcut: command `uv run casrt review-case-status .casrt/experiments/all8-front120-review-cases/case-index.json --include-reference-audits --reference-channel-threshold-db 2 --reference-channel-quiet-max-dbfs none --json -o .casrt/experiments/all8-front120-review-cases/status-current-after-energy-apply-ui.json`. Result: `reference_review_count=15`, `reference_review_duration_ms=163066`, `reference_audit.item_count=15`, `reference_channel_audit.item_count=48`, reason counts `{reference-channel-energy-mismatch:30, reference-channel-energy-uncertain:18}`, `next_review_case_id=01-front120-existing-srt`. 판단: UI shortcut only reduces human correction friction; it does not auto-change the real pseudo-gold set or clear review blockers without explicit human edits.
     - 2026-08-30 all8 human-review-aware channel audit: command `uv run casrt audit-review-case-channels .casrt/experiments/all8-front120-review-cases/case-index.json --threshold-db 2 --quiet-channel-max-dbfs none --json -o .casrt/experiments/all8-front120-review-cases/reference-channel-audit-human-review-aware.json --review-effort-output .casrt/experiments/all8-front120-review-cases/reference-channel-audit-human-review-aware-review-effort.json`. Result: raw `match=34`, `mismatch=30`, `uncertain=18`, `channel_reviewed_count=0`, `reviewed_exception_count=0`; unresolved counts are therefore still `mismatch=30`, `uncertain=18`, total `48`, and review-effort item count remains `48`. 판단: schema/queue migration does not auto-clear any real-data blocker. Only explicit WebUI human action can reduce unresolved counts.
     - 2026-08-30 all8 human-review-aware readiness: output `.casrt/experiments/all8-front120-pipeline-readiness-human-review-aware.json`, using the new channel audit plus the existing product-structure reference audit, gated VAD comparison, local model comparison, reference-copy alignment comparison, and candidate energy audit. Result: `asr_only_blocking_stages=[reference]`, `quality_blocking_stages=[reference,text_asr]`; VAD/alignment/channel remain pass. Reference reasons are review flags `15`, pseudo-gold authority, unresolved mismatch `30`, unresolved uncertain `18`. Text remains practical CER `59.4%`, segments needing edit `100%`, candidate review `100%`. 판단: non-text automation has not become final truth; it is conditionally pass pending human reference review. The current next stage remains reference, not model-only tuning.
     - 판단: 모델 승격용 gold가 아니라, 사람이 WebUI에서 audio/reference를 열어 검수하고 `freeze-case-references --reference-type human-reviewed`로 올릴 시작점이다.
   - `/data/uploads`, `/data/outputs`에서 30초~2분 단위 reference case를 늘릴 때는 `custom-asmr-case-slice-plan-v1` plan으로 재현 가능하게 기록한다.
   - 사람이 검수한 단일 파일은 `casrt freeze-reference`, prepared case set은 `casrt freeze-case-references`로 stable id와 `needs_review=false`를 고정한다.
   - manifest에 `reference_type=human-reviewed`와 검수 메모를 기록한다.
   - CER, timing error, channel accuracy, human edit count를 manifest report로 기록한다.
   - `review-effort` export는 `priority_score` 내림차순 큐로 운영한다. missing/extra/text/timing/channel 실패를 큰 것부터 들어 human-reviewed gold 제작 시간을 줄인다.

2. 일본어 평가 정규화 확장
   - strict/practical CER는 분리됐다.
   - 다음 단계에서는 장음/감탄/소형 kana 차이를 별도 옵션으로 추가할지 평가한다.

3. VAD 후보 추가
   - VAD command hook은 추가됐다.
   - `casrt vad whisper-asmr-onnx` command는 추가됐다.
   - `casrt vad coverage` command는 추가됐다.
   - `casrt vad coverage-cases` command는 추가됐다.
   - 현재 energy splitter 500/200은 fallback-free baseline이다.
   - 2026-07-01 durable all8 01-front120 energy coverage smoke: output `.casrt/experiments/all8-front120-review-cases/01-energy-vad-coverage.json`, `audio_duration_ms=120000`, `reference_segment_count=10`, `reference_interval_count=1`, `detected_interval_count=18`, `reference_speech_duration_ms=119969`, `detected_speech_duration_ms=109400`, `overlap_duration_ms=109400`, `reference_recall=91.2%`, `detected_precision=100.0%`. 판단: 이 pseudo-reference는 거의 전체 120초가 speech union이라 energy baseline은 약 10.6초를 missed reference로 남기며, ONNX VAD/energy tuning은 이 command로 먼저 coverage를 비교한 뒤 실제 ASR 평가로 이어간다.
   - 2026-07-02 durable all8 front120 energy coverage suite: command `uv run casrt vad coverage-cases .casrt/experiments/all8-front120-review-cases/case-index.json --json -o .casrt/experiments/all8-front120-review-cases/all8-energy-vad-coverage.json`. Result: `format=custom-asmr-vad-coverage-suite-v1`, `case_count=8`, `audio_duration_ms=960000`, `reference_segment_count=82`, `reference_interval_count=11`, `detected_interval_count=164`, `detected_max_interval_ms=32800`, `detected_mean_interval_ms=5272`, `reference_speech_duration_ms=956301`, `detected_speech_duration_ms=864600`, `overlap_duration_ms=862674`, `missed_reference_duration_ms=93627`, `extra_detected_duration_ms=1926`, `reference_recall=90.2%`, `detected_precision=99.8%`. Top missed cases by duration: 07 `25969ms` recall `78.1%`, 08 `15569ms` recall `87.0%`, 01 `10569ms` recall `91.2%`, 03 `10169ms` recall `91.5%`, 06 `9369ms` recall `92.1%`. 판단: 내장 energy VAD는 ASMR pseudo-reference 대비 과검출은 거의 없고 chunk 평균은 약 5.3초지만 약 93.6초의 reference speech를 놓친다. 따라서 파이프라인은 아직 오디오->텍스트 모델만 튜닝하면 되는 단계가 아니며, VAD/chunk 후보를 batch coverage와 missed interval diagnostics로 먼저 비교한다.
   - 2026-07-02 all8 coverage comparison smoke: full-audio/no-VAD baseline output `.casrt/experiments/all8-front120-review-cases/all8-full-audio-vad-coverage.json`, comparison `.casrt/experiments/all8-front120-review-cases/all8-vad-coverage-comparison.json`. Full-audio baseline result: `detected_interval_count=8`, `detected_max_interval_ms=120000`, `detected_mean_interval_ms=120000`, `missed_reference_duration_ms=0`, `extra_detected_duration_ms=3699`, `reference_recall=100.0%`, `detected_precision=99.6%`. Energy result in the same comparison: `detected_interval_count=164`, `detected_max_interval_ms=32800`, `detected_mean_interval_ms=5272`, `missed_reference_duration_ms=93627`, `extra_detected_duration_ms=1926`, `reference_recall=90.2%`, `detected_precision=99.8%`. 판단: pseudo-reference가 거의 전체 speech union이라 coverage-only ranking은 full-audio baseline을 1위로 둔다. Detected max/mean interval은 full-audio가 120초 chunk임을 드러내므로, 실제 기본 승격은 chunk length, ASR text, timing, channel attribution 평가를 이어서 통과해야 한다.
   - 2026-07-02 all8 coverage comparison gated smoke: command `uv run casrt vad compare-coverage ... --max-detected-interval-ms 60000 --fail-on-gate --json -o .casrt/experiments/all8-front120-review-cases/all8-vad-coverage-comparison-gated.json`. Result: expected failure after writing output because full-audio baseline `gate_passed=false`, failure `detected max interval 120000ms > 60000ms`; energy baseline `gate_passed=true`, `detected_max_interval_ms=32800`. 판단: chunk duration gate는 coverage-only full-audio baseline을 기계적으로 걸러내며, `--fail-on-gate`로 batch 실험을 다음 ASR 평가 전에 멈출 수 있다.
   - 2026-07-02 all8 energy parameter sweep: output comparison `.casrt/experiments/all8-front120-review-cases/all8-energy-sweep-vad-coverage-comparison.json`, gate `--max-detected-interval-ms 30000`. `casrt vad coverage-cases --energy-threshold-dbfs -54 --energy-pad-ms 800 --energy-max-chunk-ms 30000` 후보가 gated candidates 중 coverage 최상위다: `detected_interval_count=143`, `detected_max_interval_ms=30000`, `detected_mean_interval_ms=6674`, `missed_reference_duration_ms=4697`, `extra_detected_duration_ms=2796`, `reference_recall=99.5%`, `detected_precision=99.7%`, `gate_passed=true`, source settings recorded in `.casrt/experiments/all8-front120-review-cases/all8-energy-t54-pad800-max30s-vad-coverage.json`. Current energy baseline은 같은 30s gate에서 `detected_max_interval_ms=32800`로 실패하고 `missed_reference_duration_ms=93627`, `reference_recall=90.2%`다. Follow-up Qwen ASR eval `.casrt/experiments/all8-front120-qwen-official-energy-t54-pad800-max30s-eval-report.json` showed practical CER 60.2% and time-aligned 500ms 15.2%, worse than baseline Qwen practical CER 59.7% and time-aligned 16.0%. 판단: coverage 기준 개선은 실제 Qwen text/timing/channel 품질로 이어지지 않았고, t54/pad800/max30s도 기본값으로 승격하지 않는다.
   - `TransWithAI/Whisper-Vad-EncDec-ASMR-onnx`는 단독 chunker로 비교했고 기본 교체하지 않는다.
   - `--energy-rescue-min-ms 500` hybrid도 비교했고 기본 교체하지 않는다.
   - Silero VAD, TEN VAD wrapper는 ASMR ONNX VAD보다 후순위로 둔다.
   - VAD도 WebUI 옵션으로 노출하지 않고 고정/내부 설정으로 둔다.

4. Channel attribution 튜닝
   - 현재 8dB threshold + quiet-side -40dBFS gate는 보수적 baseline이다.
   - `casrt attribute-channels --diagnostics-output`은 segment별 L/R dBFS, 판정 이유, reason/channel count summary를 JSON으로 저장해 사람이 channel threshold 실패 패턴을 확인할 수 있게 한다.
   - `casrt sweep-channel-attribution`은 eval manifest와 audio map으로 threshold/quiet-side setting을 반복 적용하고 setting별 candidate, eval report, comparison을 만든다. 이미 L/R/MIX가 붙은 workflow 후보는 `--reset-speech-channels-to-mix`로 sweep copy 안에서만 speech channel을 MIX로 되돌려 threshold를 재평가한다. `--quiet-channel-max-dbfs none`은 quieter-side gate를 끄는 CLI-only 실험 후보이며 WebUI 옵션으로 노출하지 않는다. Setting item의 `reason_counts`와 attributed channel counts를 함께 보고, 기본값 변경은 sweep output과 human-reviewed gold gate를 보고 별도 결정한다.
   - gold set 기준으로 threshold와 MIX 유지 비율을 조정한다.
   - 필요하면 segment별 channel confidence를 debug metadata로만 저장한다.

5. Forced alignment 재평가
   - VAD chunk와 candidate segment/text 구조가 안정된 뒤 Qwen3-ForcedAligner를 timing 보정에 다시 사용한다.
   - `CASRT_QWEN_ALIGNER_CONTEXT_MS`는 기존 segment 밖 boundary를 탐색하는 내부 실험값이다. Qwen official all8 context500/2000은 timing을 악화시켰으므로 기본값은 0ms로 유지한다.
   - word/char alignment는 번역용 JSON에 넣지 않는다.

6. 모델 비교
   - `Qwen/Qwen3-ASR-1.7B`는 현재 로컬 주력 비교 후보지만 all8 batch CLI 기준 practical CER 59.7%, review effort 100%로 기본 승격하지 않는다.
   - `zhifeixie/Mega-ASR`는 검증 완료 후보지만 기본 승격하지 않는다.
   - `neosophie/Qwen3-ASR-1.7B-JA`는 all8 기준 practical CER 59.4%, review effort 100%라 검증 완료 후보지만 기본 승격하지 않는다.
   - `Qwen/Qwen3-ASR-1.7B-hf`는 Transformers main에서 검증했지만 기본 승격하지 않는다. 공식 release에 `qwen3_asr`가 들어오면 runtime 안정성만 재확인하고, 품질 재평가는 human-reviewed gold가 늘어난 뒤에 한다.
   - `microsoft/VibeVoice-ASR`와 `microsoft/VibeVoice-ASR-HF`는 일본어 tag가 있는 최신 로컬 후보지만 현재 repo env의 Transformers 5.12.1에서 전용 class가 없어 보류한다. 공식 release 지원 또는 별도 runtime 검토 후 exact revision local snapshot으로만 평가한다.
   - `mistralai/Voxtral-Mini-4B-Realtime-2602`는 remote code 없이 검증했지만 07 whisper 구간에서 실패해 기본 승격하지 않는다.
   - `ibm-granite/granite-speech-4.1-2b`는 native Transformers/safetensors/ja 후보로 `local-granite-asr` adapter, exact snapshot download, `casrt model digest`, 10초 smoke, 01/04/07 front120 pseudo-gold 평가를 완료했다. Non-Japanese hallucination filter 후 practical CER 23.6%, Qwen3-ForcedAligner 적용 후 time-aligned 500ms 32.7%지만 text/review gate 실패로 기본 승격하지 않는다. Granite Plus timestamp prompt도 practical CER 84.1%로 악화되어 기본 승격하지 않는다.
   - `Atotti/llm-jp-4-8b-speech-asr`는 ASR 특화 일본어 후보지만 third-party runtime package가 필요하므로 사용자 명시 승인 후 비교한다.
   - `AutoArk-AI/ARK-ASR-3B`, `CohereLabs/cohere-transcribe-03-2026`, `OpenMOSS-Team/MOSS-Transcribe-preview-2B`는 성능 후보로 남기되, custom code/gated/runtime 접근 조건을 먼저 해결해야 한다.
   - `Qwen/Qwen3-ASR-0.6B`는 속도/저사양 후보로 비교한다.
   - Gemma 4 E4B는 공식 오디오 입력과 smoke 전사는 성공했지만 01/04/07 front120 gold 기준을 만족하지 못해 기본 승격하지 않는다.
   - Whisper 계열 도메인 fine-tune은 제품 기본이 아니라 비교 baseline으로만 본다.

## 2026 공개 조사 출처

- Mega-ASR model card: https://huggingface.co/zhifeixie/Mega-ASR
- Mega-ASR runtime repository: https://github.com/xzf-thu/Mega-ASR
- WhisperJAV README: https://github.com/meizhong986/WhisperJAV
- WhisperJAV vocal separation issue: https://github.com/meizhong986/WhisperJAV/issues/224
- ASMR-trained Whisper VAD discussion: https://github.com/CrispStrobe/CrispASR/issues/36
- Qwen3-ASR user report thread: https://www.reddit.com/r/LocalLLaMA/comments/1rq118c/qwen3_asr_seems_to_outperform_whisper_in_almost/
- VibeVoice-ASR model card: https://huggingface.co/microsoft/VibeVoice-ASR
- VibeVoice-ASR-HF model card: https://huggingface.co/microsoft/VibeVoice-ASR-HF
- MOSS Transcribe preview model card: https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-preview-2B
- MOSS Transcribe GGUF model card: https://huggingface.co/cstr/MOSS-Transcribe-preview-2B-GGUF
- MiMo V2.5 ASR model card: https://huggingface.co/XiaomiMiMo/MiMo-V2.5-ASR
- Granite Speech 4.1 2B model card: https://huggingface.co/ibm-granite/granite-speech-4.1-2b
- Granite Speech 4.1 2B Plus model card: https://huggingface.co/ibm-granite/granite-speech-4.1-2b-plus
- Cohere ASR Japanese fine-tune model card: https://huggingface.co/efwkjn/cohere-asr-ja
- ARK-ASR-3B model card: https://huggingface.co/AutoArk-AI/ARK-ASR-3B

## 문서화 규칙

파이프라인 변경, 모델 선택, threshold, 평가 결과, 다음 작업 계획은 채팅에만 남기지 않는다. 변경이 확정되면 이 문서 또는 관련 제품 문서에 반영한다.
