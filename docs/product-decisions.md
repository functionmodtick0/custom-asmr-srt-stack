# 제품 결정 기록

작성일: 2026-06-27

## 목표

일본 동인 음성을 편집 가능한 텍스트/자막 데이터로 만드는 로컬 WebUI 도구를 만든다.

제품의 핵심 약속은 다음이다.

```text
오디오 드롭 -> 자동 JSON 생성 -> JSON/SRT 내보내기
```

사람 검토는 필수 실행 단계가 아니라 optional diagnostics/benchmark workflow다. 정상 제품 경로는 사람이 후보를 선택하거나 승인하지 않아도 끝나야 한다.

## 비목표

- 번역하지 않는다.
- 범용 자막 편집기를 만들지 않는다.
- 연구용 모델 튜닝 옵션을 기본 화면에 노출하지 않는다.
- WhisperX를 핵심 파이프라인으로 쓰지 않는다.
- 외부 프론티어 API를 고품질 기본 경로로 사용하지 않는다.
- 사용자 데이터나 별도 corpus로 모델을 fine-tuning, LoRA, adapter training하지 않는다.
- 같은 목적에 사용할 수 있는 최신 로컬 ASR/aligner 모델이 있다면 구형/소형 ASR 모델을 추천하지 않는다.

번역은 명시적으로 범위 밖이다. 번역은 외부 번역 도구가 담당한다.

학습은 명시적으로 범위 밖이다. 다른 주체가 공개한 pretrained/fine-tuned checkpoint를 exact local snapshot으로 평가하고 사용하는 것은 허용하지만, 이 프로젝트는 학습 dataset 구축, training loop, checkpoint 생성이나 사용자별 adapter 최적화를 제공하지 않는다. 공개 모델과 inference-time pipeline이 product gate를 만족하지 못하면 학습을 다음 단계로 제안하지 않고 관측된 품질 ceiling과 사람 검수 비용을 그대로 보고한다.

Runtime human adjudication도 범위 밖이다. 정상 제품 실행은 사용자가 선택한 메인 모델 하나와 고정 pipeline만 사용해 숨은 추가 ASR/verifier 비용 없이 완료한다. Multi-model consensus/MBR와 audio-conditioned verifier는 모델·pipeline을 고르는 개발/benchmark 도구일 뿐 production WebUI 기본 경로나 자동 fallback에 넣지 않는다. Transcript-only fluency judge를 음성 정답 판정자로 간주하지 않으며, 단일 모델 결과가 불확실해도 사용자 승인을 요구하지 않고 결과와 machine uncertainty를 저장한다. `content_reviewed`/`channel_reviewed`와 Review UI는 평가·감사 도구로 유지하되 정상 전사 완료 조건이 아니다.

사람을 runtime에서 배제하는 것과 absolute quality 검증은 별개다. 독립 human-reviewed gold가 전혀 없으면 model agreement, proxy reference, energy consistency만 보고할 수 있으며 실제 CER나 정답률을 증명했다고 표현하지 않는다.

## 기본 산출물

JSON을 기본 제품 포맷으로 사용한다.

SRT는 호환성, 가져오기, 내보내기를 위해 지원하지만 내부 기준 데이터는 아니다.

```text
master.json          내부 기준 데이터
translation.json     외부 번역 도구용 clean JSON
translated.json      외부 번역 도구에서 돌아온 번역 결과
export.srt           생성된 자막 출력
```

`master.json` segment의 사람 검수 증거는 자동 경고와 분리한다. `needs_review`는 자동/구조 검수 경고이고, `content_reviewed`는 사람이 audio를 들으며 text, start/end, kind를 확인했다는 명시적 증거이며, `channel_reviewed`는 L/R/MIX 판정 증거다. 이전 JSON처럼 `content_reviewed`가 없으면 `false`로 읽어 검수 완료를 추정하지 않는다. Text 또는 start/end가 바뀌거나 slice 경계에서 segment가 잘리면 `content_reviewed=false`가 되고, channel 변경은 `channel_reviewed=false`만 만든다. `needs_review=true`로 다시 표시하면 내용 검수 증거도 무효화한다.

WebUI는 이 계약을 위한 새 옵션을 추가하지 않는다. 기존 `검수 완료` 동작이 내용 검수 대상에서 `needs_review=false`, `content_reviewed=true`를 함께 저장하고 다음 `needs_review=true 또는 content_reviewed!=true` segment로 이동한다. Channel-only queue의 `Channel 검수 완료`는 `channel_reviewed=true`만 저장하며 내용 검수를 대신하지 않는다.

## 기본 워크플로우

1. 사용자가 WebUI에 오디오 파일을 드롭한다.
2. 시스템이 오디오를 좌측, 우측, 믹스 트랙으로 분리한다.
3. 사용자가 지정한 메인 모델 하나가 stereo 입력은 L/R을 각각, mono 입력은 MIX를 20초 비중첩 청크로 전사한다.
4. 모델이 반환한 청크 상대 타이밍을 원본 타임라인으로 offset하고 시간순 정렬한다. Production alignment는 no-op이다.
5. 시스템이 `master.json`을 저장한다.
6. 사용자는 필요한 경우 segment 텍스트를 검토하고 수정한다.
7. 사용자는 외부 번역 도구에 넣을 `translation.json`을 내보낸다.
8. 사용자는 외부 번역 도구가 만든 `translated.json`을 가져온다.
9. 시스템은 원본 타이밍과 번역 텍스트를 합쳐 SRT/JSON을 내보낸다.

## 모델 정책

고품질 제품 경로는 로컬 처리만 사용한다.

외부 프론티어 API는 제품 방향이 아니다. OpenAI-compatible / Gemini 어댑터는 기존 호환성과 로컬 HTTP 서버 연동을 위해 남길 수 있지만, 제품 기본값과 품질 목표는 로컬 모델 파일 또는 로컬 런타임이다.

메인 전사 모델은 사용자가 직접 지정한다.

모델 품질 후보 탐색은 공식 benchmark/leaderboard 순위를 기준으로 하지 않는다. 일본 동인 음성, whisper/ASMR, low-volume speech처럼 제품과 가까운 실제 파일을 사용했다는 최근 사용자 커뮤니티 후기를 우선하고, exact model/quantization/runtime/실패 사례나 출력 비교가 있는 firsthand report를 단순 추천보다 높게 본다. 서로 독립적인 사용자 보고가 반복되는지도 확인한다. 공식 model card와 benchmark는 architecture, 지원 modality/language, license, revision, runtime 요구사항 확인에만 사용하며 품질 승격 근거로 삼지 않는다.

최종 개발 단계 선택은 사용자의 실제 동인 음성에서 단일 모델 후보를 실행한 뒤 Codex가 출력 일본어의 붕괴, 반복, 누락 징후, L/R 중복, timestamp/segment 구조와 community에서 보고된 실패 양상을 제한적으로 점검한다. Codex가 audio를 직접 듣지 못하는 현재 세션에서는 이 검증을 실제 발화 정답 판정이나 CER로 표현하지 않는다.

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
- Local Qwen HF ASR worker

vLLM, SGLang 등으로 서빙되는 로컬 모델은 가능하면 OpenAI-compatible 어댑터로 연결할 수 있다. 다만 고품질 일본 ASMR 경로는 HTTP API가 필수가 아니며, 애플리케이션이 직접 로컬 worker subprocess를 띄울 수 있어야 한다.

전사 endpoint는 오디오 입력을 실제로 받아야 한다. vision-only 또는 text-only multimodal endpoint는 모델 이름이 최신이어도 ASR 경로로 사용할 수 없다.

로컬 Transformers 모델은 HTTP 서버를 필수로 요구하지 않는다. 애플리케이션은 `local-transformers` adapter를 통해 별도 Python subprocess worker를 띄우고 JSON Lines로 통신한다.

```text
WebUI/CLI -> local-transformers adapter -> transformers worker subprocess -> Gemma/Qwen/etc.
```

worker는 첫 전사 요청 때 lazy start하며, 같은 애플리케이션 프로세스 안에서 모델을 메모리에 유지한다. worker가 죽거나 응답 계약을 어기면 fallback으로 조용히 넘기지 않고 오류를 표시한다. `local-transformers` worker는 `CASRT_LOCAL_WORKER_ENV_MODE=offline`에서만 모델을 로드하고, `model_id`가 실제 local snapshot directory가 아니면 실패한다. 모델과 processor는 `local_files_only=True`, `trust_remote_code=False`, `use_safetensors=True`로만 로드한다. Remote code가 필요한 모델/runtime은 실행 전 별도 외부 runtime 검토 대상으로 둔다.

모든 production adapter는 동일한 20초 비중첩 청크 계약을 사용한다. Adapter가 더 짧은 입력 상한을 요구하면 그 상한으로만 추가 분할한다. Worker가 세부 timestamp를 만들 수 없으면 청크 전체를 하나의 speech segment로 반환하고 `needs_review`를 표시한다. Production은 별도 aligner로 이를 조용히 수정하지 않는다.

모든 adapter는 stereo 입력에서 L/R을 각각 전사하고 두 결과를 모두 보존하며, mono 입력에서만 MIX를 전사한다. 모델 종류별 MIX-first 분기, `CASRT_LOCAL_ASR_CHANNEL_MODE`, energy 기반 channel attribution, 반대 채널 duplicate 자동 삭제는 production에 없다. 이들은 과거 실험과 명시적 개발 CLI에만 남는다. 2026-08-30 pseudo-gold 실험은 global CER와 channel-aware CER가 서로 반대 결론을 냈기 때문에 자동 재귀속/삭제 근거로 쓰지 않는다.

Stereo bleed duplicate 제거는 반대 채널에 시간/text가 충분히 덮이고 원 채널 energy가 명확히 작은 segment만 지우는 bounded 후보로도 평가한다. Coverage는 제거될 segment 자체를 denominator로 사용해 짧은 반대편 조각 때문에 긴 고유 text를 통째로 지우지 않는다. 2026-08-30 all8 sweep에서는 낮은 energy threshold가 global practical CER를 개선했지만 channel-aware CER를 악화했고, 높은 threshold도 raw stereo를 두 지표에서 동시에 이기지 못했다. 따라서 자동 bleed 삭제를 기본 pipeline, production CLI, WebUI 옵션으로 추가하지 않고 human-reviewed L/R reference에서 같은 후보를 재평가한다.

`google/gemma-4-E4B-it`의 full HF `model.safetensors`는 약 16GB라 16GB VRAM 환경에서 full precision 로딩을 기본값으로 쓰기 부적절하다. Gemma 4 E4B 로컬 실행은 `CASRT_TRANSFORMERS_QUANTIZATION=4bit` runtime quantization을 권장한다. VRAM 여유가 있으면 `8bit`도 품질 비교 대상으로 사용한다.

Gemma 4 audio path는 audio tower까지 pre-quantize된 bnb checkpoint에서 깨질 수 있다. 실제 smoke에서 `unsloth/gemma-4-E4B-it-unsloth-bnb-4bit`는 audio tower의 `bitsandbytes` `AssertionError`로 실패했다. 따라서 worker의 runtime 4-bit quantization은 `lm_head`와 `model.audio_tower`를 quantization에서 제외한다.

로컬 Transformers worker는 chunk 전사마다 generation을 기본 256 new tokens로 제한한다. 30초 이하 ASMR chunk에서 1024 tokens는 반복 생성과 장시간 block을 만들 수 있으므로 정상 경로가 아니다.

## 일본 ASMR 고품질 로컬 파이프라인

일본 ASMR/동인음성의 기본 파이프라인은 다음 순서를 목표로 한다.

```text
원본 오디오
-> WAV 정규화
-> L/R/MIX 생성
-> 고정 20초 비중첩 chunking
-> stereo L/R 각각 단일 모델 ASR, mono MIX ASR
-> 결과 보존 및 시간순 정렬
-> no-op alignment
-> master.json
```

고정 production 경로는 음성 누락과 불확실한 channel 재귀속을 피하기 위해 L/R 결과를 삭제하지 않는다. Energy VAD, 외부 VAD, channel attribution, forced alignment는 후보를 비교하는 개발 도구이며 WebUI/server/project 전사의 숨은 단계가 아니다.

우선 구현 대상은 다음이다.

- `Qwen/Qwen3-ASR-1.7B`: 주력 로컬 ASR 후보
- `neosophie/Qwen3-ASR-1.7B-JA`: Qwen3-ASR-1.7B 기반 일본어 fine-tune 후보. 2026-06-28 01/04/07 front120 확장 gold에서 practical CER 29.6%, time-aligned 500ms ratio 29.5%, channel time-aligned accuracy 73.1%라 기본 승격하지 않는다. 2026-07-02 all8 front120 pseudo-gold에서는 persistent local snapshot `.casrt/models/neosophie-qwen3-asr-1.7b-ja-987bda160f2dabfa6757550bcff7cdda2ba0648c`로 practical CER 59.4%, Japanese relaxed CER 58.4%, time-aligned 500ms ratio 16.0%, channel time-aligned accuracy 53.3%, candidate MIX ratio 63.0%, review effort 100%였다. Qwen official all8보다 text만 아주 조금 낫지만 모든 promotion gate를 실패하므로 기본 승격하지 않는다.
- `Qwen/Qwen3-ASR-1.7B-hf`: native Transformers 후보. 현재 설치된 Transformers 5.12.1에서는 `qwen3_asr` 아키텍처가 remote code 없이 로딩되지 않는다. 공식 Transformers main `5.13.0.dev0` commit `45b004d7bb505a258542d1965b0f9e0d8b03b89d`에서는 실행됐지만, 2026-06-30 01/04/07 front120 pseudo-gold에서 practical CER 29.4%, time-aligned 500ms ratio 27.3%, channel time-aligned accuracy 68.2%, review effort 75/75라 기본 승격하지 않는다.
- `Junlaii/Bro-ASR-1.7B`: Qwen3-ASR-1.7B-hf 기반 noisy Japanese full-merged safetensors 후보. Exact revision `6e8f651ad6137457a2a6e813b810349a7e272f4b`와 official Transformers exact commit runtime만 사용한다. 2026-08-30 external runtime 정적 보안 검토는 `PASS_WITH_CONSTRAINTS`이며, persistent snapshot `.casrt/models/bro-asr-1.7b-6e8f651ad6137457a2a6e813b810349a7e272f4b`은 preflight를 통과했다. Snapshot digest는 `12d96810a8d5146f7da52d50c8c9576ea76d84cc56d65aca95e95e5c377fd565`다. 동일 HF runtime/beam 5 all8 pseudo-gold 비교에서 practical CER `58.84%`로 official base `59.87%`보다 `1.03` percentage points 낮고 8개 중 7개 case를 개선해 현재 Qwen HF text 후보로 유지한다. 그러나 timing `16.05%`, channel accuracy `53.33%`, review effort `100%`로 product gate를 실패하므로 기본 승격하지 않는다. Pinned chat template hash, offline/local-only/network block, persistent digest/runtime provenance를 유지한다.
- `mistralai/Voxtral-Mini-4B-Realtime-2602`: remote model code 없이 native Transformers에서 실행 가능한 최신 로컬 후보. 2026-06-28 01/04/07 front120 확장 gold에서 practical CER 40.0%, time-aligned 500ms ratio 28.7%, channel time-aligned accuracy 63.6%라 기본 승격하지 않는다.
- `mistralai/Voxtral Mini Transcribe 2.0`: Mistral API batch transcription 제품으로 확인됐다. open-weight 로컬 checkpoint가 확인되지 않았고 외부 API는 제품 방향이 아니므로 기본 경로에서 제외한다.
- `Atotti/llm-jp-4-8b-speech-asr`: 일본어 ASR 특화 8B 후보. 현재 official Transformers에서 `LlamaForSpeechLM`을 제공하지 않고 model card의 third-party runtime package가 필요하므로 사용자 명시 승인 전에는 자동 검증하지 않는다.
- `AutoArk-AI/ARK-ASR-3B`: 최신 로컬 후보. model metadata상 custom code가 필요하므로 사용자 명시 승인 또는 first-party package 경로가 확인된 뒤 실제 benchmark 대상으로 삼는다.
- `CohereLabs/cohere-transcribe-03-2026`: 2026년 2B ASR 후보. 공식 card는 일본어 포함 14개 언어, Transformers native, safetensors, no timestamps/diarization을 명시한다. Root Transformers 5.12.1에 `CohereAsrForConditionalGeneration`와 `CohereAsrProcessor`가 있어 remote model code 없이 구현 가능하다. 다만 gated/custom_code repo라 실제 download/evaluation은 exact revision pin과 file digest 기록 후, local snapshot path + `trust_remote_code=False` + `local_files_only=True` 조건에서만 수행한다.
- `ibm-granite/granite-speech-4.1-2b`: 2026-04-29 공개된 multilingual ASR/AST 후보. HF metadata 기준 `transformers`, `safetensors`, `ja`, Apache-2.0이고 root Transformers 5.12.1에 `granite_speech` native class가 있어 remote model code 없이 실행 가능하다. Exact revision은 `de575db64086f84fdc79da4932d1076e965bc546`다. `local-granite-asr` adapter로 추가했고 local snapshot은 gitignored `.casrt/models/granite-speech-4.1-2b-de575db64086f84fdc79da4932d1076e965bc546`에 둔다. Snapshot SHA-256은 `67c7d69184b53bae7a2bec077fbc88d8695a72f043fd70831f4e4830dc4752ca`다. 2026-07-01 01/04/07 front120 pseudo-gold 평가에서는 non-Japanese hallucination filter 후 practical CER 23.6%, time-aligned 500ms 21.8%, review effort 100%라 기본 승격하지 않는다. Qwen3-ForcedAligner를 붙인 재정렬은 time-aligned 500ms를 32.7%로 올렸지만 practical CER 23.6%, candidate MIX 54.5%, review effort 100%라 여전히 product gate를 실패한다. Granite Plus timestamp prompt도 같은 3-case에서 practical CER 84.1%, time-aligned 500ms 22.2%, review effort 98.6%라 더 나쁘다. Human-reviewed manifest에서는 최신 후보 비교 때 다시 평가할 수 있지만 현재 기본값은 아니다.
- stable-ts/Whisper계 산출물은 2026-06-28 01/04/07 front120 pseudo-gold에서 practical CER 16.1%, time-aligned 500ms ratio 56.7%였다. reference 자체가 stable-ts 유래이므로 실제 품질 근거로 승격하지 않고, 비교 baseline으로만 유지한다.
- `google/gemma-4-E4B-it`: 공식 오디오 입력을 지원하는 최신 로컬 multimodal 후보. 2026-06-28 smoke 전사는 성공했지만 01/04/07 front120 gold에서 4-bit practical CER 42.3%, 8-bit practical CER 46.1%라 기본 승격하지 않는다.
- `zhifeixie/Mega-ASR`: Qwen3-ASR-1.7B 기반 robust ASR 후보. 2026-06-28 01/04/07 front120 gold에서 routed practical CER 30.9%, base-only threshold 1.1 practical CER 30.8%, forced LoRA practical CER 77.6%라 기본 승격하지 않는다.
- `microsoft/VibeVoice-ASR`와 `microsoft/VibeVoice-ASR-HF`: 2026년 공개된 일본어 tag 포함 최신 local ASR 후보. Exact revisions는 각각 `d0c9efdb8d614685062c04425d91e01b6f37d944`, `f22241c2062b3b25272bf117397e03d73381037a`다. 현재 repo env의 Transformers 5.12.1에는 필요한 VibeVoice class가 없으므로 기본 경로에 넣지 않는다. 공식 release 지원 또는 별도 runtime 검토 후 exact revision local snapshot으로만 평가한다.
- `OpenMOSS-Team/MOSS-Transcribe-preview-2B`와 `cstr/MOSS-Transcribe-preview-2B-GGUF`: 2026-06 신규 ASR 후보지만 일본어 tag가 없거나 `custom_code`/별도 GGUF runtime이 필요하다. 일본 ASMR 우선 후보가 아니며 실행 전 외부 코드/runtime 검토가 필요하다.
- `TransWithAI/Whisper-Vad-EncDec-ASMR-onnx`: ASR 모델이 아니라 VAD 후보. 공개 discussion 기준 일본어 ASMR 약 500시간으로 학습된 Whisper encoder 기반 VAD다. `casrt vad whisper-asmr-onnx` command로 붙이며, WebUI 옵션으로 노출하지 않고 `CASRT_VAD_COMMAND` 뒤에 붙일 내부 후보로 둔다. 실행은 `gpt-5.4 xhigh` 정적 보안 검토의 `PASS_WITH_CONSTRAINTS` 조건을 따른다. 2026-06-28 01/04/07 front120 gold에서 단독 chunker default practical CER 30.2%, tuned practical CER 33.4%, energy-rescue hybrid practical CER 31.0%라 energy baseline 29.6%보다 나빠 기본 교체하지 않는다.
- Energy VAD t54/pad800/max30s: 2026-07-02 all8 coverage sweep에서 reference recall 99.5%로 좋아졌지만, actual Qwen ASR eval에서는 practical CER 60.2%, time-aligned 500ms 15.2%로 baseline Qwen보다 악화됐다. 2026-08-30 Bro beam5 stereo에서도 segment 수와 global CER/timing은 개선했지만 channel-aware CER와 candidate energy match ratio가 악화돼 raw stereo를 지배하지 못했다. 따라서 production에서는 제거하고 개발 CLI 재평가 후보로만 보존한다.
- `Qwen/Qwen3-ASR-0.6B`: 빠른 비교/저사양 후보
- `Qwen/Qwen3-ForcedAligner-0.6B`: 고정 forced alignment 후보. 기본은 기존 segment 내부만 재정렬하고, `CASRT_QWEN_ALIGNER_CONTEXT_MS`는 기존 segment 밖 boundary 보정 실험용 내부 env로만 둔다. 2026-07-02 Qwen official all8 context 500/2000ms 실험은 time-aligned 500ms를 baseline 16.0%보다 낮은 11.1%/6.9%로 악화시켜 기본 승격하지 않는다. 같은 날 all8 reference-copy oracle에서도 baseline reference-copy의 time-aligned 500ms 95.1%를 Qwen aligner 적용 후 51.2%로 낮추고 timing edit ratio를 9.8%에서 75.6%로 악화시켰으므로, 현재 Qwen3-ForcedAligner는 ASMR 기본 alignment 계층으로 쓰지 않는다. 현 alignment 기본 정책은 `CASRT_ALIGNER_COMMAND`를 설정하지 않는 no-op이며, 새 aligner 후보는 reference-copy no-op baseline 95.1%를 넘어야 승격 대상이 된다.

보안 검토는 외부 코드/런타임 실행에만 요구한다. 예를 들어 third-party repository code, `trust_remote_code`, 새 runtime package, unsafe model format, unreviewed downloaded tooling이 해당한다. 우리 저장소의 일반 wrapper, 테스트, 문서 변경은 일반 구현 리뷰와 behavior test로 검증한다. 외부 런타임을 benchmark할 때는 repo id를 직접 실행하지 않고 고정 snapshot directory를 model id로 사용하며, offline/local-only env를 켜서 cache miss/network fallback/custom remote code를 실패로 만든다.

Gemma 4 E4B 같은 general multimodal 모델은 실험 대상으로 유지하되, 제품의 일본 ASMR 품질 기준 모델 승격 여부는 동일한 gold set 평가 결과로만 결정한다. 현재 Gemma 4 E4B는 기준 미달이다.

평가 없이 모델을 기본값으로 승격하지 않는다. 2026-06-30 audit에서 현재 01/04/07 front120 reference가 stable-ts 기반 pseudo-gold임을 확인했다. pseudo-gold 결과는 regression/상대 비교로만 쓰고, 기본 모델 승격은 `reference_type=human-reviewed` manifest에서 다시 판단한다.

검수 완료 SRT/master는 `casrt freeze-reference`로 시간순 정렬, stable id 재부여, `needs_review=false` 저장을 거쳐 기준본 master JSON으로 고정한다. Prepared case set을 `human-reviewed`로 freeze하거나 eval manifest로 만들 때는 모든 reference segment의 `content_reviewed=true`가 필수이며, 증거가 하나라도 없으면 output을 만들기 전에 실패한다. `needs_review=false` 또는 검수 메모만으로 사람 검수를 추정하지 않는다.

모델 승격용 `eval-manifest` 실행에는 `--require-reference-type human-reviewed`를 함께 사용한다. pseudo-gold manifest는 regression/상대 비교 report를 남길 수 있지만 이 gate를 통과하지 못해야 정상이다.
반복 실험에서는 `--product-gate`를 사용해 human-reviewed reference gate와 문서화된 metric threshold를 한 번에 적용한다. 개별 threshold 인자를 같이 주면 그 값이 product 기본값보다 우선한다.

로컬 일본어 ASR worker는 정리 후 일본어 문자가 하나도 없는 segment를 hallucination으로 보고 버린다. 이 필터는 `!`, `?`, English-only fragment처럼 일본어 전사로 볼 수 없는 출력에 한정하며, 일본어 문자가 포함된 segment의 문장 품질 판단은 평가/검수 단계에 맡긴다.

최소 평가 기준은 다음이다.

- character error rate
- Japanese relaxed CER: practical CER에서 장음류 문자 `ー〜～`를 추가로 제거한 보조 지표. 모델 승격 gate에는 쓰지 않는다.
- Channel-aware practical CER: L/R/MIX별 practical edit distance를 micro-average한다. Overlap segment의 전역 interleave 순서 벌점은 제거하지만 잘못된 채널 배치는 deletion+insertion으로 벌점이 남는 text+channel 진단 지표다. Human-reviewed reference 전에는 기존 product gate나 기본 모델 결정을 대체하지 않는다.
- Channel-aware time-aligned timing: 같은 L/R/MIX channel 안에서만 최대 overlap pair를 골라 boundary error를 계산한다. Stereo alignment의 반대 채널 pairing 왜곡을 제거하는 진단 지표이며, 기존 timing/channel product gate는 유지한다.
- segment boundary error와 threshold ratio
- channel attribution accuracy
- L/R/MIX channel confusion과 candidate MIX 유지 비율
- candidate `needs_review` segment 비율. 모델 승격 gate에서는 unresolved candidate review flag가 0이어야 한다.
- `review_effort`: practical text mismatch, channel mismatch, 500ms 초과 timing mismatch, missing reference, extra candidate를 합친 사람이 실제로 고쳐야 하는 구간 수와 같은 denominator 기준 breakdown ratio
- `asr_artifacts`: candidate speech segment에서 일본어 문자가 없는 출력, 15 chars/sec 초과 고밀도 text, 12자 이상 반복 패턴을 보조 진단 지표로 센다. 이 값은 ASMR식 hallucination/repetition/chunking 실패를 분리해 보기 위한 metric이며 product gate에는 쓰지 않는다.
- `review_effort.items` export: 평가 report에서 `custom-asmr-review-effort-v1` 수정 큐 JSON을 생성해 사람이 볼 다음 검수/개선 후보를 보존한다. Export queue는 `priority_score` 내림차순으로 정렬해 missing/extra/text/timing/channel 실패를 큰 것부터 듣게 한다.
- `compare-evals`: 여러 eval report를 `custom-asmr-eval-comparison-v1` 비교표로 정렬해 다음 후보 선택을 돕는다. Item은 dominant review-effort reason/ranking을 포함해 text, timing, channel, missing/extra 중 다음 병목을 바로 보이게 한다. 이 명령은 모델/heuristic 자동 승격을 하지 않는다.
- `compare-review-effort`: 여러 eval report의 `review_effort.items`를 reference segment 기준으로 묶어 후보별 pass/fail과 text/channel/timing/missing/extra reason을 비교한다. Output은 `custom-asmr-review-effort-comparison-v1`이며, 후보 간 보완 가능성과 공통 실패 segment를 찾기 위한 CLI-only 진단 도구다. Transcript를 수정하거나 모델/heuristic을 자동 승격하지 않는다.
- `merge-review-effort`: 여러 `custom-asmr-review-effort-v1` queue를 하나의 priority queue로 합친다. 같은 case/reference/candidate/time range issue는 reason과 evidence를 병합하고, root `case_summaries`, `case_count`, `next_case_id`로 사람이 어느 case부터 검수할지 표시한다. Channel-energy reason만 있는 item은 짧은 `review_clip_*` focus를 유지할 수 있지만, content/structure reason이 함께 있으면 전체 segment를 들어야 하므로 focus와 focus-only energy fields를 제거한다. 같은 `source_case_index`는 보존하며 서로 다른 값은 실패한다.
- `pipeline-readiness`: reference audit, optional reference channel audit, VAD coverage comparison, eval comparison을 읽어 `custom-asmr-pipeline-readiness-v1` 상태 요약을 만든다. `asr_only_ready`는 reference, VAD/chunking, alignment, channel attribution이 모두 pass일 때만 true이며, `text_asr`는 별도 product quality stage로 둔다. Reference audit에 `content_unreviewed_count`가 있으면 0이 아닐 때 명시적 blocker로 표시하고, 없는 legacy report는 기존 지표로 읽는다. `--reference-channel-audit`은 사람 검수되지 않은 `unresolved_*` count만 blocker로 사용한다. Separate alignment/channel comparisons와 candidate energy audit의 기존 분리 정책, product thresholds, downstream VAD no-regression 정책은 유지한다. 이 명령은 report를 읽기만 하며 WebUI 옵션을 늘리지 않는다.
- Product-gated `pipeline-readiness`는 VAD coverage만으로 `vad_chunking`을 pass시키지 않는다. 같은 case/model/mode의 baseline/candidate eval을 `--vad-baseline-eval`/`--vad-candidate-eval`로 함께 받아 downstream no-regression을 요구한다. 관측 가능한 practical/channel-aware CER, timing, channel, MIX, review metric 중 하나라도 악화되면 VAD stage를 fail로 둔다. Coverage-only 일반 진단은 downstream 미검증 warning을 남긴다.
- `review-pack`: 수정 큐와 원본 audio를 결합해 `custom-asmr-review-pack-v1` index와 WAV clips를 만들고, priority queue 순서와 score/rank, root `item_count`/`reason_counts`/`case_summaries`/`case_count`/`next_case_id`를 보존해 human-reviewed gold 제작을 빠르게 한다. Prepared review case set에서 나온 후보 실패 pack은 CLI `--source-case-index` 또는 review-effort의 embedded `source_case_index`로 source `case-index.json`을 붙여 WebUI `case 열기`가 원 reference segment로 이동할 수 있게 한다. 같은 case index의 `items[].audio`를 audio source로도 사용하므로 prepared case 기반 queue는 별도 `audio-map.json` 없이 pack을 만들 수 있다. Item에 optional `review_clip_start_ms/review_clip_end_ms`가 있으면 사람이 들을 WAV만 그 focus 구간으로 줄이고, original `start_ms/end_ms`는 source segment ownership과 편집 이동 계약으로 보존한다. Focus range가 item bounds 밖이면 fallback하지 않고 실패한다. Root `duration_summary`는 원래 item duration, focus 적용 후 effective duration, 실제 WAV clip duration을 기록해 검수 부담 감소를 실험마다 비교 가능하게 한다.
- `attribute-channels`: 기존 SRT/master transcript와 stereo audio를 받아 `MIX` speech segment에만 L/R energy channel attribution을 적용한다. 기존 text/timing은 변경하지 않는다. Diagnostics는 reason counts와 original/attributed channel counts를 포함해 threshold 실패 패턴을 기록한다. CLI-only `--quiet-channel-max-dbfs none`은 quieter-side gate를 끄는 실험 후보이며, 기본값은 계속 8dB + quieter side -40dBFS다.
- `slice-case`: 긴 원본 audio와 SRT/master에서 matching WAV/master case를 만들고, 경계에 걸쳐 잘린 segment를 `needs_review=true`로 표시해 human-reviewed gold 제작을 돕는다.
- `review-case-status`: `case-index.json`에서 준비된 case set의 파일 존재 여부, stale segment/review/content-review count, 남은 reference `content_reviewed=false`, reference/candidate `needs_review` 수와 duration, candidate attach 진행률을 실제 파일 기준으로 다시 계산한다. Report에는 `next_content_unreviewed_case_id`, `next_review_case_id`, `cases_missing_candidate`, `cases_with_candidate_review`, item별 `first_review_segment`를 포함한다. `first_review_segment`는 자동 flag가 있거나 명시적 내용 검수가 없는 첫 segment다. `--include-reference-audits`는 structure/channel audit summary를 status report에 붙이고, 기존 fail 옵션은 각 계약을 유지한다.
- `audit-review-case-references`: prepared reference set의 explicit content-review evidence, overlap, exact-boundary overlap, long segment, near-full speech coverage, 남은 review flag를 text 없이 segment id/time/channel 중심으로 진단한다. `content_reviewed!=true`는 `reference-content-unreviewed` queue item과 readiness blocker가 된다. 기본 overlap/long threshold와 cross-channel 정책은 유지한다. `--review-effort-output`은 content authority와 구조 문제를 함께 내보내며, 같은 segment의 `needs_review`와 content reason은 한 item에 합친다. `--fail-on-audit`은 이 authority/structure queue가 남아 있을 때 실패한다. Reference를 수정하거나 자동으로 human-reviewed로 승격하지 않는다.
- `audit-review-case-channels`: prepared review case reference의 L/R label을 stereo energy와 비교해 `custom-asmr-reference-channel-audit-suite-v1`을 만든다. Transcript text는 저장하지 않고 segment id/time/channel, L/R dBFS, energy channel, match/mismatch/uncertain status만 저장한다. Raw mismatch/uncertain count는 진단용으로 보존하고, `channel_reviewed=true`인 사람 검수 예외를 제외한 `unresolved_*` count만 review queue와 승격 gate를 막는다. `--review-effort-output`은 unresolved mismatch/uncertain segment를 기존 `review-pack`에 넣을 수 있는 queue로 만든다. Long channel audit item은 최대 5초 evidence window를 `review_clip_*` fields로 싣지만, 이는 검수 부담을 줄이는 listening aid일 뿐 reference 수정이나 energy-channel 정답 승격이 아니다. 이 명령은 energy channel을 정답으로 승격하지 않고, human-reviewed 전 channel label 검수 우선순위를 정하는 CLI-only 진단 도구다.
- `audit-candidate-channels`: eval manifest candidate의 L/R/MIX label을 stereo energy와 비교해 `custom-asmr-candidate-channel-audit-suite-v1`을 만든다. Transcript text와 reference label은 저장하지 않고 segment id/time, candidate channel, energy channel, L/R dBFS, match/missed_attribution/wrong_side/mix_match/over_attribution status만 저장한다. 이 명령은 pseudo-gold reference channel label과 channel attribution heuristic 품질을 분리하기 위한 CLI-only proxy 진단 도구이며, energy channel을 human-reviewed 정답으로 승격하지 않는다.
- `review-case-pack`: 준비된 case set에서 `needs_review=true` 또는 `content_reviewed!=true`인 모든 reference segment를 잘라 기존 `custom-asmr-review-pack-v1` clip queue를 만든다. Reasons는 각각 `reference-needs-review`, `reference-content-unreviewed`로 보존하며 두 조건이면 둘 다 기록한다. 따라서 자동 flag가 없던 pseudo-gold segment도 human-reviewed 승격 전에 반드시 듣게 된다. 새 WebUI 옵션은 추가하지 않는다.
- `vad coverage`: built-in energy intervals, precomputed VAD interval JSON, 또는 고정 `--vad-command`를 reference speech union과 비교해 recall/precision/missed/extra duration, missed/extra interval 목록, detected max/mean interval을 계산한다. Built-in energy sweep 값은 CLI-only `--energy-*` 옵션으로 지정하고 `source_settings`에 기록한다. Coverage recall/precision은 union interval 기준이고, detected max/mean interval은 max-chunk split이 보이도록 union merge 전 detected chunk 기준이다. `vad coverage-cases`는 prepared review case set 전체를 같은 VAD source로 집계하고, `vad compare-coverage`는 여러 coverage report를 missed reference duration 우선으로 정렬한다. `compare-coverage --max-detected-interval-ms`, `--max-missed-reference-ms`, `--min-reference-recall`, `--min-detected-precision`은 chunk length, missed speech, recall, precision 후보를 gate failure로 표시하고, `--fail-on-gate`는 비교표를 출력/저장한 뒤 exit 1로 자동 실험을 멈춘다. VAD/chunking 후보 비교용 CLI-only 진단 도구이며, ASR text 품질 gate나 WebUI 옵션으로 쓰지 않는다. Coverage만으로 기본 VAD를 자동 승격하지 않으며, ASMR 파이프라인은 이 VAD/chunk/channel/alignment 검증을 통과하기 전까지 오디오->텍스트 모델만 남은 상태로 보지 않는다.
- `save-review-case-reference`: 편집한 단일 SRT/master를 준비된 case reference로 저장하고 `case-index.json`의 segment/review/content-review count를 갱신한다. 이 명령도 reference authority를 바꾸지 않는다.
- `transcribe-review-case-candidates`: prepared case audio를 기존 project workflow로 일괄 전사해 case id별 candidate master JSON을 만든다. VAD/chunking/channel attribution/optional aligner 동작은 project transcription과 같은 implementation path를 사용한다.
- `align-review-case-candidates`: prepared case set의 기존 candidate들을 `CASRT_ALIGNER_COMMAND`로 일괄 재정렬해 aligned candidate directory, diagnostics, attach plan, eval manifest를 만든다. 원본 case set과 원본 candidate는 수정하지 않는다. Diagnostics는 boundary count, max/mean absolute boundary delta, 250ms/500ms 이내 boundary 비율을 포함해 aligner 후보가 timing을 얼마나 크게 흔드는지 기록한다.
- `build-candidate-attach-plan`: candidate directory에서 case id와 같은 이름의 SRT/master JSON을 찾아 `custom-asmr-case-candidate-attach-plan-v1`을 생성한다. 모든 case가 정확히 하나의 candidate file과 매칭되어야 하며, 누락/모호한 중복은 output 없이 실패한다.
- `attach-review-case-candidates`: 이미 준비된 review case set에 case-local candidate SRT/master를 붙여 `candidates/*.master.json`과 `case-index.json` candidate fields를 만든다. Reference는 수정하지 않고, candidate 없는 human-reviewed set을 eval manifest로 넘기기 위한 CLI-only 단계다.
- `freeze-case-references`: 준비된 case set의 reference들을 batch로 stable id와 `needs_review=false` 상태로 고정하고 새 case set을 만든다. 사람이 실제 검수한 reference에만 `--reference-type human-reviewed`를 사용하며, 승격 전에는 `--fail-on-review`, `--fail-on-reference-audit`, `--fail-on-reference-channel-audit`로 남은 검수 flag, 구조 검수 queue, reference L/R energy mismatch/uncertain queue를 막는다.
- `build-eval-manifest`: candidate가 포함된 준비 case set에서 `custom-asmr-eval-manifest-v1`을 다시 만든다. 사람이 reference를 검수한 뒤 `--reference-type human-reviewed --fail-on-review --fail-on-reference-audit --fail-on-reference-channel-audit`로 모델 승격 평가 manifest를 만들 때 사용한다.

구현 세부 값, 실험 결과, 다음 작업 계획은 [local-asr-pipeline.md](local-asr-pipeline.md)에 기록한다.

2026년 1월 이후 공개 후기/툴링 조사는 scene detection, ASMR-friendly VAD, forced alignment, hallucination filtering의 잠재적 영향을 보여줬다. 하지만 이 프로젝트의 실제 pseudo-gold 반복에서는 어느 후보도 단순 20초 L/R 경로를 일관되게 지배하지 못했다. 따라서 이 기능들은 개발 실험으로 보존하고 production 경로에는 넣지 않는다.

## Alignment 정책

Production alignment는 no-op으로 고정하며 기본 UI에서 사용자가 선택하지 않는다.

현재 의도한 방향은 다음이다.

```text
메인 모델의 청크 상대 timing -> 원본 timeline offset -> segment timing
```

`CASRT_ALIGNER_COMMAND`와 Qwen3-ForcedAligner는 `align-transcript`, `align-review-case-candidates` 같은 명시적 개발 CLI에서만 실행한다. Server와 `project transcribe`는 이 환경변수를 읽지 않는다.

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
      "needs_review": false,
      "channel_reviewed": false
    }
  ]
}
```

segment 필수 동작은 다음이다.

- `id`는 export/import 전체에서 안정적으로 유지한다.
- `start_ms`와 `end_ms`는 segment 단위 timing이다.
- `text`에는 검토/내보내기 대상 원문 텍스트만 넣는다.
- 채널 라벨, 모델 메타데이터, 검토 플래그를 `text`에 섞지 않는다.
- `channel_reviewed=true`는 사람이 해당 segment의 L/R channel을 오디오로 확인했다는 명시적 판정이다. Stereo energy와 다른 기존 라벨을 유지하는 근거로만 사용하며 모델/heuristic이 자동으로 설정하지 않는다.
- Segment channel이나 start/end가 바뀌어 energy evidence가 달라지면 `channel_reviewed`를 다시 `false`로 만든다. Case slicing이 segment boundary를 자를 때도 기존 판정을 무효화한다.

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
선택 segment start/end/channel/review flag 수정
선택 segment 재전사
JSON 내보내기
translated JSON 가져오기
SRT 내보내기
```

Review pack/case loader와 검수 전용 control은 primary WebUI에 두지 않는다. 개발·평가 산출물 검수는 별도 `/diagnostics.html` 화면에서만 제공한다.

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
- segment start/end ms, channel, `needs_review`를 수정한다.
- 오디오를 먼저 연 뒤 SRT 또는 `master.json`을 열면, transcript가 아직 없는 현재 audio project에 붙여 검수 중 segment 재생과 저장을 유지한다.
- 선택 segment를 재전사한다.
- JSON을 export/import한다.
- SRT를 export한다.
- CLI가 만든 `review-pack` directory 또는 `index.json` 경로를 열어 priority queue와 clip을 확인한다.
- CLI가 만든 review case directory 또는 `case-index.json` 경로를 열어 case audio/reference master를 편집하고 reference file에 저장한다.

split/merge는 MVP 필수 요구사항으로 만들지 않는다. Start/end 직접 수정과 재전사로 처리하지 못하는 실제 번역/자막 검토 문제가 확인될 때만 추가한다.

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

- production chunk interval 최대 길이는 20초이며 비중첩이다. 이전 분석 metadata의 더 긴 chunk도 전사 시점에 다시 20초 이하로 나눈다.
- 오디오는 분석 전에 WAV로 정규화한다.
- SRT import는 cue text 선두의 `[L]`, `[R]`, `[LR]`, `[MIX]`, `[L:SPEAKER_00]`, `[R:SPEAKER_00]`, `[SPEAKER_00]`를 metadata로 읽고 본문 텍스트에서 제거한다.
- `[LR]`은 현재 channel model에서 `MIX`로 저장한다.
- `slice-case`는 audio와 transcript를 같은 구간으로 자르고 transcript timestamp를 0 기준으로 rebase한다. 경계에서 잘린 segment는 검수 필요 상태로 남긴다.
- `prepare-review-cases`는 여러 `slice-case` 작업을 plan 파일로 재현 가능하게 실행하고, `audio-map.json`, `case-index.json`, reference/candidate master, eval manifest 산출물을 만든다. 이 명령도 검수 완료를 판정하지 않는다.
- `review-case-status`는 준비된 `case-index.json`의 audio/reference/candidate 파일 존재 여부와 실제 segment/review count를 다시 읽어 `custom-asmr-review-case-status-v1` report를 만든다. Report에는 `next_review_case_id`, `cases_missing_candidate`, `cases_with_candidate_review`, item별 `first_review_segment`를 포함한다. `--include-reference-audits`는 structure/channel audit summary를 추가한다. 운영 gate로 쓸 때는 `--fail-on-issues`, `--fail-on-review`, `--fail-on-missing-candidates`, `--fail-on-candidate-review`, `--fail-on-reference-audit`, `--fail-on-reference-channel-audit`를 사용한다.
- `review-case-pack`은 `case-index.json`의 reference `needs_review=true` 또는 `content_reviewed!=true` segment를 기존 review pack 형식으로 잘라 WebUI loader에서 들을 수 있게 한다. 이 산출물은 편의용 queue이고 reference 편집/승격 source of truth는 case set이다.
- `transcribe-review-case-candidates`는 `case-index.json`의 audio file을 project workflow로 분석/전사해 candidate directory에 `<case-id>.master.json`을 쓴다. Output directory는 비어 있어야 하며, project artifact는 기본적으로 output 아래 `projects/`에 보존한다.
- `align-review-case-candidates`는 candidate가 붙은 `case-index.json`을 받아 aligned candidate directory, diagnostics, attach plan, eval manifest를 만든다. 원본 case set과 원본 candidate는 수정하지 않는다.
- `build-candidate-attach-plan`은 `case-index.json` case id와 candidate directory의 `<case-id>.master.json`, `<case-id>.json`, `<case-id>.srt`를 매칭해 attach plan을 만든다. Candidate path는 plan file 기준 상대경로로 저장하고, 누락/모호한 중복은 output 없이 실패한다.
- `attach-review-case-candidates`는 모든 case id에 대한 case-local candidate transcript plan을 받아 candidate master files를 쓰고 index를 갱신한다. 누락/중복/알 수 없는 case id나 기존 candidate overwrite는 output side effect 전에 실패한다.
- `freeze-case-references`는 준비된 `case-index.json`의 reference들을 새 output directory의 `references/*.master.json`으로 고정하고, 새 `case-index.json`, `audio-map.json`, optional `eval-manifest.json`을 쓴다.
- `build-eval-manifest`는 `case-index.json`의 candidate paths를 평가 manifest로 재생성한다. stale count나 missing file이 있으면 manifest를 쓰지 않는다.
- `sweep-channel-attribution`은 eval manifest와 audio map으로 여러 L/R attribution threshold를 비교하고 setting별 eval report와 comparison을 만든다. 이미 channel이 붙은 후보는 `--reset-speech-channels-to-mix`로 sweep copy 안에서만 speech channel을 MIX로 되돌려 threshold를 공정하게 재평가한다. Setting item에는 reason counts와 attributed channel counts를 보존한다. `--product-gate`를 함께 쓰면 comparison에 gate 결과를 남기지만, 기본 threshold를 자동 변경하지 않고 WebUI 옵션으로 노출하지 않는다.
- WAV가 아닌 입력은 ffmpeg로 16-bit PCM WAV로 변환한다.
- stereo WAV는 `L`, `R`, `MIX` 세 channel 파일로 분리한다.
- mono WAV는 `MIX`만 만든다.
- 모델 adapter는 `openai-compatible`, `gemini`, `local-transformers`, `local-qwen-asr`, `local-qwen-hf-asr`, `local-cohere-asr`, `local-granite-asr`다.
- 모델 설정은 UI에서 사용자가 직접 입력한다.
- 모든 ASR adapter는 stereo 입력의 L/R을 각각 전사하고 mono 입력만 MIX를 전사한다.
- Production은 energy/external VAD, energy channel attribution, automatic bleed deletion을 실행하지 않는다.
- `CASRT_VAD_COMMAND`, `CASRT_QWEN_ENERGY_*`, `CASRT_LOCAL_ASR_CHANNEL_MODE`, `CASRT_ALIGNER_COMMAND`는 server와 `project transcribe`에서 무시한다. 관련 구현은 명시적 개발/평가 CLI 전용이다.
- Production local worker 환경은 Qwen forced-aligner와 Granite timestamp parsing 실험 변수를 제거한다. Worker 모듈 직접 실행을 통한 개발 실험만 허용한다.
- `CASRT_QWEN_ASR_ALIGNER_MODEL_ID`는 고정 forced aligner 실험/내부 보정 경로다. `CASRT_QWEN_ASR_MIN_ALIGNED_DURATION_MS`보다 짧은 aligned span은 clip bounds로 되돌리며, WebUI 옵션으로 노출하지 않는다. 2026-06-30 01/04/07 front120 guard80 평가에서는 time-aligned 500ms가 29.5% -> 36.1%, channel time-aligned가 73.1% -> 75.0%로 개선됐지만 practical CER 29.5%는 변하지 않아 단독 기본 승격하지 않는다.
- Generic Qwen aligner worker는 stereo master의 segment channel에 맞는 L/R/MIX mono clip을 사용한다. 이전 full-stereo clip 경로의 결과는 MIX 후보에는 유효하지만 stereo 후보 alignment 근거로 재사용하지 않는다. 수정 worker의 Bro stereo all8 재평가에서도 같은 채널 전용 500ms timing이 `15.79% -> 8.00%`로 악화했으므로 Qwen3-ForcedAligner는 기본 승격하지 않고 no-op alignment를 유지한다.
- `custom_asmr_srt_stack.qwen_aligner_worker`는 Qwen3-ForcedAligner를 generic aligner command로 쓰는 고정 내부 경로다. 이 경로는 기존 master text/channel/kind를 변경하지 않고 segment 내부 start/end만 재정렬한다. 실행은 local snapshot, offline env scrub, network-disabled guard, `qwen-asr==0.0.6` RECORD hash, per-file RECORD hash, import origin 검증 조건을 만족해야 한다. `CASRT_QWEN_ALIGNER_MIN_ALIGNED_DURATION_MS=80`과 `CASRT_QWEN_ALIGNER_MIN_COVERAGE_RATIO=0.5` guard로 비현실적으로 짧거나 원 segment 절반 미만으로 잘린 span만 원래 timing으로 유지한다. 이 fallback은 외부 aligner over-trim에 한정하며 text/channel 수정 실패를 숨기지 않는다.
- `CASRT_GRANITE_ASR_PARSE_TIMESTAMPS=1`은 Granite Plus timestamp prompt 실험용 내부 경로다. `[T:N]` centisecond tag는 segment end로 unwrap하고 `_` silence marker는 split/trim 경계로 사용한다. Tag가 없거나 speech segment를 만들 수 없으면 기존 chunk-bound segment로 남기며 `needs_review=true`는 유지한다. 이 값은 WebUI 옵션으로 노출하지 않는다.
- alignment 튜닝 정보는 `align-transcript --diagnostics-output`의 별도 debug JSON으로만 저장하고, 자동 보존하거나 master JSON data contract에 넣지 않는다.
- `custom_asmr_srt_stack.qwen_hf_asr_worker`는 HF-native Qwen3-ASR worker다. repo id가 아니라 exact revision local snapshot directory를 받으며, offline env, local path-only, `local_files_only=True`, `trust_remote_code=False`, `use_safetensors=True`, network-disabled guard를 만족해야 한다. offline local worker env는 `PYTHONPATH`를 제거하고 `PYTHONNOUSERSITE=1`을 강제한다. timestamp를 반환하지 않으므로 chunk 전체 timing과 `needs_review=true`를 반환한다.
- 로컬 ASR adapter는 L/R energy 차이와 quieter-side gate로 channel attribution을 수행한다. 현재 기본값은 8dB 차이와 quieter side -40dBFS 이하이며, 같은 구현을 `casrt attribute-channels` CLI에서 기존 SRT/master 후처리에도 사용한다.
- channel attribution 튜닝 정보는 `--diagnostics-output`의 별도 debug JSON으로만 저장하고, master JSON data contract에는 넣지 않는다.
- VAD는 UI에서 선택하지 않는다.
- VAD command는 stdin으로 `{ audio_file, audio_info }` JSON을 받고 stdout으로 `{ intervals: [{ start_ms, end_ms }] }` JSON을 반환한다.
- VAD interval이 정렬되지 않았거나 겹치거나 audio duration을 넘으면 실패한다.
- alignment는 UI에서 선택하지 않는다.
- `CASRT_ALIGNER_COMMAND`는 명시적 alignment 개발 CLI에서만 실행하며 앱의 정상 전사 경로에는 영향을 주지 않는다.
- aligner command는 stdin으로 `{ audio_file, master }` JSON을 받고 stdout으로 `{ segments: [{ id, start_ms, end_ms }] }` JSON을 반환한다.
- aligner output이 id를 누락하거나 중복하면 실패한다.
- `needs_review`는 WebUI segment row에서 표시하고 직접 토글할 수 있다. `content_reviewed`와 `channel_reviewed`는 기존 검수 동작이 관리하며 별도 옵션으로 노출하지 않는다.
- Review case WebUI는 `needs_review=true` 또는 `content_reviewed!=true` segment row를 시각적으로 표시하고 첫 미검수 segment를 우선 선택한다.
- WebUI review-pack viewer는 `custom-asmr-review-pack-v1` `index.json`을 path로 열고, item별 `clip_url`을 통해 pack directory 내부 `clips/*.wav`만 재생한다. Candidate가 없는 `review-case-pack` item은 reference segment id를 표시하고 빈 `CAND` 줄은 숨긴다. Reference audit overlap item은 두 번째 reference segment를 후보 전사처럼 보이지 않게 `REF2`로 표시한다. Reference channel energy audit item은 `candidate_channel`을 `ENERGY` verdict로 표시하고 L/R dBFS와 delta evidence를 같은 줄에 표시한다. Pack root `duration_summary`가 있으면 header에 실제 listening duration과 focus/source duration을 표시해 검수 부담을 JSON 없이 확인하게 하고, item `review_clip_*`가 원 segment bounds와 다르면 row에 focus time range를 표시한다. `다음 clip`은 선택이 없으면 첫 priority item, 선택이 있으면 다음 priority item을 재생한다. Item에 `source_case_index`, `case_id`, `reference_id`가 있으면 `case 열기`로 source case editor를 열고 해당 reference segment를 선택하며, reference audit item에서 넘어간 경우 status에 같은 overlap/long-segment 구조 evidence를, channel audit item에서 넘어간 경우 status에 같은 energy evidence와 focus range를 유지한다. Source case가 focus item에서 열렸다면 selected source segment 재생은 focus range를 쓰되 segment `start_ms/end_ms`는 수정하지 않는다. Channel audit source segment에서는 사람이 `ENERGY L/R 적용`을 눌러 suggested channel을 저장하거나, 기존 `검수 완료` button이 context에 맞게 표시하는 `Channel 검수 완료`로 현재 reference channel을 유지하는 사람 판정을 저장할 수 있다. Energy 적용만으로 `needs_review` 또는 channel 검수를 자동 완료하지 않는다. Pack에서 연 source case는 return context를 보존해 `pack 목록`으로 원래 review pack queue에 돌아갈 수 있고, 기존 `다음 case` control은 `다음 issue`로 바뀌어 저장 후 다음 source issue editor를 priority 순서로 직접 연다. Pack root에 `next_case_id`가 있고 아직 clip을 선택하지 않았다면 `case 열기`는 해당 case의 첫 source item을 연다. Reference overlap audit item은 source editor에서 `REF2` segment row도 보조 표시해 두 segment를 빠르게 비교할 수 있게 한다.
- WebUI audio API는 `Accept-Ranges: bytes`, single-byte `Range`의 `206 Content-Range`, invalid/unsatisfiable range의 `416`을 지원한다. Segment 재생은 media data가 준비된 뒤 start timestamp를 적용하므로 긴 WAV를 0초부터 다시 받거나 source case를 연 직후 0초에서 잘못 재생하지 않는다.
- VAD/core overlap-context runtime은 제품에 두지 않는다. Bro/Qwen 계열은 context audio의 text를 한 owned timestamp segment 안에 합쳐 반환할 수 있어 midpoint ownership만으로 중복 text를 안전하게 제거할 수 없었고 실제 3-case 평가에서 text/timing이 크게 악화했다. 더 세밀한 token/word timestamp ownership 근거가 없는 한 hard interval bounds를 유지한다.
- VAD coverage 진단의 channel mode는 실제 ASR channel mode와 일치해야 한다. `mix`는 L/R 평균 mono MIX waveform과 전체 reference speech union을, `stereo`는 L/R mono audio와 same-channel+MIX reference를 각각 평가한다. Source case 수와 channel coverage unit 수는 별도 필드로 기록해 stereo를 case 두 배로 오해하지 않게 한다. 과거 원 stereo interleaved RMS report는 새 runtime-parity report를 대신할 수 없다.
- Built-in energy VAD가 반환하는 padded chunk는 정렬·비중첩이어야 한다. Padding으로 인접 range가 overlap하거나 접촉하면 core에서 다시 병합한 뒤 max-chunk split을 적용하며, workflow/evaluator가 중복 audio를 정상 입력으로 받아들이게 두지 않는다.
- Source-linked review pack의 `index.json`은 재현 가능한 불변 queue다. Loader는 item reason을 content, review-flag, channel requirement로 해석하고 source reference master의 현재 evidence를 결합해 response-only `source_review_requirements`, `source_*_reviewed`, `source_needs_review`, `source_review_resolved`와 root `total/pending/resolved_item_count`를 반환한다. WebUI는 resolved item을 목록과 다음 탐색에서 제외하고 부분 완료 시 남은 clip 청취 시간만 다시 계산한다. Source case 저장 성공 후 같은 상태를 메모리에도 반영하며, 후속 text/time/channel 편집이 evidence를 무효화하면 item이 다시 pending이 된다. Source case 또는 segment가 누락된 contract drift는 항목을 숨기지 않고 pack load를 실패시킨다. 별도 진행률 옵션이나 sidecar state는 만들지 않는다.
- Canonical review case의 attached candidate는 reference authority가 아닌 read-only 검수 보조 초안이다. 같은 local model/scope에서 text, channel-aware text, timing, 검수 밀도가 개선된 후보로 `--replace`할 수 있지만 reference master, review evidence, `reference_type`, freeze gate는 절대 자동 변경하지 않는다.
- Review-pack viewer는 pack 생성 옵션, VAD, threshold, 모델 선택을 추가로 노출하지 않는다. Pack 생성과 priority queue 정렬은 CLI가 담당하고 WebUI는 사람이 듣고 비교하는 화면만 담당한다.
- WebUI review case editor는 `custom-asmr-review-case-set-v1` `case-index.json`을 path로 열고, case를 선택하면 audio와 reference master를 기존 segment editor에 붙인다.
- Review case item에 optional `candidate`가 연결돼 있으면 서버는 candidate master를 같은 transcript contract로 검증해 반환한다. Editor는 각 editable reference row 아래에 시간이 겹치는 같은-channel/MIX candidate segment의 channel/time/text를 읽기 전용 `CAND` context로 표시하고 header에 reference/candidate segment 수를 분리해 표시한다. MIX reference는 겹치는 모든 channel candidate를 표시한다. Candidate context는 reference 수정 source of truth에 섞지 않으며, 연결된 candidate file이 누락되거나 malformed이면 숨기지 않고 case load를 실패시킨다. 별도 비교 mode나 WebUI 옵션은 추가하지 않는다.
- Review case reference 수정은 원 reference master JSON과 `case-index.json`의 `segments`/`review_count`에 자동 저장한다. 일반 `검수 완료`는 현재 선택 segment의 `needs_review`를 false로 바꾸고 다음 검수 segment로 이동한 뒤 저장한다. Channel audit에서 연 target에서는 같은 button이 `Channel 검수 완료`로 표시되고 `channel_reviewed=true`를 함께 저장한다. Channel 또는 start/end를 다시 수정하면 이 판정을 `false`로 무효화한다. `case 목록`으로 돌아가거나 `다음 case`로 이동할 때도 저장을 즉시 flush한다.
- Review case 목록은 전체 `needs_review` flag 수와 남은 review duration, flag가 남은 case, 각 case의 첫 미검수 segment 시간/텍스트 preview를 표시한다. 이 표시도 human-reviewed 여부를 판정하지 않고 남은 flag 상태만 보여준다.
- `다음 case`는 현재 case 뒤의 `review_count > 0` case를 우선 열고, 없으면 다음 순서 case를 연다. Human-reviewed 여부는 자동 판정하지 않으며, 승격은 여전히 `freeze-case-references --reference-type human-reviewed`와 manifest gate가 담당한다.

## 열린 결정

- breath/SFX를 기본 export에 포함할지, 요청 시에만 포함할지.
- 실제 파일 테스트 후 split/merge가 필요한지.
- Qwen3-ForcedAligner를 기본 timing 보정 경로로 승격할지.
- Silero/TEN VAD 중 어떤 로컬 VAD 구현을 기본 command로 고정할지.
