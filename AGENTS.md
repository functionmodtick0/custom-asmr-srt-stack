# Repository Instructions

## Documentation Authority

Plans, decisions, and experiment results must not live only in chat.

When work changes or establishes any of the following, update the relevant document in the same change set:

- product direction or non-goals
- data contracts
- CLI/WebUI behavior
- model or adapter selection
- local runtime setup
- ASR chunking, alignment, channel attribution, or threshold values
- evaluation metrics or benchmark results
- next-step implementation plans

Primary documentation targets:

- `docs/product-decisions.md`: product scope, non-goals, data contracts, UI policy, and stable product decisions
- `docs/cli-product-decisions.md`: CLI command contracts and CLI-visible behavior
- `docs/local-asr-pipeline.md`: local ASR pipeline details, model/runtime choices, ASMR-specific heuristics, experiments, and evaluation plans
- `README.md`: current user-facing setup and usage summary

If a decision is made during implementation, document it before considering the task complete. If the decision is only tentative, record it as an open decision or next-step plan rather than leaving it implicit.

## Security Review Scope

Use subagent security review only before executing or adopting external code/runtime surfaces, such as third-party repository code, `trust_remote_code`, new external runtime packages, unsafe model formats, or unreviewed downloaded tooling.

Do not require subagent security review for ordinary in-repository code changes, wrappers, tests, or documentation. Handle those with normal implementation review and behavior tests.
