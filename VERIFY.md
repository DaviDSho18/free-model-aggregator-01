# Verification

Requires Node 22. No packages or network services are needed for the local tests.

```sh
env -u MAKEFUN_PROMPT_FILE node main.ts --list
# stdout: {"providers":[]}
# exit: 0 (a missing-brief diagnostic on stderr is expected)

node --check main.ts
node --check lib/providers.ts
node --check lib/routing.ts
node --check lib/wire.ts
node --check lib/aggregator.test.ts
node --test lib/aggregator.test.ts
```

The tests use only the standard library. They cover quality sorting and stable ties, free/capability
filtering, shared provider budgets, future capability reservations, weighted assignment versus an
exhaustive oracle, paginated catalogs, request translations, multimodal tool results, native streaming
usage requests, byte-split SSE parsing, tools, usage totals, finish reasons, and all twelve
upstream/client/streaming combinations against an in-process mock environment. The integration test
checks 402/429/500/503 failover, exact call limits, concurrent clients without provider overlap,
catalog/register/list equality, clean `/work` shutdown, and byte-identical catalogs and provider-call
traces across two replays.

This is a local contract suite, not the sealed make.fun scorer; it cannot establish a bounty score
or validate an actual environment state export. In this editing session, the test command, the
`node --check main.ts` command, and the required empty-catalog command above were all blocked by the
tool approval gate before execution. No passing test or syntax-check result is claimed. Run the
commands above in an execution-enabled session before submission.
