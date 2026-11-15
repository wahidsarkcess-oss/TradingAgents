# Running TradingAgents unattended in the cloud

> **Status: the image build and container runtime are not yet verified.** This
> configuration was checked with `docker compose config` and its entrypoint was
> tested directly, but no Docker daemon was available to build and run it. Before
> relying on it, build and run it once on a Docker host and confirm: the `appuser`
> uid, that the run works on the read-only filesystem, that `secrets.env` is
> readable with the documented ownership, and that no key appears in
> `docker inspect`.

This folder runs **one analysis per invocation** in a hardened container, on any
host that runs Docker: a VM you already have, a home server, or a container
platform. Nothing here needs a paid service beyond the LLM provider you choose,
and nothing here schedules runs, connects to a broker, or places orders. A run
writes the same reports as the CLI, into a Docker volume.

| File | Purpose | Committed? |
|---|---|---|
| `docker-compose.cloud.yml` | The hardened, non-interactive service | yes |
| `entrypoint.sh` | Loads keys from the mounted secrets file, then runs the CLI | yes |
| `cloud.env.example` → `cloud.env` | Provider, models, depth, limits. **No secrets.** | example only |
| `secrets.env.example` → `secrets.env` | API keys | **never** (gitignored, kept out of the image) |
| `pricing.example.json` → `pricing.json` | Your provider's prices, for cost limits | your choice (not secret) |

## 1. Configure

```bash
cd deploy/cloud
cp cloud.env.example cloud.env
cp secrets.env.example secrets.env
chmod 600 secrets.env
```

- In `cloud.env`, set `TRADINGAGENTS_LLM_PROVIDER`, `TRADINGAGENTS_QUICK_THINK_LLM`
  and `TRADINGAGENTS_DEEP_THINK_LLM`. They are blank on purpose; a run without
  them stops before calling any model and names what is missing.
- In `secrets.env`, fill in only the key for that provider. Edit it in an editor;
  don't `echo` a key on the command line, where it lands in shell history.
- The container runs as `appuser`, normally uid 1000; confirm with
  `docker compose -f docker-compose.cloud.yml run --rm --entrypoint id analysis`.
  `secrets.env` is bind-mounted as a Docker secret with its host permissions, so
  with `chmod 600` it must be owned by that uid on the host
  (`sudo chown 1000 secrets.env`), or the entrypoint stops with "not readable".

**Subscriptions:** only a ChatGPT Plus/Pro plan can stand in for an API key
(`TRADINGAGENTS_LLM_PROVIDER=chatgpt`, after `tradingagents login chatgpt`; the
sign-in lives in the data volume, and one instance at a time should use it).
Other subscriptions, including Claude Pro/Max, do not include API credits: an
`ANTHROPIC_API_KEY` is billed through an Anthropic API account.

## 2. Spending limits

`cloud.env` starts with these, enforced before every LLM request by one budget
that the quick and deep models and the parallel analysts share:

| Variable | Start value | What happens at the limit |
|---|---|---|
| `TRADINGAGENTS_MAX_LLM_CALLS` | 40 | The next call is refused. Failed requests and SDK retries count. |
| `TRADINGAGENTS_MAX_TOKENS` | 4096 | Output limit per call (too low truncates reasoning models). |
| `TRADINGAGENTS_MAX_TOKENS_PER_RUN` | 200000 | A call that could push the run past it is refused. |
| `TRADINGAGENTS_MAX_COST_PER_RUN` / `_PER_DAY` | unset | Estimated USD; needs a pricing file (below). |
| `TRADINGAGENTS_MAX_REFLECTION_CALLS` / `_TOKENS` | unset (= the analysis values) | Reflections on past decisions use their own allowance, so a backlog of them never starves an analysis. A refused reflection still records the decision's outcome (it is scored), without the lesson text, and the run says so. |

An SDK retry is checked like a call before it is sent, so the call limit holds
exactly. A checkpointed analysis keeps its usage beside the checkpoint, so
resuming it continues from what it already spent; a resume whose usage record is
missing or damaged is refused while a limit is set (`--clear-checkpoints` starts
fresh). TypeSafe screening, a paid API outside the LLM clients, is counted and is
skipped entirely while a cost limit is set.

A refused call stops the run with **exit code 3** and a usage summary; any other
failure exits 1. Each saved state log (`logs/<TICKER>/TradingAgentsStrategy_logs/`)
records `llm_usage`, which keeps three figures apart:

- **provider-reported tokens**: the counts the provider returned;
- **estimated cost**: those tokens × your pricing file (an upper bound, not a bill);
- **provider-reported cost**: only when the provider returns one (OpenRouter does).

### Cost limits fail closed

1. Copy `pricing.example.json` to `pricing.json`. From your provider's official
   pricing page, fill in the USD per 1M-token prices for exactly the models in
   `cloud.env`, set `as_of` to today and `source` to the page URL.
2. Uncomment `TRADINGAGENTS_PRICING_FILE` and a cost limit in `cloud.env`, then
   rebuild (`pricing.json` is copied into the image).

With a cost limit set, the run refuses to start if the file is missing, invalid,
dated in the future, older than `TRADINGAGENTS_MAX_PRICING_AGE_DAYS` (90), or
missing a model; if `TRADINGAGENTS_MAX_TOKENS` is unset (a reply's cost would
have no upper bound); or if a tier uses a provider whose usage reporting cannot
be relied on (`chatgpt`, `ollama`, `openai_compatible`) or any provider at a
custom endpoint (a relay or proxy URL). A run also stops if a provider returns a response with no token
usage, since its spend can no longer be bounded. **Also set a hard spending limit
in your provider's billing settings**: these limits are local estimates.


### The daily ledger

Whenever prices are known, every call's spend goes into
`<cache dir>/spend/<UTC date>.json` (in the container, inside the data volume),
including runs with only a per-run cost limit or no limit at all. Before each
call its worst-case cost is reserved in the ledger under a lock shared by every
process using that cache directory, and settled when the call ends, so concurrent
runs cannot jointly pass `TRADINGAGENTS_MAX_COST_PER_DAY`. Scope and limits:

- one ledger per cache directory: containers or hosts without shared storage each
  keep their own, and file locks are not reliable on network filesystems;
- a run that dies mid-call leaves its reservation counted until the day ends;
- an unreadable or unwritable ledger stops every later call while a cost limit is
  set, and is logged as an error otherwise. It is never overwritten or reset.

## 3. Run

```bash
docker compose -f docker-compose.cloud.yml build
TICKER=NVDA ANALYSIS_DATE=2026-11-13 docker compose -f docker-compose.cloud.yml run --rm analysis
```

`ANALYSTS` defaults to `market,news,fundamentals` (add `social` for sentiment).
Use a date whose market close has passed; a run dated today reads a session that
may still be trading.

Reports, the memory log and the spend ledger persist in the
`tradingagents_cloud_data` volume. Copy them out with:

```bash
docker run --rm -v tradingagents-cloud_tradingagents_cloud_data:/data -v "$PWD":/out alpine cp -r /data/logs /out/
```

## What the container does to stay safe

- **No secrets in the image:** `.dockerignore` excludes `.env*`, `secrets.env`,
  `*.secrets.env`, `chatgpt-auth.json`, keys, `data/` and `cloud.env`.
- **Keys only in memory:** keys come in through a Docker secret, are exported by
  `entrypoint.sh` (which never prints them), and don't appear in
  `docker inspect` or on the command line. The entrypoint accepts only credential
  names (`*_API_KEY`, `*_KEY`, `*_TOKEN`, `*_SECRET`, `*_PASSWORD`, plus
  `AWS_ACCESS_KEY_ID`, `AWS_BEARER_TOKEN_BEDROCK`); `PATH`, `LD_PRELOAD`,
  `PYTHON*` and `TRADINGAGENTS_*` settings in the secrets file are skipped.
- **What this does not protect against:** anyone with access to the Docker
  daemon or root on the host. Exported keys can be read from the process's
  environment (`/proc/1/environ`) by root, including through `docker exec`. A key
  put in `cloud.env` by mistake is visible in `docker inspect`.
- **Redacted output:** a run without a terminal redacts stdout, stderr and log
  output: key values, OAuth tokens and anything shaped like a credential. Error
  messages are scrubbed, and tracebacks never print local variables.
- **No update check, no prompts:** `TRADINGAGENTS_NO_UPDATE_CHECK=1`, and there
  is no TTY.
- **Locked down:** read-only root filesystem with tmpfs `/tmp`, all Linux
  capabilities dropped, `no-new-privileges`, and memory/CPU/PID limits.

## Container platforms

The image and entrypoint work unchanged on managed platforms (Cloud Run jobs,
ECS/Fargate tasks, Kubernetes Jobs). Those are paid services; check their pricing
before you use them. On such a platform:

- Inject keys through its secret manager, either as environment variables or as
  a file at `/run/secrets/tradingagents.env`.
- Put the `cloud.env` settings in the job definition.
- Mount persistent storage at `/home/appuser/.tradingagents`, with a single
  writer: the memory log and spend ledger use file locks, which are not reliable
  on network filesystems shared by many writers.
- Allow outbound HTTPS to your LLM provider and to the data sources:
  `query1.finance.yahoo.com`, `query2.finance.yahoo.com`, `www.sec.gov`,
  `data.sec.gov`, and optionally `api.stlouisfed.org`, `www.reddit.com`,
  `api.stocktwits.com`, `gamma-api.polymarket.com`, `www.alphavantage.co`.

Running analyses on a schedule spends money every time it fires, so this folder
deliberately ships no scheduler. Add one only once the limits above are set and
tested.
