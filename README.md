# Routine Graph

Discover daily habits and routines from persistent activity JSONs, retain their supporting observations, and answer questions through activity, habit, and routine retrieval channels.

A **habit** is a recurring individual activity. A **routine** is a recurring ordered activity subsequence. Each profile belongs to one participant and requires evidence from at least two distinct recording days. DBSCAN noise remains available as individual activities.

## Setup

Use Python 3.11+ and [uv](https://docs.astral.sh/uv/):

```sh
uv sync
cp .env.example .env
```

If `.env` already exists, add the example's missing settings to it rather than replacing it. Set `EMBEDDING_API_KEY` and `EMBEDDING_BASE_URL` locally for your OpenAI-compatible embedding endpoint. The embedding client also accepts `OPENAI_API_KEY` and `OPENAI_BASE_URL`. Keep the existing `OPENROUTER_API_KEY` for the default summary and question-answering models.

Embeddings default to `qwen3-embedding-8b`. Use `EMBEDDING_MODEL=qwen3-embedding-4b` to select a smaller model. Requests send plain text to `/embeddings`; query embeddings include Qwen retrieval instructions. Embedding dimensions are inferred from responses unless explicitly requested with `--embedding-dim`.

Chat models remain configurable through `--chat-model` and `--chat-provider`, or `ROUTINE_CHAT_MODEL` and `ROUTINE_CHAT_PROVIDER`. The existing default is `google/gemini-3.6-flash` through OpenRouter; set another available model if needed. Installed alternative providers can be selected explicitly for embeddings with `--embedding-provider`.

## Input

The existing video extractor writes `persistent_event_<PARTICIPANT>_<DAY>.json`, for example `persistent_event_A1_JAKE_DAY1.json`:

```json
[
  {
    "label": "drinking coffee",
    "description": "The participant drinks coffee from a mug in the kitchen.",
    "start_time": "08:00:00",
    "end_time": "08:10:00",
    "location": "kitchen",
    "interacted_objects": ["mug"],
    "date": "2026-11-02"
  }
]
```

Only label, description, start_time, and end_time are required. Times are local `HH:MM:SS`; an end earlier than the start represents crossing midnight. Optional context and additional fields are preserved. Participant names can contain underscores. JSON order identifies each event within its source file; activities are ordered chronologically for sequence discovery.

Each source contains one participant's activities for one recording day. The following day's activities belong to a separate participant/day input. An activity ending after midnight keeps its source day membership; it does not join the two daily timelines.

An alternative envelope permits arbitrary filenames and a date shared by all events:

```json
{
  "participant": "A1_JAKE",
  "day": "DAY1",
  "date": "2026-11-02",
  "events": []
}
```

`DAY1` alone does not identify a calendar weekday. Weekday statistics are populated only from supplied dates. An empty list replaces that participant/day with no observations. A corrected file replaces the same participant/day, removing trailing activities absent from the correction. Reordering or inserting source events can change which observation an index-based event ID identifies; profile and subsequence IDs are local to a rebuilt snapshot.

## Run

```sh
# Files, quoted globs, and directories are supported.
uv run routine-graph ingest '/path/to/persistent_event_*.json'

# Small examples may need a lower density threshold than the default of five.
uv run routine-graph ingest examples --habit-min-samples 2 --routine-min-samples 2 --habit-eps 0.1 --routine-eps 0.1

uv run routine-graph stats
uv run routine-graph search 'drinking coffee' --channel habit --participant A1_JAKE
uv run routine-graph search 'washing dishes' --channel activity --day DAY1
uv run routine-graph show routine 0 --participant A1_JAKE
uv run routine-graph timeline 1 --direction after
uv run routine-graph ask 'What do I usually do after drinking coffee?' --participant A1_JAKE
```

All commands produce JSON. Use `--db` and `--index` after the subcommand to override `data/routine_graph.db` and `data/events.faiss`. The module entry point also works: `uv run python -m routine_graph --help`. The earlier ingestion launcher delegates to the same CLI: `uv run python -m routine_graph.ingest_script <inputs>`.

`search --channel all` searches each of the three channels separately. Filtering applies before the top results are selected. `show` traverses profile-to-activity links and returns routine subsequence IDs and step positions; use `--limit` and `--offset` for pagination. `timeline` follows start-time order within one participant/day and can move before or after an activity. Overlapping events remain in that order; a next link does not assert that the earlier event ended first. None of these three inspection commands needs a chat model. `show`, `timeline`, and `stats` also work without an embedding key.

`ask` uses bounded LangChain tool calls to search channels, inspect supporting evidence, and follow activity order. With multiple participants, specify `--participant`. Answers cite node IDs, and the JSON includes the nodes observed during retrieval. Similarity scores measure embedding proximity, not confidence or frequency. Insufficient-evidence wording is instructed in the agent prompt; factual accuracy still requires validation with a real model.

## Discovery and tuning

- Habits use DBSCAN on normalized semantic embeddings plus weighted sine/cosine coordinates for both start and end time. Routines use the same features in ordered subsequence matrices, compared by the existing custom DTW with mean cosine distance along its alignment path.
- `--temporal-weight` defaults to `0.5`. `0` disables temporal influence. Midnight wraps naturally.
- `--habit-eps`, `--routine-eps`, `--habit-min-samples`, and `--routine-min-samples` control DBSCAN. Density counts observations, including the point itself. `--min-days` is an additional recurrence filter, defaults to `2`, and cannot be lower than `2`.
- The longest possible sequence is one participant's complete day event sequence. `--max-sequence-length` defaults to `5` as an additional cap. `0` generates all contiguous subsequences of at least two activities, up to the full sequence of each day. Windows and chronological graph links do not cross participant/day boundaries, even when an activity ends after midnight. For example, days `[a, b]` and `[c, d]` yield `[a, b]` and `[c, d]`, never `[b, c]` or `[a, b, c, d]`.
- All-pairs DTW remains quadratic in the number of candidate subsequences. `--max-routine-candidates` defaults to `2000` per participant and stops oversized runs with an actionable error. Raise it deliberately, or use `0` to disable the guard. This is a small-dataset implementation, not a validated million-node ingestion system.
- A routine's observation count counts matched subsequences, which can overlap. Its day count is distinct supporting recording days. Neither is a claim about behavior outside the recordings.
- Summaries use bounded, day-balanced samples and bounded context. Exact observation/day counts and temporal statistics are stored separately. Weekday counts describe supported occurrences, not recording coverage or probabilities.

Generate tuning plots for one participant using the features saved at ingestion:

```sh
uv run python -m routine_graph.utils.k_distance --db data/routine_graph.db --faiss data/events.faiss --participant A1_JAKE --min_samples 5 --output output
```

## Persistence and failures

SQLite contains observations, membership links, summaries, temporal statistics, embedding provenance, and vectors. Its `graph_edges` view exposes chronological `next` links, habit `occurrence` links, and routine `step` links carrying subsequence IDs and positions.

Each ingestion publishes its SQLite changes in one transaction. Invalid input or a model failure rolls back that run. Unchanged activity embeddings are cached by content and model configuration; profiles and their summaries are rebuilt on every run. Changes to the embedding model or dimensions rebuild the embeddings consistently.

FAISS indexes for activities, habits, and routines are derived caches. A revision manifest detects outdated indexes; missing, stale, or unreadable caches are rebuilt from committed SQLite vectors. A cache-write failure after the SQLite commit leaves a recoverable committed snapshot. Retrieval rejects a model that differs from the stored embedding configuration.

Existing databases receive additive schema migrations preserving event IDs. Run ingestion again to populate vectors, participant-specific profiles, and the new metadata. Generated files under `data/` are ignored by Git.

## Validation

```sh
uv run python -m unittest discover -s tests -v
```

Tests exercise real SQLite, FAISS, clustering, CLI commands, and the LangChain tool loop using deterministic embedding/chat substitutes. Embedding HTTP requests are checked with mocked responses, including request format, batching, response order, missing credentials, and invalid vectors. No API keys or model downloads are needed for tests.

The included two-day synthetic example also passed a live smoke test with hosted `qwen3-embedding-8b` (4096 dimensions) and the configured OpenRouter chat model: four activities, two habits, one routine, results from all three retrieval channels, and an answer citing the observed coffee-to-dishwashing sequence. This verifies the integration on a small example; it does not establish accuracy on real recordings. The EgoMemReason benchmark has not been run.

## Fresh raw EgoLife extraction with Gemma

`persistent_memory.extract_egolife` reads A1_JAKE's original videos and sends freshly preprocessed video and audio to the local Gemma tunnel. It plans windows using the source filenames and available timing metadata; it does not reuse earlier event descriptions or processed video pixels.

```sh
# Test one real window before replacing older results.
PYTHONPATH=src uv run python -m persistent_memory.extract_egolife --max-batches 1 --workers 1

# Resume the pilot and process every raw source file.
PYTHONPATH=src caffeinate -i .venv/bin/python -u -m persistent_memory.extract_egolife --workers 4 --replace-old-results --publish-dir /Volumes/TEW_SSD/dataset/EgoLife/persistent_events
```

The endpoint defaults to `http://localhost:8000/v1`, model `gemma4-12b-it`, and the local `EMPTY` key placeholder. Override the endpoint/model with `--base-url` / `--model`. No OpenRouter key is needed for this extraction path.

Defaults are 120-second windows within a single participant/day and continuous recording segment, 0.25 video fps, a 320-pixel maximum frame dimension preserving aspect ratio, and mono 16 kHz / 32 kbps MP3 audio. FFmpeg uses automatic hardware decoding where available, one decoder/encoder thread per process, and at most two concurrent preprocessing processes. Temporary media is removed after each successful request. `--workers` controls concurrent day requests; windows from each day are always processed sequentially with carried activity state.

The default output is `data/egolife_A1_JAKE_raw_gemma4/`. `extraction.sqlite` commits each model response and raw-file coverage before continuing; `progress.json` reports completion against the full input inventory. The same command resumes after interruption. `--replace-old-results` removes only the earlier A1_JAKE partial event files and their generated workspace results, requires a successful pilot, and records cleanup so resumes preserve newly extracted files. Complete day files are published to `--publish-dir`; local exports for a day still in progress are partial until the progress file marks it complete.

At a recording gap or the end of a day, ongoing observations are retained with `end_observed: false` and an `end_reason`; these boundaries do not assert that the real activity ended. A model/preprocessing failure stops coverage advancement, and completion requires every raw source file to be accounted for. Review the resulting events before choosing clustering thresholds for a graph build.

See [GLOSSARY.md](GLOSSARY.md), [implementation plan](docs/implementation-plan.md), and [architecture decisions](docs/adr/) for definitions and the accepted design.
