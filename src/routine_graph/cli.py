import argparse
from contextlib import contextmanager
import glob
import json
import os
from pathlib import Path
import sys

from dotenv import load_dotenv
import httpx

from routine_graph.agent.query_agent import QueryAgent
from routine_graph.database.sqlite_controller import Database
from routine_graph.ingestion.ingest import IngestionPipeline
from routine_graph.retrieval import Retriever


def resolve_inputs(patterns: list[str]) -> list[str]:
    paths = []
    for pattern in patterns:
        path = Path(pattern)
        matches = sorted(path.rglob("persistent_event_*.json")) if path.is_dir() else [Path(p) for p in sorted(glob.glob(pattern))]
        if not matches:
            raise ValueError(f"No JSON inputs found for {pattern}")
        for match in matches:
            if not match.is_file() or match.suffix.lower() != ".json":
                raise ValueError(f"Expected a JSON file: {match}")
            resolved = str(match.resolve())
            if resolved not in paths:
                paths.append(resolved)
    return paths


@contextmanager
def open_retriever(args):
    if not Path(args.db).is_file():
        raise ValueError(f"Database does not exist: {args.db}; run ingest first")
    db = Database(args.db)
    retriever = Retriever(db, args.index)
    try:
        yield retriever
    finally:
        try:
            retriever.close()
        finally:
            db.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Discover and query daily habits and routines")
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("ingest", help="Ingest persistent-event JSON files and rebuild profiles")
    ingest.add_argument("inputs", nargs="+", help="JSON files, quoted glob patterns, or input directories")
    ingest.add_argument("--embedding-model", default=None, help="Embedding model ID; default EMBEDDING_MODEL or qwen3-embedding-8b")
    ingest.add_argument("--embedding-provider", default="openai-compatible")
    ingest.add_argument("--embedding-base-url", default=None, help="Override EMBEDDING_BASE_URL")
    ingest.add_argument("--embedding-dim", type=int, default=None, help="Optional output dimensions; otherwise inferred from the model")
    ingest.add_argument("--habit-eps", type=float, default=0.5)
    ingest.add_argument("--habit-min-samples", type=int, default=5)
    ingest.add_argument("--routine-eps", type=float, default=0.5)
    ingest.add_argument("--routine-min-samples", type=int, default=5)
    ingest.add_argument("--min-days", type=int, default=2)
    ingest.add_argument("--temporal-weight", type=float, default=0.5)
    ingest.add_argument("--max-sequence-length", type=int, default=5, help="Maximum activities per subsequence; 0 means all lengths")
    ingest.add_argument("--max-routine-candidates", type=int, default=2000, help="Per-participant DTW safety limit; 0 disables it")
    search = commands.add_parser("search", help="Search activity, habit, and routine channels")
    search.add_argument("query")
    search.add_argument("--channel", choices=["activity", "habit", "routine", "all"], default="all")
    search.add_argument("--day", help="Recording day; requires --channel activity")
    search.add_argument("--limit", type=int, default=5)
    ask = commands.add_parser("ask", help="Answer a question using graph retrieval tools")
    ask.add_argument("question")
    ask.add_argument("--max-steps", type=int, default=20)
    show = commands.add_parser("show", help="Inspect a node and traverse to its evidence")
    show.add_argument("channel", choices=["activity", "habit", "routine"])
    show.add_argument("node_id", type=int)
    show.add_argument("--limit", type=int, default=20)
    show.add_argument("--offset", type=int, default=0)
    timeline = commands.add_parser("timeline", help="Follow activities before or after an event")
    timeline.add_argument("event_id", type=int)
    timeline.add_argument("--direction", choices=["before", "after"], default="after")
    timeline.add_argument("--limit", type=int, default=5)
    stats = commands.add_parser("stats", help="Show graph counts and saved discovery configuration")
    for command in (ingest, search, ask, show, timeline, stats):
        command.add_argument("--db", default="data/routine_graph.db")
        command.add_argument("--index", default="data/events.faiss", help="Activity index path; profile index paths are derived")
    for command in (search, ask, show, timeline):
        command.add_argument("--participant", default=None)
    for command in (ingest, ask):
        command.add_argument("--chat-model", default=os.getenv("ROUTINE_CHAT_MODEL", "google/gemini-3.6-flash"))
        command.add_argument("--chat-provider", default=os.getenv("ROUTINE_CHAT_PROVIDER", "openrouter"))
    return parser


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "ingest":
            paths = resolve_inputs(args.inputs)
            with IngestionPipeline(db_path=args.db, index_path=args.index, embedding_model=args.embedding_model,
                embedding_provider=args.embedding_provider, embedding_dim=args.embedding_dim,
                embedding_base_url=args.embedding_base_url, summarize_model=args.chat_model,
                summarize_model_provider=args.chat_provider, habit_eps=args.habit_eps, habit_min_samples=args.habit_min_samples,
                routine_eps=args.routine_eps, routine_min_samples=args.routine_min_samples, min_days=args.min_days,
                temporal_weight=args.temporal_weight, max_sequence_length=args.max_sequence_length,
                max_routine_candidates=args.max_routine_candidates) as pipeline:
                events, sequences = pipeline.run(paths)
                result = {"events_ingested": events, "stored_subsequences": sequences,
                          "habits": len(pipeline.db.get_habits()), "routines": len(pipeline.db.get_routines())}
        else:
            with open_retriever(args) as retriever:
                if args.command == "search":
                    if args.day is not None and args.channel != "activity":
                        raise ValueError("--day requires --channel activity")
                    channels = ["activity", "habit", "routine"] if args.channel == "all" else [args.channel]
                    result = {channel: retriever.search(channel, args.query, participant=args.participant,
                              day=args.day, limit=args.limit) for channel in channels}
                elif args.command == "ask":
                    result = QueryAgent(retriever, args.chat_model, args.chat_provider, max_steps=args.max_steps).ask(args.question, args.participant)
                elif args.command == "show":
                    result = retriever.inspect(args.channel, args.node_id, participant=args.participant, limit=args.limit, offset=args.offset)
                elif args.command == "timeline":
                    result = retriever.follow(args.event_id, participant=args.participant, direction=args.direction, limit=args.limit)
                else:
                    db = retriever.db
                    events = db.get_events()
                    result = {"activities": len(events), "subsequences": len(db.get_sequences()), "habits": len(db.get_habits()),
                              "routines": len(db.get_routines()), "participants": sorted({row[1] for row in events}),
                              "discovery": db.setting("discovery"), "embedding": db.setting("embedding_configuration")}
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except httpx.HTTPStatusError as error:
        parser.exit(1, f"API request failed with HTTP {error.response.status_code}; check your endpoint, model, and environment key.\n")
    except (ValueError, OSError, httpx.RequestError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main(sys.argv[1:])
