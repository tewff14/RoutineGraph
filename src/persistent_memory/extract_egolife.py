"""Resumable, resource-bounded raw EgoLife extraction through a local vLLM API."""

import argparse
import base64
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import tempfile
from threading import BoundedSemaphore, Event, Lock
import time

import httpx
from pydantic import BaseModel, ConfigDict, Field

from persistent_memory.agent import SYSTEM_PROMPT
from routine_graph.models import clock_seconds

CLOCK = r"^(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d$"
# The full system prompt sent with every window; its hash is part of the saved settings, so changing it requires a new output directory.
EXTRACTION_PROMPT = SYSTEM_PROMPT.replace("(about 30 seconds each, low frame rate, with audio)",
                                          "(up to two minutes each, sampled video frames, with audio when available)") + (
    "\nThe input video is deliberately sampled. Infer routine-level activities using the audio and visible evidence; do not invent unobserved atomic actions. Each response must preserve EVERY previous ongoing label and start_time in exactly one output list. Use the supplied absolute HH:MM:SS window timestamps, not elapsed video timestamps."
    "\n\n## Re-check every ongoing activity against THIS chunk\n"
    "The ongoing list only says what the subject was doing before this chunk; it is not evidence that they still are. "
    "For each ongoing activity, keep it in on_going_activities only if this chunk's frames or audio show the subject still doing it. "
    "If this chunk shows the subject doing a different activity for more than about a minute (different place, objects, people or task), "
    "end the ongoing activity at the moment of the switch and add the new activity. "
    "Rule 3 (UNCERTAIN) applies only when the chunk is genuinely unreadable; if you can see what the subject is doing and it is something else, the previous activity has ended."
    "\n\n## Keep the timeline continuous\n"
    "Whenever you end an activity, decide what the subject does next and add it as a new activity starting at that moment. "
    "Using a phone, talking with others, getting dressed, washing up, eating and moving between places are activities too. "
    "A change of task is a new activity even in the same room. "
    "Leave on_going_activities empty only if the footage is genuinely unreadable or the subject is doing nothing identifiable."
    "\n\n## Label and description must agree\n"
    "Each description describes the labelled activity itself. Do not describe what the subject did after it ended; that belongs in the next activity.")


class Ongoing(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str = Field(min_length=1, max_length=160)
    description: str = Field(min_length=1, max_length=1200)
    start_time: str = Field(pattern=CLOCK)


class Ended(Ongoing):
    end_time: str = Field(pattern=CLOCK)


class ChunkActivities(BaseModel):
    model_config = ConfigDict(extra="forbid")
    on_going_activities: list[Ongoing] = Field(max_length=3)
    add_ended_activities: list[Ended]


def clock(value: float) -> str:
    seconds = min(86399, max(0, int(round(value))))
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


@dataclass(frozen=True)
class Source:
    path: str
    day: str
    start: int
    duration: float
    size: int
    mtime_ns: int


@dataclass(frozen=True)
class Batch:
    day: str
    index: int
    sources: tuple[Source, ...]

    @property
    def start(self):
        return self.sources[0].start

    @property
    def end(self):
        return self.sources[-1].start + self.sources[-1].duration


def probe(path: Path) -> dict:
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type",
                             "-of", "json", str(path)], capture_output=True, text=True, check=True, timeout=30)
    return json.loads(result.stdout)


def inventory(dataset: Path, participant: str, output: Path) -> list[Source]:
    """Reuse only timing metadata; every inference reads pixels/audio from original files."""
    hints = {}
    cache = dataset / "processed" / "dataset.json"
    if cache.is_file():
        for row in json.loads(cache.read_text()):
            if row.get("name") == participant:
                name = Path(row["clip_path"]).stem
                if name.endswith("_0"):
                    duration = (clock_seconds(row["timestamp_end"]) - clock_seconds(row["timestamp_start"])) % 86400
                    if duration > 0:
                        hints[name.removesuffix("_0")] = duration
    sources = []
    for directory in sorted((dataset / participant).glob("DAY*"), key=lambda p: int(p.name.removeprefix("DAY"))):
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.mp4")):
            if path.name.startswith("."):
                continue
            match = re.fullmatch(r"(DAY\d+)_" + re.escape(participant) + r"_(\d{6})\d{2}", path.stem)
            if not match or match[1] != directory.name:
                raise ValueError(f"Unexpected raw EgoLife filename: {path}")
            timestamp = match[2]
            start = clock_seconds(f"{timestamp[:2]}:{timestamp[2:4]}:{timestamp[4:]}")
            stat = path.stat()
            duration = hints.get(path.stem)
            if duration is None:
                duration = float(probe(path)["format"]["duration"])
            sources.append(Source(str(path.resolve()), directory.name, start, duration, stat.st_size, stat.st_mtime_ns))
    if not sources:
        raise ValueError(f"No raw videos found for {participant}")
    atomic_json(output / "source_manifest.json", [asdict(source) for source in sources])
    return sources


def build_batches(sources: list[Source], seconds: float = 120, gap: float = 2) -> dict[str, list[Batch]]:
    groups = {}
    for source in sorted(sources, key=lambda item: (int(item.day.removeprefix("DAY")), item.start, item.path)):
        batches = groups.setdefault(source.day, [])
        if batches:
            last = batches[-1]
            delta = source.start - last.end
            if -2 <= delta <= gap and source.start + source.duration - last.start <= seconds + 1:
                batches[-1] = Batch(source.day, last.index, (*last.sources, source))
                continue
        batches.append(Batch(source.day, len(batches), (source,)))
    return groups


def preprocess(batch: Batch, directory: Path, fps: float, size: int) -> tuple[Path, Path | None, float]:
    directory.mkdir(parents=True, exist_ok=True)
    for source in batch.sources:
        stat = Path(source.path).stat()
        if (stat.st_size, stat.st_mtime_ns) != (source.size, source.mtime_ns):
            raise ValueError(f"Raw input changed during extraction: {source.path}")
    listing = directory / "sources.ffconcat"
    listing.write_text("ffconcat version 1.0\n" + "".join("file '" + item.path.replace("'", "'\\''") + "'\n" for item in batch.sources))
    video, audio = directory / "video.mp4", directory / "audio.mp3"
    last_probe = probe(Path(batch.sources[-1].path))
    end = batch.sources[-1].start + float(last_probe["format"]["duration"])
    has_audio = any(stream["codec_type"] == "audio" for stream in last_probe["streams"])
    # A window shorter than one frame interval would otherwise encode zero frames, which the server cannot open.
    keep_frame = ":eof_action=pass" if end - batch.start < 1 / fps else ""
    command = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-hwaccel", "auto", "-threads", "1",
               "-f", "concat", "-safe", "0", "-i", str(listing), "-map", "0:v:0", "-an",
               "-vf", f"fps={fps}{keep_frame},scale={size}:{size}:force_original_aspect_ratio=decrease:force_divisible_by=2",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "30", "-threads", "1",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video)]
    if has_audio:
        command += ["-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000", "-b:a", "32k",
                    "-c:a", "libmp3lame", "-threads", "1", str(audio)]
    try:
        subprocess.run(command, capture_output=True, text=True, check=True, timeout=180)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"Preprocessing failed for {batch.day} window {batch.index}: {error.stderr[:1500]}") from error
    return video, audio if has_audio else None, end


def validate_response(result: ChunkActivities, ongoing: list[dict], start: float, end: float):
    previous = {(item["label"], item["start_time"]) for item in ongoing}
    seen, carried, dropped = set(), [], []
    for item in [*result.on_going_activities, *result.add_ended_activities]:
        key = (item.label, item.start_time)
        if key in seen:
            raise ValueError(f"Duplicate activity {item.label!r} starting {item.start_time}: include this key exactly once, either ongoing or ended")
        seen.add(key)
        activity_start = clock_seconds(item.start_time)
        if key not in previous and math.ceil(end) + 2 < activity_start <= 2 * math.ceil(end) - start:
            # Starts shortly after the footage ends, so this window cannot observe it; the next window records it from its own footage.
            # Starts later than one window length past the end are timestamp errors and still fail below.
            dropped.append(item)
            continue
        if key not in previous and not start - 2 <= activity_start <= math.ceil(end) + 2:
            raise ValueError(f"New activity {item.label!r} starts {item.start_time} outside {clock(start)}–{clock(math.ceil(end))}")
        if isinstance(item, Ended):
            activity_end = clock_seconds(item.end_time)
            if activity_end > math.ceil(end) + 2 and activity_start <= math.ceil(end) + 2:
                # Footage stops at the window end, so an activity ending later is still ongoing; the next window ends it.
                carried.append(item)
                continue
            if activity_end < activity_start or not start - 2 <= activity_end <= math.ceil(end) + 2:
                raise ValueError(f"Activity {item.label!r} ends {item.end_time} outside {clock(start)}–{clock(math.ceil(end))}, or before {item.start_time}")
    if not previous <= seen:
        raise ValueError(f"The response omitted ongoing activities {sorted(previous - seen)}; preserve each existing label and start_time exactly once")
    if dropped:
        result.on_going_activities = [item for item in result.on_going_activities if all(item is not other for other in dropped)]
        result.add_ended_activities = [item for item in result.add_ended_activities if all(item is not other for other in dropped)]
    if carried:
        if len(result.on_going_activities) + len(carried) > 3:
            raise ValueError(f"Activities ending after {clock(math.ceil(end))} are still ongoing, which would exceed 3 ongoing activities; end the others by the window end")
        result.add_ended_activities = [item for item in result.add_ended_activities if all(item is not other for other in carried)]
        result.on_going_activities = [*result.on_going_activities, *(Ongoing(**item.model_dump(exclude={"end_time"})) for item in carried)]


def gemini_request(payload: dict, fps: float | None, thinking_level: str | None = None) -> dict:
    """Translate the OpenAI-style payload into a native Gemini generateContent request (the OpenAI-compatible endpoint rejects video)."""
    def part(item):
        if isinstance(item, str) or item["type"] == "text":
            return {"text": item if isinstance(item, str) else item["text"]}
        if item["type"] == "video_url":
            mime, data = item["video_url"]["url"].removeprefix("data:").split(";base64,", 1)
            return {"inlineData": {"mimeType": mime, "data": data}, **({"videoMetadata": {"fps": fps}} if fps else {})}
        return {"inlineData": {"mimeType": "audio/mp3", "data": item["input_audio"]["data"]}}
    system = [m for m in payload["messages"] if m["role"] == "system"]
    contents = [{"role": "model" if m["role"] == "assistant" else "user",
                 "parts": [part(item) for item in (m["content"] if isinstance(m["content"], list) else [m["content"]])]}
                for m in payload["messages"] if m["role"] != "system"]
    config = {"temperature": payload["temperature"], "maxOutputTokens": 8192,  # room for Gemini's thinking tokens
              "responseMimeType": "application/json", "responseJsonSchema": payload["response_format"]["json_schema"]["schema"]}
    if thinking_level:
        config["thinkingConfig"] = {"thinkingLevel": thinking_level}
    return {"systemInstruction": {"parts": [{"text": m["content"]} for m in system]}, "contents": contents, "generationConfig": config}


class GemmaExtractor:
    """Sends one window to the model: an OpenAI-compatible server (vLLM) or, for Google's API, native Gemini."""
    def __init__(self, model: str, base_url: str, api_key: str, timeout: float = 180, client: httpx.Client | None = None,
                 diagnostic_dir: Path | None = None, fps: float | None = None, thinking_level: str | None = None):
        self.model, self.base_url, self.api_key = model, base_url.rstrip("/"), api_key
        self.client = client or httpx.Client(timeout=timeout)
        self._owns_client = client is None
        self.diagnostic_dir = diagnostic_dir
        self.gemini = "generativelanguage.googleapis.com" in self.base_url
        self.fps, self.thinking_level = fps, thinking_level

    def complete(self, payload: dict) -> tuple[str | None, bool, dict]:
        """Return the reply text, whether it was cut off by the token limit, and token usage."""
        if self.gemini:
            response = self.client.post(f"{self.base_url}/models/{self.model}:generateContent",
                                        headers={"x-goog-api-key": self.api_key}, json=gemini_request(payload, self.fps, self.thinking_level))
        else:
            response = self.client.post(self.base_url + "/chat/completions", headers={"Authorization": f"Bearer {self.api_key}"}, json=payload)
        if response.status_code >= 400:
            raise RuntimeError(f"Model HTTP {response.status_code}: {response.text[:800]}")
        body = response.json()
        if self.gemini:
            if not body.get("candidates"):
                raise RuntimeError(f"Gemini returned no candidate: {json.dumps(body.get('promptFeedback', body))[:800]}")
            candidate = body["candidates"][0]
            reply = "".join(p.get("text", "") for p in candidate.get("content", {}).get("parts", []) if not p.get("thought")) or None
            return reply, candidate.get("finishReason") == "MAX_TOKENS", body.get("usageMetadata", {})
        choice = body["choices"][0]
        return choice["message"]["content"], choice.get("finish_reason") == "length", body.get("usage", {})

    def infer(self, batch: Batch, ongoing: list[dict], video: Path, audio: Path | None, end: float) -> tuple[ChunkActivities, dict]:
        content = [{"type": "text", "text": json.dumps({"subject": "A1_JAKE", "day": batch.day,
            "timestamp_start": clock(batch.start), "timestamp_end": clock(math.ceil(end)),
            "ongoing_activities": ongoing, "audio_provided": audio is not None,
            "source_starts": [clock(source.start) for source in batch.sources]}, ensure_ascii=False)},
            {"type": "video_url", "video_url": {"url": "data:video/mp4;base64," + base64.b64encode(video.read_bytes()).decode()}}]
        if audio is not None:
            content.append({"type": "input_audio", "input_audio": {"data": base64.b64encode(audio.read_bytes()).decode(), "format": "mp3"}})
        payload = {"model": self.model, "messages": [{"role": "system", "content": EXTRACTION_PROMPT}, {"role": "user", "content": content}],
                   "temperature": 0, "max_tokens": 2048,
                   "response_format": {"type": "json_schema", "json_schema": {"name": "ChunkActivities", "schema": ChunkActivities.model_json_schema(), "strict": True}}}
        if self.model.startswith("gemma"):
            payload["chat_template_kwargs"] = {"enable_thinking": False}  # vLLM Gemma chat-template option; hosted APIs reject it
        last_error = None
        for attempt in range(3):
            reply = None
            try:
                reply, truncated, usage = self.complete(payload)
                if truncated:
                    raise ValueError("Model output exceeded its token limit")
                if reply is None:
                    raise ValueError("Model returned an empty response")
                result = ChunkActivities.model_validate_json(reply)
                validate_response(result, ongoing, batch.start, end)
                return result, usage
            except (ValueError, httpx.RequestError, RuntimeError) as error:
                last_error = error
                if self.diagnostic_dir is not None:
                    atomic_json(self.diagnostic_dir / f"{batch.day}-{batch.index}-{time.time_ns()}.json",
                        {"day": batch.day, "batch_index": batch.index, "attempt": attempt + 1,
                         "previous_ongoing": ongoing, "error": str(error)[:1500], "model_reply": reply})
                if attempt == 2:
                    raise
                if reply:
                    payload["messages"].append({"role": "assistant", "content": reply})
                payload["messages"].append({"role": "user", "content":
                    "Correct your previous JSON response. Validation error: " + str(error)[:800] +
                    ". Return the corrected full response. Preserve each previous ongoing activity exactly once; do not repeat an activity in both lists."})
                time.sleep(2 ** attempt)
        raise RuntimeError(str(last_error))

    def close(self):
        if self._owns_client:
            self.client.close()


STORE_SCHEMA = """
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS day_state (day TEXT PRIMARY KEY, next_batch INTEGER NOT NULL DEFAULT 0,
    ongoing TEXT NOT NULL DEFAULT '[]', last_end REAL, complete INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS batches (day TEXT, batch_index INTEGER, sources TEXT NOT NULL, source_count INTEGER NOT NULL,
    start REAL NOT NULL, end REAL NOT NULL, response TEXT NOT NULL, usage TEXT NOT NULL,
    elapsed_seconds REAL NOT NULL, media_bytes INTEGER NOT NULL, PRIMARY KEY(day,batch_index));
CREATE TABLE IF NOT EXISTS events (event_id INTEGER PRIMARY KEY, day TEXT NOT NULL,
    label TEXT NOT NULL, start_time TEXT NOT NULL, end_time TEXT NOT NULL, payload TEXT NOT NULL,
    UNIQUE(day,label,start_time,end_time));
"""


class Store:
    def __init__(self, path: Path):
        self.connection = sqlite3.connect(path, timeout=30)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.executescript(STORE_SCHEMA)

    def events(self, day: str) -> list[dict]:
        return [json.loads(row[0]) for row in self.connection.execute("SELECT payload FROM events WHERE day=? ORDER BY start_time,event_id", (day,))]

    def add_events(self, day: str, events: list[dict]):
        self.connection.executemany("INSERT OR IGNORE INTO events(day,label,start_time,end_time,payload) VALUES (?,?,?,?,?)",
            [(day, item["label"], item["start_time"], item["end_time"], json.dumps(item, ensure_ascii=False)) for item in events])

    def close(self):
        self.connection.close()


class ExtractionRunner:
    def __init__(self, output: Path, groups: dict[str, list[Batch]], configuration: dict, *, fps: float = .25,
                 size: int = 320, workers: int = 2, model: str = "gemma4-12b-it", base_url: str = "http://localhost:8000/v1",
                 api_key: str = "EMPTY", publish_dir: Path | None = None, max_batches: int = 0, infer_factory=None, media_builder=None,
                 thinking_level: str | None = None):
        self.output, self.groups, self.configuration = Path(output), groups, configuration
        self.output.mkdir(parents=True, exist_ok=True)
        self.path = self.output / "extraction.sqlite"
        self.fps, self.size, self.workers = fps, size, workers
        self.publish_dir, self.max_batches = publish_dir, max_batches
        self.infer_factory = infer_factory or (lambda: GemmaExtractor(model, base_url, api_key, diagnostic_dir=self.output / "failed_responses", fps=fps,
                                                                      thinking_level=thinking_level))
        self.media_builder = media_builder or (lambda batch, directory: preprocess(batch, directory, fps, size))
        self.stop = Event()
        self.media_slots = BoundedSemaphore(2)
        self.limit_lock, self.new_batches = Lock(), 0
        store = Store(self.path)
        try:
            existing = store.connection.execute("SELECT value FROM settings WHERE key='configuration'").fetchone()
            encoded = json.dumps(configuration, sort_keys=True)
            if existing and existing[0] != encoded:
                raise ValueError("Extraction settings or raw inputs changed; use a new output directory")
            with store.connection:
                store.connection.execute("INSERT OR IGNORE INTO settings VALUES ('configuration',?)", (encoded,))
                store.connection.executemany("INSERT OR IGNORE INTO day_state(day) VALUES (?)", [(day,) for day in groups])
        finally:
            store.close()

    def status(self) -> dict:
        store = Store(self.path)
        try:
            complete = [row[0] for row in store.connection.execute("SELECT day FROM day_state WHERE complete=1 ORDER BY day")]
            done, media_bytes, seconds = store.connection.execute("SELECT COALESCE(SUM(source_count),0),COALESCE(SUM(media_bytes),0),COALESCE(SUM(end-start),0) FROM batches").fetchone()
            batches = store.connection.execute("SELECT COUNT(*) FROM batches").fetchone()[0]
            events = store.connection.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            return {"status": "complete" if len(complete) == len(self.groups) else "running", "completed_days": complete,
                    "expected_days": len(self.groups), "source_clips_done": done,
                    "source_clips_expected": sum(len(batch.sources) for batches in self.groups.values() for batch in batches),
                    "batches_done": batches, "batches_expected": sum(len(batches) for batches in self.groups.values()),
                    "events": events, "video_hours_observed": round(seconds / 3600, 3), "uploaded_media_bytes": media_bytes}
        finally:
            store.close()

    def export(self, store: Store, day: str, complete: bool = False):
        events = store.events(day)
        atomic_json(self.output / "events" / f"persistent_event_A1_JAKE_{day}.json", events)
        if complete and self.publish_dir is not None:
            atomic_json(self.publish_dir / f"persistent_event_A1_JAKE_{day}.json", events)

    @staticmethod
    def close_observations(ongoing: list[dict], end: float, reason: str) -> list[dict]:
        return [{**item, "end_time": clock(end), "end_observed": False, "end_reason": reason} for item in ongoing]

    def process_day(self, day: str):
        store, extractor = Store(self.path), self.infer_factory()
        active_index = None
        try:
            index, encoded, last_end, complete = store.connection.execute("SELECT next_batch,ongoing,last_end,complete FROM day_state WHERE day=?", (day,)).fetchone()
            if complete:
                self.export(store, day, True)
                return
            ongoing = json.loads(encoded)
            self.export(store, day)
            for batch in self.groups[day][index:]:
                active_index = batch.index
                if self.stop.is_set():
                    return
                with self.limit_lock:
                    if self.max_batches and self.new_batches >= self.max_batches:
                        return
                    self.new_batches += 1
                if last_end is not None and batch.start - last_end > 2 and ongoing:
                    with store.connection:
                        store.add_events(day, self.close_observations(ongoing, last_end, "recording_gap"))
                        store.connection.execute("UPDATE day_state SET ongoing='[]' WHERE day=?", (day,))
                    ongoing = []
                started = time.monotonic()
                with tempfile.TemporaryDirectory(prefix=f"{day}-", dir=self.output) as directory:
                    with self.media_slots:
                        video, audio, end = self.media_builder(batch, Path(directory))
                    media_ready = time.monotonic()
                    result, usage = extractor.infer(batch, ongoing, video, audio, end)
                    usage = {**usage, "pipeline_timing": {"preprocessing_seconds": round(media_ready - started, 3),
                                                        "inference_seconds": round(time.monotonic() - media_ready, 3)}}
                    added = [item.model_dump() for item in result.add_ended_activities]
                    ongoing = [item.model_dump() for item in result.on_going_activities]
                    media_bytes = video.stat().st_size + (audio.stat().st_size if audio else 0)
                with store.connection:
                    store.add_events(day, added)
                    store.connection.execute("INSERT INTO batches VALUES (?,?,?,?,?,?,?,?,?,?)", (day, batch.index,
                        json.dumps([source.path for source in batch.sources]), len(batch.sources), batch.start, end,
                        result.model_dump_json(), json.dumps(usage), time.monotonic() - started, media_bytes))
                    store.connection.execute("UPDATE day_state SET next_batch=?,ongoing=?,last_end=? WHERE day=?",
                                             (batch.index + 1, json.dumps(ongoing), end, day))
                last_end = end
                self.export(store, day)
            with store.connection:
                store.add_events(day, self.close_observations(ongoing, last_end, "end_of_recording"))
                store.connection.execute("UPDATE day_state SET complete=1,ongoing='[]' WHERE day=?", (day,))
            self.export(store, day, True)
        except Exception as error:
            self.stop.set()
            raise RuntimeError(f"{day} window {active_index}: {error}") from error
        except BaseException:
            self.stop.set()
            raise
        finally:
            extractor.close()
            store.close()

    def run(self) -> dict:
        with open(self.output / "run.lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            atomic_json(self.output / "configuration.json", self.configuration)
            started = time.monotonic()
            try:
                with ThreadPoolExecutor(max_workers=self.workers) as executor:
                    days = sorted(self.groups, key=lambda day: len(self.groups[day]), reverse=True)
                    pending = {executor.submit(self.process_day, day) for day in days}
                    while pending:
                        finished, pending = wait(pending, timeout=15, return_when=FIRST_COMPLETED)
                        for future in finished:
                            future.result()
                        status = self.status()
                        status["elapsed_seconds_this_run"] = round(time.monotonic() - started, 1)
                        atomic_json(self.output / "progress.json", status)
                        print(json.dumps(status), flush=True)
            except BaseException as error:
                self.stop.set()
                status = self.status()
                status.update(status="stopped", error_type=type(error).__name__, error_message=str(error)[:1000])
                atomic_json(self.output / "progress.json", status)
                raise
            status = self.status()
            if status["status"] != "complete":
                status["status"] = "partial"
            if status["status"] == "complete" and status["source_clips_done"] != status["source_clips_expected"]:
                raise AssertionError("Extraction completed without accounting for every raw source clip")
            atomic_json(self.output / "progress.json", status)
            return status


def remove_previous_results(dataset: Path, output: Path, old_directory: Path | None = None):
    """Remove only the earlier A1_JAKE outputs explicitly selected by this run."""
    marker = output / "old_results_removed.json"
    if marker.exists() and json.loads(marker.read_text()).get("status") == "removed":
        return
    checkpoint = Store(output / "extraction.sqlite")
    try:
        if checkpoint.connection.execute("SELECT COUNT(*) FROM batches").fetchone()[0] < 1:
            raise ValueError("Validate one raw-video window before removing the previous results")
    finally:
        checkpoint.close()
    old_directory = old_directory or Path(__file__).resolve().parents[2] / "data" / "egolife_A1_JAKE_gemma4"
    paths = sorted((dataset / "persistent_events").glob("persistent_event_A1_JAKE_DAY*.json"))
    manifest = {"status": "removing", "requested_at": datetime.now(timezone.utc).isoformat(),
                "event_files": [{"path": str(path.resolve()), "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths],
                "workspace_results": str(old_directory)}
    atomic_json(marker, manifest)
    if old_directory.is_symlink():
        raise ValueError("Refusing to remove a symlink as the previous result directory")
    for path in paths:
        path.unlink()
    if old_directory.exists():
        shutil.rmtree(old_directory)
    manifest.update(status="removed", completed_at=datetime.now(timezone.utc).isoformat())
    atomic_json(marker, manifest)
    print(json.dumps({"phase": "removed_previous_results", "old_event_files_removed": len(paths),
                      "old_workspace_results_removed": str(old_directory)}), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="/Volumes/TEW_SSD/dataset/EgoLife")
    parser.add_argument("--participant", default="A1_JAKE", choices=["A1_JAKE"])
    parser.add_argument("--output", default="data/egolife_A1_JAKE_raw_gemma4")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="gemma4-12b-it")
    parser.add_argument("--fps", type=float, default=.25)
    parser.add_argument("--size", type=int, default=320)
    parser.add_argument("--batch-seconds", type=float, default=120)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--publish-dir")
    parser.add_argument("--thinking-level", choices=["minimal", "low", "medium", "high"], help="Gemini API reasoning level; default is the model's own")
    parser.add_argument("--replace-old-results", action="store_true", help="Remove the earlier partial A1_JAKE event files and generated graph after a successful raw-video pilot")
    args = parser.parse_args(argv)
    if not 0 < args.fps or args.size < 64 or args.batch_seconds < 30 or args.fps * args.batch_seconds > 32:
        parser.error("Use positive fps, size >=64, windows >=30 seconds, and fps * batch-seconds <=32 frames")
    if not 1 <= args.workers <= 7 or args.max_batches < 0:
        parser.error("Use 1..7 workers and a nonnegative batch limit")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    sources = inventory(Path(args.dataset), args.participant, output)
    groups = build_batches(sources, args.batch_seconds)
    configuration = {"participant": args.participant, "model": args.model, "base_url": args.base_url,
                     "fps": args.fps, "size": args.size, "batch_seconds": args.batch_seconds,
                     "audio": {"sample_rate": 16000, "channels": 1, "bitrate": "32k"},
                     "source_manifest_hash": hashlib.sha256(json.dumps([asdict(source) for source in sources], sort_keys=True).encode()).hexdigest(),
                     "prompt_hash": hashlib.sha256(EXTRACTION_PROMPT.encode()).hexdigest(), "schema": "ChunkActivities-v1",
                     **({"thinking_level": args.thinking_level} if args.thinking_level else {})}
    print(json.dumps({"phase": "inventory", "raw_source_clips": len(sources),
                      "batches": {day: len(batches) for day, batches in groups.items()},
                      "expected_video_hours": round(sum(source.duration for source in sources) / 3600, 2)}), flush=True)
    runner = ExtractionRunner(output, groups, configuration, fps=args.fps, size=args.size, workers=args.workers,
        model=args.model, base_url=args.base_url, publish_dir=Path(args.publish_dir) if args.publish_dir else None,
        max_batches=args.max_batches, api_key=os.getenv("EXTRACTION_API_KEY", "EMPTY"), thinking_level=args.thinking_level)
    if args.replace_old_results:
        remove_previous_results(Path(args.dataset), output)
    print(json.dumps(runner.run()), flush=True)


if __name__ == "__main__":
    main()
