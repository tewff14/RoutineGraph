from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage
from typing_extensions import TypedDict
from typing import Annotated
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

EVENTS_SYSTEM_PROMPT = """You summarize the daily-life activities of one person ("the subject") recorded by an egocentric (first-person) camera.

## Input
A list of events. Each event has an activity label, a description and a start_time / end_time (HH:MM:SS wall-clock time). The events were grouped together by a clustering step because they are similar, so they are usually repeated occurrences of the same habit on different days or at different times of the day.

## Task
Describe the single habit these events represent:
- label: a short, lowercase verb phrase (2-5 words) that names what is common to the events, e.g. "eating lunch", "washing dishes". Prefer the most frequent wording of the input labels when it fits all events.
- description: 2-4 objective sentences: what the subject does, where and with which key objects or people, and when it usually happens (typical time of day and duration, from the timestamps). Mention a notable variation only if several events share it.

## Rules
- Use only the information in the events. Do not invent places, objects, people, feelings or intentions.
- If a few events do not fit the rest, describe the majority and ignore the outliers.
- Times are approximate: describe them as ranges or time of day (e.g. "around 12:00-13:00", "in the evening"), not as exact timestamps.
"""

SEQUENCES_SYSTEM_PROMPT = """You summarize the daily routines of one person ("the subject") recorded by an egocentric (first-person) camera.

## Input
A list of sequences. Each sequence is a run of consecutive events from one day, ordered by start_time; each event has an activity label, a description and a start_time / end_time (HH:MM:SS wall-clock time). The sequences were grouped together by a clustering step because their order of activities is similar, so they are usually the same routine repeated on different days.

## Task
Describe the single routine these sequences represent:
- label: a short, lowercase phrase (2-6 words) that names the routine as a whole, e.g. "morning breakfast routine", "after-dinner cleanup".
- description: 2-5 objective sentences: the typical order of activities (e.g. "cooking breakfast, then eating breakfast, then washing dishes"), when the routine usually happens and how long it takes (from the timestamps), and the key places, objects or people involved.

## Rules
- Use only the information in the sequences. Do not invent places, objects, people, feelings or intentions.
- Describe the order shared by most sequences. Mention a step as optional only if it is missing in several sequences, and ignore steps that appear only once.
- Times are approximate: describe them as ranges or time of day (e.g. "around 07:30-08:30", "in the morning"), not as exact timestamps.
"""

#one event given to the summarizer (same fields as an ended activity in the persistent_event json)
class Event(TypedDict):
    label: Annotated[str, "The name of activity"]
    description: Annotated[str, "Description of the activity"]
    start_time: Annotated[str, "start time of the activity"]
    end_time: Annotated[str, "end timestamp of the activity"]

#structured output of the LLM for both events and sequences
class Summary(BaseModel):
    label: str = Field(description="Short lowercase name of the habit / routine")
    description: str = Field(description="Summary of the habit / routine")

def format_event(event: Event) -> str:
    """One line per event: label (start_time - end_time): description"""
    return f'{event["label"]} ({event["start_time"]} - {event["end_time"]}): {event["description"]}'

class SummarizeAgent:
    def __init__(self, model: str = "google/gemini-3.6-flash", model_provider: str = "openrouter", **kwargs):
        """Initialize the chat model, model / model_provider / kwargs are passed to langchain init_chat_model"""
        self.model = init_chat_model(model, model_provider=model_provider, **kwargs)
        self.structured_llm = self.model.with_structured_output(Summary)

    def summarize_events(self, events: list[Event]) -> Summary:
        """Summarize a list of events (e.g. one habit cluster) into a label + description"""

        events_context = "\n".join(f"Event No.{idx}: {format_event(event)}" for idx, event in enumerate(events))

        system_message = SystemMessage(EVENTS_SYSTEM_PROMPT)
        human_message = HumanMessage("events:\n" + events_context)
        return self.structured_llm.invoke([system_message, human_message])

    def summarize_sequences(self, sequences: list[list[Event]]) -> Summary:
        """Summarize a list of sequences, each a list of events ordered by start_time (e.g. one routine cluster), into a label + description"""

        sequences_context = ""
        for idx, sequence in enumerate(sequences):
            #one block per sequence, its events in order
            body = f"Sequence No.{idx}\n" + "\n".join(f"  {step}. {format_event(event)}" for step, event in enumerate(sequence, 1))
            sequences_context += body + "\n----------------------------------------\n"

        system_message = SystemMessage(SEQUENCES_SYSTEM_PROMPT)
        human_message = HumanMessage("sequences:\n" + sequences_context)
        return self.structured_llm.invoke([system_message, human_message])
