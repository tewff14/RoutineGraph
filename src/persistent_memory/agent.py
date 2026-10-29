from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig
from typing_extensions import TypedDict
from typing import Annotated
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import base64
from langchain_core.messages import SystemMessage, HumanMessage
from pathlib import Path

from dataset.utils import Clip

load_dotenv()

SYSTEM_PROMPT = """You are an expert annotator of egocentric (first-person) video. You track the daily-life activities of the camera wearer ("the subject") across a full day of recording. The day is processed as a sequence of consecutive short video chunks (about 30 seconds each, low frame rate, with audio). You see ONE chunk at a time, together with the list of activities that were still ongoing at the end of the previous chunk. Your job is to update that list so that, chained together, the chunks produce an accurate timeline of the subject's day.

## Input
- Chunk metadata: subject, day, timestamp_start and timestamp_end (HH:MM:SS, wall-clock time of the first and last moment of this chunk).
- Ongoing activities from the previous chunk: label, description, start_time. This list may be empty (e.g. at the start of the day).
- The video chunk itself, including audio.

## What counts as an activity
An activity is a goal-directed, routine-level thing the subject is doing that normally lasts from about one minute to a few hours, for example: "cooking breakfast", "washing dishes", "eating lunch", "working on laptop", "assembling furniture", "having a conversation with others", "watching TV", "grocery shopping", "commuting by car", "doing laundry", "cleaning the kitchen".
- NOT an activity: atomic actions that are only steps of a larger activity (opening a fridge, picking up a cup, cutting one onion, walking to the table). Fold these into the activity they serve.
- NOT an activity: whole-day or vague states ("being at home", "daily life", "morning routine").
- Only describe what the SUBJECT is doing. Other people's actions matter only if the subject takes part (e.g. talking with them, cooking together).
- The subject can do at most two or three activities at the same time, and only when they clearly overlap (e.g. "eating lunch" while "having a conversation with others"). Prefer one main activity.

## How to update the list
Go through every ongoing activity from the previous chunk, then look for new ones:

1. CONTINUES: the subject is still doing it in this chunk (even with short pauses, interruptions under about a minute, or partial occlusion). Keep it in on_going_activities with EXACTLY the same label and the SAME start_time. You may refine the description with new information, but keep it about the same activity.
2. ENDED: there is clear evidence that the subject stopped it (left the place, put the tools away, finished eating, switched to a different activity, said goodbye). Remove it from on_going_activities and add it to add_ended_activities with the SAME label and start_time, and an end_time equal to the moment within this chunk when it stopped. If it had already stopped before this chunk began, use this chunk's timestamp_start as end_time.
3. UNCERTAIN: if the chunk is ambiguous (dark, blurry, camera pointing away) and nothing indicates a switch to another activity, assume the activity CONTINUES. Do not end activities just because one chunk is hard to read, and do not let one brief distraction (checking the phone, answering a short question) end a longer activity.
4. NEW: an activity that is not in the ongoing list starts in this chunk. Add it to on_going_activities with a start_time equal to the moment within this chunk when it began (timestamp_start if it is already in progress when the chunk begins). If a new activity both starts and clearly finishes inside this chunk and lasted long enough to matter, put it directly in add_ended_activities.

## Field rules
- label: short, lowercase verb phrase (2-5 words), e.g. "washing dishes". Never rename an activity that continues; the label is how the activity is tracked across chunks.
- description: 1-2 objective sentences: what the subject is doing, where, with which key objects or people (by role, e.g. "a friend", unless a name is clearly spoken). No guesses about feelings or intentions you cannot see or hear.
- start_time / end_time: absolute wall-clock time in HH:MM:SS. New times must fall within [timestamp_start, timestamp_end] of the current chunk; estimate the moment from the position in the chunk (timestamp_start + seconds into the chunk). Never invent times outside the chunk, and never change the start_time of an existing activity. end_time must not be earlier than start_time.

## Consistency checks before answering
- Every ongoing activity from the input appears exactly once: either in on_going_activities or in add_ended_activities.
- No activity appears in both lists, and there are no duplicates or near-duplicates (e.g. "cooking" and "preparing food" for the same thing).
- Use the audio: speech (who is talking, what about), appliance sounds (water running, microwave, vacuum), and ambient sound (street, store) help identify and confirm activities.
- If nothing meaningful is happening and the ongoing list is empty, return two empty lists.
"""

model = init_chat_model(
    "google/gemini-3.6-flash",
    model_provider="openrouter"
)

#represent an event that is not end yet (no end timestamp)
class OngoingActivity(TypedDict):
    label: Annotated[str, "The name of activity"]
    description: Annotated[str, "Description of the activity"]
    start_time: Annotated[str, "start time of the activity"]

#represent an event that is already ended
class EndedActivity(TypedDict):
    label: Annotated[str, "The name of activity"]
    description: Annotated[str, "Description of the activity"]
    start_time: Annotated[str, "start time of the activity"]
    end_time: Annotated[str, "end timestamp of the activity"]
    # end_time: str = Field(description="end timestamp  of the activity")

#represent structured output of the LLM: 1.ongoing activities list after generation  2. ended activities list after generation
class ActivityOutput(BaseModel):
    on_going_activities: list[OngoingActivity] = Field(description="The list of activities that haven't finished")
    add_ended_activities: list[EndedActivity] = Field(description="The list of activities that are ended in this round of generation")
     
#the class show the state of ongoing activities that will be pass through the graph (adjust by the result from ActivityOutput in each LLM call)
class CarryOnState(TypedDict):
    video: Clip #video path
    on_going_activities: list[OngoingActivity]
    ended_activities: list[EndedActivity]
    llm_calls: int

structured_llm = model.with_structured_output(ActivityOutput)

def event_extraction(state: CarryOnState):
    ongoing_activities = state["on_going_activities"]

    ongoing_activities_context = ""

    for idx, activity in enumerate(ongoing_activities):

        #prepare context for the ongoing event
        body = f"""
        Ongoing Event No.{idx}
        Activity label: {activity["label"]}
        Description: {activity["description"]}
        start_time: {activity["start_time"]}
        ----------------------------------------\n
        """

        #append the ongoing activity context to the overall context string
        ongoing_activities_context += body

    clip = state["video"] #extract Clip obj from state (input from dataset)

    video_input_context = f"""current chunk video:
    subject: {clip.name}
    day: {clip.day}
    timestamp_start: {clip.timestamp_start}
    timestamp_end: {clip.timestamp_end}
    """

    #preprocess video to base64 for LLM to process
    video_path = Path(clip.clip_path)
    video_bytes = video_path.read_bytes()
    video_base64 = base64.b64encode(video_bytes).decode("utf-8")

    #construct message systempompt + context + video
    system_message = SystemMessage(SYSTEM_PROMPT)
    human_message = HumanMessage(content=[
        {"type": "text", "text": video_input_context },
        {"type": "text", "text": "ongoing activities from the previous chunk:\n" + (ongoing_activities_context or "None")},
        {"type": "file",
         "mime_type": "video/mp4",
         "base64": video_base64}
    ])

    response: ActivityOutput = structured_llm.invoke([system_message, human_message])

    # replace state's ongoing activities list with new list from the llm
    # state["on_going_activities"] = response["ongoing_activities"]

    # add ended activities in the round to the state
    # state["ended_activities"].extend(response["add_ended_activities"])

    return {"on_going_activities": response.on_going_activities,
            "ended_activities": state["ended_activities"] + response.add_ended_activities,
            "llm_calls": state["llm_calls"] +1}

def get_graph():
    workflow = StateGraph(CarryOnState)
    workflow.add_node(event_extraction)
    workflow.add_edge(START, "event_extraction")
    workflow.add_edge("event_extraction", END)

    checkpointer = InMemorySaver()
    graph = workflow.compile(checkpointer=checkpointer)
    # config: RunnableConfig = {"configurable": {"thread_id": 1}} # use 1 as a thread ID because we wil have only one thread anyway
    return graph #don't forget to call graph with the config

def get_config(config_unique: str):
    config: RunnableConfig  = {"configurable": {"thread_id": config_unique}}
    return config