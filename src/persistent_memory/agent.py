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

from dataset.hd_epic import Clip #TODO Move Clip class inside hd_epic to utilities module for using as interface for every dataset 

load_dotenv()

#TODO add system prompt
SYSTEM_PROMPT = ""

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
        Ongoing Event No.{idx}\n"
        Activity label: {activity["label"]}
        Description: {activity["description"]}
        start_time: {activity["start_time"]}
        ----------------------------------------\n
        """

        #append the ongoing activity context to the overall context string
        ongoing_activities_context += body

    clip = state["video"] #extract Clip obj from state (input from dataset)

    video_input_context = f"""current chunk video:
    subject: {clip["name"]}
    day: {clip["day"]}
    timestamp_start: {clip["timestamp_start"]}
    timestamp_end: {clip["timestamp_end"]}
    """

    #preprocess video to base64 for LLM to process
    video_path = Path(clip["clip_path"])
    video_bytes = video_path.read_bytes()
    video_base64 = base64.b64encode(video_bytes)

    #construct message systempompt + context + video
    system_message = SystemMessage(SYSTEM_PROMPT)
    human_message = HumanMessage(content=[
        {"type": "text", "text": video_input_context },
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
            "ended_activities": state["ended_activities"].extend(response.add_ended_activities),
            "ll_calls": state["llm_calls"] +1}

def get_graph():
    workflow = StateGraph(CarryOnState)
    workflow.add_node(event_extraction)
    workflow.add_edge(START, "event_extraction")
    workflow.add_edge("event_extraction", END)

    checkpointer = InMemorySaver()
    graph = workflow.compile(checkpointer=checkpointer)
    config: RunnableConfig = {"configurable": {"thread_id": 1}} # use 1 as a thread ID because we wil have only one thread anyway
    return graph, config #don't forget to call graph with the config