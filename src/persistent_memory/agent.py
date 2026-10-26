from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig
from typing_extensions import TypedDict
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

#TODO add system prompt
SYSTEM_PROMPT = ""

class ActivityOutput(BaseModel):
    labels: list[str]


model = init_chat_model(
    "google/gemini-3.6-flash",
    model_provider="openrouter"
)

#represent an event that is not end yet (no end timestamp)
class OngoingActivity(TypedDict):
    label:str = Field(description="The name of activity")
    description:str = Field(description="Description of the activity")
    start_time: str = Field(description="start time of the activity")

#represent an event that is already ended
class EndedActivity(TypedDict):
    label:str = Field(description="The name of activity")
    description:str = Field(description="Description of the activity")
    start_time: str = Field(description="start timestamp of the activity")
    end_time: str = Field(description="end timestamp  of the activity")

#represent structured output of the LLM: 1.ongoing activities list after generation  2. ended activities list after generation
class ActivityOutput(TypedDict):
    on_going_activities: list[OngoingActivity] = Field(description="The list of activities that haven't finished")
    add_ended_activities: list[EndedActivity] = Field(description="The list of activities that are ended in this round of generation")
     
#the class show the state of ongoing activities that will be pass through the graph (adjust by the result from ActivityOutput in each LLM call)
class CarryOnState(TypedDict):
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
        Activity labels: {activity["label"]}
        Description: {activity["description"]}
        start_time: {activity["start_time"]}
        ----------------------------------------\n
        """

        #append the ongoing activity context to the overall context string
        ongoing_activities_context += body

    response: ActivityOutput = structured_llm.invoke(
        SYSTEM_PROMPT + ongoing_activities_context #system prompt + ongoing activities context
    )

    # replace state's ongoing activities list with new list from the llm
    state["ongoing_activities"] = response["ongoing_activities"]

    # add ended activities in the round to the state
    state["ended_activities"].extend(response["add_ended_activities"])


def get_graph():
    workflow = StateGraph(CarryOnState)
    workflow.add_node(event_extraction)
    workflow.add_edge(START, "event_extraction")
    workflow.add_edge("event_extraction", END)

    checkpointer = InMemorySaver()
    graph = workflow.compile(checkpointer=checkpointer)
    config: RunnableConfig = {"Configurable": {"thread_id": 1}} # use 1 as a thread ID because we wil have only one thread anyway
    return graph