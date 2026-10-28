import json
from tqdm import tqdm
from dataset.hd_epic import HDepicDataset
from persistent_memory.agent import get_graph
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Persistent Video Extraction")
    parser.add_argument("--dataset", help="Specify dataset path")
    parser.add_argument("--output", help="Specify output path")
    parser.add_argument("--size", help="Specify chunk duration")

    args = parser.parse_args()

    dataset = HDepicDataset(args.dataset, args.output)
    print(dataset)
    graph, config = get_graph()

    #input the video into LLM one by one
    for idx in tqdm(range(len(dataset)), desc="Extracting events"):
        try:
            if idx == 0:
                graph.invoke({
                    "video": dataset[idx],
                    "on_going_activities": [],
                    "ended_activities": [],
                    "llm_calls": 0 #input Clip obj to the grah
                }, config=config)
            else:
                graph.invoke({
                    "video": dataset[idx],
                }, config=config)
        except Exception as e:
            print("Can't extract video no.", idx)
            print(e)


    #---dump graph state into json file---

    #fetch data from graph state
    final_state = graph.get_state(config)    
    final_state = final_state.values
    print(final_state)

    #construct json content
    json_content_ended_activity = []

    for activity in final_state["ended_activities"]:
        content = {
            "label": activity["label"],
            "description": activity["description"],
            "start_time": activity["start_time"],
            "end_time": activity["end_time"],
        }
        json_content_ended_activity.append(content)

    with open("persistent_event.json", "w", encoding="utf-8") as f:
        json.dump(json_content_ended_activity, f, indent=4)
        
    

    