import json
from tqdm import tqdm
from dataset.hd_epic import HDepicDataset
from persistent_memory.agent import get_graph, get_config
from collections import defaultdict
import argparse
import threading

def list_unique_participants(dataset: HDepicDataset):
    participants = [] #participant list

    for clip in dataset:
        if not clip.name in participants:
            participants.append(clip.name)
    return participants


def group_by_participant_and_day(dataset: HDepicDataset):
    participant_day_video_pairs = defaultdict(list)
    for clip in dataset:
        participant_day_video_pairs[f"{clip.name}_{clip.day}"].append(clip)

    return participant_day_video_pairs


#task that process one day of video
def task(graph, config_str: str, clips):
    config = get_config(config_str)

    for idx in tqdm(range(len(dataset)), desc=f"Extracting events for {config_str}"):
        try:
            if idx == 0:
                graph.invoke({
                    "video": clips[idx],
                    "on_going_activities": [],
                    "ended_activities": [],
                    "llm_calls": 0 #input Clip obj to the grah
                }, config=config)
            else:
                graph.invoke({
                    "video": clips[idx],
                }, config=config)
        except Exception as e:
                print("Can't extract video no.", idx)
                print(e)

    final_state = graph.get_state(config)    
    final_state = final_state.values
    print(final_state)

    json_content_ended_activity = []
    
    for activity in final_state["ended_activities"]:
        content = {
            "label": activity["label"],
            "description": activity["description"],
            "start_time": activity["start_time"],
            "end_time": activity["end_time"],
        }
        json_content_ended_activity.append(content)
    
    with open(f"persistent_event_{config_str}.json", "w", encoding="utf-8") as f:
        json.dump(json_content_ended_activity, f, indent=4)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Persistent Video Extraction")
    parser.add_argument("--dataset", help="Specify dataset path")
    parser.add_argument("--output", help="Specify output path")
    parser.add_argument("--size", help="Specify chunk duration")
    # parser.add_argument("-p", help="specify participant number to be processed")

    args = parser.parse_args()

    dataset = HDepicDataset(args.dataset, args.output, args.size)

    participants = list_unique_participants(dataset)
    clip_dict = group_by_participant_and_day(dataset)
    graph = get_graph()
    #construct thread
    threads = []
    for k, v in clip_dict.items():
        thread = threading.Thread(target=task, args=(graph, k, v))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()
    
        
    

    