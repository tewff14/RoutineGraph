import json
from tqdm import tqdm
from torch.utils.data import Dataset
from dataset.hd_epic import HDepicDataset
from dataset.egolife import EgoLifeDataset
from persistent_memory.agent import get_graph, get_config
from collections import defaultdict
import argparse
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

load_dotenv()

def list_unique_participants(dataset: Dataset):
    participants = [] #participant list

    for clip in dataset:
        if not clip.name in participants:
            participants.append(clip.name)
    return participants


def group_by_participant_and_day(dataset: Dataset):
    participant_day_video_pairs = defaultdict(list)
    for clip in dataset:
        participant_day_video_pairs[f"{clip.name}_{clip.day}"].append(clip)

    #sort clips of each day by time because the graph carries state from one clip to the next
    for clips in participant_day_video_pairs.values():
        clips.sort(key=lambda clip: clip.timestamp_start)

    return participant_day_video_pairs


#task that process one day of video
def task(graph, config_str: str, clips, position: int = 0):
    config = get_config(config_str)

    for idx in tqdm(range(len(clips)), desc=f"Extracting events for {config_str}", position=position):
        try:
            if not graph.get_state(config).values: #initialize state on first successful call of this thread
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

    for activity in final_state.get("ended_activities", []):
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
    parser.add_argument("--dataset_name", choices=["hd_epic", "egolife"], default="egolife", help="Specify dataset type")
    parser.add_argument("--dataset", help="Specify dataset path")
    parser.add_argument("--output", help="Specify output path")
    parser.add_argument("--size", type=int, default=30, help="Specify chunk duration")
    parser.add_argument("--workers", type=int, default=4, help="Specify number of participant-days processed in parallel")
    # parser.add_argument("-p", help="specify participant number to be processed")

    args = parser.parse_args()

    dataset_classes = {"hd_epic": HDepicDataset, "egolife": EgoLifeDataset}
    dataset = dataset_classes[args.dataset_name](args.dataset, args.output, args.size)

    participants = list_unique_participants(dataset)
    clip_dict = group_by_participant_and_day(dataset)
    graph = get_graph()
    #run one thread per participant-day, at most args.workers at the same time (avoid LLM rate limit)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(task, graph, k, v, position): k for position, (k, v) in enumerate(clip_dict.items())}

        for future, k in futures.items():
            try:
                future.result() #re-raise exception from the thread
            except Exception as e:
                print(f"Task {k} failed: {e}")
    
        
    

    