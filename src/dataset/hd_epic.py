from torch.utils.data import Dataset
from pathlib import Path
import subprocess
import tempfile
import os
import re
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from dataset.utils import Clip, split_video, duration, save_clip_json, fetch_clip_json

#TODO remove the merge_and_split function later
def merge_and_split_videos(video_path: list, output_path: str, output_pattern: str, chunk_duration_s: int):
    """Merge videos into a long video by using FFMPEG on path list"""

    #create ouptut folder if not availiable
    if not Path(output_path).is_dir():
        os.mkdir(output_path)

    #store all video paths into one .txt file for letting ffmpeg read
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        for p in video_path:
            f.write(f"file '{p}'\n")

        list_path = f.name

    #read all video as one stream and split into n-second clips
    command = ["ffmpeg", 
               "-y", 
               "-f", "concat", 
               "-safe", "0", 
               "-i", list_path, 
               "-map", "0", 
               "-c", "copy",
               "-reset_timestamps", "1",
               "-f", "segment",
               "-segment_time", str(chunk_duration_s),
               f"{output_path}/{output_pattern}"]

    subprocess.run(command, check=True)
    os.remove(list_path)

class HDepicDataset(Dataset):
    def __init__(self, dataset_path: str, output_path: str = f'output/hd_epic/processed/', chunk_duration_s: int = 30, num_workers: int = 6):

        self.dataset_path = Path(dataset_path)
        self.video_path = Path(dataset_path + "/Videos") 
        self.participants =  [p for p in self.video_path.glob(f"**/P0*") if p.is_dir()] #list all availiable participants
        self.participants_raw_video = {} # store pair participant name: video list
        print(self.participants)
        for idx, path in enumerate(self.participants):
            self.participants_raw_video[f"p{idx+1}"] = [p for p in path.glob("*.mp4") if not p.name.startswith("._")] #list all mp4 files for each participant

        #if have json file fetch from the file and skip preprocess pipeline
        if len(list(Path(output_path).glob("dataset.json"))) > 0:
            print("Dataset already processed. Fetch from json...")
            self.clips = fetch_clip_json(f"{output_path}/dataset.json")
            return

        #create output folder if not availiable
        if not Path(output_path).is_dir():
            print(f"Create output path: {output_path}")
            os.makedirs(output_path)

        #split videos for each participant, num_workers ffmpeg processes at the same time
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for name, paths in self.participants_raw_video.items():
                #create clip_path for each name
                if not Path(f"{output_path}/{name}").is_dir():
                    print(f"Create clip path for participant: {name}")
                    os.mkdir(f"{output_path}/{name}")

                for path in paths:
                    filename = path.stem
                    futures.append(executor.submit(split_video, path, f"{output_path}/{name}", f"{filename}_%d.mp4", chunk_duration_s=chunk_duration_s))

            for future in tqdm(futures, desc="Split videos"):
                future.result() #re-raise ffmpeg error from the thread

        #pair name with corresponding list of clips
        self.clips = []

        for name, _ in self.participants_raw_video.items():
            participant_clip_list = [p for p in Path(f"{output_path}/{name}").rglob("*.mp4") if not p.name.startswith("._")]
            print(participant_clip_list)
            for path in participant_clip_list:
                clip_name = Path(path).stem
                p_name = re.search(r"P0[0-9]", clip_name).group()
                timestamp_start_text = re.search(r"-[0-9]+_", clip_name).group().lstrip("-").rstrip("_") #get timestamp str from filename
                timestamp_start_datetime = datetime.strptime(timestamp_start_text, "%H%M%S") #transform timestamp text to datetime
                clip_number = int(re.search(r"_[0-9]+", clip_name).group().lstrip("_")) #get number in the sequence of the video's chunks
                timestamp_start = timestamp_start_datetime + timedelta(seconds=chunk_duration_s * clip_number) 
                timestamp_end = timestamp_start + timedelta(seconds=int(duration(path)))

                print(clip_name)
                print(p_name)
                print(timestamp_start)
                print(timestamp_end)
                print("\n\n")

                #data row of the clip
                clip = Clip(
                    name=p_name,
                    clip_path=str(path),
                    day = re.search(r"-[0-9]+-", str(path)).group().strip("-"),
                    timestamp_start = timestamp_start.strftime("%H:%M:%S"),
                    timestamp_end = timestamp_end.strftime("%H:%M:%S")
                )

                self.clips.append(clip)


        save_clip_json(self.clips, f"{output_path}/dataset.json")

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, index):
        return self.clips[index]