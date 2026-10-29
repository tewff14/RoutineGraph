from torch.utils.data import Dataset
from pathlib import Path
import os
import re
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from dataset.utils import Clip, split_video, duration, save_clip_json, fetch_clip_json

class EgoLifeDataset(Dataset):
    def __init__(self, dataset_path: str, output_path: str = f'output/egolife/processed/', chunk_duration_s: int = 30, num_workers: int = 6):

        self.dataset_path = Path(dataset_path)
        self.participants = sorted([p for p in self.dataset_path.glob("A[0-9]_*") if p.is_dir()]) #list all availiable participants (e.g. A1_JAKE)
        self.participants_raw_video = {} # store pair participant name: video list
        print(self.participants)
        for path in self.participants:
            self.participants_raw_video[path.name] = sorted([p for p in path.glob("DAY*/*.mp4") if not p.name.startswith("._")]) #list all mp4 files of every day for each participant

        #if have json file fetch from the file and skip preprocess pipeline
        if len(list(Path(output_path).glob("dataset.json"))) > 0:
            print("Dataset already processed. Fetch from json...")
            self.clips = fetch_clip_json(f"{output_path}/dataset.json")
            return

        #create output folder if not availiable
        if not Path(output_path).is_dir():
            print(f"Create output path: {output_path}")
            os.makedirs(output_path)

        #split (and downsample) videos for each participant, num_workers ffmpeg processes at the same time
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for name, paths in self.participants_raw_video.items():
                #create clip_path for each name
                if not Path(f"{output_path}/{name}").is_dir():
                    print(f"Create clip path for participant: {name}")
                    os.makedirs(f"{output_path}/{name}")

                for path in paths:
                    filename = path.stem
                    futures.append(executor.submit(split_video, path, f"{output_path}/{name}", f"{filename}_%d.mp4", chunk_duration_s=chunk_duration_s))

            for future in tqdm(futures, desc="Split videos"):
                future.result() #re-raise ffmpeg error from the thread

        #pair name with corresponding list of clips
        self.clips = []

        for name, _ in self.participants_raw_video.items():
            participant_clip_list = sorted([p for p in Path(f"{output_path}/{name}").rglob("*.mp4") if not p.name.startswith("._")])
            for path in participant_clip_list:
                clip_name = Path(path).stem #e.g. DAY1_A1_JAKE_11094208_0
                day, timestamp_text, clip_number = re.fullmatch(r"(DAY[0-9]+)_A[0-9]_[A-Z]+_([0-9]{8})_([0-9]+)", clip_name).groups()
                timestamp_start_datetime = datetime.strptime(timestamp_text[:6], "%H%M%S") #HHMMSSFF -> drop the FF part
                timestamp_start = timestamp_start_datetime + timedelta(seconds=chunk_duration_s * int(clip_number))
                timestamp_end = timestamp_start + timedelta(seconds=int(duration(path)))

                #data row of the clip
                clip = Clip(
                    name=name,
                    clip_path=str(path),
                    day=day,
                    timestamp_start=timestamp_start.strftime("%H:%M:%S"),
                    timestamp_end=timestamp_end.strftime("%H:%M:%S")
                )

                self.clips.append(clip)

        save_clip_json(self.clips, f"{output_path}/dataset.json")

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, index):
        return self.clips[index]
