from torch.utils.data import Dataset
from pathlib import Path
import subprocess
import tempfile
import os
from pydantic import BaseModel
import re
from datetime import datetime, timedelta

def duration(path:str):
    command = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", path]
    out = subprocess.run(command, capture_output=True, text=True, check=True)
    return float(out.stdout)

def merge_and_split_videos(video_path: list, output_pattern: str, chunk_duration_s: int):
    """Merge videos into a long video by using FFMPEG on path list"""

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
               "-f", "segment",
               "-segment_time", str(chunk_duration_s),
               f"{output_pattern}"]

    subprocess.run(command, check=True)
    os.remove(list_path)

def split_video(video_path: str, output_pattern: str, chunk_duration_s: int):
    """Split a video into n-second clips"""

    #read all video as one stream and split into n-second clips
    command = ["ffmpeg",  
               "-safe", "0", 
               "-i", video_path, 
               "-map", "0", 
               "-c", "copy",
               "-f", "segment",
               "-segment_time", str(chunk_duration_s),
               f"{output_pattern}"]

    subprocess.run(command, check=True)

class Clip(BaseModel):
    name: str
    clip_path: str
    day: str
    timestamp_start = str
    timestamp_end = str

class HDepicDataset(Dataset):
    def __init__(self, dataset_path: str, output_path: str = 'output/hd_epic/', chunk_duration_s: int = 30):
        self.dataset_path = Path(dataset_path)
        self.video_path = Path(dataset_path + "/Videos") 
        self.participants =  [p for p in self.video_path.rglob(f"**/P0*")] #list all availiable participants
        self.participants_raw_video = {} # store pair participant name: video list
        for idx, path in enumerate(self.participants):
            self.participants_raw_video[f"p{idx}"] = list(path.rglob("*.mp4")) #list all mp4 files for each participant


        #merge and split videos for each participant
        for name, paths in self.participants_raw_video:
            # merge_and_split_videos(paths, f"{output_path}/{name}/clip%03d.mp4", chunk_duration_s) #store clips iside the participant's name directory
            for path in paths:
                filename = Path(path).stem
                split_video(path, f"{output_path}/{name}/{filename}_%d.mp4", chunk_duration_s)

        #pair name with corresponding list of clips
        self.clips = []
        for name, _ in self.participants_raw_video:
            for path in list(path.rglob(f"{output_path}/{name}/*.mp4")): #list all clips for each participant
                timestamp_start_text = re.search(r"-[0-9]+_").lstrip("-").rstrip("_") #get timestamp str from filename
                timestamp_start_datetime = datetime.strptime(timestamp_start_text, "%H%M%S").time() #transform timestamp text to datetime
                clip_number = int(re.search(r"_[0-9]+")).rstrip("_") #get number in the sequence of the video's chunks
                timestamp_start = timestamp_start_datetime + timedelta(seconds=chunk_duration_s * clip_number) 
                timestamp_end = timestamp_start + timedelta(seconds=chunk_duration_s)

                #data row of the clip
                clip = Clip(
                    name=name,
                    clip_path=path,
                    day = re.search(r"-[0-9]+-", path),
                    timestamp_start = timestamp_start,
                    timestamp_end = timestamp_end
                ) 

                self.clips.append(clip)

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, index):
        return self.clips[index]








        
        