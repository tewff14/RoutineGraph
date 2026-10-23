from torch.utils.data import Dataset
from pathlib import Path
import subprocess
import tempfile
import os

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
            merge_and_split_videos(paths, f"{output_path}/{name}/clip%03d.mp4", chunk_duration_s) #store clips iside the participant's name directory

        #pair name with corresponding list of clips
        self.participants_clip = {}
        for i in range(self.participants):
            name = f"p{i}"
            self.participants_clip[name] = list(path.rglob(f"{output_path}/{name}/*.mp4")) #list all clips for each participant


    def __len__(self):
        return sum(len(clips) for clips in self.self.participants_clip)

    def __getitem__(self, index):
        name, clips = self.participants_clip[index]
        return name, clips








        
        