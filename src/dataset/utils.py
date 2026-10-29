import subprocess
from pydantic import BaseModel
import json

def duration(path:str):
    command = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", path]
    out = subprocess.run(command, capture_output=True, text=True, check=True)
    return float(out.stdout)

def split_video(video_path: str, output_path: str, output_pattern: str, chunk_duration_s: int):
    """Split a video into n-second clips"""

    #read all video as one stream and split into n-second clips
    command = ["ffmpeg",
           "-n",                                     # global: skip if exists
           "-i", video_path,                         # input
           "-map", "0",
           "-vf", "fps=2,scale=448:448",             # output options from here on
           "-c:v", "h264_videotoolbox", "-b:v", "300k",
           "-force_key_frames", "expr:gte(t,n_forced*30)",
           "-c:a", "aac", "-ac", "1", "-b:a", "32k",
           "-f", "segment",
           "-segment_time", str(chunk_duration_s),
           "-reset_timestamps", "1",
           f"{output_path}/{output_pattern}"]


    subprocess.run(command, check=True)


class Clip(BaseModel):
    name: str 
    clip_path: str
    day: str
    timestamp_start: str
    timestamp_end: str

def save_clip_json(clip_list: list[Clip], json_path:str):
    """Save processed dataset information into a json file on output path"""

    #construct json content (dict)
    json_contents = []

    for clip in clip_list:
        content = {
            "name": clip.name,
            "clip_path": clip.clip_path,
            "day": clip.day,
            "timestamp_start": clip.timestamp_start,
            "timestamp_end" : clip.timestamp_end
        }

        json_contents.append(content)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_contents, f, indent=4)

def fetch_clip_json(json_path:str):
    '''Fetch processed dataset information from the json file'''
    with open(json_path, "r", encoding="utf-8") as f:
        clip_list_dict = json.load(f)

    #Construct clip list and obj
    clip_list = []
    for clip in clip_list_dict:
        clip_obj = Clip(
            name=clip["name"],
            clip_path=clip["clip_path"],
            day=clip["day"],
            timestamp_start=clip["timestamp_start"],
            timestamp_end=clip["timestamp_end"]
        )

        clip_list.append(clip_obj)
    return clip_list
