import os
import pandas as pd
import numpy as np
import json
import argparse
import moviepy.editor as mp

file = open("details.json")
details = json.load(file)
emotion = details["emotion"]
statement = details["statement"]
modality = details["modality"]
emotional_intensity = details["emotional_intensity"] # There is no strong intensity for the 'neutral' emotion
vocal_channel = details["vocal_channel"]
repetition = details["repetition"]


def createDataframe(videos, begin, end):
    df = pd.DataFrame()
    for key in list(videos.keys())[begin:end]:
        for vidpath in videos[key]:
            row = {}
            vidname = vidpath.split("/")[-1][:-4]
            row['ID'] = vidname
            row["Path"] = vidpath
            labels = vidname.split("-")
            if modality[labels[0]] == "video-only":
                continue
            row["Modality"] = modality[labels[0]]
            row["Vocal Channel"] = vocal_channel[labels[1]]
            row["Emotion"] = emotion[labels[2]]
            row["Emotional intensity"] = emotional_intensity[labels[3]]
            row["Statement"] = statement[labels[4]]
            row["Repetition"] = repetition[labels[5]]
            row["Actor"] = labels[6]
            df = df.append(row, ignore_index=True)
    return df

def saveAudio(df, destination):
    os.makedirs(destination, exist_ok=True)
    for x in range(df.shape[0]):
        filename = df["ID"].iloc[x]
        filepath = df["Path"].iloc[x]
        my_clip = mp.VideoFileClip(filepath)
        my_clip.audio.write_audiofile("{}/{}.mp3".format(destination, filename))
    print("Audios have been saved at {} !".format(destination))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description ="Extract Audio from Videos")
    parser.add_argument('--src', type=str, default="audio-video", help="Source of Video Files")
    args = parser.parse_args()
    src = args.src
    videos = {}
    actor_list = os.listdir(src)
    actor_list.sort(key=lambda x: int(x[-2:]))
    for actor in actor_list:
        files = []
        actor_dir = "{}/{}".format(src, actor)
        for file in os.listdir(actor_dir):
            files.append("{}/{}".format(actor_dir, file))
        videos[actor_dir.split("/")[-1]] = files 
    Train = createDataframe(videos, 0, 20)
    Train.to_excel("Train.xlsx", index=False)
    Valid = createDataframe(videos, 20, 25)
    Valid.to_excel("Valid.xlsx", index=False)
    saveAudio(Train, "audio")
    saveAudio(Valid, "audio")





