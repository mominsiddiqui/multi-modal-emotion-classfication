import os
import cv2
import pandas as pd
import numpy as np
import json
import argparse
from tqdm import tqdm

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
            row["Modality"] = modality[labels[0]]
            row["Vocal Channel"] = vocal_channel[labels[1]]
            row["Emotion"] = emotion[labels[2]]
            row["Emotional intensity"] = emotional_intensity[labels[3]]
            row["Statement"] = statement[labels[4]]
            row["Repetition"] = repetition[labels[5]]
            row["Actor"] = labels[6]
            df = df.append(row, ignore_index=True)
    return df

def saveFrame(df, destination, frame_count, path2cascade):
    os.makedirs(destination, exist_ok=True)
    frames_per_vid = {}
    cascade = cv2.CascadeClassifier(path2cascade)
    prev = ()
    for x in tqdm(range(df.shape[0])):
        filename = df["ID"].iloc[x]
        filepath = df["Path"].iloc[x]
        os.makedirs("{}/{}".format(destination, filename), exist_ok=True)
        v_cap = cv2.VideoCapture(filepath)
        v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_list= np.linspace(0, v_len-1, frame_count+1, dtype=np.int16)
        for fn in range(v_len):
            success, frame = v_cap.read()
            if success is False:
                continue
            if (fn in frame_list):
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = cascade.detectMultiScale(gray, 1.3, 5)
                if len(faces) == 0:
                    x, y, w, h = prev
                else:
                    x, y, w, h = faces[0]
                    prev = (x, y, w, h)
                frame = frame[y:y + h, x:x + w]
                cv2.imwrite("{}/{}/{}.jpg".format(destination, filename, fn), frame)
        v_cap.release()
        frames_per_vid[filename] = v_len
    print("Images have been saved at {} !".format(destination))
    return frames_per_vid


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description ="Extract Frames from Videos")
    parser.add_argument('--src', type=str, default="audio-video", help="Source of Video Files")
    parser.add_argument('--fc', type=int, default=25, help="Number of frames to extract")
    args = parser.parse_args()
    src = args.src
    frame_count = args.fc
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
    frames_train = saveFrame(Train, "frames", frame_count, "haarcascade_frontalface_default.xml")
    frames_valid = saveFrame(Valid, "frames", frame_count, "haarcascade_frontalface_default.xml")





