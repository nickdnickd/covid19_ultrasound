import cv2
import os
import json
import numpy as np
import matplotlib.pyplot as plt

DISPLAY_IMG = False

# input path with uncropped videos
path = "soft_tissue_study"

# load json with crop
with open("crop.json", "r") as infile:
    crop_dir = json.load(infile)

for key in crop_dir.keys():
    # I/O paths
    vid_path = os.path.join(path, key)
    save_video_path = vid_path.replace("Videos", "CroppedVideos")
    save_path_dir = os.path.dirname(save_video_path)
    if not os.path.exists(save_path_dir):
        print(f'making dir {save_path_dir}')
        os.makedirs(save_path_dir)

    # get crop and trimming
    start, end = crop_dir[key][1]
    bottom, left, size = crop_dir[key][0]

    print(key, crop_dir[key])

    # read video
    cap = cv2.VideoCapture(vid_path)

    # test whether it's okay
    ret, test = cap.read()
    if test is None:
        print(f"Problem reading file: {vid_path}")
        continue

    # reset cap
    cap = cv2.VideoCapture(vid_path)

    # Image processing
    if cap.get(7) < 2:
        ret, frame = cap.read()
        frame = frame[bottom:bottom + size, left:left + size]
        cv2.imwrite(save_video_path, frame)
        continue

    video_array = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    for i in range(int(end - start)):
        ret, frame = cap.read()
        cropped = frame[bottom:bottom + size, left:left + size]
        if i == 0 and DISPLAY_IMG:
            plt.imshow(cropped)
            plt.show()
        video_array.append(cropped)

    # write video
    print(np.array(video_array).shape)
    video_path = ".".join(save_video_path.split(".")[:-1])
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # XVID
    writer = cv2.VideoWriter(
        video_path + '.mp4', fourcc, cap.get(5), cropped.shape[:2]
    )
    for x in video_array:
        writer.write(x.astype("uint8"))
    writer.release()
