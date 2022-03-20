import pandas as ps
import numpy as np
import cv2  # for capturing videos
import os
import shutil
import argparse

TAKE_CLASSES = ["Abscess", "Cellulitis", "Normal"]


def label_to_dir(lab):
    if lab == "Abs":
        label = "Abscess"
    elif lab == "Cel" or lab == "cel":
        label = "Cellulitis"
    elif lab == "Nor":
        label = "Normal"
    else:
        raise ValueError("Wrong label! " + lab)
    return label


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-imgs', type=str, default="../data/pocus_images")
    parser.add_argument('-out', type=str, default="../data/soft_tissue_study")
    parser.add_argument('-vids', type=str, default="../data/soft_tissue_study")
    parser.add_argument(
        '-fr',
        help="framerate - at how much Hz to sample",
        type=int,
        default=3
    )
    parser.add_argument(
        '-max',
        help="maximum of frames to sample from one video",
        type=int,
        default=30
    )
    args = parser.parse_args()

    FRAMERATE = args.fr
    # saves automatically <FRAMERATE> frames per second
    MAX_FRAMES = args.max
    POCUS_IMAGE_DIR = args.imgs
    POCUS_VIDEO_DIR = args.vids
    out_image_dir = args.out

    if not os.path.exists(out_image_dir):
        print("Output directory depends on input directory")
        exit(1)
    for take_class in TAKE_CLASSES:
        class_dir = os.path.join(out_image_dir, take_class)
        class_image_dir = os.path.join(class_dir, f"{take_class}Images")
        if not os.path.exists(class_image_dir):
            os.makedirs(class_image_dir)


    # process all videos
    # Collect All of the input videos per category
    vid_files = {}
    for take_class in TAKE_CLASSES:
        class_dir = os.path.join(POCUS_VIDEO_DIR, take_class)
        class_vid_dir = os.path.join(class_dir, f"{take_class}CroppedVideos")

        for vid_file in os.listdir(class_vid_dir):

            # skip non video files
            if vid_file[-3:].lower() not in [
                "peg", "gif", "mp4", "m4v", "avi", "mov"
            ]:
                continue

            # define video path
            video_path = os.path.join(class_vid_dir, vid_file)
            # determine out path based on label
            out_path = os.path.join(class_dir, f"{take_class}Images" )

            # read and write if video
            cap = cv2.VideoCapture(video_path)
            frameRate = cap.get(5)  # video frame rate
            every_x_image = int(frameRate / FRAMERATE)
            print(
                vid_file, "framerate", cap.get(5), "width", cap.get(3),
                "height", cap.get(4), "number frames:", cap.get(7)
            )
            print("--> taking every ", every_x_image, "th image")
            x = 1
            nr_selected = 0
            while cap.isOpened() and nr_selected < MAX_FRAMES:
                frameId = cap.get(1)  #current frame number
                ret, frame = cap.read()
                if (ret != True):
                    break
                if (frameId % every_x_image == 0):
                    # storing the frames in a new folder named test_1
                    filename = os.path.join(
                        out_path, vid_file.replace('.mp4', '') + "_frame%d.jpg" % frameId
                    )
                    cv2.imwrite(filename, frame)
                    nr_selected += 1
            cap.release()
