from typing import List, Dict, Set
from collections import defaultdict
import os
import argparse
import numpy as np
import shutil
import json
import csv

# NOTE: To use the default parameters, execute this from the main directory of
# the package.

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument(
    "-d",
    "--data_dir",
    type=str,
    default="../data/soft_tissue_study",
    help=("Raw data path. Expects 3 or 4 subfolders with classes"),
)
ap.add_argument(
    "-o",
    "--output_dir",
    type=str,
    default="../data/cross_validation/",
    help=("Output path where images for cross validation will be stored."),
)
ap.add_argument(
    "-v",
    "--video_dir",
    type=str,
    default="../data/pocus_videos/convex/",
    help=("Path where the videos of the database are stored"),
)
ap.add_argument(
    "-w",
    "--white_list",
    type=str,
    default="../data/soft_tissue_study/soft_tissue_whitelist.csv",
    help=("Full path to a whitelist file of approved videos"),
)
ap.add_argument(
    "-s", "--splits", type=int, default=5, help="Number of folds for cross validation"
)
args = vars(ap.parse_args())

NUM_FOLDS = args["splits"]
DATA_DIR = args["data_dir"]
OUTPUT_DIR = args["output_dir"]
WHITELIST_FULLPATH = args["white_list"]

# MAKE DIRECTORIES
for split_ind in range(NUM_FOLDS):
    # make directory for this split
    split_path = os.path.join(OUTPUT_DIR, "split" + str(split_ind))
    if not os.path.exists(split_path):
        os.makedirs(split_path)


def came_from_video(in_file: str):
    return in_file.find("_frame") != -1


def build_whitelist(whitelist_path: str = WHITELIST_FULLPATH):
    whitelist = defaultdict(lambda: set())
    with open(whitelist_path, newline="\n") as csvfile:
        csv_reader = csv.reader(csvfile)
        for whitelist_row in csv_reader:
            whitelist_frame_file: str = whitelist_row[0]
            whitelist_video = whitelist_frame_file.split("_")[0]
            whitelist[whitelist_video].add(whitelist_frame_file)

    return whitelist


def should_skip_file(filename: str, frame_whitelist):
    video_name = filename.split("_")[0]
    if video_name in frame_whitelist:
        return not filename in frame_whitelist[video_name]

    # We didn't remove frames from this video, so use all of the frames
    return False


frame_whitelist = build_whitelist()

# MAKE SPLIT
copy_dict = {}
for data_class in os.listdir(DATA_DIR):
    if data_class[0] == "." or data_class.endswith(".xlsx") or data_class.endswith("whitelist.csv"):
        # Hidden file/dir or excel file
        continue
    # make directories:
    for split_ind in range(NUM_FOLDS):
        mod_path = os.path.join(OUTPUT_DIR, "split" + str(split_ind), data_class)
        if not os.path.exists(mod_path):
            os.makedirs(mod_path)

    unique_videos: List[str] = []
    unique_images: List[str] = []
    class_images_dir = os.path.join(
        os.path.join(DATA_DIR, data_class), f"{data_class}Images"
    )
    for in_file in os.listdir(class_images_dir):
        if in_file[0] == ".":
            continue
        if came_from_video(in_file):  # Be careful if this is too general
            # this is a video
            unique_videos.append(in_file.split("_frame")[0])
        else:
            # this is an image
            # Soft tissue study is currently only sourcing from videos
            # we shouldn't be here
            raise Exception("file is determined to come from an image")
            # uni_images.append(in_file.split(".")[0])
    # construct dict of file to fold mapping
    file_to_fold = {}
    # consider images and videos separately
    # We shouldn't have any standalone images yet for soft tissue
    assert len(unique_images) == 0

    # for k, uni in enumerate([unique_videos]):
    unique_files = list(set(unique_videos))
    # s is number of files in one split
    files_per_fold = len(unique_files) // NUM_FOLDS
    for fold in range(NUM_FOLDS):
        for f in unique_files[fold * files_per_fold : (fold + 1) * files_per_fold]:
            file_to_fold[f] = fold
    # distribute the rest randomly
    for f in unique_files[NUM_FOLDS * files_per_fold :]:
        file_to_fold[f] = np.random.choice(np.arange(NUM_FOLDS))

    copy_dict[data_class] = file_to_fold
    for in_file in os.listdir(class_images_dir):
        fold_to_put = file_to_fold[in_file.split("_frame")[0]]
        split_path = os.path.join(OUTPUT_DIR, "split" + str(fold_to_put), data_class)
        # print(os.path.join(DATA_DIR, classe, file), split_path)

        if should_skip_file(in_file, frame_whitelist):
            continue

        shutil.copy(os.path.join(class_images_dir, in_file), split_path)


def check_crossval(cross_val_directory="../data/cross_validation"):
    """
    Test method to check a cross validation split (prints number of unique f)
    """
    check = cross_val_directory
    file_list = []
    for folder in os.listdir(check):
        if folder[0] == ".":
            continue
        for classe in os.listdir(os.path.join(check, folder)):
            if classe[0] == "." or classe[0] == "u":
                continue
            uni = []
            is_image = 0
            for file in os.listdir(os.path.join(check, folder, classe)):
                if file[0] == ".":
                    continue
                if len(file.split(".")) == 2:
                    is_image += 1
                file_list.append(file)
                uni.append(file.split(".")[0])
            print(folder, classe, len(np.unique(uni)), len(uni), is_image)
    if len(file_list) != len(np.unique(file_list)):
        print("PROBLEM: FILES THAT APPEAR TWICE")
        # print(len(file_list), len(np.unique(file_list)))
        uni, counts = np.unique(file_list, return_counts=True)
        for i in range(len(counts)):
            if counts[i] > 1:
                print(uni[i])
    else:
        print("Fine, every file is unique")


# check whether all files are unique
check_crossval()

# MAKE VIDEO CROSS VAL FILE --> corresponds to json cross val

check = OUTPUT_DIR
videos_dir = args["video_dir"]

file_list = []
video_cross_val = {}
# This is creating a label file for the cross validation set
vid_to_class = {}
for split in range(NUM_FOLDS):
    train_test_dict = {"test": [[], []], "train": [[], []]}
    for folder in os.listdir(check):
        if folder[0] == ".":
            continue
        for classe in os.listdir(os.path.join(check, folder)):
            if classe[0] == "." or classe[0] == "u":
                continue
            uni = []
            for file in os.listdir(os.path.join(check, folder, classe)):
                if file[0] == "." or not came_from_video(file):
                    continue
                # parts = file.split("frame_")
                # if not os.path.exists(
                #     os.path.join(videos_dir, parts[0] + "." + parts[1].split("_")[0])
                # ):
                #     # Previous paper used butterfly videos but they seem to be checking for them
                #     # and skipping certain kinds?
                #     butterfly_name = (
                #         parts[0][:3] + "_Butterfly_" + parts[0][4:] + ".avi"
                #     )
                #     if not os.path.exists(os.path.join(videos_dir, butterfly_name)):
                #         print("green dots in video or aibronch", file)
                #         continue
                #     uni.append(butterfly_name)
                # else:
                uni.append(file)
                vid_to_class[file] = classe[:3]

            uni_files_in_split = list(set(uni))
            uni_labels = [vid_to_class.get(vid) for vid in uni_files_in_split]

            if folder[-1] == str(split):
                train_test_dict["test"][0].extend(uni_files_in_split)
                train_test_dict["test"][1].extend(uni_labels)
            else:
                train_test_dict["train"][0].extend(uni_files_in_split)
                train_test_dict["train"][1].extend(uni_labels)
    video_cross_val[split] = train_test_dict

with open(os.path.join("..", "data", "cross_val.json"), "w") as outfile:
    json.dump(video_cross_val, outfile)


print("Validating that the class and labels exist")
this_class = {"Abs": "Abscess", "Cel": "Cellulitis", "Nor": "Normal"}
for fold in range(NUM_FOLDS):
    files, labels = video_cross_val[fold]["test"]
    for j, file in enumerate(files):
        target_file = files[j] # + "_frame0.jpg"
        if should_skip_file(target_file, frame_whitelist):
            continue
        assert os.path.exists(
            os.path.join(
                OUTPUT_DIR,
                "split" + str(fold),
                this_class[labels[j]],
                target_file,
            )
        ), (
            files[j] + "  in  " + str(fold)
        )

print("Complete!")
