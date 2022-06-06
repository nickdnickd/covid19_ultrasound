from collections import defaultdict
import os
import numpy as np
import csv

# import seaborn as sn
import matplotlib.pyplot as plt
import pickle


OUT_DIR = "../data/backup/full_run_6_holdout_test/holdout_results"  # "../results_oct/plots/new"
IN_DIR = "../data/backup/full_run_6_holdout_test/holdout_results"
BEST_MODEL = "voting_results.dat"

CLASSES = ["Abscess", "Cellulitis", "Normal"]

compare_model_list = [
    "voting_results.dat",
]
name_dict = {
    "voting_results": "VGG16",
    # "cam_3": "VGG-CAM",
    # "nasnet_3": "NASNetMobile",
    # "encoding_3": "Segment-Enc",
    # "segmented_3": "VGG-Segment",
}


def write_results_to_csv(save_file_path: str, video_to_votes, video_to_truth):
    with open(save_file_path, "w") as f:
        writer = csv.writer(f)
        total_videos = len(video_to_truth.keys())
        correct_videos = 0
        writer.writerow(("Video", "CorrectClass", *CLASSES))
        for video, correct_label_idx in video_to_truth.items():
            votes = video_to_votes[video]
            # votes is dictionary
            winner = -1
            highest_votes = 0

            for class_idx, class_votes in votes.items():
                if class_votes > highest_votes:
                    winner = class_idx
                    highest_votes = highest_votes

            writer.writerow(
                (video, CLASSES[correct_label_idx], votes[0], votes[1], votes[2])
            )

            if winner == correct_label_idx:
                correct_videos += 1

        print(
            f"Total videos {total_videos} Correct videos {correct_videos}. Score {correct_videos/total_videos}"
        )


def sumarize_results(video_to_votes, video_to_truth):
    total_videos = len(video_to_truth.keys())
    correct_videos = 0
    tied_videos = 0
    incorrect_videos = 0

    for video, correct_label_idx in video_to_truth.items():
        votes = video_to_votes[video]

        # votes is dictionary
        winner = -1
        highest_votes = 0

        for class_idx, class_votes in votes.items():
            if class_votes > highest_votes:
                winner = class_idx
                highest_votes = highest_votes

        # (video, CLASSES[correct_label_idx], votes[0], votes[1], votes[2])
        # other_classes = set(range(len(CLASSES))).remove(winner)
        # breakpoint()
        if video_to_votes[video][correct_label_idx] == highest_votes:
            tied_videos += 1
        elif winner == correct_label_idx:
            correct_videos += 1
        else:
            incorrect_videos += 1

    print(
        f"Total videos {total_videos}\n"
        f"Correct videos {correct_videos}\n"
        f"Tied videos {tied_videos}\n"
        f"Incorrect videos {incorrect_videos}\n"
        f"Accuracy (Excluding Ties) {correct_videos/(correct_videos+incorrect_videos)}"
    )


def evaluate_video_voting(
    saved_logits, saved_gt, saved_files, classes=CLASSES, save_path=OUT_DIR
):
    files_to_videos = np.vectorize(lambda filename: filename.split("_frame")[0])
    init_dict = lambda: {idx: 0 for idx in range(len(classes))}
    video_to_votes = defaultdict(init_dict)  # vid name -> class_idx -> vote_count
    video_to_truth = {}

    # Start with total aggregate vote over each of the fold weights
    for saved_logit, ground_truths, file_labels in zip(
        saved_logits, saved_gt, saved_files
    ):

        source_videos = files_to_videos(file_labels)
        pred_idxs = np.argmax(np.array(saved_logit), axis=1)

        # Convert this into votes
        for source_video, pred_idx, ground_truth in zip(
            source_videos, pred_idxs, ground_truths
        ):
            video_to_votes[source_video][pred_idx] += 1
            video_to_truth[source_video] = ground_truth

        # convert these indicies to votes

        # Start with just correct incorrect videos
        pred_idxs == ground_truths

    votes_file_path = os.path.join(save_path, "voting_results.csv")
    sumarize_results(video_to_votes, video_to_truth)
    # write_results_to_csv(votes_file_path, video_to_votes, video_to_truth)


if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

from matplotlib import rc

plt.rcParams["legend.title_fontsize"] = 20
plt.rcParams["axes.facecolor"] = "white"
# activate latex text rendering
rc("text", usetex=False)


with open(os.path.join(IN_DIR, BEST_MODEL), "rb") as outfile:
    (saved_logits, saved_gt, saved_files) = pickle.load(outfile)


evaluate_video_voting(saved_logits, saved_gt, saved_files, CLASSES)
