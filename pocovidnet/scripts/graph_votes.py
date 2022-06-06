import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os

import matplotlib.pyplot as plt
import pickle


OUT_DIR = "../data/backup/test_run_7_full_holdout/results"  # "../results_oct/plots/new"
IN_DIR = "../data/backup/test_run_7_full_holdout/results"
BEST_MODEL = "output.dat"

CLASSES = ["Abscess", "Cellulitis", "Normal"]
CLASS_COLORS = category_colors = plt.colormaps["RdYlGn"](
    np.linspace(0.15, 0.85, len(CLASSES))
)

compare_model_list = [
    "voting_results.dat",
]
name_dict = {
    "voting_results": "VGG16",
}


def roll_list(input_list, roll_amount, axis=None):
    return np.roll(np.array(input_list), roll_amount, axis)


def plot_results(results, category_truth_idx, category_names=CLASSES):
    """
    Parameters
    ----------
    results : dict
        A mapping from video labels to a list of answers per category.
    category_truth_idx : int
        The current index of result. The graph should start from here.
    category_names : list of str
        The category labels.
    """
    # We want to roll the values to the right by negative category index to
    # display it first

    labels = [label for label in results.keys()]
    # Determine the ordering here
    num_classes = len(category_names)
    rolled_names = roll_list(category_names, -category_truth_idx)
    values = []
    for result_vid in results.keys():
        values.append(
            roll_list(
                [results[result_vid][idx] for idx in range(num_classes)],
                -category_truth_idx,
            )
        )

    # values = [list(value.values()) for value in results.values()]
    data = np.array(values)
    data_cum = data.cumsum(axis=1)
    category_colors = roll_list(CLASS_COLORS, -category_truth_idx, axis=(0,))

    fig, ax = plt.subplots(figsize=(9.2, 5))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())
    ax.set_title(
        f"Vote Results for {category_names[category_truth_idx]} Videos",
        loc="right",
    )

    # for i, (colname, color) in enumerate(zip(category_names, category_colors)):
    for idx in range(len(CLASSES)):
        widths = data[:, idx]
        starts = data_cum[:, idx] - widths
        rects = ax.barh(
            labels,
            widths,
            left=starts,
            height=0.7,
            label=rolled_names[idx],
            color=category_colors[idx],
        )

        r, g, b, _ = category_colors[idx]
        text_color = "white" if r * g * b < 0.5 else "darkgrey"
        ax.bar_label(rects, label_type="center", color=text_color)
    ax.legend(
        ncol=len(category_names),
        bbox_to_anchor=(0, 1),
        loc="lower left",
        fontsize="small",
    )

    return fig, ax


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

    for idx, class_name in enumerate(classes):
        class_video_to_votes = {
            video: votes
            for video, votes in video_to_votes.items()
            if video_to_truth.get(video) == idx
        }
        plot_results(class_video_to_votes, idx)


if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)


with open(os.path.join(IN_DIR, BEST_MODEL), "rb") as outfile:
    (saved_logits, saved_gt, saved_files) = pickle.load(outfile)


evaluate_video_voting(saved_logits, saved_gt, saved_files, CLASSES)

plt.show()
