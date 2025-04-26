from pathlib import Path

import click
import json
import pickle
import shutil
import numpy as np

from talknet_asd.asd_pipeline import Pipeline as ASDPipeline


def get_data_from_pickle(file_path: str):
    with open(file_path, "rb") as f:
        loaded = pickle.load(f)
    return loaded


def save_to_json_file(data: dict | list, file_path: str):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)


def get_active_speakers(
    tracks: list[dict], scores: list[dict], is_active_threshold=-0.5
) -> list[dict]:
    """
    Scene Definition:
    - A scene is a continuous sequence of frames without any cuts/edits.
    - Example format:
        scenes = [
            (00:00:00.000 [frame=0, fps=25.000], 00:00:07.640 [frame=191, fps=25.000]),
            (00:00:07.640 [frame=191, fps=25.000], 00:00:18.640 [frame=466, fps=25.000]),
            (00:00:18.640 [frame=466, fps=25.000], 00:00:30.000 [frame=750, fps=25.000])
        ]

    Track Definition:
    - A track represents one face appearing in a specific scene. The same face in another scene would be a different track.
    - Data structure:
        tracks = [
            {
                'track': {
                    'frame': array([0, 1, 2, ...]),  # Frame numbers where this face appears
                    'bbox': array([[...], [...]])    # Bounding box coordinates (x1,y1,x2,y2) for each frame
                },
                'proc_track': {
                    'x': array([...]),  # Smoothed X-coordinates of bbox centers
                    'y': array([...]),  # Smoothed Y-coordinates of bbox centers
                    's': array([...])   # Smoothed bbox sizes (half width/height)
                }
            },
            ...
        ]

    Scores Definition:
    - Confidence scores indicating whether the person in a track is speaking in each frame.
    - Data structure:
        scores = [
            # Track 0
            array([-1.6, -1.7, -1.7, ..., -1.5]),  # Scores for each frame in scene
            # Track 1
            array([-0.6, -0.7, -0.8, ..., -0.5]),
            ...
        ]

    Faces Definition:
    - Array containing all detected faces per frame
    - Data structure:
        faces = [
            # Frame 0
            [
                {
                    'frame': 0,
                    'bbox': [1116.42, 341.39, 1216.44, 481.27],
                    'conf': 0.999857  # Detection confidence
                },
                {
                    'frame': 0,
                    'bbox': [272.09, 282.71, 368.19, 442.03],
                    'conf': 0.999051
                },
                ...
            ],
            # Frame 1
            [...],
            ...
        ]

    Conversion Task:
    Transform tracks and scores into a simplified format:
    [
        {
            "frame_idx": frame_number,
            "faces": [
                {
                    "bbox": (x1, y1, x2, y2),
                    "score": speaker_detection_confidence,
                    "is_active": 1/0  # Determined by score > `is_active_threshold`
                },
                ...
            ]
        },
        ...
    ]
    """

    assert len(tracks) == len(scores), "tracks size and scores size have to be equal!!!"

    max_frame = max([max(track["track"]["frame"]) for track in tracks]) if tracks else 0

    active_speakers = [{"frame_idx": i, "faces": []} for i in range(max_frame + 1)]

    for track_idx, track in enumerate(tracks):
        frames = track["track"]["frame"]
        bboxes = track["track"]["bbox"]
        track_scores = scores[track_idx]

        # Because the number of audio and video frames may not be equal
        if len(track_scores) < len(frames):
            last_score = track_scores[-1]
            track_scores = np.append(track_scores, last_score)

        for frame_idx, frame in enumerate(frames):
            x1, y1, x2, y2 = bboxes[frame_idx]
            active_score = track_scores[frame_idx]

            face_data = {
                "bbox": (float(x1), float(y1), float(x2), float(y2)),
                "score": float(active_score),
                "is_active": int(active_score > is_active_threshold),
            }

            active_speakers[frame]["faces"].append(face_data)

    return active_speakers


def delete_dir(dir_to_delete: Path):
    shutil.rmtree(dir_to_delete)


@click.command()
@click.option(
    "--video-file",
    type=click.Path(exists=True),
    required=True,
)
@click.option(
    "--output-file",
    type=click.Path(exists=False),
    required=True,
)
@click.option(
    "--delete-asd-dir",
    type=bool,
    default=True,
    help="Whether to delete the asd dir",
)
@click.option(
    "--device",
    type=str,
    default="auto",
    help="Device to run the ASD pipeline on",
)
def main(video_file: str, output_file: str, delete_asd_dir: bool, device: str):
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    asd_pipeline = ASDPipeline(video_path=video_file, device=device)
    asd_pipeline.run()

    tracks = get_data_from_pickle(
        file_path=f"{asd_pipeline.save_path}/pywork/tracks.pckl"
    )
    scores = get_data_from_pickle(
        file_path=f"{asd_pipeline.save_path}/pywork/scores.pckl"
    )

    active_speakers = get_active_speakers(tracks=tracks, scores=scores)
    save_to_json_file(data=active_speakers, file_path=output_file)

    if delete_asd_dir:
        delete_dir(dir_to_delete=asd_pipeline.save_path)


if __name__ == "__main__":
    main()
