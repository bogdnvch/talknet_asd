import shutil
import sys
import glob
import time
import os
import tqdm
import subprocess
import warnings
import pickle
import math
from pathlib import Path
from typing import Literal, Optional, Union
from datetime import datetime
import copy
import queue
import threading

import torch
import numpy
import python_speech_features
import cv2
from scipy import signal
from scipy.io import wavfile
from scipy.interpolate import interp1d

from scenedetect import (
    open_video,
    ContentDetector,
    SceneManager,
    StatsManager,
    SceneDetector,
    SceneList,
    FrameTimecode,
)


from huggingface_hub import hf_hub_download
from batch_face import RetinaFace

from talknet_asd.talkNet import talkNet
from talknet_asd.utils.resolve_device import resolve_device

warnings.filterwarnings("ignore")

cache_dir = Path.home() / ".cache" / "talknet_asd"
cache_dir.mkdir(parents=True, exist_ok=True)


def visualization(tracks, scores, args):
    cap = cv2.VideoCapture(args.video_file_path)
    if not cap.isOpened():
        sys.stderr.write(
            f"Error: Could not open video file {args.video_file_path} for visualization.\\r\\n"
        )
        return

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if num_frames == 0 or fw == 0 or fh == 0:
        sys.stderr.write(
            f"Error: Video file {args.video_file_path} appears to have no frames or invalid dimensions for visualization.\\r\\n"
        )
        cap.release()
        return

    faces = [[] for _ in range(num_frames)]
    for tidx, track in enumerate(tracks):
        score = scores[tidx]
        # Ensure track["track"]["frame"] contains valid frame indices within num_frames
        for fidx_in_track, frame_num in enumerate(track["track"]["frame"].tolist()):
            if 0 <= frame_num < num_frames:
                s = score[
                    max(fidx_in_track - 2, 0) : min(fidx_in_track + 3, len(score) - 1)
                ]  # average smoothing
                s = numpy.mean(s)
                faces[frame_num].append(
                    {
                        "track": tidx,
                        "score": float(s),
                        "s": track["proc_track"]["s"][fidx_in_track],
                        "x": track["proc_track"]["x"][fidx_in_track],
                        "y": track["proc_track"]["y"][fidx_in_track],
                    }
                )
            else:
                sys.stderr.write(
                    f"Warning: Invalid frame number {frame_num} in track {tidx}. Max frames: {num_frames}. Skipping this entry.\r\n"
                )

    vOut = cv2.VideoWriter(
        os.path.join(args.pyavi_path, "video_only.avi"),
        cv2.VideoWriter_fourcc(*"XVID"),
        25,  # Assuming 25 FPS output, consistent with original
        (fw, fh),
    )
    colorDict = {0: 0, 1: 255}

    for fidx in tqdm.tqdm(range(num_frames), total=num_frames, desc="Visualizing"):
        ret, image = cap.read()
        if not ret:
            sys.stderr.write(
                f"Warning: Could not read frame {fidx} during visualization. Stopping.\\r\\n"
            )
            break

        for face_info in faces[fidx]:
            clr = colorDict[int((face_info["score"] >= 0))]
            txt = round(face_info["score"], 1)
            cv2.rectangle(
                image,
                (
                    int(face_info["x"] - face_info["s"]),
                    int(face_info["y"] - face_info["s"]),
                ),
                (
                    int(face_info["x"] + face_info["s"]),
                    int(face_info["y"] + face_info["s"]),
                ),
                (0, clr, 255 - clr),
                10,
            )
            cv2.putText(
                image,
                "%s" % (txt),
                (
                    int(face_info["x"] - face_info["s"]),
                    int(face_info["y"] - face_info["s"]),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, clr, 255 - clr),
                5,
            )
        vOut.write(image)

    vOut.release()
    cap.release()

    command = (
        "ffmpeg -y -i %s -i %s -threads %d -c:v copy -c:a copy %s -loglevel panic"
        % (
            os.path.join(args.pyavi_path, "video_only.avi"),
            os.path.join(args.pyavi_path, "audio.wav"),
            args.n_data_loader_thread,
            os.path.join(args.pyavi_path, "video_out.avi"),
        )
    )
    subprocess.call(command, shell=True, stdout=None)


class VideoPreprocessor:
    def __init__(self, args):
        self.args = args

    def extract_video(self):
        """Extract video to frames. Converts to 25 FPS if necessary."""
        cap = cv2.VideoCapture(self.args.video_path)
        fps = 0.0
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()

        # Use a small tolerance for floating point comparison of FPS
        if fps > 0 and abs(fps - 25.0) < 0.01:
            self.args.video_file_path = self.args.video_path
            sys.stderr.write(
                time.strftime("%Y-%m-%d %H:%M:%S")
                + f" Input video '{self.args.video_path}' is already {fps:.2f} FPS. Using original video directly.\\r\\n"
            )
        else:
            self.args.video_file_path = os.path.join(self.args.pyavi_path, "video.avi")
            if fps > 0:
                sys.stderr.write(
                    time.strftime("%Y-%m-%d %H:%M:%S")
                    + f" Input video '{self.args.video_path}' has {fps:.2f} FPS. Converting to 25 FPS.\\r\\n"
                )
            else:
                sys.stderr.write(
                    time.strftime("%Y-%m-%d %H:%M:%S")
                    + f" Could not determine FPS for input video '{self.args.video_path}'. Attempting conversion to 25 FPS.\\r\\n"
                )

            command = (
                "ffmpeg -y -i %s -threads %d -async 1 -r 25 %s -loglevel panic"
                % (
                    self.args.video_path,
                    self.args.n_data_loader_thread,
                    self.args.video_file_path,
                )
            )
            subprocess.call(command, shell=True)
            sys.stderr.write(
                time.strftime("%Y-%m-%d %H:%M:%S")
                + " Processed video saved in %s \\r\\n" % self.args.video_file_path
            )

    def extract_audio(self):
        """Extract audio track"""
        self.args.audio_file_path = os.path.join(self.args.pyavi_path, "audio.wav")
        command = (
            "ffmpeg -y -i %s -qscale:a 0 -ac 1 -vn -threads %d -ar 16000 %s -loglevel panic"
            % (
                self.args.video_file_path,
                self.args.n_data_loader_thread,
                self.args.audio_file_path,
            )
        )
        subprocess.call(command, shell=True, stdout=None)
        sys.stderr.write(
            time.strftime("%Y-%m-%d %H:%M:%S")
            + " Extract the audio and save in %s \r\n" % self.args.audio_file_path
        )

    def extract_frames(self):
        """Extract individual frames"""
        command = "ffmpeg -y -i %s -threads %d -f image2 %s -loglevel panic" % (
            self.args.video_file_path,
            self.args.n_data_loader_thread,
            os.path.join(self.args.pyframes_path, "%06d.png"),
        )
        subprocess.call(command, shell=True, stdout=None)
        sys.stderr.write(
            time.strftime("%Y-%m-%d %H:%M:%S")
            + " Extract the frames and save in %s \r\n" % self.args.pyframes_path
        )


class FaceProcessor:
    def __init__(self, args):
        self.args = args

    def scenes_detect(self):
        sceneList = self.custom_detect_scenes(
            video_path=self.args.video_file_path,
            detector=ContentDetector(),
            show_progress=True,
            backend=self.args.scene_detector_backend,
        )
        savePath = os.path.join(self.args.pywork_path, "scene.pckl")
        with open(savePath, "wb") as fil:
            pickle.dump(sceneList, fil)
            sys.stderr.write(
                "%s - scenes detected %d\n"
                % (self.args.video_file_path, len(sceneList))
            )
        sys.stderr.write(
            time.strftime("%Y-%m-%d %H:%M:%S")
            + " Scene detection and save in %s \r\n" % self.args.pywork_path
        )
        return sceneList

    def custom_detect_scenes(
        self,
        video_path: str,
        detector: SceneDetector,
        stats_file_path: Optional[str] = None,
        show_progress: bool = False,
        backend: str = "opencv",
        start_time: Optional[Union[str, float, int]] = None,
        end_time: Optional[Union[str, float, int]] = None,
        start_in_scene: bool = False,
    ) -> SceneList:
        video = open_video(video_path, backend=backend)
        if start_time is not None:
            start_time = video.base_timecode + start_time
            video.seek(start_time)
        if end_time is not None:
            end_time = video.base_timecode + end_time
        # To reduce memory consumption when not required, we only add a StatsManager if we
        # need to save frame metrics to disk.
        scene_manager = SceneManager(StatsManager() if stats_file_path else None)
        scene_manager.add_detector(detector)
        scene_manager.detect_scenes(
            video=video,
            show_progress=show_progress,
            end_time=end_time,
        )
        if scene_manager.stats_manager is not None:
            scene_manager.stats_manager.save_to_csv(csv_file=stats_file_path)

        current_scene_list = scene_manager.get_scene_list(start_in_scene=start_in_scene)

        if current_scene_list:
            # Manually adjust the end of the last scene to be the end of the video
            # This is necessary because some backends (e.g. pyav) might not set the last scene's end correctly
            video_for_props = None  # Renamed from video to avoid conflict with the video object used for detection
            try:
                # We need to re-open the video or ensure the original video object is seekable and still valid.
                # For simplicity and to ensure we get accurate duration regardless of prior processing,
                # we open it again. Note: this assumes `video_path` is accessible here.
                video_for_props = open_video(
                    video_path, backend=backend
                )  # Use `video_path` and `backend` from function args
                video_duration_frames = video_for_props.duration.get_frames()
                video_fps = video_for_props.frame_rate

                last_scene_start, last_scene_end = current_scene_list[-1]

                if last_scene_end.get_frames() < video_duration_frames:
                    corrected_last_scene_end = FrameTimecode(
                        timecode=video_duration_frames, fps=video_fps
                    )
                    current_scene_list[-1] = (
                        last_scene_start,
                        corrected_last_scene_end,
                    )
                elif last_scene_end.get_frames() > video_duration_frames:
                    corrected_last_scene_end = FrameTimecode(
                        timecode=video_duration_frames, fps=video_fps
                    )
                    current_scene_list[-1] = (
                        last_scene_start,
                        corrected_last_scene_end,
                    )

            except Exception as e:
                print(
                    f"Warning: Could not adjust the last scene's end time during detection: {e}"
                )

        return current_scene_list

    def detect_faces(self):
        """Run face detection on all frames"""

        cap_for_count = cv2.VideoCapture(self.args.video_file_path)
        if not cap_for_count.isOpened():
            sys.stderr.write(
                f"Error: Could not open video file {self.args.video_file_path} to get frame count.\\r\\n"
            )
            return False, []
        total_frames = int(cap_for_count.get(cv2.CAP_PROP_FRAME_COUNT))
        cap_for_count.release()

        if total_frames == 0:
            sys.stderr.write(
                f"Error: Video file {self.args.video_file_path} appears to have no frames for detection count.\\r\\n"
            )
            return False, []

        detections = [[] for _ in range(total_frames)]
        total_detection_time_gpu_and_assembly = 0.0  # Renamed for clarity

        threshold = self.args.face_detection_threshold
        max_size = self.args.face_detection_max_size
        batch_size = self.args.face_detection_batch_size

        gpu_id = 0 if self.args.device == torch.device("cuda") else -1
        fp16_enabled = self.args.dtype == torch.float16
        detector_instance = RetinaFace(
            gpu_id=gpu_id,
            fp16=fp16_enabled,
            network="resnet50",
        )
        print(f"Detector instance device: {detector_instance.device}")
        print(f"Detector instance model dtype: fp16={detector_instance.fp16}")
        # if self.args.device == torch.device("cuda"):
        #     detector_instance.model.compile(mode="reduce-overhead", fullgraph=True)

        frame_queue = queue.Queue(
            maxsize=batch_size * 8
        )  # Queue for (fidx, img_rgb) or None
        loader_exception = None  # To capture exceptions from the loader thread

        def _frame_loader_worker():
            nonlocal loader_exception
            try:
                cap_loader = cv2.VideoCapture(self.args.video_file_path)
                if not cap_loader.isOpened():
                    # Cannot directly write to sys.stderr from thread easily without GIL issues with print
                    # Instead, signal error through an exception or a dedicated error queue/variable
                    raise IOError(
                        f"Frame loader: Could not open video file {self.args.video_file_path}"
                    )

                for fidx_loader in range(total_frames):
                    ret_loader, img_loader = cap_loader.read()
                    if not ret_loader:
                        # Signal that loading is done or an error occurred
                        frame_queue.put(None)  # Sentinel for end or error
                        # print(f"Frame loader: Failed to read frame {fidx_loader}", file=sys.stderr) # Not thread-safe for direct print
                        return  # Stop loader thread

                    img_rgb_loader = cv2.cvtColor(img_loader, cv2.COLOR_BGR2RGB)
                    frame_queue.put((fidx_loader, img_rgb_loader))

                frame_queue.put(
                    None
                )  # Sentinel to indicate successful completion of all frames
                cap_loader.release()
            except Exception as e:
                loader_exception = e  # Store exception
                frame_queue.put(None)  # Ensure consumer unblocks if loader crashes

        loader_thread = threading.Thread(target=_frame_loader_worker)
        loader_thread.start()

        frames_batch = []
        frame_indices_batch = []

        processed_frames_count = 0

        with tqdm.tqdm(total=total_frames, desc="Detecting faces (Batched)") as pbar:
            while processed_frames_count < total_frames:
                try:
                    # Timeout helps prevent indefinite blocking if loader thread has an issue not caught by sentinel
                    queued_item = frame_queue.get(timeout=20)
                except queue.Empty:
                    # This could happen if loader thread is stuck or died without putting sentinel
                    if loader_thread.is_alive():
                        sys.stderr.write(
                            "Warning: Frame queue timeout, loader thread might be stuck.\\r\\n"
                        )
                        # Potentially break or implement more robust error handling
                        continue  # Try getting again or break
                    else:
                        # Loader died, check for exception
                        if loader_exception:
                            sys.stderr.write(
                                f"Error: Frame loader thread exited with exception: {loader_exception}\\r\\n"
                            )
                        else:
                            sys.stderr.write(
                                "Error: Frame loader thread died unexpectedly.\\r\\n"
                            )
                        # Process any remaining items in frames_batch just in case, then break
                        if frames_batch:  # Process final partial batch if any
                            # This block is duplicated below, consider refactoring into a helper
                            batch_loop_start_time = time.time()
                            try:
                                all_faces_in_batch = detector_instance(
                                    frames_batch,
                                    threshold=threshold,
                                    max_size=max_size,
                                    return_dict=True,
                                )
                                for i, (
                                    current_fidx_in_batch,
                                    faces_in_frame,
                                ) in enumerate(
                                    zip(frame_indices_batch, all_faces_in_batch)
                                ):
                                    # ... (result processing logic - same as below)
                                    frame_detections_for_frame = []
                                    if (
                                        isinstance(faces_in_frame, list)
                                        and len(faces_in_frame) > 0
                                    ):
                                        for face in faces_in_frame:
                                            # ... (detailed face processing)
                                            facial_area = face["box"]
                                            if face["score"] < threshold:
                                                continue
                                            if isinstance(facial_area, dict):
                                                if not all(
                                                    key in facial_area
                                                    for key in ["x", "y", "w", "h"]
                                                ):
                                                    continue
                                                x1, y1, x2, y2 = (
                                                    facial_area["x"],
                                                    facial_area["y"],
                                                    facial_area["x"] + facial_area["w"],
                                                    facial_area["y"] + facial_area["h"],
                                                )
                                            else:
                                                x1, y1, x2, y2 = (
                                                    facial_area[0],
                                                    facial_area[1],
                                                    facial_area[2],
                                                    facial_area[3],
                                                )
                                            if x2 - x1 <= 0 or y2 - y1 <= 0:
                                                continue
                                            bbox = [x1, y1, x2, y2]
                                            frame_detections_for_frame.append(
                                                {
                                                    "frame": current_fidx_in_batch,
                                                    "bbox": bbox,
                                                    "conf": face["score"],
                                                }
                                            )
                                    if 0 <= current_fidx_in_batch < len(detections):
                                        detections[current_fidx_in_batch] = (
                                            frame_detections_for_frame
                                        )
                            except Exception as e:
                                print(
                                    f"Error processing final batch from queue timeout: {str(e)}"
                                )
                            total_detection_time_gpu_and_assembly += (
                                time.time() - batch_loop_start_time
                            )
                        return False, detections  # Abort

                if loader_exception:  # Check for exception from loader thread
                    sys.stderr.write(
                        f"Error: Frame loader thread failed: {loader_exception}\\r\\n"
                    )
                    # Process any items already batched
                    if frames_batch:  # Process final partial batch if any
                        batch_loop_start_time = time.time()
                        try:
                            all_faces_in_batch = detector_instance(
                                frames_batch,
                                threshold=threshold,
                                max_size=max_size,
                                return_dict=True,
                            )
                            # ... (result processing logic - duplicated)
                            for i, (current_fidx_in_batch, faces_in_frame) in enumerate(
                                zip(frame_indices_batch, all_faces_in_batch)
                            ):
                                frame_detections_for_frame = []
                                if (
                                    isinstance(faces_in_frame, list)
                                    and len(faces_in_frame) > 0
                                ):
                                    for face in faces_in_frame:
                                        facial_area = face["box"]
                                        if face["score"] < threshold:
                                            continue
                                        if isinstance(facial_area, dict):
                                            if not all(
                                                key in facial_area
                                                for key in ["x", "y", "w", "h"]
                                            ):
                                                continue
                                            x1, y1, x2, y2 = (
                                                facial_area["x"],
                                                facial_area["y"],
                                                facial_area["x"] + facial_area["w"],
                                                facial_area["y"] + facial_area["h"],
                                            )
                                        else:
                                            x1, y1, x2, y2 = (
                                                facial_area[0],
                                                facial_area[1],
                                                facial_area[2],
                                                facial_area[3],
                                            )
                                        if x2 - x1 <= 0 or y2 - y1 <= 0:
                                            continue
                                        bbox = [x1, y1, x2, y2]
                                        frame_detections_for_frame.append(
                                            {
                                                "frame": current_fidx_in_batch,
                                                "bbox": bbox,
                                                "conf": face["score"],
                                            }
                                        )
                                if 0 <= current_fidx_in_batch < len(detections):
                                    detections[current_fidx_in_batch] = (
                                        frame_detections_for_frame
                                    )
                        except Exception as e:
                            print(
                                f"Error processing batch after loader exception: {str(e)}"
                            )
                        total_detection_time_gpu_and_assembly += (
                            time.time() - batch_loop_start_time
                        )
                    return False, detections  # Abort

                if queued_item is None:  # Sentinel: loader finished or error
                    # Process any remaining frames in the current batch
                    if frames_batch:
                        batch_loop_start_time = time.time()
                        try:
                            all_faces_in_batch = detector_instance(
                                frames_batch,
                                threshold=threshold,
                                max_size=max_size,
                                return_dict=True,
                            )
                            # ... (result processing logic - duplicated)
                            for i, (current_fidx_in_batch, faces_in_frame) in enumerate(
                                zip(frame_indices_batch, all_faces_in_batch)
                            ):
                                frame_detections_for_frame = []
                                if (
                                    isinstance(faces_in_frame, list)
                                    and len(faces_in_frame) > 0
                                ):
                                    for face in faces_in_frame:
                                        facial_area = face["box"]
                                        if face["score"] < threshold:
                                            continue
                                        if isinstance(facial_area, dict):
                                            if not all(
                                                key in facial_area
                                                for key in ["x", "y", "w", "h"]
                                            ):
                                                continue
                                            x1, y1, x2, y2 = (
                                                facial_area["x"],
                                                facial_area["y"],
                                                facial_area["x"] + facial_area["w"],
                                                facial_area["y"] + facial_area["h"],
                                            )
                                        else:
                                            x1, y1, x2, y2 = (
                                                facial_area[0],
                                                facial_area[1],
                                                facial_area[2],
                                                facial_area[3],
                                            )
                                        if x2 - x1 <= 0 or y2 - y1 <= 0:
                                            continue
                                        bbox = [x1, y1, x2, y2]
                                        frame_detections_for_frame.append(
                                            {
                                                "frame": current_fidx_in_batch,
                                                "bbox": bbox,
                                                "conf": face["score"],
                                            }
                                        )
                                if 0 <= current_fidx_in_batch < len(detections):
                                    detections[current_fidx_in_batch] = (
                                        frame_detections_for_frame
                                    )
                        except Exception as e:
                            print(f"Error processing final batch: {str(e)}")
                        finally:  # Ensure batch clear and time update
                            total_detection_time_gpu_and_assembly += (
                                time.time() - batch_loop_start_time
                            )
                            frames_batch = []
                            frame_indices_batch = []
                    break  # Exit the while loop, all frames processed or loader stopped

                fidx, img_rgb = queued_item
                frames_batch.append(img_rgb)
                frame_indices_batch.append(fidx)

                # pbar.update(1) # Update pbar for each frame received from queue

                if len(frames_batch) == batch_size:
                    batch_loop_start_time = time.time()
                    try:
                        all_faces_in_batch = detector_instance(
                            frames_batch,
                            threshold=threshold,
                            max_size=max_size,
                            return_dict=True,
                        )
                        for i, (current_fidx_in_batch, faces_in_frame) in enumerate(
                            zip(frame_indices_batch, all_faces_in_batch)
                        ):
                            frame_detections_for_frame = []
                            if (
                                isinstance(faces_in_frame, list)
                                and len(faces_in_frame) > 0
                            ):
                                for face in faces_in_frame:
                                    facial_area = face["box"]
                                    if face["score"] < threshold:
                                        continue
                                    if isinstance(facial_area, dict):
                                        if not all(
                                            key in facial_area
                                            for key in ["x", "y", "w", "h"]
                                        ):
                                            continue
                                        x1, y1 = facial_area["x"], facial_area["y"]
                                        x2, y2 = (
                                            facial_area["x"] + facial_area["w"],
                                            facial_area["y"] + facial_area["h"],
                                        )
                                    else:
                                        x1, y1, x2, y2 = (
                                            facial_area[0],
                                            facial_area[1],
                                            facial_area[2],
                                            facial_area[3],
                                        )

                                    if x2 - x1 <= 0 or y2 - y1 <= 0:
                                        continue
                                    bbox = [x1, y1, x2, y2]
                                    frame_detections_for_frame.append(
                                        {
                                            "frame": current_fidx_in_batch,
                                            "bbox": bbox,
                                            "conf": face["score"],
                                        }
                                    )
                            if (
                                0 <= current_fidx_in_batch < len(detections)
                            ):  # Boundary check
                                detections[current_fidx_in_batch] = (
                                    frame_detections_for_frame
                                )
                            else:
                                sys.stderr.write(
                                    f"Warning: Frame index {current_fidx_in_batch} out of bounds for detections list. Skipping.\\r\\n"
                                )
                    except Exception as e:
                        print(
                            f"Error processing batch starting with frame index {frame_indices_batch[0] if frame_indices_batch else 'unknown'}: {str(e)}"
                        )
                        # For frames in this failed batch, detections will remain as empty lists
                    finally:  # Ensure batch clear and time update
                        total_detection_time_gpu_and_assembly += (
                            time.time() - batch_loop_start_time
                        )
                        pbar.update(
                            len(frames_batch)
                        )  # Update pbar by number of frames processed in this batch
                        processed_frames_count += len(frames_batch)
                        frames_batch = []
                        frame_indices_batch = []

                # Slow detection check logic
                # This check uses processed_frames_count which is updated after each batch
                if (
                    processed_frames_count
                    >= self.args.face_detection_min_frames_for_avg
                    and processed_frames_count > 0
                ):
                    avg_time_per_frame = (
                        total_detection_time_gpu_and_assembly / processed_frames_count
                    )
                    if (
                        self.args.enable_skip_slow_face_detection
                        and avg_time_per_frame
                        > self.args.face_detection_avg_time_threshold
                    ):
                        sys.stderr.write(
                            time.strftime("%Y-%m-%d %H:%M:%S")
                            + f" Face detection too slow (avg {avg_time_per_frame:.2f}s/frame > "
                            + f"{self.args.face_detection_avg_time_threshold:.2f}s/frame). "
                            + f"Skipping video {self.args.video_path}.\\r\\n"
                        )
                        # Ensure loader thread is stopped
                        # frame_queue.put(None) # Not strictly necessary if we are breaking, but good for completeness
                        loader_thread.join(timeout=5)  # Attempt to join loader thread
                        return False, detections

        loader_thread.join(timeout=10)  # Wait for loader thread to finish
        if loader_thread.is_alive():
            sys.stderr.write(
                "Warning: Frame loader thread did not exit cleanly after processing.\\r\\n"
            )
        if loader_exception:  # Final check if an error occurred late in loader
            sys.stderr.write(
                f"Error: Frame loader thread failed with: {loader_exception}\\r\\n"
            )
            return False, detections

        with open(os.path.join(self.args.pywork_path, "faces.pckl"), "wb") as f:
            pickle.dump(detections, f)

        sys.stderr.write(
            time.strftime("%Y-%m-%d %H:%M:%S")
            + " Face detection and save in %s \\r\\n" % (self.args.pywork_path)
        )
        return True, detections

    def track_faces(self, scenes, face_detections):
        all_tracks = []
        for i, shot in enumerate(scenes):
            if (
                shot[1].frame_num - shot[0].frame_num >= self.args.min_track
            ):  # Discard the shot frames less than min_track frames
                # Original slice, used for debug prints or if original needed before copy
                # shot[1].frame_num is an EXCLUSIVE end frame index.
                # The actual frames for the scene are shot[0].frame_num to shot[1].frame_num - 1.
                # So, the slice should go up to shot[1].frame_num (exclusive end for slice).
                current_shot_detections_segment = face_detections[
                    shot[0].frame_num : shot[
                        1
                    ].frame_num  # Corrected slice
                ]

                # Create a deepcopy of the segment for track_shot to prevent modifying shared lists
                current_shot_detections_for_track_shot = copy.deepcopy(
                    current_shot_detections_segment
                )

                shot_tracks = self.track_shot(
                    current_shot_detections_for_track_shot  # Pass the deepcopied segment
                )
                all_tracks.extend(shot_tracks)

        sys.stderr.write(
            time.strftime("%Y-%m-%d %H:%M:%S")
            + " Face track and detected %d tracks \r\n" % len(all_tracks)
        )
        return all_tracks

    def track_shot(self, sceneFaces):
        iouThres = 0.5  # Minimum IOU between consecutive face detections
        tracks = []
        while True:
            track = []
            for i, frameFaces in enumerate(sceneFaces):
                for face in frameFaces:
                    if track == []:
                        track.append(face)
                        frameFaces.remove(face)
                    elif face["frame"] - track[-1]["frame"] <= self.args.num_failed_det:
                        iou = self.bb_intersection_over_union(
                            face["bbox"], track[-1]["bbox"]
                        )
                        if iou > iouThres:
                            track.append(face)
                            frameFaces.remove(face)
                            continue
                    else:
                        break
            if track == []:
                break
            elif len(track) > self.args.min_track:
                frameNum = numpy.array([f["frame"] for f in track])
                bboxes = numpy.array([numpy.array(f["bbox"]) for f in track])
                frameI = numpy.arange(frameNum[0], frameNum[-1] + 1)
                confidences = numpy.array([f["conf"] for f in track])
                bboxesI = []
                for ij in range(0, 4):
                    interpfn = interp1d(frameNum, bboxes[:, ij])
                    bboxesI.append(interpfn(frameI))
                bboxesI = numpy.stack(bboxesI, axis=1)

                confInterpFn = interp1d(
                    frameNum,
                    confidences,
                )
                confidencesI = confInterpFn(frameI)

                if (
                    max(
                        numpy.mean(bboxesI[:, 2] - bboxesI[:, 0]),
                        numpy.mean(bboxesI[:, 3] - bboxesI[:, 1]),
                    )
                    > self.args.min_face_size
                ):
                    tracks.append(
                        {
                            "frame": frameI,
                            "bbox": bboxesI,
                            "confidence": confidencesI,
                        }
                    )
        return tracks

    @staticmethod
    def bb_intersection_over_union(boxA, boxB, evalCol=False):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        if evalCol == True:
            iou = interArea / float(boxAArea)
        else:
            iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def video_tracks(self, all_tracks):
        video_tracks = []
        main_video_cap = cv2.VideoCapture(self.args.video_file_path)

        if not main_video_cap.isOpened():
            sys.stderr.write(
                f"Error: Could not open video file {self.args.video_file_path} in video_tracks method.\\r\\n"
            )
            return []

        try:
            for ii, track in tqdm.tqdm(
                enumerate(all_tracks), total=len(all_tracks), desc="Cropping tracks"
            ):
                video_tracks.append(
                    self.crop_video(
                        main_video_cap,
                        track,
                        os.path.join(self.args.pycrop_path, f"{ii:05d}"),
                    )
                )
        finally:
            main_video_cap.release()  # Ensure the video capture is released

        print(f"\n{time.strftime('%Y-%m-%d %H:%M:%S')} Face Crop completed")

        return video_tracks

    def crop_video(self, main_video_cap, track, cropFile):
        self.args.audio_file_path = os.path.join(self.args.pyavi_path, "audio.wav")

        vOut = cv2.VideoWriter(
            cropFile + "t.avi", cv2.VideoWriter_fourcc(*"XVID"), 25, (224, 224)
        )  # Write video
        dets = {"x": [], "y": [], "s": []}
        for det in track["bbox"]:  # Read the tracks
            dets["s"].append(max((det[3] - det[1]), (det[2] - det[0])) / 2)
            dets["y"].append((det[1] + det[3]) / 2)  # crop center x
            dets["x"].append((det[0] + det[2]) / 2)  # crop center y
        dets["s"] = signal.medfilt(dets["s"], kernel_size=13)  # Smooth detections
        dets["x"] = signal.medfilt(dets["x"], kernel_size=13)
        dets["y"] = signal.medfilt(dets["y"], kernel_size=13)

        original_frames_to_crop = track["frame"]

        for fidx_in_track, original_frame_num in enumerate(original_frames_to_crop):
            # Seek to the specific frame in the main video
            main_video_cap.set(cv2.CAP_PROP_POS_FRAMES, original_frame_num)
            ret, image = main_video_cap.read()

            if not ret:
                sys.stderr.write(
                    f"Warning: Could not read frame {original_frame_num} (index {fidx_in_track} in track) from {self.args.video_file_path} during crop_video. Skipping frame.\\r\\n"
                )
                continue  # Skip this frame if not readable

            cs = self.args.crop_scale
            bs = dets["s"][
                fidx_in_track
            ]  # Detection box size, use fidx_in_track for dets
            bsi = int(bs * (1 + 2 * cs))  # Pad videos by this amount

            # Pad the image correctly
            # Ensure image is not None before padding
            if image is None:
                sys.stderr.write(
                    f"Warning: Image for frame {original_frame_num} is None before padding. Skipping frame.\\r\\n"
                )
                continue

            padded_image = numpy.pad(
                image,
                ((bsi, bsi), (bsi, bsi), (0, 0)),
                "constant",
                constant_values=(
                    110,
                    110,
                ),  # Using default padding color from original code
            )
            my = dets["y"][fidx_in_track] + bsi  # BBox center Y
            mx = dets["x"][fidx_in_track] + bsi  # BBox center X

            # Ensure cropping indices are valid
            y_start, y_end = int(my - bs), int(my + bs * (1 + 2 * cs))
            x_start, x_end = int(mx - bs * (1 + cs)), int(mx + bs * (1 + cs))

            if not (
                y_start < y_end
                and x_start < x_end
                and y_start >= 0
                and x_start >= 0
                and y_end <= padded_image.shape[0]
                and x_end <= padded_image.shape[1]
            ):
                sys.stderr.write(
                    f"Warning: Invalid crop dimensions for frame {original_frame_num}. Skipping frame. \
                    y_start={y_start}, y_end={y_end}, x_start={x_start}, x_end={x_end}, padded_shape={padded_image.shape}\\r\\n"
                )
                continue

            face = padded_image[
                y_start:y_end,
                x_start:x_end,
            ]

            if face.size == 0:
                sys.stderr.write(
                    f"Warning: Cropped face for frame {original_frame_num} is empty. Skipping frame.\\r\\n"
                )
                continue

            vOut.write(cv2.resize(face, (224, 224)))

        audioTmp = cropFile + ".wav"
        # Use the first and last original frame numbers for audio start/end times
        audioStart = (
            (original_frames_to_crop[0]) / 25.0
        )  # Assuming 25 FPS for audio sync
        audioEnd = (
            original_frames_to_crop[-1] + 1
        ) / 25.0  # +1 to include the last frame's duration
        vOut.release()
        command = (
            "ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 -threads %d -ss %.3f -to %.3f %s -loglevel panic"
            % (
                self.args.audio_file_path,
                self.args.n_data_loader_thread,
                audioStart,
                audioEnd,
                audioTmp,
            )
        )
        subprocess.call(command, shell=True, stdout=None)  # Crop audio file
        _, audio = wavfile.read(audioTmp)
        command = (
            "ffmpeg -y -i %st.avi -i %s -threads %d -c:v copy -c:a copy %s.avi -loglevel panic"
            % (cropFile, audioTmp, self.args.n_data_loader_thread, cropFile)
        )  # Combine audio and video file
        subprocess.call(command, shell=True, stdout=None)
        os.remove(cropFile + "t.avi")
        return {"track": track, "proc_track": dets}

    def save_results(self, video_tracks):
        save_path = os.path.join(self.args.pywork_path, "tracks.pckl")
        with open(save_path, "wb") as f:
            pickle.dump(video_tracks, f)


class ActiveSpeakerDetector:
    def __init__(self, args, dtype=torch.float32):
        self.args = args
        self.dtype = dtype

    @staticmethod
    def _pad_audio_features(features, num_target_samples, num_mfcc_channels=13):
        current_samples = features.shape[0]

        if current_samples == 0:  # No audio to begin with
            return numpy.zeros(
                (num_target_samples, num_mfcc_channels), dtype=numpy.float32
            )

        if current_samples >= num_target_samples:  # Enough or more samples than target
            return features[:num_target_samples, :]

        # Determine the base for padding (the last 40ms chunk = 4 samples, or whatever is available if less)
        if current_samples >= 4:
            base_padding_chunk = features[-4:, :]
        else:  # current_samples is 1, 2, or 3
            repeats_needed = (4 + current_samples - 1) // current_samples
            base_padding_chunk = numpy.tile(features, (repeats_needed, 1))[:4, :]

        samples_to_add = num_target_samples - current_samples
        # How many 4-sample chunks we need to generate to fill the gap
        num_chunks_to_add = (samples_to_add + 3) // 4

        padding_elements = [features]
        for _ in range(num_chunks_to_add):
            padding_elements.append(base_padding_chunk)

        padded_features = numpy.vstack(padding_elements)
        return padded_features[:num_target_samples, :]

    @staticmethod
    def _pad_video_frames(frames, num_target_frames, frame_height=112, frame_width=112):
        current_frames_count = frames.shape[0]

        if current_frames_count == 0:  # No video frames to begin with
            return numpy.zeros(
                (num_target_frames, frame_height, frame_width), dtype=numpy.uint8
            )

        if current_frames_count >= num_target_frames:  # Enough or more frames
            return frames[:num_target_frames, :, :]

        frames_to_add = num_target_frames - current_frames_count
        last_frame = frames[-1:, :, :]  # Keep dimension for vstack/tile

        padding_list = [frames]
        # Efficiently create padding by tiling the last frame
        padding_block = numpy.tile(last_frame, (frames_to_add, 1, 1))
        padding_list.append(padding_block)

        padded_frames = numpy.vstack(padding_list)
        return padded_frames[:num_target_frames, :, :]

    def evaluate_network(self):
        # GPU: active speaker detection by pretrained TalkNet
        model = talkNet(device=self.args.device, dtype=self.dtype)
        model.loadParameters(self.args.pretrain_model)
        model.eval()
        files = glob.glob("%s/*.avi" % self.args.pycrop_path)
        files.sort()
        all_scores = []
        duration_set = {1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6}

        for file_idx, file in tqdm.tqdm(
            enumerate(files), total=len(files), desc="Evaluating ASD"
        ):
            file_name = os.path.splitext(os.path.basename(file))[0]

            _, audio_samples_raw_data = wavfile.read(
                os.path.join(self.args.pycrop_path, file_name + ".wav")
            )
            audio_feature_raw = python_speech_features.mfcc(
                audio_samples_raw_data, 16000, numcep=13, winlen=0.025, winstep=0.010
            )

            video_cap = cv2.VideoCapture(
                os.path.join(self.args.pycrop_path, file_name + ".avi")
            )
            video_feature_list_raw = []
            while video_cap.isOpened():
                ret, frame_img = video_cap.read()
                if ret:
                    face = cv2.cvtColor(frame_img, cv2.COLOR_BGR2GRAY)
                    face = cv2.resize(face, (224, 224))
                    face = face[
                        int(112 - (112 / 2)) : int(112 + (112 / 2)),
                        int(112 - (112 / 2)) : int(112 + (112 / 2)),
                    ]
                    video_feature_list_raw.append(face)
                else:
                    break
            video_cap.release()
            video_feature_raw = numpy.array(video_feature_list_raw)

            num_audio_mfcc_samples_raw = audio_feature_raw.shape[0]
            num_video_frames_raw = video_feature_raw.shape[0]

            # Default shape parameters (used if raw features are empty for padding)
            audio_mfcc_channels = (
                audio_feature_raw.shape[1] if num_audio_mfcc_samples_raw > 0 else 13
            )
            video_frame_h = (
                video_feature_raw.shape[1] if num_video_frames_raw > 0 else 112
            )
            video_frame_w = (
                video_feature_raw.shape[2] if num_video_frames_raw > 0 else 112
            )

            if num_video_frames_raw == 0:
                print(
                    f"[WARN] evaluate_network: {file_name} - No video frames found. Appending empty scores array."
                )
                all_scores.append(numpy.array([]))
                continue

            model_input_audio_chunks = num_audio_mfcc_samples_raw // 4
            model_input_video_chunks = num_video_frames_raw
            target_model_chunks = max(
                model_input_audio_chunks, model_input_video_chunks
            )

            if target_model_chunks == 0:
                print(
                    f"[WARN] evaluate_network: {file_name} - Both audio and video inputs are too short (target_model_chunks is 0). Scoring as all zeros for {num_video_frames_raw} frames."
                )
                all_scores.append(numpy.zeros(num_video_frames_raw))
                continue

            num_target_audio_samples_for_model = target_model_chunks * 4
            audio_feature_for_model = self._pad_audio_features(
                audio_feature_raw,
                num_target_audio_samples_for_model,
                audio_mfcc_channels,
            )

            num_target_video_frames_for_model = target_model_chunks
            video_feature_for_model = self._pad_video_frames(
                video_feature_raw,
                num_target_video_frames_for_model,
                video_frame_h,
                video_frame_w,
            )

            current_processing_length_seconds = target_model_chunks * 0.04
            model_scores_accumulator = []

            for duration in duration_set:
                if current_processing_length_seconds == 0:
                    break
                batch_size = int(
                    math.ceil(current_processing_length_seconds / duration)
                )
                if batch_size == 0:
                    continue

                scores_for_duration = []
                with torch.no_grad():
                    for i in range(batch_size):
                        seg_start_sec = i * duration
                        seg_end_sec = min(
                            (i + 1) * duration, current_processing_length_seconds
                        )
                        actual_segment_duration_sec = seg_end_sec - seg_start_sec

                        if actual_segment_duration_sec <= 0:
                            continue

                        audio_start_idx = int(
                            seg_start_sec * 100
                        )  # 100 MFCC vectors per second (1 per 10ms)
                        audio_end_idx = int(seg_end_sec * 100)

                        video_start_idx = int(
                            seg_start_sec * 25
                        )  # 25 video frames per second
                        video_end_idx = int(seg_end_sec * 25)

                        input_a_segment = audio_feature_for_model[
                            audio_start_idx:audio_end_idx, :
                        ]
                        input_v_segment = video_feature_for_model[
                            video_start_idx:video_end_idx, :, :
                        ]

                        if (
                            input_a_segment.shape[0] == 0
                            or input_v_segment.shape[0] == 0
                        ):
                            print(
                                f"[WARN] evaluate_network: {file_name} - Empty segment for duration {duration}, batch {i}. Audio shape: {input_a_segment.shape}, Video shape: {input_v_segment.shape}. Skipping."
                            )
                            continue

                        input_a = (
                            torch.FloatTensor(input_a_segment)
                            .unsqueeze(0)
                            .to(self.args.device, dtype=self.dtype)
                        )
                        input_v = (
                            torch.FloatTensor(input_v_segment)
                            .unsqueeze(0)
                            .to(self.args.device, dtype=self.dtype)
                        )

                        embed_a = model.model.forward_audio_frontend(input_a)
                        embed_v = model.model.forward_visual_frontend(input_v)

                        if embed_a.size(1) != embed_v.size(1):
                            print(
                                f"[INFO] evaluate_network: {file_name} - Embed lengths differ (A:{embed_a.size(1)}, V:{embed_v.size(1)}) Duration:{duration}s, Batch:{i}. Padding shorter."
                            )
                            if embed_a.size(1) < embed_v.size(1):
                                diff = embed_v.size(1) - embed_a.size(1)
                                last_vector_a = embed_a[
                                    :, -1:, :
                                ]  # Keep dim for repeat
                                padding_a = last_vector_a.repeat(1, diff, 1)
                                embed_a = torch.cat((embed_a, padding_a), dim=1)
                            else:  # embed_v.size(1) < embed_a.size(1)
                                diff = embed_a.size(1) - embed_v.size(1)
                                last_vector_v = embed_v[
                                    :, -1:, :
                                ]  # Keep dim for repeat
                                padding_v = last_vector_v.repeat(1, diff, 1)
                                embed_v = torch.cat((embed_v, padding_v), dim=1)

                        if (
                            embed_a.size(1) == 0
                        ):  # This check should ideally be hit less or not at all if padding works
                            print(
                                f"[WARN] evaluate_network: {file_name} - Zero sequence length for embeddings after potential truncation. Skipping batch."
                            )
                            continue

                        context_a, context_v = model.model.forward_cross_attention(
                            embed_a, embed_v
                        )
                        out = model.model.forward_audio_visual_backend(
                            context_a, context_v
                        )
                        score_from_model = model.lossAV.forward(out, labels=None)
                        scores_for_duration.extend(score_from_model)

                if scores_for_duration:
                    model_scores_accumulator.append(scores_for_duration)

            computed_model_scores = numpy.array([])
            if model_scores_accumulator:
                # Average scores across different duration runs
                # This part assumes each duration run ideally produces `target_model_chunks` scores
                min_len_across_durations = (
                    min(len(s) for s in model_scores_accumulator)
                    if model_scores_accumulator
                    else 0
                )
                if (
                    min_len_across_durations != target_model_chunks
                    and min_len_across_durations > 0
                ):  # if any list is shorter but not empty
                    print(
                        f"[WARN] evaluate_network: {file_name} - Score lists from durations have inconsistent lengths. Shortest: {min_len_across_durations}, Expected: {target_model_chunks}. Truncating all to shortest for averaging."
                    )

                # Truncate all score lists to the minimum consistent length (ideally target_model_chunks)
                processed_accumulator = []
                for s_list in model_scores_accumulator:
                    if (
                        len(s_list) >= min_len_across_durations
                        and min_len_across_durations > 0
                    ):
                        processed_accumulator.append(s_list[:min_len_across_durations])

                if processed_accumulator:
                    try:
                        computed_model_scores = numpy.round(
                            (numpy.mean(numpy.array(processed_accumulator), axis=0)), 1
                        ).astype(float)
                    except Exception as e:
                        print(
                            f"[ERROR] evaluate_network: {file_name} - Error during score averaging: {e}. Computed scores will be empty."
                        )
                        computed_model_scores = numpy.array(
                            []
                        )  # Ensure it's an array for .size
                else:  # All score lists were empty or became empty after truncation
                    computed_model_scores = numpy.array([])

            final_scores_this_file = numpy.zeros(
                num_video_frames_raw
            )  # Default to zeros

            if computed_model_scores.size > 0:
                # computed_model_scores corresponds to target_model_chunks
                # final_scores_this_file needs to be num_video_frames_raw long

                len_to_copy = min(computed_model_scores.size, num_video_frames_raw)
                final_scores_this_file[:len_to_copy] = computed_model_scores[
                    :len_to_copy
                ]

                # If original video was longer than the scores we have (after mapping from target_model_chunks)
                # This happens if num_video_frames_raw > computed_model_scores.size (which is min_len_across_durations)
                if num_video_frames_raw > len_to_copy and len_to_copy > 0:
                    last_valid_score = final_scores_this_file[len_to_copy - 1]
                    final_scores_this_file[len_to_copy:] = last_valid_score
            # If computed_model_scores.size is 0, final_scores_this_file remains all zeros (num_video_frames_raw long)

            all_scores.append(final_scores_this_file)

        print(time.strftime("%Y-%m-%d %H:%M:%S") + " Scores extracted")
        return all_scores

    def save_results(self, scores):
        save_path = os.path.join(self.args.pywork_path, "scores.pckl")
        with open(save_path, "wb") as fil:
            pickle.dump(scores, fil)


class Pipeline:
    def __init__(
        self,
        video_path: str,
        n_data_loader_thread: int = 10,
        facedet_scale: float = 0.25,
        min_track: int = 10,
        num_failed_det: int = 10,
        min_face_size: int = 1,
        crop_scale: float = 0.40,
        device: Literal["auto", "cpu", "cuda"] = "auto",
        scene_detector_backend: Literal["opencv", "pyav"] = "opencv",
        dtype: Literal["float32", "float16"] = "float16",
        face_detection_avg_time_threshold: float = 1.0,
        face_detection_min_frames_for_avg: int = 10,
        enable_skip_slow_face_detection: bool = False,
        face_detection_threshold: float = 0.6,
        face_detection_max_size: int = 1920,
        face_detection_batch_size: int = 32,
        **kwargs,
    ):
        self.device = resolve_device(device=device)
        self.dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
        }[dtype]
        self.scene_detector_backend: str = scene_detector_backend

        self.video_path = video_path

        video_name = self._get_filename(file_path=video_path)
        now_iso = datetime.now().isoformat(timespec="seconds")
        self.save_path = cache_dir / f"{video_name}-{now_iso}"
        self.pretrain_model = hf_hub_download(
            repo_id="AlekseyKorshuk/talknet-asd",
            filename="pretrain_TalkSet.model",
        )

        self.n_data_loader_thread = n_data_loader_thread
        self.facedet_scale = facedet_scale
        self.min_track = min_track
        self.num_failed_det = num_failed_det
        self.min_face_size = min_face_size
        self.crop_scale = crop_scale

        self.face_detection_avg_time_threshold = face_detection_avg_time_threshold
        self.face_detection_min_frames_for_avg = face_detection_min_frames_for_avg
        self.enable_skip_slow_face_detection = enable_skip_slow_face_detection
        self.face_detection_threshold = face_detection_threshold
        self.face_detection_max_size = face_detection_max_size
        self.face_detection_batch_size = face_detection_batch_size

        self.pyavi_path = None
        self.pywork_path = None
        self.pycrop_path = None

        self._setup_paths()

        try:
            self.video_preprocessor = VideoPreprocessor(args=self)
            self.face_processor = FaceProcessor(args=self)
            self.speaker_detector = ActiveSpeakerDetector(args=self, dtype=self.dtype)
        except Exception:
            print("Error while trying to initialize ASD pipeline")
            self._cleanup_cache()
            raise

    def _setup_paths(self):
        self.pyavi_path = self.save_path / "pyavi"
        self.pywork_path = self.save_path / "pywork"
        self.pycrop_path = self.save_path / "pycrop"

        self._cleanup_cache()

        self.pyavi_path.mkdir(parents=True, exist_ok=True)
        self.pywork_path.mkdir(parents=True, exist_ok=True)
        self.pycrop_path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _get_filename(file_path: str):
        base_name = os.path.basename(file_path)
        filename, _ = os.path.splitext(base_name)
        return filename

    def run(self):
        """Run complete pipeline"""
        try:
            # 1. Extract and prepare media
            self.video_preprocessor.extract_video()
            self.video_preprocessor.extract_audio()

            # 2. Face processing
            scenes = self.face_processor.scenes_detect()
            detection_successful, faces = self.face_processor.detect_faces()

            if not detection_successful:
                sys.stderr.write(
                    time.strftime("%Y-%m-%d %H:%M:%S")
                    + f" Skipping video {self.video_path} due to slow face detection.\r\n"
                )
                self._cleanup_cache()
                return

            all_tracks = self.face_processor.track_faces(
                scenes=scenes, face_detections=faces
            )
            video_tracks = self.face_processor.video_tracks(all_tracks=all_tracks)
            self.face_processor.save_results(video_tracks=video_tracks)

            # 3. Active speaker detection
            scores = self.speaker_detector.evaluate_network()
            self.speaker_detector.save_results(scores=scores)

            visualization(scores=scores, tracks=video_tracks, args=self)
        except Exception:
            print(f"Error while trying to process video {self.video_path}")
            self._cleanup_cache()
            raise

    def _cleanup_cache(self):
        if self.save_path.exists():
            print(f"Deleting {self.save_path}")
            shutil.rmtree(self.save_path)
