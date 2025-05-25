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
    print(f"[DEBUG] visualization: Total frames for visualization: {num_frames}")
    # Example of checking a specific frame, e.g., frame 1161 as in the user's example
    if num_frames > 1161:
        print(f"[DEBUG] visualization: Faces content for frame 1161: {faces[1161]}")
    if num_frames > 0:
        print(f"[DEBUG] visualization: Faces content for last frame ({num_frames-1}): {faces[num_frames-1]}")


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
        self.detector_backend: str = args.detector_backend

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
                    print(
                        f"Adjusting end of last scene from frame {last_scene_end.get_frames()} to {video_duration_frames}"
                    )
                    corrected_last_scene_end = FrameTimecode(
                        timecode=video_duration_frames, fps=video_fps
                    )
                    current_scene_list[-1] = (
                        last_scene_start,
                        corrected_last_scene_end,
                    )
                elif last_scene_end.get_frames() > video_duration_frames:
                    print(
                        f"Warning: Last scene end ({last_scene_end.get_frames()}) is beyond video duration ({video_duration_frames}). "
                        f"Attempting to cap it."
                    )
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
        detections = []
        total_detection_time = 0.0
        threshold = 0.6
        max_size = 1920
        resize = 1

        is_cuda_available = torch.cuda.is_available()
        gpu_id = 0 if is_cuda_available else -1
        detector = RetinaFace(gpu_id=gpu_id, fp16=True)

        cap = cv2.VideoCapture(self.args.video_file_path)
        if not cap.isOpened():
            sys.stderr.write(
                f"Error: Could not open video file {self.args.video_file_path}\\r\\n"
            )
            return False, detections

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            sys.stderr.write(
                f"Error: Video file {self.args.video_file_path} appears to have no frames or is corrupted.\\r\\n"
            )
            cap.release()
            return False, detections

        pbar = tqdm.tqdm(
            range(total_frames), total=total_frames, desc="Detecting faces"
        )

        for fidx in pbar:
            frame_start_time = time.time()
            frame_detections = []

            ret, img = cap.read()
            if not ret:
                sys.stderr.write(
                    f"Warning: Could not read frame at index {fidx} from {self.args.video_file_path}. Stopping detection.\\r\\n"
                )
                break  # Stop if a frame cannot be read

            try:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                faces = detector(
                    img,
                    threshold=threshold,
                    resize=resize,
                    max_size=max_size,
                    return_dict=True,
                )
                if isinstance(faces, list) and len(faces) > 0:
                    for face in faces:
                        facial_area = face["box"]
                        if face["score"] < threshold:
                            continue
                        if isinstance(facial_area, dict):
                            if not all(
                                key in facial_area for key in ["x", "y", "w", "h"]
                            ):
                                continue

                            x1 = facial_area["x"]
                            y1 = facial_area["y"]
                            x2 = facial_area["x"] + facial_area["w"]
                            y2 = facial_area["y"] + facial_area["h"]
                        else:
                            x1 = facial_area[0]
                            y1 = facial_area[1]
                            x2 = facial_area[2]
                            y2 = facial_area[3]

                        if x2 - x1 <= 0 or y2 - y1 <= 0:
                            continue

                        bbox = [x1, y1, x2, y2]

                        frame_detections.append(
                            {"frame": fidx, "bbox": bbox, "conf": face["score"]}
                        )

            except Exception as e:
                print(f"Error processing frame at index {fidx}: {str(e)}")
                # Continue to the next frame even if one frame fails
                detections.append(
                    []
                )  # Append empty list for this failed frame to keep indices correct
                total_detection_time += (
                    time.time() - frame_start_time
                )  # count this frame's attempt time
                continue

            detections.append(frame_detections)
            frame_end_time = time.time()
            total_detection_time += frame_end_time - frame_start_time

            if fidx + 1 >= self.args.face_detection_min_frames_for_avg:
                avg_time_per_frame = total_detection_time / (fidx + 1)
                if (
                    self.args.enable_skip_slow_face_detection
                    and avg_time_per_frame > self.args.face_detection_avg_time_threshold
                ):
                    sys.stderr.write(
                        time.strftime("%Y-%m-%d %H:%M:%S")
                        + f" Face detection too slow (avg {avg_time_per_frame:.2f}s/frame > "
                        + f"{self.args.face_detection_avg_time_threshold:.2f}s/frame). "
                        + f"Skipping video {self.args.video_path}.\\r\\n"
                    )
                    cap.release()
                    return False, detections  # Abort and signal failure

        cap.release()

        with open(os.path.join(self.args.pywork_path, "faces.pckl"), "wb") as f:
            pickle.dump(detections, f)

        sys.stderr.write(
            time.strftime("%Y-%m-%d %H:%M:%S")
            + " Face detection and save in %s \r\n" % (self.args.pywork_path)
        )
        return True, detections

    def track_faces(self, scenes, face_detections):
        all_tracks = []
        print(f"[DEBUG] track_faces: Input scenes: {scenes}")
        print(f"[DEBUG] track_faces: Input face_detections length: {len(face_detections)}")
        for i, shot in enumerate(scenes):
            print(f"[DEBUG] track_faces: Processing shot {i}: Start frame {shot[0].frame_num}, End frame {shot[1].frame_num}")
            if (
                shot[1].frame_num - shot[0].frame_num >= self.args.min_track
            ):  # Discard the shot frames less than min_track frames
                current_shot_detections = face_detections[shot[0].frame_num : shot[1].frame_num + 1]
                print(f"[DEBUG] track_faces: Detections for shot {i} (frames {shot[0].frame_num}-{shot[1].frame_num}): Count = {len(current_shot_detections)}")
                if shot[1].frame_num < len(face_detections):
                    print(f"[DEBUG] track_faces: Last frame detections for shot {i} (frame {shot[1].frame_num}): {face_detections[shot[1].frame_num]}")
                else:
                    print(f"[DEBUG] track_faces: Last frame index {shot[1].frame_num} is out of bounds for face_detections (len {len(face_detections)})")

                shot_tracks = self.track_shot(
                       current_shot_detections
                    )
                print(f"[DEBUG] track_faces: Tracks generated for shot {i}: {len(shot_tracks)}")
                if shot_tracks:
                    for t_idx, tr in enumerate(shot_tracks):
                        print(f"[DEBUG] track_faces: Shot {i}, Track {t_idx}: Frames {tr['frame'][0]} to {tr['frame'][-1]}, Length {len(tr['frame'])}")
                all_tracks.extend(shot_tracks)
            else:
                print(f"[DEBUG] track_faces: Shot {i} (frames {shot[0].frame_num}-{shot[1].frame_num}) discarded as too short.")
        sys.stderr.write(
            time.strftime("%Y-%m-%d %H:%M:%S")
            + " Face track and detected %d tracks \r\n" % len(all_tracks)
        )
        return all_tracks

    def track_shot(self, sceneFaces):
        iouThres = 0.5  # Minimum IOU between consecutive face detections
        tracks = []
        print(f"[DEBUG] track_shot: Input sceneFaces length: {len(sceneFaces)}")
        if sceneFaces:
            print(f"[DEBUG] track_shot: First frame in sceneFaces (frame {sceneFaces[0][0]['frame' ]if sceneFaces[0] else 'N/A'}): {sceneFaces[0]}")
            print(f"[DEBUG] track_shot: Last frame in sceneFaces (frame {sceneFaces[-1][0]['frame'] if sceneFaces[-1] else 'N/A'}): {sceneFaces[-1]}")

        while True:
            track = []
            for i, frameFaces in enumerate(sceneFaces):
                #print(f"[DEBUG] track_shot: Processing frame {i} within scene. Num faces: {len(frameFaces)}")
                for face in frameFaces:
                    if track == []:
                        track.append(face)
                        frameFaces.remove(face)
                        #print(f"[DEBUG] track_shot: Started new track with face: {face}")
                    elif face["frame"] - track[-1]["frame"] <= self.args.num_failed_det:
                        iou = self.bb_intersection_over_union(
                            face["bbox"], track[-1]["bbox"]
                        )
                        if iou > iouThres:
                            track.append(face)
                            frameFaces.remove(face)
                            #print(f"[DEBUG] track_shot: Added face to track: {face}, IOU: {iou}")
                            continue
                    else:
                        #print(f"[DEBUG] track_shot: Face skipped, condition not met: face_frame={face[\"frame\"]}, track_last_frame={track[-1][\"frame\"]}, num_failed_det={self.args.num_failed_det}")
                        break
            if track == []:
                #print("[DEBUG] track_shot: No more faces to form a track.")
                break
            elif len(track) > self.args.min_track:
                print(f"[DEBUG] track_shot: Track created with {len(track)} faces. Frames {track[0]['frame']} to {track[-1]['frame']}")
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
                    print(f"[DEBUG] track_shot: Track appended. Interpolated frames: {frameI[0]} to {frameI[-1]}, Total: {len(frameI)}")
        print(f"[DEBUG] track_shot: Total tracks created for this shot: {len(tracks)}")
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

        print(f"[DEBUG] crop_video: Processing track for cropFile {cropFile}")
        print(f"[DEBUG] crop_video: Track details - Frames: {track['frame'][0]} to {track['frame'][-1]}, Length: {len(track['frame'])}")

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
        print(f"[DEBUG] crop_video: Original frames to crop: {original_frames_to_crop[0]} to {original_frames_to_crop[-1]}, Count: {len(original_frames_to_crop)}")

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
        self.model = talkNet(device=args.device, dtype=dtype)
        self.model.loadParameters(args.pretrain_model)
        self.model.eval()

    def evaluate_network(self):
        # GPU: active speaker detection by pretrained TalkNet
        files = glob.glob("%s/*.avi" % self.args.pycrop_path)
        files.sort()
        all_scores = []
        duration_set = {1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6}
        for file in tqdm.tqdm(files, total=len(files)):
            file_name = os.path.splitext(file.split("/")[-1])[0]  # Load audio and video
            print(f"[DEBUG] evaluate_network: Processing file {file_name}")
            _, audio = wavfile.read(
                os.path.join(self.args.pycrop_path, file_name + ".wav")
            )
            audio_feature = python_speech_features.mfcc(
                audio, 16000, numcep=13, winlen=0.025, winstep=0.010
            )
            video = cv2.VideoCapture(
                os.path.join(self.args.pycrop_path, file_name + ".avi")
            )
            video_feature = []
            while video.isOpened():
                ret, frames = video.read()
                if ret == True:
                    face = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
                    face = cv2.resize(face, (224, 224))
                    face = face[
                        int(112 - (112 / 2)) : int(112 + (112 / 2)),
                        int(112 - (112 / 2)) : int(112 + (112 / 2)),
                    ]
                    video_feature.append(face)
                else:
                    break
            video.release()
            video_feature = numpy.array(video_feature)
            print(f"[DEBUG] evaluate_network: {file_name} - Audio feature shape: {audio_feature.shape}, Video feature shape: {video_feature.shape}")
            length = min(
                audio_feature.shape[0] / 100,
                video_feature.shape[0] / 25,
            )
            print(f"[DEBUG] evaluate_network: {file_name} - Calculated length: {length}")
            audio_feature = audio_feature[: int(round(length * 100)), :]
            video_feature = video_feature[: int(round(length * 25)), :, :]
            print(f"[DEBUG] evaluate_network: {file_name} - Trimmed Audio feature shape: {audio_feature.shape}, Trimmed Video feature shape: {video_feature.shape}")
            all_score = []  # Evaluation use TalkNet
            for duration in duration_set:
                batch_size = int(math.ceil(length / duration))
                scores = []
                with torch.no_grad():
                    for i in range(batch_size):
                        input_a = (
                            torch.FloatTensor(
                                audio_feature[
                                    i * duration * 100 : (i + 1) * duration * 100, :
                                ]
                            )
                            .unsqueeze(0)
                            .to(self.args.device, dtype=self.dtype)
                        )
                        input_v = (
                            torch.FloatTensor(
                                video_feature[
                                    i * duration * 25 : (i + 1) * duration * 25, :, :
                                ]
                            )
                            .unsqueeze(0)
                            .to(self.args.device, dtype=self.dtype)
                        )
                        embed_a = self.model.model.forward_audio_frontend(input_a)
                        embed_v = self.model.model.forward_visual_frontend(input_v)
                        embed_a, embed_v = self.model.model.forward_cross_attention(
                            embed_a, embed_v
                        )
                        out = self.model.model.forward_audio_visual_backend(
                            embed_a, embed_v
                        )
                        score = self.model.lossAV.forward(out, labels=None)
                        scores.extend(score)
                all_score.append(scores)
            all_score = numpy.round(
                (numpy.mean(numpy.array(all_score), axis=0)), 1
            ).astype(float)
            print(f"[DEBUG] evaluate_network: {file_name} - Final scores for this file (first 5): {all_score[:5]}, Length: {len(all_score)}")
            all_scores.append(all_score)
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
        device: Literal["auto", "cpu", "cuda", "mps"] = "auto",
        detector_backend: str = "yolov8",
        scene_detector_backend: Literal["opencv", "pyav"] = "opencv",
        dtype: Literal["float32", "float16", "bfloat16"] = "float32",
        face_detection_avg_time_threshold: float = 1.0,
        face_detection_min_frames_for_avg: int = 10,
        enable_skip_slow_face_detection: bool = False,
        **kwargs,
    ):
        self.device = resolve_device(device=device)
        self.dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[dtype]
        self.detector_backend: str = detector_backend
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
