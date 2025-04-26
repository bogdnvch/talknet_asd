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
from typing import Literal
from datetime import datetime

import torch
import numpy
import python_speech_features
import cv2
from scipy import signal
from scipy.io import wavfile
from scipy.interpolate import interp1d

from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector

from huggingface_hub import hf_hub_download
from ultralytics import YOLO

from talknet_asd.talkNet import talkNet
from talknet_asd.utils.resolve_device import resolve_device

warnings.filterwarnings("ignore")

cache_dir = Path.home() / ".cache" / "talknet_asd"
cache_dir.mkdir(parents=True, exist_ok=True)


def visualization(tracks, scores, args):
    flist = glob.glob(os.path.join(args.pyframes_path, "*.jpg"))
    flist.sort()
    faces = [[] for i in range(len(flist))]
    for tidx, track in enumerate(tracks):
        score = scores[tidx]
        for fidx, frame in enumerate(track["track"]["frame"].tolist()):
            s = score[
                max(fidx - 2, 0) : min(fidx + 3, len(score) - 1)
            ]  # average smoothing
            s = numpy.mean(s)
            faces[frame].append(
                {
                    "track": tidx,
                    "score": float(s),
                    "s": track["proc_track"]["s"][fidx],
                    "x": track["proc_track"]["x"][fidx],
                    "y": track["proc_track"]["y"][fidx],
                }
            )
    firstImage = cv2.imread(flist[0])
    fw = firstImage.shape[1]
    fh = firstImage.shape[0]
    vOut = cv2.VideoWriter(
        os.path.join(args.pyavi_path, "video_only.avi"),
        cv2.VideoWriter_fourcc(*"XVID"),
        25,
        (fw, fh),
    )
    colorDict = {0: 0, 1: 255}
    for fidx, fname in tqdm.tqdm(enumerate(flist), total=len(flist)):
        image = cv2.imread(fname)
        for face in faces[fidx]:
            clr = colorDict[int((face["score"] >= 0))]
            txt = round(face["score"], 1)
            cv2.rectangle(
                image,
                (int(face["x"] - face["s"]), int(face["y"] - face["s"])),
                (int(face["x"] + face["s"]), int(face["y"] + face["s"])),
                (0, clr, 255 - clr),
                10,
            )
            cv2.putText(
                image,
                "%s" % (txt),
                (int(face["x"] - face["s"]), int(face["y"] - face["s"])),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, clr, 255 - clr),
                5,
            )
        vOut.write(image)
    vOut.release()
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
        """Extract video to frames"""
        self.args.video_file_path = os.path.join(self.args.pyavi_path, "video.avi")
        command = (
            "ffmpeg -y -i %s -qscale:v 2 -threads %d -async 1 -r 25 %s -loglevel panic"
            % (
                self.args.video_path,
                self.args.n_data_loader_thread,
                self.args.video_file_path,
            )
        )
        subprocess.call(command, shell=True)
        sys.stderr.write(
            time.strftime("%Y-%m-%d %H:%M:%S")
            + " Extract the video and save in %s \r\n" % self.args.video_file_path
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
        command = (
            "ffmpeg -y -i %s -qscale:v 2 -threads %d -f image2 %s -loglevel panic"
            % (
                self.args.video_file_path,
                self.args.n_data_loader_thread,
                os.path.join(self.args.pyframes_path, "%06d.jpg"),
            )
        )
        subprocess.call(command, shell=True, stdout=None)
        sys.stderr.write(
            time.strftime("%Y-%m-%d %H:%M:%S")
            + " Extract the frames and save in %s \r\n" % self.args.pyframes_path
        )


class FaceProcessor:
    def __init__(self, args):
        self.args = args
        self.detector = YOLO(
            hf_hub_download(
                repo_id="AlekseyKorshuk/yolov11-face",
                filename="yolov11n-face.pt",
            ),
            task="detect",
            verbose=False,
        )
        self.detector.to(self.args.device)

    def scenes_detect(self):
        videoManager = VideoManager([self.args.video_file_path])
        statsManager = StatsManager()
        sceneManager = SceneManager(statsManager)
        sceneManager.add_detector(ContentDetector())
        baseTimecode = videoManager.get_base_timecode()
        videoManager.set_downscale_factor()
        videoManager.start()
        sceneManager.detect_scenes(frame_source=videoManager)
        sceneList = sceneManager.get_scene_list(baseTimecode)
        savePath = os.path.join(self.args.pywork_path, "scene.pckl")
        if sceneList == []:
            sceneList = [
                (videoManager.get_base_timecode(), videoManager.get_current_timecode())
            ]
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

    def detect_faces(self):
        """Run face detection on all frames"""
        flist = glob.glob(os.path.join(self.args.pyframes_path, "*.jpg"))
        flist.sort()
        detections = []

        pbar = tqdm.tqdm(flist, total=len(flist), desc="Detecting faces")

        for fidx, fname in enumerate(pbar):
            image = cv2.imread(fname)
            image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.detector(image_np, verbose=False)

            frame_detections = []
            for result in results:
                for box in result.boxes:
                    frame_detections.append(
                        {
                            "frame": fidx,
                            "bbox": box.xyxy.tolist()[0],
                            "conf": float(box.conf.tolist()[0]),
                        }
                    )
            detections.append(frame_detections)

        with open(os.path.join(self.args.pywork_path, "faces.pckl"), "wb") as f:
            pickle.dump(detections, f)

        sys.stderr.write(
            time.strftime("%Y-%m-%d %H:%M:%S")
            + " Face detection and save in %s \r\n" % (self.args.pywork_path)
        )
        return detections

    def track_faces(self, scenes, face_detections):
        all_tracks = []
        for shot in scenes:
            if (
                shot[1].frame_num - shot[0].frame_num >= self.args.min_track
            ):  # Discard the shot frames less than min_track frames
                all_tracks.extend(
                    self.track_shot(
                        face_detections[shot[0].frame_num : shot[1].frame_num]
                    )
                )  # 'frames' to present this tracks' timestep, 'bbox' presents the location of the faces
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
            for frameFaces in sceneFaces:
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
                bboxesI = []
                for ij in range(0, 4):
                    interpfn = interp1d(frameNum, bboxes[:, ij])
                    bboxesI.append(interpfn(frameI))
                bboxesI = numpy.stack(bboxesI, axis=1)
                if (
                    max(
                        numpy.mean(bboxesI[:, 2] - bboxesI[:, 0]),
                        numpy.mean(bboxesI[:, 3] - bboxesI[:, 1]),
                    )
                    > self.args.min_face_size
                ):
                    tracks.append({"frame": frameI, "bbox": bboxesI})
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
        for ii, track in enumerate(all_tracks):
            video_tracks.append(
                self.crop_video(track, os.path.join(self.args.pycrop_path, f"{ii:05d}"))
            )

        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} Face Crop")

        return video_tracks

    def crop_video(self, track, cropFile):
        self.args.audio_file_path = os.path.join(self.args.pyavi_path, "audio.wav")

        flist = glob.glob(
            os.path.join(self.args.pyframes_path, "*.jpg")
        )  # Read the frames
        flist.sort()
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
        for fidx, frame in enumerate(track["frame"]):
            cs = self.args.crop_scale
            bs = dets["s"][fidx]  # Detection box size
            bsi = int(bs * (1 + 2 * cs))  # Pad videos by this amount
            image = cv2.imread(flist[frame])
            frame = numpy.pad(
                image,
                ((bsi, bsi), (bsi, bsi), (0, 0)),
                "constant",
                constant_values=(110, 110),
            )
            my = dets["y"][fidx] + bsi  # BBox center Y
            mx = dets["x"][fidx] + bsi  # BBox center X
            face = frame[
                int(my - bs) : int(my + bs * (1 + 2 * cs)),
                int(mx - bs * (1 + cs)) : int(mx + bs * (1 + cs)),
            ]
            vOut.write(cv2.resize(face, (224, 224)))
        audioTmp = cropFile + ".wav"
        audioStart = (track["frame"][0]) / 25
        audioEnd = (track["frame"][-1] + 1) / 25
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
            length = min(
                (audio_feature.shape[0] - audio_feature.shape[0] % 4) / 100,
                video_feature.shape[0] / 25,
            )
            audio_feature = audio_feature[: int(round(length * 100)), :]
            video_feature = video_feature[: int(round(length * 25)), :, :]
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
        dtype: Literal["float32", "float16", "bfloat16"] = "float32",
        **kwargs,
    ):
        self.device = resolve_device(device=device)
        self.dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[dtype]

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

        self.pyavi_path = None
        self.pyframes_path = None
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
        self.pyframes_path = self.save_path / "pyframes"
        self.pywork_path = self.save_path / "pywork"
        self.pycrop_path = self.save_path / "pycrop"

        self._cleanup_cache()

        self.pyavi_path.mkdir(parents=True, exist_ok=True)
        self.pyframes_path.mkdir(parents=True, exist_ok=True)
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
            self.video_preprocessor.extract_frames()

            # 2. Face processing
            scenes = self.face_processor.scenes_detect()
            faces = self.face_processor.detect_faces()
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
