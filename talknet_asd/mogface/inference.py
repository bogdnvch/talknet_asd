import gc
import os
import sys

from huggingface_hub import hf_hub_download
import torch
import numpy as np
import cv2

from tqdm import tqdm
import threading
import queue

from talknet_asd.mogface.model import build_mogface_e

# Optional GPU NMS (torchvision); will gracefully fall back to CPU NMS
try:
    from torchvision.ops import nms as tv_nms  # type: ignore

    _HAS_TV_NMS = True
except Exception:
    tv_nms = None  # type: ignore
    _HAS_TV_NMS = False

# Cap external thread pools to avoid CPU oversubscription (must run after imports)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
try:
    cv2.setNumThreads(1)
except Exception:
    pass


# -----------------------------
# Utilities
# -----------------------------


def preprocess_image(bgr_img):
    img = bgr_img.astype(np.float32)
    img_mean = (np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255)[::-1]
    img_std = (np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255)[::-1]
    img -= img_mean
    img /= img_std
    img /= 255.0
    img = img[:, :, ::-1].copy()
    return img


def anchors_cxcywh_to_xyxy(anchors):
    return np.concatenate(
        (
            anchors[:, :2] - (anchors[:, 2:] - 1) / 2,
            anchors[:, :2] + (anchors[:, 2:] - 1) / 2,
        ),
        axis=1,
    )


def anchors_xyxy_to_cxcywh(anchors):
    centers = (anchors[:, :2] + anchors[:, 2:]) / 2
    wh = anchors[:, 2:] - anchors[:, :2] + 1
    return np.concatenate((centers, wh), axis=1)


def _compute_resize_ratio(height, width, max_long_side):
    """Compute aspect-preserving resize ratio so that the longest side <= max_long_side.
    Returns 1.0 if no resizing is needed."""
    if max_long_side is None or max_long_side <= 0:
        return 1.0
    longest = max(int(height), int(width))
    if longest <= int(max_long_side):
        return 1.0
    return float(max_long_side) / float(longest)


def _maybe_resize_image(bgr_img, max_long_side, verbose=False, tag="image"):
    """Resize BGR image if its longest side exceeds max_long_side, preserving aspect ratio.
    Returns (resized_img, ratio). The ratio is processed/original (<= 1.0)."""
    h, w = int(bgr_img.shape[0]), int(bgr_img.shape[1])
    ratio = _compute_resize_ratio(h, w, max_long_side)
    if ratio < 1.0:
        new_w = max(1, int(round(w * ratio)))
        new_h = max(1, int(round(h * ratio)))
        resized = cv2.resize(bgr_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        if verbose:
            print(
                f"Resized {tag} from ({h}x{w}) to ({new_h}x{new_w}) because longest side {max(h, w)} > max_source_image_size {max_long_side}"
            )
        return resized, ratio
    return bgr_img, 1.0


def generate_prior_boxes(
    img_height,
    img_width,
    scale_list=(0.68,),
    aspect_ratio_list=(1.0,),
    stride_list=(4, 8, 16, 32, 64, 128),
    anchor_size_list=(16, 32, 64, 128, 256, 512),
):
    final_anchor_list = []
    for idx, stride in enumerate(stride_list):
        anchor_list = []
        cur_img_height = img_height
        cur_img_width = img_width
        tmp_stride = stride
        while tmp_stride != 1:
            tmp_stride = tmp_stride // 2
            cur_img_height = (cur_img_height + 1) // 2
            cur_img_width = (cur_img_width + 1) // 2
        for i in range(cur_img_height):
            for j in range(cur_img_width):
                for scale in scale_list:
                    cx = (j + 0.5) * stride
                    cy = (i + 0.5) * stride
                    side_x = anchor_size_list[idx] * scale
                    side_y = anchor_size_list[idx] * scale
                    for ratio in aspect_ratio_list:
                        anchor_list.append(
                            [cx, cy, side_x / np.sqrt(ratio), side_y * np.sqrt(ratio)]
                        )
        final_anchor_list.append(anchor_list)
    final_anchor_arr = np.concatenate(final_anchor_list, axis=0).astype(np.float32)
    return anchors_cxcywh_to_xyxy(final_anchor_arr)


def decode_boxes(loc_tensor, anchors_cxcywh_tensor):
    boxes = torch.cat(
        (
            anchors_cxcywh_tensor[:, :2]
            + loc_tensor[:, :2] * anchors_cxcywh_tensor[:, 2:],
            anchors_cxcywh_tensor[:, 2:] * torch.exp(loc_tensor[:, 2:]),
        ),
        1,
    )
    boxes[:, 0] -= (boxes[:, 2] - 1) / 2
    boxes[:, 1] -= (boxes[:, 3] - 1) / 2
    boxes[:, 2] += boxes[:, 0] - 1
    boxes[:, 3] += boxes[:, 1] - 1
    return boxes


def nms_numpy(dets, thresh=0.3):
    if dets.shape[0] == 0:
        return []
    x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep


def bbox_vote(det, vote_th, max_per_img):
    if det.size == 0:
        return det
    order = det[:, 4].ravel().argsort()[::-1]
    det = det[order, :]
    det[:, :4] = np.round(det[:, :4])
    dets = None
    while det.shape[0] > 0:
        box_w = np.maximum(det[:, 2] - det[:, 0], 0)
        box_h = np.maximum(det[:, 3] - det[:, 1], 0)
        area = box_w * box_h
        xx1 = np.maximum(det[0, 0], det[:, 0])
        yy1 = np.maximum(det[0, 1], det[:, 1])
        xx2 = np.minimum(det[0, 2], det[:, 2])
        yy2 = np.minimum(det[0, 3], det[:, 3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        union = area[0] + area[:] - inter
        union[union <= 0] = 1
        o = inter / union
        o[0] = 1
        merge_index = np.where(o >= vote_th)[0]
        det_accu = det[merge_index, :]
        det = np.delete(det, merge_index, 0)
        if merge_index.shape[0] <= 1:
            if dets is None:
                dets = det_accu
            else:
                dets = np.vstack((dets, det_accu))
            continue
        det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:], (1, 4))
        max_score = np.max(det_accu[:, 4])
        det_accu_sum = np.zeros((1, 5))
        det_accu_sum[:, 0:4] = np.sum(det_accu[:, 0:4], axis=0) / np.sum(
            det_accu[:, -1:]
        )
        det_accu_sum[:, 4] = max_score
        dets = dets if dets is not None else det_accu_sum
        if dets is not det_accu_sum:
            dets = np.vstack((dets, det_accu_sum))
    if dets.shape[0] > max_per_img:
        dets = dets[0:max_per_img, :]
    return dets


# -------------------------------------------------
# Multiprocessing workers (CPU-only) and helpers
# -------------------------------------------------


def _compute_num_threads(total_procs):
    N = os.cpu_count() or 1
    return max(1, N // max(1, total_procs))


def _batch_preparer_worker(proc_idx, cfg, batch_q, prepped_q):
    """
    CPU worker (thread-safe): receives (batch, max_h, max_w) with per-variant dicts,
    builds a padded NumPy batch and sends (batch, max_h, max_w, xt_np) to the GPU stage.
    Uses only CPU and OpenCV. No Torch tensors here.
    """
    import numpy as _np  # local import for fork-safety
    import cv2 as _cv2

    pad_to_multiple = cfg.get("pad_to_multiple", None)

    while True:
        try:
            item = batch_q.get()
        except (EOFError, OSError):
            break
        if item is None:
            # Forward sentinel downstream and exit
            try:
                prepped_q.put(None)
            except (EOFError, OSError):
                pass
            break
        batch, max_h, max_w = item
        if pad_to_multiple and int(pad_to_multiple) > 1:
            m = int(pad_to_multiple)
            max_h = ((int(max_h) + m - 1) // m) * m
            max_w = ((int(max_w) + m - 1) // m) * m

        B = len(batch)
        # Always prepare as float32 NumPy (autocast will handle dtype on device)
        xt_np = _np.zeros((B, int(max_h), int(max_w), 3), dtype=_np.float32)
        for bi, v in enumerate(batch):
            if v.get("dummy", False):
                # leave as zeros; already allocated
                continue
            img_in = v["img_norm"]
            if v["flip"]:
                img_in = _cv2.flip(img_in, 1)
            s = v["shrink"]
            if s != 1:
                img_in = _cv2.resize(
                    img_in,
                    None,
                    None,
                    fx=s,
                    fy=s,
                    interpolation=_cv2.INTER_LINEAR,
                )
            pad = xt_np[bi]
            pad[: img_in.shape[0], : img_in.shape[1], :] = img_in
        try:
            prepped_q.put((batch, max_h, max_w, xt_np))
        except (EOFError, OSError):
            break


# ---------------------------------------------
# Detector class with __call__ for three modes
# ---------------------------------------------


class MogFaceDetector:
    def __init__(
        self,
        device: torch.device = None,
        precision: str = "bf16",
        max_source_image_size: int = 1280,
        max_pixels_per_batch: int = 60_000_000,
        max_frame_queue_size: int = 32,
        max_batch_queue_size: int = 128,
        min_frames_for_pack: int = 2,
        min_pending_variants: int = 64,
        target_fill_ratio: float = 0.9,
        num_preproc_workers: int = 2,
        num_postprocess_workers: int = 1,
        pad_to_multiple: int = 32,
        max_prepped_queue_size: int = 4,
        max_model_out_queue_size: int = 32,
        max_minibatch_size: int = 64,
        score_th: float = 0.01,
        nms_th: float = 0.3,
        top_k: int = 5000,
        max_per_img: int = 750,
        scale_weight: float = 15.0,
        max_img_shrink: float = 2.6,
        vote_th: float = 0.6,
        flip_ratio: float = None,
        test_hard: int = 0,
        repo_id: str = "AlekseyKorshuk/MogFace",
        weights_filename: str = "model_140000.pth",
        verbose: bool = False,
    ):
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.max_pixels_per_batch = max_pixels_per_batch
        self.score_th = score_th
        self.nms_th = nms_th
        self.top_k = top_k
        self.max_per_img = max_per_img
        self.scale_weight = scale_weight
        self.max_img_shrink = max_img_shrink
        self.vote_th = vote_th
        self.flip_ratio = flip_ratio
        self.test_hard = test_hard
        self.verbose = verbose
        self.max_frame_queue_size = max_frame_queue_size
        self.max_batch_queue_size = max_batch_queue_size
        self.min_frames_for_pack = max(1, min_frames_for_pack)
        self.min_pending_variants = max(1, int(min_pending_variants))
        self.target_fill_ratio = float(target_fill_ratio)
        self.num_preproc_workers = max(1, int(num_preproc_workers))
        self.num_postprocess_workers = max(1, int(num_postprocess_workers))
        self.pad_to_multiple = int(pad_to_multiple) if pad_to_multiple else None
        self.max_prepped_queue_size = max(1, int(max_prepped_queue_size))
        self.max_model_out_queue_size = max(1, int(max_model_out_queue_size))
        self.max_source_image_size = (
            int(max_source_image_size) if max_source_image_size else None
        )
        self.max_minibatch_size = max(1, int(max_minibatch_size))

        # Autocast setup
        requested_precision = (precision or "fp32").lower()
        use_autocast = False
        amp_dtype = None
        if requested_precision == "bf16":
            if (self.device.type == "cuda" and torch.cuda.is_bf16_supported()) or (
                self.device.type == "cpu"
            ):
                use_autocast = True
                amp_dtype = torch.bfloat16
            else:
                print("BF16 not supported; falling back to FP32.")
        elif requested_precision == "fp16":
            if self.device.type == "cuda":
                use_autocast = True
                amp_dtype = torch.float16
            else:
                print("FP16 autocast requires CUDA; falling back to FP32.")
        self.use_autocast = use_autocast
        self.amp_torch_dtype = amp_dtype

        # cuDNN autotuner can stabilize throughput for fixed input shapes
        if self.device.type == "cuda":
            try:
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
            except Exception:
                pass

        # Build model and load weights
        print(f"Building MogFace-E model ({self.device.type.upper()})...")
        self.model = build_mogface_e()

        weights_path = hf_hub_download(repo_id=repo_id, filename=weights_filename)
        if not os.path.isabs(weights_path):
            repo_root = os.path.dirname(os.path.abspath(__file__))
            weights_path = os.path.join(repo_root, weights_path)
        if not os.path.exists(weights_path):
            print("Weights not found:", weights_path)
            sys.exit(1)
        print("Loading state_dict from:", weights_path)
        state_dict = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()
        self.model.compile(mode="reduce-overhead", fullgraph=True)
        if self.use_autocast:
            print(f"Autocast enabled with precision: {precision}")

        # Anchor caches to avoid recomputation and conversions
        self._anchor_cache = {}  # (H,W) -> np anchors_cxcywh
        self._anchor_torch_cpu = {}  # (H,W) -> torch.FloatTensor (CPU)
        self._anchor_torch_device = {}  # (H,W,device,idx) -> torch.FloatTensor (on device)

    def delete_model(self):
        print(
            f"Memory allocated before deleting model: {torch.cuda.memory_allocated() / 1024**2} MB"
        )
        torch.compiler.reset()
        print(
            f"Memory allocated after resetting compiler: {torch.cuda.memory_allocated() / 1024**2} MB"
        )
        self.model.to("cpu")
        print(
            f"Memory allocated after moving model to CPU: {torch.cuda.memory_allocated() / 1024**2} MB"
        )
        del self.model
        print(
            f"Memory allocated after deleting model: {torch.cuda.memory_allocated() / 1024**2} MB"
        )
        gc.collect()
        print(
            f"Memory allocated after garbage collection: {torch.cuda.memory_allocated() / 1024**2} MB"
        )
        torch.cuda.empty_cache()
        print(
            f"Memory allocated after emptying cache: {torch.cuda.memory_allocated() / 1024**2} MB"
        )
        self.model = None

    def _round_up_to_multiple(self, value, multiple):
        if multiple is None or multiple <= 1:
            return value
        return ((value + multiple - 1) // multiple) * multiple

    def _get_anchors_torch(self, H, W, device: torch.device):
        key = (H, W)
        t_cpu = self._anchor_torch_cpu.get(key)
        if t_cpu is None:
            anchors_np = self._get_anchors_cxcywh(H, W)
            t_cpu = torch.from_numpy(anchors_np).float()
            self._anchor_torch_cpu[key] = t_cpu
        if device.type == "cpu":
            return t_cpu
        dkey = (H, W, device.type, device.index if device.index is not None else -1)
        t_dev = self._anchor_torch_device.get(dkey)
        if t_dev is None:
            t_dev = t_cpu.to(device, non_blocking=True)
            self._anchor_torch_device[dkey] = t_dev
        return t_dev

    # -----------------------------
    # Public API
    # -----------------------------
    def __call__(
        self,
        images,
        mode: str = "balanced",
    ):
        # Accept a single path/array or a list
        if isinstance(images, (str, np.ndarray)):
            images = [images]

        # Load images and prepare per-image variant sets
        imgs_norm, paths, variants_per_image, valid_indices, resize_ratios = (
            self._prepare_images_and_variants(images, mode)
        )
        if len(imgs_norm) == 0:
            return []

        # Run shared micro-batched inference across images
        results = self._batched_inference_multi_images(imgs_norm, variants_per_image)

        # Reconstruct per-image outputs per mode
        outputs = []
        for local_idx in range(len(imgs_norm)):
            dets, elapsed = self._assemble_detections(results[local_idx], mode)
            ratio = resize_ratios[local_idx] if local_idx < len(resize_ratios) else 1.0
            if ratio != 1.0 and dets.size:
                scale = 1.0 / float(ratio)
                dets = dets.copy()
                dets[:, 0:4] *= scale
            outputs.append((dets, elapsed))

        # Align outputs back to the original input order
        aligned = [(None, 0.0) for _ in range(len(images))]
        for local_idx, orig_idx in enumerate(valid_indices):
            aligned[orig_idx] = outputs[local_idx]

        return aligned

    # -----------------------------
    # Internals
    # -----------------------------
    def _compute_img_scales(self, img_norm):
        max_im_shrink = (
            0x7FFFFFFF / 200.0 / (img_norm.shape[0] * img_norm.shape[1])
        ) ** 0.5
        max_im_shrink = self.max_img_shrink if max_im_shrink > 2.2 else max_im_shrink
        shrink = max_im_shrink if max_im_shrink < 1 else 1
        return max_im_shrink, shrink

    def _prepare_images_and_variants(self, images, mode):
        imgs_norm = []
        paths = []
        variants_per_image = []
        valid_indices = []
        resize_ratios = []  # processed/original ratio per image
        for idx, item in enumerate(images):
            if isinstance(item, str):
                p = item
                if not os.path.isabs(p):
                    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), p)
                orig = cv2.imread(p)
                if orig is None:
                    continue
                proc_img, ratio = _maybe_resize_image(
                    orig,
                    self.max_source_image_size,
                    verbose=self.verbose,
                    tag=f"file {os.path.basename(p)}",
                )
                img_norm = preprocess_image(proc_img)
                path_for_save = item
                resize_ratios.append(ratio)
            elif isinstance(item, np.ndarray):
                proc_img, ratio = _maybe_resize_image(
                    item, self.max_source_image_size, verbose=self.verbose, tag="array"
                )
                img_norm = preprocess_image(proc_img)
                path_for_save = None
                resize_ratios.append(ratio)
            else:
                continue

            max_im_shrink, shrink = self._compute_img_scales(img_norm)
            if mode == "one":
                variants = self._prepare_one_shot_variants()
            elif mode == "balanced":
                variants = self._prepare_balanced_variants(max_im_shrink, shrink)
            elif mode == "multi":
                variants = self._prepare_multi_variants(max_im_shrink, shrink)
            else:
                raise ValueError("mode must be one of: one|balanced|multi")

            imgs_norm.append(img_norm)
            paths.append(path_for_save)
            variants_per_image.append(variants)
            valid_indices.append(idx)

        return imgs_norm, paths, variants_per_image, valid_indices, resize_ratios

    def _prepare_one_shot_variants(self):
        return [{"key": "s0", "shrink": 1.0, "flip": False}]

    def _prepare_balanced_variants(self, max_im_shrink, shrink):
        variants = []
        variants.append({"key": "s0", "shrink": shrink, "flip": False})
        variants.append({"key": "flip_s0", "shrink": shrink, "flip": True})
        st = 0.5 if max_im_shrink >= 0.5 else max(0.25, max_im_shrink / 2)
        variants.append({"key": "ms_st", "shrink": st, "flip": False})
        if 1.25 <= max_im_shrink:
            variants.append({"key": "p_125", "shrink": 1.25, "flip": False})
        if 1.5 <= max_im_shrink:
            variants.append({"key": "ms_150", "shrink": 1.5, "flip": False})
        return variants

    def _prepare_multi_variants(self, max_im_shrink, shrink):
        variants = []
        variants.append({"key": "s0", "shrink": shrink, "flip": False})
        variants.append({"key": "flip_s0", "shrink": shrink, "flip": True})

        st = 0.5 if max_im_shrink >= 0.75 else 0.5 * max_im_shrink
        variants.append({"key": "ms_st", "shrink": st, "flip": False})
        if max_im_shrink > 0.75:
            variants.append({"key": "ms_075", "shrink": 0.75, "flip": False})

        bt = min(2, max_im_shrink) if max_im_shrink > 1 else (st + max_im_shrink) / 2
        variants.append({"key": "ms_bt", "shrink": bt, "flip": False})
        if max_im_shrink > 1.5:
            variants.append({"key": "ms_150", "shrink": 1.5, "flip": False})
        if max_im_shrink > 2:
            variants.append({"key": "ms_max", "shrink": max_im_shrink, "flip": False})

        variants.append({"key": "p_025", "shrink": 0.25, "flip": False})
        for s, k in [(1.25, "p_125"), (1.75, "p_175"), (2.25, "p_225")]:
            if s <= max_im_shrink:
                variants.append({"key": k, "shrink": s, "flip": False})

        if self.flip_ratio is not None:
            variants.append(
                {"key": "flip_extra", "shrink": self.flip_ratio, "flip": True}
            )
        return variants

    def _get_anchors_cxcywh(self, H, W):
        key = (H, W)
        arr = self._anchor_cache.get(key)
        if arr is None:
            anchors_xyxy = generate_prior_boxes(H, W)
            anchors_cxcywh = anchors_xyxy_to_cxcywh(anchors_xyxy)
            self._anchor_cache[key] = anchors_cxcywh
            return anchors_cxcywh
        return arr

    def _batched_inference_multi_images(self, images, variants_per_image):
        # Prepare resized images for all variants, metadata for packing
        resized_images = []
        meta = []  # (img_idx, key, shrink, flip, hi, wi, w0)
        for img_idx, (img, variants) in enumerate(zip(images, variants_per_image)):
            w0 = img.shape[1]
            for v in variants:
                img_in = img
                if v["flip"]:
                    img_in = cv2.flip(img_in, 1)
                if v["shrink"] != 1:
                    img_in = cv2.resize(
                        img_in,
                        None,
                        None,
                        fx=v["shrink"],
                        fy=v["shrink"],
                        interpolation=cv2.INTER_LINEAR,
                    )
                resized_images.append(img_in)
                hi, wi = img_in.shape[0], img_in.shape[1]
                meta.append((img_idx, v["key"], v["shrink"], v["flip"], hi, wi, w0))

        # Pack by area (desc)
        indices = list(range(len(meta)))
        indices.sort(key=lambda i: meta[i][4] * meta[i][5], reverse=True)
        if self.verbose:
            print(
                f"Prepared {len(meta)} variants for micro-batched inference across {len(images)} images. Pixel budget: {self.max_pixels_per_batch:,} pixels"
            )
            for i, (img_idx, k, s, f, hi, wi, w0) in enumerate(meta):
                area = hi * wi
                flip_str = ", flip" if f else ""
                print(
                    f"  [{i:02d}] img={img_idx} {k}: shrink={s:.3f}{flip_str}, size=({hi}x{wi}), area={area:,}"
                )

        results = {i: {} for i in range(len(images))}
        device = self.device
        start = 0
        batch_id = 0
        while start < len(indices):
            batch = []
            max_h, max_w = 0, 0
            i = start
            while i < len(indices):
                idx = indices[i]
                _, _, _, _, hi, wi, _ = meta[idx]
                new_max_h = max(max_h, hi)
                new_max_w = max(max_w, wi)
                new_cost = (len(batch) + 1) * new_max_h * new_max_w
                if len(batch) == 0:
                    batch.append(idx)
                    max_h, max_w = new_max_h, new_max_w
                    i += 1
                    continue
                if (
                    new_cost <= self.max_pixels_per_batch
                    and len(batch) < self.max_minibatch_size
                ):
                    batch.append(idx)
                    max_h, max_w = new_max_h, new_max_w
                    i += 1
                else:
                    break

            if self.verbose:
                est_cost = len(batch) * max_h * max_w
                extra = ""
                if len(batch) >= self.max_minibatch_size:
                    extra = f" (capped by max_minibatch_size={self.max_minibatch_size})"
                print(
                    f"[Micro-batch {batch_id}] num={len(batch)}, padded_to=({max_h}x{max_w}), est_cost={est_cost:,} pixels{extra}"
                )
                for bi, idx in enumerate(batch):
                    img_idx, k, s, f, hi, wi, _ = meta[idx]
                    flip_str = ", flip" if f else ""
                    print(
                        f"    - img={img_idx} {k}: shrink={s:.3f}{flip_str}, size=({hi}x{wi})"
                    )

            # Build padded tensors for this micro-batch (compute resized on the fly)
            batch_tensors = []
            for bi, idx in enumerate(batch):
                img_idx, _, s, f, _, _, _ = meta[idx]
                base_img = images[img_idx]
                img_in = base_img
                if f:
                    img_in = cv2.flip(img_in, 1)
                if s != 1:
                    img_in = cv2.resize(
                        img_in,
                        None,
                        None,
                        fx=s,
                        fy=s,
                        interpolation=cv2.INTER_LINEAR,
                    )
                pad = np.zeros((max_h, max_w, 3), dtype=img_in.dtype)
                pad[: img_in.shape[0], : img_in.shape[1], :] = img_in
                x = torch.from_numpy(pad).permute(2, 0, 1).unsqueeze(0).float()
                batch_tensors.append(x)
            xt = torch.cat(batch_tensors, dim=0).to(device)

            with (
                torch.no_grad(),
                torch.autocast(
                    device_type=device.type,
                    dtype=self.amp_torch_dtype,
                    enabled=self.use_autocast,
                ),
            ):
                out_conf_b, out_loc_b = self.model(xt)
            out_conf_b = out_conf_b.float()
            out_loc_b = out_loc_b.float()

            anchors_cxcywh = self._get_anchors_cxcywh(max_h, max_w)
            anchors_cxcywh_t = torch.from_numpy(anchors_cxcywh).float().to(device)
            anchor_centers = anchors_cxcywh_t[:, :2]

            for bi, idx in enumerate(batch):
                img_idx, key, shrink, flip, hi, wi, w0 = meta[idx]
                out_conf = out_conf_b[bi]
                out_loc = out_loc_b[bi]

                valid_mask = (anchor_centers[:, 0] <= wi) & (anchor_centers[:, 1] <= hi)
                decode_bbox = decode_boxes(out_loc, anchors_cxcywh_t)
                boxes = decode_bbox[valid_mask]
                scores = out_conf[valid_mask]

                v, idx_top = scores[:, 0].sort(0)
                idx_top = idx_top[-self.top_k :]
                boxes = boxes[idx_top]
                scores = scores[idx_top]

                boxes = boxes.cpu().numpy()
                w = boxes[:, 2] - boxes[:, 0] + 1
                h = boxes[:, 3] - boxes[:, 1] + 1
                boxes[:, 0] /= shrink
                boxes[:, 1] /= shrink
                boxes[:, 2] = boxes[:, 0] + w / shrink - 1
                boxes[:, 3] = boxes[:, 1] + h / shrink - 1
                scores = scores.cpu().numpy()

                inds = np.where(scores[:, 0] > self.score_th)[0]
                if len(inds) == 0:
                    c_dets = np.empty([0, 5], dtype=np.float32)
                else:
                    c_bboxes = boxes[inds]
                    c_scores = scores[inds, 0]
                    c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
                        np.float32, copy=False
                    )
                    keep = nms_numpy(c_dets, self.nms_th)
                    c_dets = c_dets[keep, :]
                    if self.max_per_img > 0:
                        image_scores = c_dets[:, -1]
                        if len(image_scores) > self.max_per_img:
                            image_thresh = np.sort(image_scores)[-self.max_per_img]
                            keep = np.where(c_dets[:, -1] >= image_thresh)[0]
                            c_dets = c_dets[keep, :]

                if flip and c_dets.size:
                    det_t = np.zeros(c_dets.shape, dtype=np.float32)
                    det_t[:, 0] = w0 - c_dets[:, 2] - 1
                    det_t[:, 1] = c_dets[:, 1]
                    det_t[:, 2] = w0 - c_dets[:, 0] - 1
                    det_t[:, 3] = c_dets[:, 3]
                    det_t[:, 4] = c_dets[:, 4]
                    c_dets = det_t

                results[img_idx][key] = c_dets

            start = i
            batch_id += 1

        return results

    def _assemble_detections(self, res, mode):
        # Returns (dets, elapsed_dummy) — elapsed is not measured here
        if mode == "one":
            dets = res.get("s0", np.empty([0, 5], dtype=np.float32))
            return dets, 0.0

        if mode == "balanced":
            det0 = res.get("s0", np.empty([0, 5], dtype=np.float32))
            det1 = res.get("flip_s0", np.empty([0, 5], dtype=np.float32))
            det_s = res.get("ms_st", np.empty([0, 5], dtype=np.float32))
            if det_s.size:
                if self.scale_weight == -1:
                    index = np.where(
                        np.maximum(
                            det_s[:, 2] - det_s[:, 0] + 1, det_s[:, 3] - det_s[:, 1] + 1
                        )
                        > 30
                    )[0]
                else:
                    index = np.where(
                        ((det_s[:, 2] - det_s[:, 0]) * (det_s[:, 3] - det_s[:, 1]))
                        > 2000
                    )[0]
                det_s = det_s[index, :] if det_s.shape[0] else det_s

            det_u_list = []
            if "p_125" in res:
                det_temp = res["p_125"]
                if det_temp.size:
                    if self.scale_weight == -1:
                        index = np.where(
                            np.maximum(
                                det_temp[:, 2] - det_temp[:, 0] + 1,
                                det_temp[:, 3] - det_temp[:, 1] + 1,
                            )
                            > 30
                        )[0]
                    else:
                        index = np.where(
                            (
                                (det_temp[:, 2] - det_temp[:, 0])
                                * (det_temp[:, 3] - det_temp[:, 1])
                            )
                            < self.scale_weight * 2000
                        )[0]
                    det_temp = det_temp[index, :] if det_temp.shape[0] else det_temp
                    det_u_list.append(det_temp)

            if "ms_150" in res:
                det_temp = res["ms_150"]
                if det_temp.size:
                    if self.scale_weight == -1:
                        index = np.where(
                            np.minimum(
                                det_temp[:, 2] - det_temp[:, 0] + 1,
                                det_temp[:, 3] - det_temp[:, 1] + 1,
                            )
                            < 100
                        )[0]
                    else:
                        index = np.where(
                            (
                                (det_temp[:, 2] - det_temp[:, 0])
                                * (det_temp[:, 3] - det_temp[:, 1])
                            )
                            < self.scale_weight * 800
                        )[0]
                    det_temp = det_temp[index, :] if det_temp.shape[0] else det_temp
                    det_u_list.append(det_temp)

            parts = [d for d in [det0, det1, det_s] + det_u_list if d.size]
            if len(parts) == 0:
                return np.empty([0, 5], dtype=np.float32), 0.0
            det = parts[0]
            for p in parts[1:]:
                det = np.vstack((det, p))
            return bbox_vote(det, self.vote_th, self.max_per_img), 0.0

        # mode == "multi"
        det0 = res.get("s0", np.empty([0, 5], dtype=np.float32))
        det1 = res.get("flip_s0", np.empty([0, 5], dtype=np.float32))

        det_s = res.get("ms_st", np.empty([0, 5], dtype=np.float32))
        if "ms_075" in res:
            if det_s.size and res["ms_075"].size:
                det_s = np.vstack((det_s, res["ms_075"]))
            else:
                det_s = det_s if det_s.size else res["ms_075"]
        if det_s.size:
            if self.scale_weight == -1:
                index = np.where(
                    np.maximum(
                        det_s[:, 2] - det_s[:, 0] + 1, det_s[:, 3] - det_s[:, 1] + 1
                    )
                    > 30
                )[0]
            else:
                index = np.where(
                    ((det_s[:, 2] - det_s[:, 0]) * (det_s[:, 3] - det_s[:, 1])) > 2000
                )[0]
            det_s = det_s[index, :] if det_s.shape[0] else det_s

        det_b = res.get("ms_bt", np.empty([0, 5], dtype=np.float32))
        if det_b.size:
            if self.scale_weight == -1:
                index = np.where(
                    np.minimum(
                        det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1
                    )
                    < 100
                )[0]
            else:
                index = np.where(
                    ((det_b[:, 2] - det_b[:, 0]) * (det_b[:, 3] - det_b[:, 1]))
                    < self.scale_weight * 600
                )[0]
            det_b = det_b[index, :] if det_b.shape[0] else det_b

        if "ms_150" in res:
            det_tmp = res["ms_150"]
            if det_tmp.size:
                if self.scale_weight == -1:
                    index = np.where(
                        np.minimum(
                            det_tmp[:, 2] - det_tmp[:, 0] + 1,
                            det_tmp[:, 3] - det_tmp[:, 1] + 1,
                        )
                        < 100
                    )[0]
                else:
                    index = np.where(
                        (
                            (det_tmp[:, 2] - det_tmp[:, 0])
                            * (det_tmp[:, 3] - det_tmp[:, 1])
                        )
                        < self.scale_weight * 800
                    )[0]
                det_tmp = det_tmp[index, :] if det_tmp.shape[0] else det_tmp
                det_b = det_b if det_b.size else det_tmp
                if det_b.size and det_tmp.size:
                    det_b = np.vstack((det_b, det_tmp))

        if "ms_max" in res:
            det_tmp = res["ms_max"]
            if det_tmp.size:
                if self.scale_weight == -1:
                    index = np.where(
                        np.minimum(
                            det_tmp[:, 2] - det_tmp[:, 0] + 1,
                            det_tmp[:, 3] - det_tmp[:, 1] + 1,
                        )
                        < 100
                    )[0]
                else:
                    index = np.where(
                        (
                            (det_tmp[:, 2] - det_tmp[:, 0])
                            * (det_tmp[:, 3] - det_tmp[:, 1])
                        )
                        < self.scale_weight * 500
                    )[0]
                det_tmp = det_tmp[index, :] if det_tmp.shape[0] else det_tmp
                det_b = det_b if det_b.size else det_tmp
                if det_b.size and det_tmp.size:
                    det_b = np.vstack((det_b, det_tmp))

        det2, det3 = det_s, det_b

        det4 = res.get("p_025", np.empty([0, 5], dtype=np.float32))
        if det4.size:
            if self.scale_weight == -1:
                index = np.where(
                    np.maximum(det4[:, 2] - det4[:, 0] + 1, det4[:, 3] - det4[:, 1] + 1)
                    > 30
                )[0]
            else:
                index = np.where(
                    ((det4[:, 2] - det4[:, 0]) * (det4[:, 3] - det4[:, 1])) > 2000
                )[0]
            det4 = det4[index, :] if det4.shape[0] else det4

        for i_key, thresh_mul in [("p_125", 2000), ("p_175", 1000), ("p_225", 600)]:
            if i_key in res:
                det_temp = res[i_key]
                if det_temp.size:
                    if i_key == "p_125":
                        if self.scale_weight == -1:
                            index = np.where(
                                np.maximum(
                                    det_temp[:, 2] - det_temp[:, 0] + 1,
                                    det_temp[:, 3] - det_temp[:, 1] + 1,
                                )
                                > 30
                            )[0]
                        else:
                            index = np.where(
                                (
                                    (det_temp[:, 2] - det_temp[:, 0])
                                    * (det_temp[:, 3] - det_temp[:, 1])
                                )
                                < self.scale_weight * thresh_mul
                            )[0]
                    else:
                        if self.scale_weight == -1:
                            index = np.where(
                                np.minimum(
                                    det_temp[:, 2] - det_temp[:, 0] + 1,
                                    det_temp[:, 3] - det_temp[:, 1] + 1,
                                )
                                < 100
                            )[0]
                        else:
                            index = np.where(
                                (
                                    (det_temp[:, 2] - det_temp[:, 0])
                                    * (det_temp[:, 3] - det_temp[:, 1])
                                )
                                < self.scale_weight * thresh_mul
                            )[0]
                    det_temp = det_temp[index, :] if det_temp.shape[0] else det_temp
                    det4 = det4 if det4.size else det_temp
                    if det4.size and det_temp.size:
                        det4 = np.vstack((det4, det_temp))

        if self.flip_ratio is not None and "flip_extra" in res:
            det5 = res.get("flip_extra", np.empty([0, 5], dtype=np.float32))
            det = (
                np.vstack((det0, det1, det2, det3, det4, det5))
                if det0.size
                else np.vstack((det1, det2, det3, det4, det5))
            )
        else:
            det = np.vstack((det0, det1, det2, det3, det4))

        return bbox_vote(det, self.vote_th, self.max_per_img), 0.0

    # -----------------------------
    # Video processing
    # -----------------------------
    def detect_video(
        self,
        video_path: str,
        mode: str = "balanced",
    ):
        """
        Producer/consumer video inference to maximize GPU utilization.
        Returns a list (per frame) of list of dicts: {"bbox": [x0,y0,x1,y1], "coef": float}
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Failed to open video:", video_path)
            return []

        # Queues: use threads for local queues and multiprocessing SimpleQueues across processes
        frame_q: "queue.Queue" = queue.Queue(maxsize=self.max_frame_queue_size)
        batch_q: "queue.Queue" = queue.Queue(maxsize=self.max_batch_queue_size)
        prepped_q: "queue.Queue" = queue.Queue(maxsize=self.max_prepped_queue_size)

        # Shared result dict: frame_idx -> {key: dets}
        results = {}
        results_lock = threading.Lock()
        expected_counts = {}
        end_of_frames = object()
        # Sentinels for mp queues: use None

        # Progress bar and live queue monitoring
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                total_frames = None
        except Exception:
            total_frames = None
        pbar = tqdm(total=total_frames, desc="Inferencing", unit="frame")
        pbar_lock = threading.Lock()

        # Compute a global padded size and constant batch size to stabilize compiled kernels
        try:
            v_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            v_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        except Exception:
            v_width, v_height = 0, 0
        if v_width <= 0 or v_height <= 0:
            try:
                ret_probe, probe_frame = cap.read()
                if not ret_probe:
                    print("Failed to read a frame for probing video size")
                    return []
                v_height, v_width = probe_frame.shape[0], probe_frame.shape[1]
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            except Exception:
                print("Failed to probe video size")
                return []

        # Compute video-level resize ratio and log if resizing is needed
        video_ratio = _compute_resize_ratio(
            v_height, v_width, self.max_source_image_size
        )
        if video_ratio < 1.0 and self.verbose:
            new_v_w = max(1, int(round(v_width * video_ratio)))
            new_v_h = max(1, int(round(v_height * video_ratio)))
            print(
                f"Resized video frames from ({v_height}x{v_width}) to ({new_v_h}x{new_v_w}) because longest side {max(v_height, v_width)} > max_source_image_size {self.max_source_image_size}"
            )

        # Use processed (possibly resized) dimensions for subsequent planning
        proc_v_h = (
            max(1, int(round(v_height * video_ratio)))
            if video_ratio < 1.0
            else v_height
        )
        proc_v_w = (
            max(1, int(round(v_width * video_ratio))) if video_ratio < 1.0 else v_width
        )

        dummy_img_norm = np.zeros((proc_v_h, proc_v_w, 3), dtype=np.float32)
        max_im_shrink, shrink = self._compute_img_scales(dummy_img_norm)
        if mode == "one":
            vlist = self._prepare_one_shot_variants()
        elif mode == "balanced":
            vlist = self._prepare_balanced_variants(max_im_shrink, shrink)
        else:
            vlist = self._prepare_multi_variants(max_im_shrink, shrink)
        var_sizes = []
        for v in vlist:
            s = v["shrink"]
            hi = int(round(proc_v_h * s)) if s != 1 else proc_v_h
            wi = int(round(proc_v_w * s)) if s != 1 else proc_v_w
            var_sizes.append((hi, wi))
        max_h0 = max(h for h, _ in var_sizes) if var_sizes else proc_v_h
        max_w0 = max(w for _, w in var_sizes) if var_sizes else proc_v_w
        if self.pad_to_multiple and self.pad_to_multiple > 1:
            m = int(self.pad_to_multiple)
            global_eff_h = ((max_h0 + m - 1) // m) * m
            global_eff_w = ((max_w0 + m - 1) // m) * m
        else:
            global_eff_h, global_eff_w = max_h0, max_w0
        pixels_per_item = int(global_eff_h) * int(global_eff_w)
        constant_B = max(1, int(self.max_pixels_per_batch) // max(1, pixels_per_item))
        if constant_B > self.max_minibatch_size:
            if self.verbose:
                print(
                    f"Capping constant_B from {constant_B} to max_minibatch_size={self.max_minibatch_size}"
                )
            constant_B = self.max_minibatch_size
        if self.verbose:
            print(
                f"Global padded size=({global_eff_h}x{global_eff_w}), per-item pixels={pixels_per_item:,}, constant_B={constant_B}, budget={self.max_pixels_per_batch:,}"
            )

        def reader_thread():
            idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # Resize frame if needed and preprocess
                if video_ratio < 1.0:
                    new_w = max(1, int(round(frame.shape[1] * video_ratio)))
                    new_h = max(1, int(round(frame.shape[0] * video_ratio)))
                    proc_frame = cv2.resize(
                        frame, (new_w, new_h), interpolation=cv2.INTER_AREA
                    )
                else:
                    proc_frame = frame
                img_norm = preprocess_image(proc_frame)
                frame_q.put(
                    (idx, img_norm, proc_frame.shape[1], video_ratio)
                )  # (frame_idx, img_norm, proc_w, ratio)
                idx += 1
            frame_q.put(end_of_frames)

        def prepare_variants_for_frame(img_norm):
            max_im_shrink, shrink = self._compute_img_scales(img_norm)
            if mode == "one":
                variants = self._prepare_one_shot_variants()
            elif mode == "balanced":
                variants = self._prepare_balanced_variants(max_im_shrink, shrink)
            else:
                variants = self._prepare_multi_variants(max_im_shrink, shrink)
            return variants

        def packer_thread():
            # pending variant items with computed sizes
            pending = []  # list of dicts: {frame_idx, img_norm, key, shrink, flip, hi, wi, w0}
            finished = False
            while True:
                if not finished:
                    item = frame_q.get()
                    if item is end_of_frames:
                        finished = True
                    else:
                        frame_idx, img_norm, w0_proc, ratio = item
                        h0, w_base = img_norm.shape[0], img_norm.shape[1]
                        variants = prepare_variants_for_frame(img_norm)
                        with results_lock:
                            expected_counts[frame_idx] = len(variants)
                        for v in variants:
                            s = v["shrink"]
                            hi = int(round(h0 * s)) if s != 1 else h0
                            wi = int(round(w_base * s)) if s != 1 else w_base
                            pending.append(
                                {
                                    "frame_idx": frame_idx,
                                    "img_norm": img_norm,
                                    "key": v["key"],
                                    "shrink": s,
                                    "flip": v["flip"],
                                    "hi": hi,
                                    "wi": wi,
                                    "w0": w0_proc,
                                    "ratio": ratio,
                                }
                            )
                    frame_q.task_done()
                # If not finished, wait until we have enough pending to make a constant-size batch
                if (not finished) and (len(pending) < constant_B):
                    continue

                # Build constant-shape batches (global_eff_h x global_eff_w) of size constant_B
                build_budget = max(1, self.max_prepped_queue_size)
                while build_budget > 0:
                    if not finished and len(pending) < constant_B:
                        break
                    if not pending and not finished:
                        break
                    take = min(constant_B, len(pending))
                    batch = pending[:take]
                    pending = pending[take:]
                    if finished and take < constant_B:
                        need = constant_B - take
                        dummy = {
                            "frame_idx": -1,
                            "img_norm": None,
                            "key": "dummy",
                            "shrink": 1.0,
                            "flip": False,
                            "hi": global_eff_h,
                            "wi": global_eff_w,
                            "w0": 0,
                            "dummy": True,
                        }
                        for _ in range(need):
                            batch.append(dummy.copy())
                    batch_q.put((batch, global_eff_h, global_eff_w))
                    build_budget -= 1

                if finished and not pending:
                    break

            # Send sentinel for each preparer process
            for _ in range(num_pre):
                batch_q.put(None)

        # Removed legacy gpu_model_thread; inference runs on main thread now

        # Compute per-process CPU threads to avoid oversubscription
        num_pre = getattr(self, "num_preproc_workers", 2)
        total_procs = max(1, num_pre)
        per_proc_threads = _compute_num_threads(total_procs)

        # Prepare worker configs
        amp_dtype_str = (
            "fp16"
            if self.amp_torch_dtype == torch.float16
            else ("bf16" if self.amp_torch_dtype == torch.bfloat16 else None)
        )
        prep_cfg = {
            "num_threads": per_proc_threads,
            "pad_to_multiple": self.pad_to_multiple,
            "use_autocast": self.use_autocast,
            "amp_dtype": amp_dtype_str,
        }
        # post_cfg removed as postprocess workers are no longer used

        # Launch threads/processes
        t_reader = threading.Thread(target=reader_thread, daemon=True)
        t_packer = threading.Thread(target=packer_thread, daemon=True)
        preparers = [
            threading.Thread(
                target=_batch_preparer_worker,
                args=(i, prep_cfg, batch_q, prepped_q),
                daemon=True,
            )
            for i in range(num_pre)
        ]
        t_reader.start()
        t_packer.start()
        for p in preparers:
            p.start()

        # Main-thread inference: consume prepared batches and run the model, store results
        device = self.device
        sentinels_seen = 0
        with (
            torch.no_grad(),
            torch.autocast(
                device_type=device.type,
                dtype=self.amp_torch_dtype,
                enabled=self.use_autocast,
            ),
        ):
            while True:
                try:
                    item = prepped_q.get()
                except (EOFError, OSError, FileNotFoundError):
                    break
                if item is None:
                    sentinels_seen += 1
                    if sentinels_seen == num_pre:
                        break
                    else:
                        continue
                batch, max_h, max_w, xt_np = item
                if self.verbose:
                    try:
                        est_cost = int(xt_np.shape[0]) * int(max_h) * int(max_w)
                        print(
                            f"GPU batch: num={xt_np.shape[0]}, padded=({max_h}x{max_w}), pixels={est_cost:,} / budget={self.max_pixels_per_batch:,}"
                        )
                    except Exception:
                        pass
                # Convert NumPy NHWC to Torch NCHW and move to device
                xt = torch.from_numpy(xt_np).permute(0, 3, 1, 2).contiguous()
                xt = xt.to(device, non_blocking=(device.type == "cuda"))
                if self.verbose and device.type == "cuda":
                    torch.cuda.synchronize()
                out_conf_b, out_loc_b = self.model(xt)
                if self.verbose and device.type == "cuda":
                    torch.cuda.synchronize()

                anchors_dev = self._get_anchors_torch(max_h, max_w, device)
                anchor_centers = anchors_dev[:, :2]
                B = out_conf_b.shape[0]
                for bi in range(B):
                    v = batch[bi]
                    if v.get("dummy", False):
                        continue
                    hi, wi = v["hi"], v["wi"]
                    s = v["shrink"]
                    valid_mask = (anchor_centers[:, 0] <= wi) & (
                        anchor_centers[:, 1] <= hi
                    )
                    out_conf = out_conf_b[bi]
                    out_loc = out_loc_b[bi]
                    decode_bbox = decode_boxes(out_loc, anchors_dev)
                    boxes = decode_bbox[valid_mask]
                    scores = out_conf[valid_mask]
                    scores_flat = (
                        scores[:, 0]
                        if (scores.dim() == 2 and scores.size(1) == 1)
                        else scores.reshape(-1)
                    )
                    k = int(min(self.top_k, int(scores_flat.numel())))
                    if k <= 0:
                        c_dets = np.empty([0, 5], dtype=np.float32)
                    else:
                        topk_vals, topk_idx = torch.topk(scores_flat, k)
                        boxes_sel = boxes[topk_idx]
                        # scale back to original processed resolution (shrink factor)
                        w = boxes_sel[:, 2] - boxes_sel[:, 0] + 1
                        h = boxes_sel[:, 3] - boxes_sel[:, 1] + 1
                        b0 = boxes_sel[:, 0] / s
                        b1 = boxes_sel[:, 1] / s
                        b2 = b0 + w / s - 1
                        b3 = b1 + h / s - 1
                        boxes_adj = torch.stack([b0, b1, b2, b3], dim=1)
                        score_mask = topk_vals > float(self.score_th)
                        if score_mask.any():
                            boxes_adj = boxes_adj[score_mask]
                            vals = topk_vals[score_mask]
                            if _HAS_TV_NMS:
                                try:
                                    keep = tv_nms(boxes_adj, vals, float(self.nms_th))
                                except Exception:
                                    keep = None
                            else:
                                keep = None
                            if keep is None:
                                # fallback to CPU NMS
                                b_np = boxes_adj.float().detach().cpu().numpy()
                                v_np = (
                                    vals.float().detach().cpu().numpy().reshape(-1, 1)
                                )
                                dets_np = np.hstack((b_np, v_np)).astype(
                                    np.float32, copy=False
                                )
                                keep_idx = nms_numpy(dets_np, float(self.nms_th))
                                dets_np = dets_np[keep_idx, :]
                                if (
                                    self.max_per_img > 0
                                    and dets_np.shape[0] > self.max_per_img
                                ):
                                    image_scores = dets_np[:, -1]
                                    image_thresh = np.sort(image_scores)[
                                        -self.max_per_img
                                    ]
                                    keep2 = np.where(dets_np[:, -1] >= image_thresh)[0]
                                    dets_np = dets_np[keep2, :]
                                c_dets = dets_np
                            else:
                                if self.max_per_img > 0:
                                    keep = keep[: int(self.max_per_img)]
                                boxes_kept = (
                                    boxes_adj[keep].float().detach().cpu().numpy()
                                )
                                scores_kept = (
                                    vals[keep]
                                    .float()
                                    .detach()
                                    .cpu()
                                    .numpy()
                                    .reshape(-1, 1)
                                )
                                c_dets = np.hstack((boxes_kept, scores_kept)).astype(
                                    np.float32, copy=False
                                )
                        else:
                            c_dets = np.empty([0, 5], dtype=np.float32)

                    if v["flip"] and c_dets.size:
                        det_t = np.zeros(c_dets.shape, dtype=np.float32)
                        det_t[:, 0] = v["w0"] - c_dets[:, 2] - 1
                        det_t[:, 1] = c_dets[:, 1]
                        det_t[:, 2] = v["w0"] - c_dets[:, 0] - 1
                        det_t[:, 3] = c_dets[:, 3]
                        det_t[:, 4] = c_dets[:, 4]
                        c_dets = det_t

                    frame_idx = v["frame_idx"]
                    key = v["key"]
                    ratio = v.get("ratio", 1.0)
                    if ratio != 1.0 and c_dets.size:
                        scale = 1.0 / float(ratio)
                        c_dets = c_dets.copy()
                        c_dets[:, 0:4] *= scale
                    with results_lock:
                        fr = results.setdefault(frame_idx, {})
                        fr[key] = c_dets
                        if len(fr) == expected_counts.get(frame_idx, 0):
                            with pbar_lock:
                                pbar.update(1)
                                pbar.set_postfix(
                                    batch_q_size=batch_q.qsize(),
                                    prepped_q_size=prepped_q.qsize(),
                                )

        t_reader.join()
        frame_q.join()
        t_packer.join()
        for p in preparers:
            p.join()

        # Assemble outputs in order
        if not results:
            cap.release()
            pbar.close()
            return []
        max_idx = max(results.keys())
        outputs = []
        for i in range(max_idx + 1):
            res = results.get(i, {})
            dets, _ = self._assemble_detections(res, mode)
            frame_out = []
            for j in range(dets.shape[0]):
                x0, y0, x1, y1, s = dets[j]
                frame_out.append(
                    {
                        "bbox": [float(x0), float(y0), float(x1), float(y1)],
                        "coef": float(s),
                    }
                )
            outputs.append(frame_out)
        cap.release()
        pbar.close()
        return outputs

    def _detect_frames_chunk(self, frames, mode):
        # Convert frames to normalized images with optional resizing
        imgs_norm = []
        resize_ratios = []
        for i, f in enumerate(frames):
            f_proc, ratio = _maybe_resize_image(
                f,
                self.max_source_image_size,
                verbose=self.verbose,
                tag=f"chunk_frame_{i}",
            )
            imgs_norm.append(preprocess_image(f_proc))
            resize_ratios.append(ratio)
        variants_per_image = []
        for img_norm in imgs_norm:
            max_im_shrink, shrink = self._compute_img_scales(img_norm)
            if mode == "one":
                variants = self._prepare_one_shot_variants()
            elif mode == "balanced":
                variants = self._prepare_balanced_variants(max_im_shrink, shrink)
            else:
                variants = self._prepare_multi_variants(max_im_shrink, shrink)
            variants_per_image.append(variants)

        res = self._batched_inference_multi_images(imgs_norm, variants_per_image)
        outputs = []
        for local_idx in range(len(imgs_norm)):
            dets, _ = self._assemble_detections(res[local_idx], mode)
            ratio = resize_ratios[local_idx]
            if ratio != 1.0 and dets.size:
                scale = 1.0 / float(ratio)
                dets = dets.copy()
                dets[:, 0:4] *= scale
            # Convert to required format per frame
            frame_out = []
            for i in range(dets.shape[0]):
                x0, y0, x1, y1, s = dets[i]
                frame_out.append(
                    {
                        "bbox": [float(x0), float(y0), float(x1), float(y1)],
                        "coef": float(s),
                    }
                )
            outputs.append(frame_out)
        return outputs
