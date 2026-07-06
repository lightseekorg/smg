#!/usr/bin/env python3
"""Generate compact HuggingFace reference fingerprints for Qwen preprocessing.

The processor classes are constructed locally, so this script does not download
model weights or configuration. Its output is checked into the Rust integration
test as an external correctness oracle for resize, normalization, and patchify.
"""

import json

import numpy as np
from PIL import Image
from PIL import __version__ as pillow_version
from transformers import Qwen2VLImageProcessor
from transformers import __version__ as transformers_version
from transformers.models.qwen3_vl.video_processing_qwen3_vl import smart_resize

CASES = ((37, 23), (259, 194))


def make_image(width: int, height: int, seed: int = 0) -> Image.Image:
    y, x = np.indices((height, width), dtype=np.uint32)
    pixels = np.stack(
        (
            (seed + x * 7 + y * 3) % 256,
            (seed + x * 5 + y * 11) % 256,
            (seed + x + y * 2) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    return Image.fromarray(pixels, mode="RGB")


def fingerprint_bytes(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    value = 0xCBF29CE484222325
    for byte in contiguous.tobytes():
        value ^= byte
        value = value * 0x100000001B3 & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


def processor_cases(name: str, processor: Qwen2VLImageProcessor) -> list[dict]:
    results = []
    for width, height in CASES:
        output = processor(
            images=make_image(width, height),
            do_normalize=False,
            return_tensors="np",
        )
        values = output["pixel_values"]
        patch_u8 = np.rint(values * 255.0).astype(np.uint8)
        results.append(
            {
                "model": name,
                "width": width,
                "height": height,
                "shape": list(values.shape),
                "grid_thw": output["image_grid_thw"][0].tolist(),
                "fnv1a_patch_u8": fingerprint_bytes(patch_u8),
            }
        )
    return results


def qwen3_video_case() -> dict:
    width, height = 37, 35
    seeds = (3, 101, 177)
    patch_size = 16
    merge_size = 2
    temporal_patch_size = 2
    target_height, target_width = smart_resize(
        len(seeds),
        height,
        width,
        temporal_factor=temporal_patch_size,
        factor=patch_size * merge_size,
        min_pixels=65536,
        max_pixels=16777216,
    )
    frames = [
        np.asarray(
            make_image(width, height, seed).resize(
                (target_width, target_height),
                Image.Resampling.BICUBIC,
            )
        )
        for seed in seeds
    ]
    while len(frames) % temporal_patch_size:
        frames.append(frames[-1])

    grid_t = len(frames) // temporal_patch_size
    grid_h = target_height // patch_size
    grid_w = target_width // patch_size
    patch_rows = grid_h // merge_size
    patch_cols = grid_w // merge_size
    patchified = []
    for temporal in range(grid_t):
        frame_window = frames[temporal * temporal_patch_size : (temporal + 1) * temporal_patch_size]
        for patch_row in range(patch_rows):
            for patch_col in range(patch_cols):
                for merge_row in range(merge_size):
                    for merge_col in range(merge_size):
                        y = (patch_row * merge_size + merge_row) * patch_size
                        x = (patch_col * merge_size + merge_col) * patch_size
                        for channel in range(3):
                            for frame in frame_window:
                                patchified.append(
                                    frame[y : y + patch_size, x : x + patch_size, channel]
                                )
    values = np.concatenate([patch.reshape(-1) for patch in patchified])
    patch_features = 3 * temporal_patch_size * patch_size * patch_size
    return {
        "model": "qwen3_vl",
        "width": width,
        "height": height,
        "frame_count": len(seeds),
        "shape": [grid_t * grid_h * grid_w, patch_features],
        "grid_thw": [grid_t, grid_h, grid_w],
        "fnv1a_patch_u8": fingerprint_bytes(values),
    }


def main() -> None:
    qwen2 = Qwen2VLImageProcessor(
        patch_size=14,
        merge_size=2,
        temporal_patch_size=2,
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
        image_mean=[0.48145466, 0.4578275, 0.40821073],
        image_std=[0.26862954, 0.26130258, 0.27577711],
        resample=Image.Resampling.BICUBIC,
    )
    qwen3 = Qwen2VLImageProcessor(
        patch_size=16,
        merge_size=2,
        temporal_patch_size=2,
        min_pixels=65536,
        max_pixels=16777216,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        resample=Image.Resampling.BICUBIC,
    )
    document = {
        "generator": "generate_qwen_preprocess_fingerprints.py",
        "transformers": transformers_version,
        "pillow": pillow_version,
        "cases": processor_cases("qwen2_vl", qwen2) + processor_cases("qwen3_vl", qwen3),
        "video_cases": [qwen3_video_case()],
    }
    print(json.dumps(document, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
