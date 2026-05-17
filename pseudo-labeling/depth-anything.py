import collections.abc
import collections
collections.Iterable = collections.abc.Iterable

import webdataset as wds
import transformers
import torch
import numpy as np
import os
import json
from pathlib import Path

from PIL import Image

from transformers import AutoImageProcessor, AutoModelForDepthEstimation

def process_shard(model, processor, input_tar, output_tar, batch_size=64):
    dataset = (
        wds.WebDataset(input_tar, shardshuffle=False)
        .decode("pil")
        .to_tuple("__key__", "jpg;png")
    )

    with wds.TarWriter(output_tar) as sink:
        batch_keys, batch_imgs = [], []

        for key, img in dataset:
            batch_keys.append(key)
            batch_imgs.append(img)

            if len(batch_keys) == batch_size:
                _run_batch(batch_keys, batch_imgs, processor, model, sink)
                batch_keys, batch_imgs = [], []

        if batch_keys:
            _run_batch(batch_keys, batch_imgs, processor, model, sink)


def _run_batch(keys, imgs, processor, model, sink):
    original_sizes = [(img.height, img.width) for img in imgs]
    imgs_resized = [img.resize((518, 518), Image.BILINEAR) for img in imgs]
    inputs = processor(
        images=imgs_resized,
        return_tensors="pt",
        size={"height": 518, "width": 518},
    ).to("cuda")

    with torch.inference_mode():
        outputs = model(**inputs)

    depths = outputs.predicted_depth  # (B, 518, 518)

    for key, depth, (orig_h, orig_w) in zip(keys, depths, original_sizes):
        depth_resized = torch.nn.functional.interpolate(
            depth.unsqueeze(0).unsqueeze(0),  # (1, 1, H, W)
            size=(orig_h, orig_w),
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy().astype(np.float32)

        sink.write({
            "__key__": key,
            "npy": depth_resized,
        })

if __name__ == "__main__":
    os.environ["HF_HOME"] = "/scratch/users/liubr/hf_cache"
    output_dir = "/scratch/users/liubr/neural-image-foundation-data/data/depth"
    model_str = "depth-anything/Depth-Anything-V2-Large-hf"
    processor = AutoImageProcessor.from_pretrained(model_str)
    model = AutoModelForDepthEstimation.from_pretrained(model_str).cuda().eval()
    for shard_input in Path("/scratch/users/liubr/neural-image-foundation-data/data/rgb").iterdir():
        print(f"Processing: {shard_input}")
        process_shard(model, processor, shard_input.absolute().as_posix(), f"{output_dir}/{shard_input.name}")