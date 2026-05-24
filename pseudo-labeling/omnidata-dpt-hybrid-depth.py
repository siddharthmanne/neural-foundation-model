import torch
import torchvision.transforms as T

from PIL import Image

import webdataset as wds
import numpy as np

import sys
import os
import argparse

from pathlib import Path

TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=0.5, std=0.5),
])

def process_shard(model, input_tar, output_tar, device='cuda', batch_size=64):
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
                _run_batch(batch_keys, batch_imgs, model, sink, device)
                batch_keys, batch_imgs = [], []

        if batch_keys:
            _run_batch(batch_keys, batch_imgs, model, sink, device)

def _run_batch(keys, imgs, model, sink, device):  # no processor arg
    original_sizes = [(img.height, img.width) for img in imgs]
    imgs_resized = [img.resize((384, 384), Image.BILINEAR) for img in imgs]  # note: 384, not 518

    inputs = torch.stack([TRANSFORM(img) for img in imgs_resized]).to(device)

    with torch.inference_mode():
        depths = model(inputs)  # (B, 1, 384, 384) or (B, 384, 384) — check shape

    # Squeeze channel dim if present
    if depths.dim() == 4:
        depths = depths.squeeze(1)

    for key, depth, (orig_h, orig_w) in zip(keys, depths, original_sizes):
        depth_resized = torch.nn.functional.interpolate(
            depth.unsqueeze(0).unsqueeze(0),
            size=(orig_h, orig_w),
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy().astype(np.float32)
        sink.write({"__key__": key, "npy": depth_resized})

if __name__ == "__main__":
    os.environ["HF_HOME"] = "/scratch/users/liubr/hf_cache"

    parser = argparse.ArgumentParser(description="Process depth normals using omnidata.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run inference on (e.g. 'cuda', 'cuda:1', 'cpu')")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Inference batch size")
    args = parser.parse_args()

    model = torch.hub.load('alexsax/omnidata_models', 'depth_dpt_hybrid_384').to(args.device).eval()
    output_dir = "/scratch/users/liubr/neural-image-foundation-data/data/train/things/depth"
    for shard_input in Path("/scratch/users/liubr/neural-image-foundation-data/data/train/things/rgb").iterdir():
        print(f"Processing: {shard_input}")
        process_shard(model, shard_input.absolute().as_posix(), f"{output_dir}/{shard_input.name}", args.device, batch_size=args.batch_size)