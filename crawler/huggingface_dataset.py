from huggingface_hub import hf_hub_download

for i in range(20, 60):  # ~5000 per shard
    hf_hub_download(
        repo_id="pixparse/cc12m-wds",
        filename=f"cc12m-train-{i:04d}.tar",
        repo_type="dataset",
        local_dir="/scratch/users/liubr/neural-image-foundation-data/data/train/cc12m/rgb/"
    )