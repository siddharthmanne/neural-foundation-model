from huggingface_hub import hf_hub_download

for i in range(20):  # downloads shards 0000-0009 (~500k images)
    hf_hub_download(
        repo_id="pixparse/cc12m-wds",
        filename=f"cc12m-train-{i:04d}.tar",
        repo_type="dataset",
        local_dir="/scratch/users/liubr/neural-image-foundation-data/downloaded"
    )