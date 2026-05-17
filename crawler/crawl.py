from img2dataset import download
import multiprocessing
multiprocessing.set_start_method("fork", force=True)

if __name__ == '__main__':
    download(url_list="/scratch/users/liubr/neural-image-foundation-data/cc12m/cc1.5m_subset.tsv", processes_count=8, input_format="tsv", url_col="url", output_folder="/scratch/users/liubr/neural-image-foundation-data/cc12m/downloaded", output_format="webdataset", resize_mode="no", enable_wandb=False, verify_hash=["sha256", "sha256"])