#Image generation pipeline
SDXL Option 2 has been tried and tested by many neural-> image papers in the past. Option 3 from black forest labs is more novel but is a much larger model and could require higher quality GPUs, take more time. 

**Option 1: 4M chained generation**

**Option 2: SDXL**
Expected input:
    - Semantic embedding (compressed into ~256 VQ tokens for 4M architecture)
        OpenCLIP ViT-H/14
            - 16x16 patches + 1 CLS token = 257 tokens, 1024 dim per patch
            - total continuous values: 257 x 1024 = 263K (smaller than MindEye2's proven 426K target)
            - directly compatible with IP-Adapter-Plus for SDXL — no projection layer needed
            - requires training a new VQ tokenizer on ViT-H/14 features (4M natively only has CLIP-B/16)
            - ViT-H/14 is significantly richer than CLIP-B/16 (1024-dim vs 512-dim) while being 
              a well-established, moderate-sized embedding — good balance of quality vs prediction difficulty
    - Depth map: 
        - any resolution grayscale image from 4M's depth detokenizer
        - resized to 1024x1024 to match SDXL output resolution
        - 4M already has a pretrained depth tokenizer (no new tokenizer needed)

Generation model and parameters:
    - SDXL (stabilityai/stable-diffusion-xl-base-1.0): base diffusion model, 3.5B params
        - runs comfortably on 8-12GB VRAM (RTX 3090/4090, A100)
    - IP-Adapter-Plus for SDXL (h94/IP-Adapter, ip-adapter-plus_sdxl_vit-h.safetensors):
        - lightweight adapter that injects OpenCLIP ViT-H/14 image embeddings into SDXL's 
          cross-attention layers, replacing text conditioning with image-embedding conditioning
        - accepts pre-computed embeddings via `ip_adapter_image_embeds` — no need to pass an actual image
        - "ip_adapter_scale" (0.0 to 1.0): how strongly CLIP drives semantics. start at 0.7.
    - ControlNet-Depth for SDXL (diffusers/controlnet-depth-sdxl-1.0-small):
        - takes depth map and injects structural conditioning into SDXL at each denoising step
        - "controlnet_conditioning_scale" (0.0 to 1.0): how strongly depth drives structure. start at 0.5-0.6.
        - lower this when brain-predicted depth is noisy — let SDXL's prior fill in details

    Tuning note: when conditioning signals come from noisy brain data, LOWER both scales. 
    Let the diffusion model's learned prior compensate for prediction errors rather than 
    rigidly following imperfect embeddings.

    (potential later additions) 
        - Multi-ControlNet for edges: add ControlNet-Canny alongside depth if edge predictions 
          from neural data are high quality. Use MultiControlNetModel in diffusers.
        - Text caption refinement (MindEye2 style): predict a caption from CLIP features 
          (via frozen GIT or CoCa model), then run a second SDXL img2img pass with the caption 
          as text conditioning and first output as initialization (strength=0.5). Cleans up artifacts.
        - Multi-sample selection: generate N=5 images per input, select best by cosine similarity 
          between generated image's CLIP embedding and the input embedding. Free quality boost.
        - Future upgrade path to Flux: swap SDXL→Flux, IP-Adapter→Redux, ControlNet→Flux-Depth, 
          ViT-H→SigLIP. Architecture stays the same, only the models change.

Setup required before this pipeline works:
    1. Train a VQ-VAE tokenizer on OpenCLIP ViT-H/14 spatial features (same process as 4M's 
       existing CLIP-B/16 tokenizer, just on ViT-H features). 
    2. Add ViT-H/14 as a modality in your 4M training config, using the new tokenizer.
    3. Retrain/finetune 4M to predict ViT-H tokens from neural data.
    4. Everything else (SDXL, IP-Adapter, ControlNet) is pretrained and frozen — just download and run.

All HuggingFace model IDs:
    pip install diffusers transformers accelerate safetensors torch pillow open_clip_torch
    - stabilityai/stable-diffusion-xl-base-1.0
    - diffusers/controlnet-depth-sdxl-1.0-small  
    - h94/IP-Adapter (subfolder: sdxl_models, weights: ip-adapter-plus_sdxl_vit-h.safetensors)
    - laion/CLIP-ViT-H-14-laion2B-s32B-b79K (or via open_clip: ViT-H-14, pretrained='laion2b_s32b_b79k')

**Option 3: Flux from Black Forest Labs**
Expected input:
    - Semantic embedding (compressed into ~196 VAE tokens for 4M architecutre)
        SigLIP SO400M 
            - 27x27 patches, 1152 dim per batch
            - can use with SOTA blackforest labs image generation pipeline
            - more lossy tokenization
            - try reducing output quality to make up for lossy tokenization
    - Depth map: (dims)

Generation model and parameters:
    - Flux1.Dev: base diffusion model
    - Flex Redux: semantic conditioner using SigLIP embeddings
    - Flux Depth: depth conditioner

    - (potential later additions) 
        - ControlNet for edges based on edge prediction quality from neural recordings
        - Text caption refinement: predict a caption from semantic embedding and run image generation pipeline conditioned on caption


This requires retraining a tokenizer for the new semantic embedding (SigLIP). 4M natively only outputs CLIPB-16, which is a 512D small CLIP embedding that may not contain enough info for 
high quality image generation. 