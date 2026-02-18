#!/usr/bin/env python3
"""
Step 2: Generate Images

Uses Flux.2 Klein 4B to generate illustrations from scene prompts.
Reads from prompts/ directory, outputs to images/ directory.

Features:
- ~0.5s per image at 1024x1024
- Resume capability: skip existing images
- Batch processing with VRAM management
- Consistent seeds for character continuity
"""

import os
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Diffusers for image generation
try:
    import torch
    from diffusers import Flux2KleinPipeline
    from PIL import Image
except ImportError:
    print("Installing required packages: diffusers, torch, pillow")
    os.system("pip install -q diffusers torch pillow")
    import torch
    from diffusers import Flux2KleinPipeline
    from PIL import Image

# Configuration
PROMPTS_DIR = Path("prompts")
IMAGES_DIR = Path("images")
CONSISTENCY_DIR = Path("consistency")
PROGRESS_DIR = Path("progress")

# Model configuration (distilled version — 4-5 steps, sub-second inference)
MODEL_NAME = "black-forest-labs/FLUX.2-klein-4B"

# Image generation settings
IMAGE_SIZE = 1024
GUIDANCE_SCALE = 1.0  # Must be 1.0 for distilled models
NUM_INFERENCE_STEPS = 5
BATCH_SIZE = 1  # Flux Klein works best with batch size 1


def load_consistency_registry() -> Dict:
    """Load character registry for seed hints."""
    char_path = CONSISTENCY_DIR / "characters.json"
    if char_path.exists():
        with open(char_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def generate_seed(scene_data: Dict, characters: Dict) -> int:
    """
    Generate deterministic seed for image generation.
    
    Uses scene_global_index as base, modified by character seeds if entities present.
    """
    base_seed = scene_data.get("scene_global_index", 0)
    
    # Mix in character seeds if present
    entities = scene_data.get("entities", [])
    for entity in entities:
        if entity in characters:
            char_seed = characters[entity].get("seed_hint", 0)
            base_seed = (base_seed * 31 + char_seed) % 2**32
    
    return base_seed % 2**32


def load_model() -> Flux2KleinPipeline:
    """Load Flux.2 Klein 4B model."""
    print(f"Loading model: {MODEL_NAME}")
    print("  This may take several minutes on first run (downloading ~16GB)...")
    
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    pipe = Flux2KleinPipeline.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        cache_dir=os.environ.get("HF_CACHE_DIR", None)
    )
    
    # Enable memory-efficient features
    pipe.enable_model_cpu_offload()
    
    # Try to compile the UNet for faster inference (PyTorch 2.0+)
    # This is a safe optimization that doesn't affect quality
    if hasattr(torch, 'compile') and hasattr(pipe, 'unet'):
        try:
            print("  Compiling model for faster inference...")
            pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=False)
        except Exception as e:
            print(f"  Note: Model compilation not available ({e})")
    
    print(f"  Model loaded (dtype: {dtype})")
    return pipe


def warmup_model(pipe: Flux2KleinPipeline):
    """Warm up the model with a dummy inference to ensure CUDA kernels are ready."""
    print("  Warming up model...")
    try:
        generator = torch.Generator(device="cpu").manual_seed(42)
        with torch.inference_mode():
            _ = pipe(
                prompt="warmup",
                height=IMAGE_SIZE,
                width=IMAGE_SIZE,
                guidance_scale=GUIDANCE_SCALE,
                num_inference_steps=1,  # Minimal steps for warmup
                generator=generator
            )
        print("  Warmup complete")
    except Exception as e:
        print(f"  Warmup skipped: {e}")


def generate_image(
    prompt: str,
    seed: int,
    pipe: Flux2KleinPipeline
) -> Optional[Image.Image]:
    """Generate a single image from prompt."""
    try:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        
        # Use inference_mode for slightly faster generation (no gradient computation)
        with torch.inference_mode():
            result = pipe(
                prompt=prompt,
                height=IMAGE_SIZE,
                width=IMAGE_SIZE,
                guidance_scale=GUIDANCE_SCALE,
                num_inference_steps=NUM_INFERENCE_STEPS,
                generator=generator
            )
        
        return result.images[0]
        
    except Exception as e:
        print(f"    ERROR generating image: {e}")
        return None


def is_scene_completed(chapter_index: int, scene_index: int) -> bool:
    """Check if image already exists for this scene."""
    image_path = IMAGES_DIR / f"chapter_{chapter_index:04d}_scene_{scene_index:02d}.png"
    return image_path.exists()


def mark_chapter_completed(chapter_index: int):
    """Mark chapter as completed in progress file."""
    PROGRESS_DIR.mkdir(exist_ok=True)
    progress_file = PROGRESS_DIR / "images_completed.txt"
    
    # Check if already marked
    if progress_file.exists():
        completed = progress_file.read_text().splitlines()
        if str(chapter_index) in completed:
            return
    
    with open(progress_file, "a") as f:
        f.write(f"{chapter_index}\n")


def get_all_prompts() -> List[Tuple[Path, Dict]]:
    """Get all prompt files sorted by chapter/scene."""
    if not PROMPTS_DIR.exists():
        return []
    
    prompts = []
    for prompt_file in sorted(PROMPTS_DIR.glob("chapter_*_scene_*.json")):
        try:
            with open(prompt_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            prompts.append((prompt_file, data))
        except Exception as e:
            print(f"Warning: Could not load {prompt_file}: {e}")
    
    return prompts


def process_scene(
    prompt_file: Path,
    scene_data: Dict,
    pipe: Flux2KleinPipeline,
    characters: Dict
) -> bool:
    """
    Process a single scene: generate and save image.
    
    Returns True on success.
    """
    chapter_index = scene_data["chapter_index"]
    scene_index = scene_data["scene_index"]
    
    # Check if already exists
    image_path = IMAGES_DIR / f"chapter_{chapter_index:04d}_scene_{scene_index:02d}.png"
    if image_path.exists():
        return True
    
    # Generate seed
    seed = generate_seed(scene_data, characters)
    
    # Get prompt
    prompt = scene_data.get("prompt", "")
    if not prompt:
        print(f"    Warning: Empty prompt for scene {scene_index}")
        return False
    
    # Generate image
    image = generate_image(prompt, seed, pipe)
    if image is None:
        return False
    
    # Save image
    IMAGES_DIR.mkdir(exist_ok=True)
    image.save(image_path, "PNG")
    
    return True


def main():
    """Main entry point."""
    print("=" * 60)
    print("Step 2: Generate Images (Flux.2 Klein 4B)")
    print("=" * 60)
    
    POLL_INTERVAL = 30  # seconds between polls for new prompts

    # Load consistency registry
    print("\nLoading consistency registry...")
    characters = load_consistency_registry()
    print(f"  Characters: {len(characters)}")

    pipe = None
    total_processed = 0
    total_failed = 0
    start_time = time.time()

    try:
        while True:
            # Get all prompts and filter to unprocessed
            all_prompts = get_all_prompts()
            remaining = [
                (pf, pd) for pf, pd in all_prompts
                if not is_scene_completed(pd["chapter_index"], pd["scene_index"])
            ]

            if not remaining:
                if total_processed > 0 and len(all_prompts) >= 2883:
                    print("\nAll images generated! Pipeline complete.")
                    break
                print(f"\r  Waiting for prompts... ({len(all_prompts)} prompts, {total_processed} images done)", end="", flush=True)
                time.sleep(POLL_INTERVAL)
                continue

            # Load model on first batch (lazy load)
            if pipe is None:
                print(f"\nFound {len(remaining)} images to generate")
                print("-" * 60)
                pipe = load_model()
                warmup_model(pipe)  # Warmup for consistent timing
                print(f"Settings: {IMAGE_SIZE}x{IMAGE_SIZE}, {NUM_INFERENCE_STEPS} steps, CFG {GUIDANCE_SCALE}")
                print("-" * 60)
                print("Note: First few images may be slower while model optimizes")
                print("-" * 60)

            # Process available scenes
            batch_processed = 0
            batch_start_time = time.time()
            
            for idx, (prompt_file, scene_data) in enumerate(remaining):
                chapter_index = scene_data["chapter_index"]
                scene_index = scene_data["scene_index"]
                
                img_start_time = time.time()

                success = process_scene(prompt_file, scene_data, pipe, characters)
                img_time = time.time() - img_start_time

                if success:
                    total_processed += 1
                    batch_processed += 1
                    mark_chapter_completed(chapter_index)
                    
                    # Show progress with correct units (s/img, not img/s)
                    if idx == 0 or (idx + 1) % 5 == 0 or (idx + 1) == len(remaining):
                        elapsed = time.time() - start_time
                        secs_per_img = elapsed / total_processed if total_processed > 0 else 0
                        remaining_imgs = len(remaining) - (idx + 1)
                        eta_mins = (remaining_imgs * secs_per_img) / 60 if secs_per_img > 0 else 0
                        print(f"  [{idx + 1}/{len(remaining)}] Ch {chapter_index:04d} Sc {scene_index:02d} "
                              f"({img_time:.1f}s, avg: {secs_per_img:.1f}s/img, ETA: {eta_mins:.0f}m)")
                else:
                    total_failed += 1
                    print(f"    FAILED: chapter_{chapter_index:04d}_scene_{scene_index:02d}")

            print(f"\nBatch done ({batch_processed} images). Polling for new prompts...")
            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\n\nStopped by user.")

    # Cleanup
    if pipe is not None:
        print("Cleaning up...")
        del pipe
        torch.cuda.empty_cache()

    total_time = time.time() - start_time
    total_done = total_processed + total_failed
    avg_time = total_time / total_done if total_done > 0 else 0
    print(f"\nStep 2: {total_processed} images ({total_failed} failed) in {total_time/3600:.1f}h")
    print(f"Average: {avg_time:.1f}s per image ({total_done / (total_time / 60):.1f} images/min)")
    print("=" * 60)


if __name__ == "__main__":
    main()
