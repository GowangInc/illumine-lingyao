#!/usr/bin/env python3
"""
Step 1: Generate Image Prompts

Uses local Qwen3-4B LLM to generate image prompts for each scene.
Reads from plan/ directory, outputs to prompts/ directory.

Features:
- 4-bit quantized inference (~2.5GB VRAM on RTX 3090)
- LLM-based entity extraction (characters, locations)
- Consistency injection from characters.json/locations.json
- Chapter title incorporated into image prompt for text rendering
- Continuous polling mode: waits for new plan files from Step 0
- Atomic chapter-level processing with resume capability
"""

import os
import re
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Transformers for local LLM
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from huggingface_hub import login
except ImportError:
    print("Installing required packages: transformers, torch, bitsandbytes, accelerate")
    os.system("pip install -q transformers torch bitsandbytes accelerate")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Configuration
PLAN_DIR = Path("plan")
CONSISTENCY_DIR = Path("consistency")
PROMPTS_DIR = Path("prompts")
PROGRESS_DIR = Path("progress")

# Model configuration
MODEL_NAME = "Qwen/Qwen3-4B"
MODEL_URL = "https://huggingface.co/Qwen/Qwen3-4B"
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Quantization config for 4-bit loading
QUANT_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# Art style prefix (prepended to every prompt — kept short for Flux.2 Klein)
ART_STYLE_PREFIX = "Ming Dynasty ink wash painting (文人画), 1628, muted earth tones with cobalt and vermillion accents, cinematic widescreen"

# System prompt for image prompt generation (kept compact for speed)
SYSTEM_PROMPT = """Create an image prompt for a chapter title card. Setting: 1628 Ming Dynasty Hainan, novel "Illumine Lingao".

Rules:
- Under 75 words. Focus on subject, environment, mood.
- Include calligraphic text of the chapter title in the image.
- Extract character and location names from the text.
- Do NOT include art style (added automatically).

Respond with JSON only:
{{"prompt": "visual description with calligraphy reading 'Chapter Title'...", "scene_description": "1-sentence summary", "mood": "tone keywords", "entities": ["Name1", "Location1"]}}"""


@dataclass
class ScenePrompt:
    chapter_index: int
    scene_index: int
    scene_global_index: int
    chapter_title: str
    text_chunk: str
    audio_start_sec: float
    audio_end_sec: float
    duration_sec: float
    prompt: str
    scene_description: str
    mood: str
    entities: List[str]


# Global tokenizer reference
_tokenizer = None


def load_consistency_registry() -> Tuple[Dict, Dict]:
    """Load character and location registries."""
    chars = {}
    locs = {}

    char_path = CONSISTENCY_DIR / "characters.json"
    if char_path.exists():
        with open(char_path, "r", encoding="utf-8") as f:
            chars = json.load(f)

    loc_path = CONSISTENCY_DIR / "locations.json"
    if loc_path.exists():
        with open(loc_path, "r", encoding="utf-8") as f:
            locs = json.load(f)

    return chars, locs


def build_entity_context(entities: List[str], characters: Dict, locations: Dict) -> str:
    """Build context string from entity registry for consistency."""
    context_parts = []

    for entity in entities:
        if entity in characters:
            char_info = characters[entity]
            desc = char_info.get("description", "")
            if desc:
                context_parts.append(f"{entity}: {desc}")

    if context_parts:
        return "\n".join(context_parts[:5])
    return ""


def load_model():
    """Load the Qwen3-8B text-only model with 4-bit quantization."""
    print(f"Loading model: {MODEL_NAME}")
    print(f"  URL: {MODEL_URL}")
    print("  This may take a few minutes on first run (downloading model weights)...")

    # Login to HuggingFace
    print("  Authenticating with HuggingFace...")
    login(token=HF_TOKEN)

    # Check available VRAM
    if torch.cuda.is_available():
        free_vram = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
        free_vram_gb = free_vram / 1024**3
        print(f"  Free VRAM: {free_vram_gb:.1f} GB")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        cache_dir=os.environ.get("HF_CACHE_DIR", None)
    )

    # Load model with 4-bit quantization
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=QUANT_CONFIG,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        cache_dir=os.environ.get("HF_CACHE_DIR", None)
    )

    print("  Model loaded successfully")

    # Show device map
    if hasattr(model, 'hf_device_map'):
        gpu_layers = sum(1 for d in model.hf_device_map.values() if d in [0, 'cuda', 'cuda:0'])
        cpu_layers = sum(1 for d in model.hf_device_map.values() if d in ['cpu', 'disk'])
        total_layers = len(model.hf_device_map)
        print(f"  Layers: {gpu_layers} on GPU, {cpu_layers} on CPU/Disk, {total_layers} total")

    # Show VRAM usage
    if torch.cuda.is_available():
        vram_mb = torch.cuda.memory_allocated() / 1024 / 1024
        print(f"  VRAM used: {vram_mb:.0f} MB")

    return model, tokenizer


def generate_prompt_for_scene(
    scene_text: str,
    chapter_title: str,
    entity_context: str,
    model,
    tokenizer
) -> Optional[Dict]:
    """
    Generate image prompt for a single scene using the text-only Qwen3-8B model.

    Returns dict with prompt, scene_description, mood, entities or None on failure.
    """
    global _tokenizer
    _tokenizer = tokenizer

    # Build user prompt
    user_content = f"Chapter: {chapter_title}\n\n"

    if entity_context:
        user_content += f"Known Character/Location Descriptions (use for visual consistency):\n{entity_context}\n\n"

    user_content += f"Scene Text:\n{scene_text[:3000]}"

    # Format as Qwen3 chat
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]

    # Apply chat template and tokenize (disable thinking for speed)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True
    ).to(model.device)

    # Generate
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode - skip the input tokens
        input_len = inputs.input_ids.shape[1]
        response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

        # Strip Qwen3 thinking tags (model defaults to thinking mode)
        think_end = response.find("</think>")
        if think_end >= 0:
            response = response[think_end + len("</think>"):].strip()

        # Extract JSON - try matching a full JSON object with nested arrays
        # Use a more robust pattern that handles arrays in the JSON
        json_match = re.search(r'\{[^{}]*(?:"entities"\s*:\s*\[[^\]]*\])?[^{}]*\}', response, re.DOTALL)
        if not json_match:
            # Fallback: find outermost braces
            start = response.find('{')
            end = response.rfind('}')
            if start >= 0 and end > start:
                json_match = type('Match', (), {'group': lambda self: response[start:end+1]})()

        if json_match:
            try:
                result = json.loads(json_match.group())
                if "prompt" in result:
                    return result
            except json.JSONDecodeError:
                pass

        # Fallback: try to parse entire response as JSON
        try:
            result = json.loads(response.strip())
            if "prompt" in result:
                return result
        except json.JSONDecodeError:
            pass

        # Last resort: manual extraction
        prompt_match = re.search(r'"prompt"\s*:\s*"([^"]+)"', response)
        desc_match = re.search(r'"scene_description"\s*:\s*"([^"]+)"', response)
        mood_match = re.search(r'"mood"\s*:\s*"([^"]+)"', response)

        if prompt_match:
            return {
                "prompt": prompt_match.group(1),
                "scene_description": desc_match.group(1) if desc_match else "",
                "mood": mood_match.group(1) if mood_match else "",
                "entities": []
            }

        print(f"    Warning: Could not parse LLM response: {response[:200]}...")
        return None

    except Exception as e:
        print(f"    Error generating prompt: {e}")
        return None


def is_chapter_completed(chapter_index: int) -> bool:
    """Check if chapter has already been processed."""
    progress_file = PROGRESS_DIR / "prompts_completed.txt"
    if not progress_file.exists():
        return False

    completed = progress_file.read_text().splitlines()
    return str(chapter_index) in completed


def mark_chapter_completed(chapter_index: int):
    """Mark chapter as completed in progress file."""
    PROGRESS_DIR.mkdir(exist_ok=True)
    progress_file = PROGRESS_DIR / "prompts_completed.txt"

    with open(progress_file, "a") as f:
        f.write(f"{chapter_index}\n")


def process_chapter(
    chapter_index: int,
    model,
    tokenizer,
    characters: Dict,
    locations: Dict,
    scene_counter: int
) -> Tuple[int, int]:
    """
    Process all scenes in a chapter.

    Atomic: either all scenes succeed or none are saved.
    Returns (number of scenes processed, updated global scene counter).
    """
    plan_path = PLAN_DIR / f"chapter_{chapter_index:04d}.json"

    if not plan_path.exists():
        print(f"  ERROR: Plan file not found: {plan_path}")
        return 0, scene_counter

    # Load plan
    with open(plan_path, "r", encoding="utf-8") as f:
        plan = json.load(f)

    chapter_title = plan.get("chapter_title", f"Chapter {chapter_index}")
    scenes = plan.get("scenes", [])
    text_content = plan.get("text_content", "")

    print(f"  Processing {len(scenes)} scene(s)...")

    # Generate prompts for all scenes
    scene_prompts = []
    current_global_index = scene_counter

    for scene_data in scenes:
        scene_idx = scene_data["scene_index"]
        text_start = scene_data["text_start_char"]
        text_end = scene_data["text_end_char"]

        # Extract text chunk
        chunk_text = text_content[text_start:text_end]

        # Get entity context from plan entities (may be empty or poor quality)
        # The LLM will extract better entities in its response
        plan_entities = scene_data.get("entities", [])
        entity_context = build_entity_context(plan_entities, characters, locations)

        # Generate prompt (LLM also extracts entities)
        result = generate_prompt_for_scene(
            chunk_text,
            chapter_title,
            entity_context,
            model,
            tokenizer
        )

        if result is None:
            print(f"    ERROR: Failed to generate prompt for scene {scene_idx}")
            return 0, scene_counter  # Atomic failure

        # Use LLM-extracted entities (much better than regex)
        llm_entities = result.get("entities", [])
        if not isinstance(llm_entities, list):
            llm_entities = []

        # Build full prompt with style prefix
        full_prompt = f"{ART_STYLE_PREFIX}\n\n{result.get('prompt', '')}"

        # Create scene prompt object
        scene_prompt = ScenePrompt(
            chapter_index=chapter_index,
            scene_index=scene_idx,
            scene_global_index=current_global_index,
            chapter_title=chapter_title,
            text_chunk=chunk_text[:1000],
            audio_start_sec=scene_data["audio_start_sec"],
            audio_end_sec=scene_data["audio_end_sec"],
            duration_sec=scene_data["duration_sec"],
            prompt=full_prompt,
            scene_description=result.get("scene_description", ""),
            mood=result.get("mood", ""),
            entities=llm_entities
        )

        scene_prompts.append(scene_prompt)
        current_global_index += 1

        print(f"    Scene {scene_idx}: {len(full_prompt)} chars, {len(llm_entities)} entities")

    # Save all scene prompts (atomic write)
    PROMPTS_DIR.mkdir(exist_ok=True)
    temp_dir = PROMPTS_DIR / f"_temp_{chapter_index:04d}"
    temp_dir.mkdir(exist_ok=True)

    try:
        for sp in scene_prompts:
            prompt_path = temp_dir / f"chapter_{sp.chapter_index:04d}_scene_{sp.scene_index:02d}.json"
            with open(prompt_path, "w", encoding="utf-8") as f:
                json.dump({
                    "chapter_index": sp.chapter_index,
                    "scene_index": sp.scene_index,
                    "scene_global_index": sp.scene_global_index,
                    "chapter_title": sp.chapter_title,
                    "text_chunk": sp.text_chunk,
                    "audio_start_sec": sp.audio_start_sec,
                    "audio_end_sec": sp.audio_end_sec,
                    "duration_sec": sp.duration_sec,
                    "prompt": sp.prompt,
                    "scene_description": sp.scene_description,
                    "mood": sp.mood,
                    "entities": sp.entities
                }, f, indent=2, ensure_ascii=False)

        # Move from temp to final (atomic)
        for sp in scene_prompts:
            src = temp_dir / f"chapter_{sp.chapter_index:04d}_scene_{sp.scene_index:02d}.json"
            dst = PROMPTS_DIR / f"chapter_{sp.chapter_index:04d}_scene_{sp.scene_index:02d}.json"
            src.replace(dst)

        # Clean up temp dir
        temp_dir.rmdir()

        print(f"  Saved {len(scene_prompts)} prompt(s)")
        return len(scene_prompts), current_global_index

    except Exception as e:
        print(f"  ERROR saving prompts: {e}")
        if temp_dir.exists():
            for f in temp_dir.glob("*"):
                f.unlink()
            temp_dir.rmdir()
        return 0, scene_counter


def main():
    """Main entry point."""
    print("=" * 60)
    print("Step 1: Generate Image Prompts")
    print("=" * 60)

    POLL_INTERVAL = 30  # seconds between polls for new plan files

    # Load consistency registry
    print("\nLoading consistency registry...")
    characters, locations = load_consistency_registry()
    print(f"  Characters: {len(characters)}")
    print(f"  Locations: {len(locations)}")

    # Load model once, reuse across polls
    model = None
    tokenizer = None
    total_processed = 0
    start_time = time.time()
    scene_counter = 0

    try:
        while True:
            # Find chapters to process
            plan_files = sorted(PLAN_DIR.glob("chapter_*.json"))
            remaining = [f for f in plan_files if not is_chapter_completed(int(f.stem.split("_")[1]))]

            if not remaining:
                if total_processed > 0 and len(plan_files) >= 2883:
                    print("\nAll 2,883 chapters processed! Pipeline complete.")
                    break
                print(f"\r  Waiting for plan files... ({len(plan_files)} plans, {total_processed} prompts done)", end="", flush=True)
                time.sleep(POLL_INTERVAL)
                continue

            # Load model on first batch (lazy load)
            if model is None:
                print(f"\nFound {len(remaining)} chapters to process")
                print("-" * 60)
                model, tokenizer = load_model()
                print("-" * 60)
                # Count already-processed scenes
                scene_counter = sum(
                    json.load(open(f, encoding="utf-8")).get("scene_count", 1)
                    for f in plan_files if is_chapter_completed(int(f.stem.split("_")[1]))
                )

            # Process available chapters
            batch_processed = 0
            for idx, plan_file in enumerate(remaining):
                chapter_index = int(plan_file.stem.split("_")[1])

                print(f"\n[{total_processed + idx + 1}] Chapter {chapter_index} ({len(remaining) - idx - 1} remaining)")

                scenes_processed, scene_counter = process_chapter(
                    chapter_index,
                    model,
                    tokenizer,
                    characters,
                    locations,
                    scene_counter
                )

                if scenes_processed > 0:
                    mark_chapter_completed(chapter_index)
                    total_processed += scenes_processed
                    batch_processed += scenes_processed

                # Progress update every 10
                if (idx + 1) % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = total_processed / elapsed if elapsed > 0 else 0
                    print(f"\n--- {total_processed} scenes done, {rate * 60:.1f}/min ---")

            print(f"\nBatch done ({batch_processed} scenes). Polling for new plan files...")
            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\n\nStopped by user.")

    # Cleanup
    if model is not None:
        print("Cleaning up...")
        del model
        torch.cuda.empty_cache()

    total_time = time.time() - start_time
    print(f"\nStep 1: {total_processed} scenes in {total_time/3600:.1f}h")
    print("=" * 60)


if __name__ == "__main__":
    main()
