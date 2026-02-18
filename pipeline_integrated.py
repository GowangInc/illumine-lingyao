#!/usr/bin/env python3
"""
Integrated Pipeline - Direct Implementation with Comprehensive Clip Validation

This script directly implements Steps 1-4 with full validation at every stage:
- Step 1: Generate prompts using Qwen3-4B
- Step 2: Generate images using Flux.2 Klein 4B  
- Step 3: Generate clips using FFmpeg (with rigorous validation)
- Step 4: Build final videos (only when ALL clips are validated)

Key Features:
- Direct model integration (no subprocess delegation)
- Rigorous clip validation: duration check, corruption check, ffprobe verification
- Automatic clip regeneration on failure
- Segment building ONLY proceeds when 100% of clips pass validation
- Comprehensive logging and progress tracking
"""

import os
import sys
import json
import time
import subprocess
import tempfile
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime

# Configuration
PLAN_DIR = Path("plan")
PROMPTS_DIR = Path("prompts")
IMAGES_DIR = Path("images")
CLIPS_DIR = Path("clips")
VIDEOS_DIR = Path("videos")
DESCRIPTIONS_DIR = Path("descriptions")
CONSISTENCY_DIR = Path("consistency")
PROGRESS_DIR = Path("progress")

CHAPTERS_AUDIO_DIR = Path("Q:/toys/audio-book-maker/outputs/Illumine Lingao (English Translation)/chapters")

PLAYLIST_URL = "https://www.youtube.com/playlist?list=PLjq2oIRxOOOmKJ54-0L-JCr46zLuNXmFK"

HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Video settings
MAX_SEGMENT_HOURS = 11.95
MAX_SEGMENT_SECONDS = MAX_SEGMENT_HOURS * 3600
OUTPUT_WIDTH = 1920
OUTPUT_HEIGHT = 1080
FPS = 30

# Validation settings
CLIP_VALIDATION_TIMEOUT = 15
MAX_CLIP_RETRIES = 3
MIN_CLIP_DURATION_TOLERANCE = 0.9  # Clip must be at least 90% of expected duration

# Art style for prompts
ART_STYLE_PREFIX = "Ming Dynasty ink wash painting (文人画), 1628, muted earth tones with cobalt and vermillion accents, cinematic widescreen"

# System prompt for Qwen
SYSTEM_PROMPT = """Create an image prompt for a chapter title card. Setting: 1628 Ming Dynasty Hainan, novel "Illumine Lingao".

Rules:
- Under 75 words. Focus on subject, environment, mood.
- Include calligraphic text of the chapter title in the image.
- Extract character and location names from the text.
- Do NOT include art style (added automatically).

Respond with JSON only:
{"prompt": "visual description with calligraphy reading 'Chapter Title'...", "scene_description": "1-sentence summary", "mood": "tone keywords", "entities": ["Name1", "Location1"]}"""

# Volume definitions
VOLUMES = [
    {"vol": 1,  "key": "vol01", "name": "Setting Sail",         "start": 0,    "end": 44},
    {"vol": 2,  "key": "vol02", "name": "New World",            "start": 45,   "end": 222},
    {"vol": 3,  "key": "vol03", "name": "New Society",          "start": 223,  "end": 558},
    {"vol": 4,  "key": "vol04", "name": "New Australia",        "start": 559,  "end": 780},
    {"vol": 5,  "key": "vol05", "name": "Entering",             "start": 781,  "end": 1223},
    {"vol": 6,  "key": "vol06", "name": "Conflict",             "start": 1224, "end": 1666},
    {"vol": 7,  "key": "vol07", "name": "Guangzhou Governance", "start": 1667, "end": 2070},
    {"vol": 8,  "key": "vol08", "name": "Two Guangs Campaign",  "start": 2071, "end": 2402},
    {"vol": 9,  "key": "vol09", "name": "Deep Cultivation",     "start": 2403, "end": 2790},
    {"vol": 10, "key": "vol10", "name": "Volume 10",            "start": 2791, "end": 2878},
    {"vol": 0,  "key": "extras","name": "Extras",               "start": 2879, "end": 2882},
]


@dataclass
class PipelineStats:
    prompts_generated: int = 0
    prompts_failed: int = 0
    images_generated: int = 0
    images_failed: int = 0
    clips_generated: int = 0
    clips_failed: int = 0
    clips_retried: int = 0
    videos_built: int = 0
    videos_failed: int = 0
    start_time: float = field(default_factory=time.time)
    
    def elapsed(self) -> str:
        mins = (time.time() - self.start_time) / 60
        if mins < 60:
            return f"{mins:.1f}m"
        else:
            return f"{mins/60:.1f}h"


@dataclass
class ChapterInfo:
    index: int
    title: str
    audio_file: str
    audio_duration: float
    scene_count: int
    scenes: List[Dict]


@dataclass
class Segment:
    volume: Dict
    part: int
    total_parts: int
    chapters: List[ChapterInfo]
    total_duration: float


class ClipValidator:
    """Comprehensive clip validation using ffprobe."""
    
    def __init__(self):
        self.cache: Dict[Path, Dict] = {}
        self.failed: Set[Path] = set()
    
    def validate(self, clip_path: Path, expected_duration: Optional[float] = None) -> Tuple[bool, str]:
        """Thoroughly validate a clip file."""
        if not clip_path.exists():
            return False, "File not found"
        
        if clip_path in self.failed:
            return False, "Previously marked as failing"
        
        # Check file size
        size = clip_path.stat().st_size
        if size == 0:
            return False, "Empty file (0 bytes)"
        
        # Use ffprobe for validation
        try:
            # Check duration
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(clip_path)],
                capture_output=True, text=True, timeout=CLIP_VALIDATION_TIMEOUT
            )
            
            if result.returncode != 0:
                return False, f"ffprobe error: {result.stderr[:100]}"
            
            if not result.stdout.strip():
                return False, "Empty duration from ffprobe"
            
            try:
                duration = float(result.stdout.strip())
            except ValueError:
                return False, f"Invalid duration: {result.stdout.strip()[:50]}"
            
            if duration <= 0:
                return False, f"Zero/negative duration: {duration}"
            
            # Check video stream exists
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=codec_name", "-of", "default=noprint_wrappers=1", str(clip_path)],
                capture_output=True, text=True, timeout=CLIP_VALIDATION_TIMEOUT
            )
            
            if result.returncode != 0 or "codec_name" not in result.stdout:
                return False, "No valid video stream"
            
            # Check duration matches expected
            if expected_duration and expected_duration > 0:
                min_dur = expected_duration * MIN_CLIP_DURATION_TOLERANCE
                max_dur = expected_duration * 1.1
                
                if duration < min_dur:
                    return False, f"Too short: {duration:.1f}s < {min_dur:.1f}s expected"
                if duration > max_dur:
                    return False, f"Too long: {duration:.1f}s > {max_dur:.1f}s expected"
            
            self.cache[clip_path] = {"duration": duration, "time": time.time()}
            return True, f"OK ({duration:.1f}s)"
            
        except subprocess.TimeoutExpired:
            return False, f"ffprobe timeout"
        except Exception as e:
            return False, f"Error: {str(e)[:100]}"
    
    def batch_validate(self, clips: List[Path], durations: Dict[Path, float] = None) -> Dict[Path, Tuple[bool, str]]:
        """Validate multiple clips."""
        durations = durations or {}
        return {c: self.validate(c, durations.get(c)) for c in clips}
    
    def mark_failed(self, clip_path: Path):
        self.failed.add(clip_path)
        self.cache.pop(clip_path, None)


class IntegratedPipeline:
    """Direct implementation of all pipeline steps."""
    
    def __init__(self):
        self.stats = PipelineStats()
        self.validator = ClipValidator()
        
        # Models (lazy loaded)
        self.qwen_model = None
        self.qwen_tokenizer = None
        self.flux_pipe = None
        self.characters = {}
        self.locations = {}
        
        # Ensure directories exist
        for d in [PROMPTS_DIR, IMAGES_DIR, CLIPS_DIR, VIDEOS_DIR, DESCRIPTIONS_DIR, PROGRESS_DIR]:
            d.mkdir(exist_ok=True)
    
    def log(self, msg: str, level: str = "INFO"):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] [{level}] {msg}")
    
    def check_ffmpeg(self) -> bool:
        try:
            r = subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
            return r.returncode == 0
        except:
            return False
    
    # ==================== Progress Tracking ====================
    
    def is_done(self, step: str, id: str) -> bool:
        f = PROGRESS_DIR / f"{step}_completed.txt"
        if not f.exists():
            return False
        return id in {l.strip() for l in f.read_text(encoding='utf-8').splitlines() if l.strip()}
    
    def mark_done(self, step: str, id: str):
        with open(PROGRESS_DIR / f"{step}_completed.txt", "a") as f:
            f.write(f"{id}\n")
    
    def get_retry_count(self, ch: int, sc: int) -> int:
        f = PROGRESS_DIR / "clip_retries.txt"
        if not f.exists():
            return 0
        key = f"{ch}_{sc}"
        for line in f.read_text(encoding='utf-8').splitlines():
            if line.startswith(key + ":"):
                return int(line.split(":")[1])
        return 0
    
    def inc_retry(self, ch: int, sc: int):
        f = PROGRESS_DIR / "clip_retries.txt"
        retries = {}
        if f.exists():
            for line in f.read_text(encoding='utf-8').splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    retries[k] = int(v)
        retries[f"{ch}_{sc}"] = retries.get(f"{ch}_{sc}", 0) + 1
        with open(f, "w", encoding='utf-8') as f:
            for k, v in sorted(retries.items()):
                f.write(f"{k}:{v}\n")
    
    # ==================== Step 1: Prompts ====================
    
    def load_qwen(self):
        if self.qwen_model:
            return
        
        self.log("Loading Qwen3-4B...")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from huggingface_hub import login
        
        login(token=HF_TOKEN)
        
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
        self.qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)
        self.qwen_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-4B",
            quantization_config=quant,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        self.log("Qwen loaded")
    
    def load_consistency(self):
        char_file = CONSISTENCY_DIR / "characters.json"
        loc_file = CONSISTENCY_DIR / "locations.json"
        if char_file.exists():
            self.characters = json.loads(char_file.read_text(encoding='utf-8'))
        if loc_file.exists():
            self.locations = json.loads(loc_file.read_text(encoding='utf-8'))
    
    def generate_prompt(self, text: str, title: str) -> Optional[Dict]:
        """Generate prompt using Qwen."""
        import torch
        
        user = f"Chapter: {title}\n\nScene Text:\n{text[:3000]}"
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user}
        ]
        
        text_input = self.qwen_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = self.qwen_tokenizer(text_input, return_tensors="pt").to(self.qwen_model.device)
        
        with torch.no_grad():
            outputs = self.qwen_model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.qwen_tokenizer.eos_token_id
            )
        
        response = self.qwen_tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # Parse JSON response
        think_end = response.find("</think>")
        if think_end >= 0:
            response = response[think_end + len("</think>"):].strip()
        
        try:
            match = re.search(r'\{[^{}]*"prompt"[^{}]*\}', response, re.DOTALL)
            if match:
                return json.loads(match.group())
            # Try full parse
            return json.loads(response.strip())
        except:
            # Manual extraction
            p = re.search(r'"prompt"\s*:\s*"([^"]+)"', response)
            d = re.search(r'"scene_description"\s*:\s*"([^"]+)"', response)
            m = re.search(r'"mood"\s*:\s*"([^"]+)"', response)
            if p:
                return {"prompt": p.group(1), "scene_description": d.group(1) if d else "", 
                        "mood": m.group(1) if m else "", "entities": []}
        return None
    
    def process_prompts(self) -> int:
        """Process pending prompt generation."""
        if not PLAN_DIR.exists():
            return 0
        
        pending = []
        for f in sorted(PLAN_DIR.glob("chapter_*.json")):
            try:
                idx = int(f.stem.split("_")[1])
                if not self.is_done("prompts", str(idx)):
                    pending.append((idx, f))
            except:
                pass
        
        if not pending:
            return 0
        
        self.load_qwen()
        self.load_consistency()
        
        processed = 0
        for idx, plan_file in pending[:20]:  # Batch size
            self.log(f"Processing chapter {idx}...")
            try:
                plan = json.loads(plan_file.read_text(encoding='utf-8'))
                title = plan.get("chapter_title", f"Chapter {idx}")
                text = plan.get("text_content", "")
                scenes = plan.get("scenes", [])
                
                if not scenes:
                    continue
                
                # Generate for first scene (each chapter has 1 scene)
                scene = scenes[0]
                chunk = text[scene.get("text_start_char", 0):scene.get("text_end_char", 1000)]
                
                result = self.generate_prompt(chunk, title)
                
                if result:
                    full_prompt = f"{ART_STYLE_PREFIX}\n\n{result.get('prompt', '')}"
                    data = {
                        "chapter_index": idx,
                        "scene_index": 0,
                        "scene_global_index": idx,
                        "chapter_title": title,
                        "text_chunk": chunk[:1000],
                        "audio_start_sec": scene.get("audio_start_sec", 0),
                        "audio_end_sec": scene.get("audio_end_sec", 60),
                        "duration_sec": scene.get("duration_sec", 60),
                        "prompt": full_prompt,
                        "scene_description": result.get("scene_description", ""),
                        "mood": result.get("mood", ""),
                        "entities": result.get("entities", [])
                    }
                    
                    out_file = PROMPTS_DIR / f"chapter_{idx:04d}_scene_00.json"
                    out_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')

                    self.mark_done("prompts", str(idx))
                    processed += 1
                    self.stats.prompts_generated += 1
                    self.log(f"✓ Generated prompt for chapter {idx}: {title}")
                else:
                    self.stats.prompts_failed += 1
                    
            except Exception as e:
                self.log(f"Error processing chapter {idx}: {e}", "ERROR")
                self.stats.prompts_failed += 1
        
        return processed
    
    # ==================== Step 2: Images ====================
    
    def load_flux(self):
        if self.flux_pipe:
            return
        
        self.log("Loading Flux.2 Klein...")
        import torch
        from diffusers import Flux2KleinPipeline
        
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.flux_pipe = Flux2KleinPipeline.from_pretrained(
            "black-forest-labs/FLUX.2-klein-4B", torch_dtype=dtype
        )
        self.flux_pipe.enable_model_cpu_offload()
        self.log("Flux loaded")
    
    def generate_image(self, prompt: str, seed: int) -> Optional["Image.Image"]:
        """Generate image using Flux."""
        import torch
        from PIL import Image
        
        generator = torch.Generator(device="cpu").manual_seed(seed)
        
        with torch.inference_mode():
            result = self.flux_pipe(
                prompt=prompt,
                height=1024,
                width=1024,
                guidance_scale=1.0,
                num_inference_steps=5,
                generator=generator
            )
        
        return result.images[0]
    
    def process_images(self) -> int:
        """Process pending image generation."""
        if not PROMPTS_DIR.exists():
            return 0

        pending = []
        for f in sorted(PROMPTS_DIR.glob("chapter_*_scene_*.json")):
            try:
                data = json.loads(f.read_text(encoding='utf-8'))
                ch, sc = data["chapter_index"], data["scene_index"]
                img_path = IMAGES_DIR / f"chapter_{ch:04d}_scene_{sc:02d}.png"
                if not img_path.exists():
                    pending.append((data, img_path))
            except:
                pass
        
        if not pending:
            return 0
        
        self.load_flux()
        
        processed = 0
        for data, img_path in pending[:10]:  # Batch size
            try:
                ch = data["chapter_index"]
                sc = data["scene_index"]
                prompt = data.get("prompt", "")
                seed = data.get("scene_global_index", 0)

                self.log(f"Generating image for chapter {ch:04d} scene {sc:02d}...")

                image = self.generate_image(prompt, seed)

                if image:
                    image.save(img_path, "PNG")
                    processed += 1
                    self.stats.images_generated += 1
                    self.mark_done("images", str(ch))
                    self.log(f"✓ Generated image: {img_path.name}")
                else:
                    self.stats.images_failed += 1
                    self.log(f"✗ Failed to generate image for chapter {ch:04d} scene {sc:02d}", "ERROR")
                    
            except Exception as e:
                self.log(f"Error generating image: {e}", "ERROR")
                self.stats.images_failed += 1
        
        return processed
    
    # ==================== Step 3: Clips ====================
    
    def get_pending_clips(self) -> List[Dict]:
        """Get scenes needing clips with validation."""
        if not PROMPTS_DIR.exists():
            return []
        
        pending = []
        for f in sorted(PROMPTS_DIR.glob("chapter_*_scene_*.json")):
            try:
                data = json.loads(f.read_text(encoding='utf-8'))
                ch, sc = data["chapter_index"], data["scene_index"]

                img = IMAGES_DIR / f"chapter_{ch:04d}_scene_{sc:02d}.png"
                if not img.exists():
                    continue
                
                clip = CLIPS_DIR / f"chapter_{ch:04d}_scene_{sc:02d}.mp4"
                expected = data.get("duration_sec", 60)
                
                if clip.exists():
                    valid, msg = self.validator.validate(clip, expected)
                    if valid:
                        continue
                    self.log(f"Invalid clip (regenerating): {clip.name} - {msg}", "WARN")
                    clip.unlink()
                    self.stats.clips_retried += 1
                
                if self.get_retry_count(ch, sc) >= MAX_CLIP_RETRIES:
                    self.validator.mark_failed(clip)
                    continue
                
                pending.append(data)
            except Exception as e:
                self.log(f"Error checking clip: {e}", "ERROR")
        
        return pending
    
    def generate_clip(self, data: Dict) -> Tuple[bool, str]:
        """Generate and validate a single clip."""
        ch, sc = data["chapter_index"], data["scene_index"]
        duration = data.get("duration_sec", 60)
        
        img = IMAGES_DIR / f"chapter_{ch:04d}_scene_{sc:02d}.png"
        clip = CLIPS_DIR / f"chapter_{ch:04d}_scene_{sc:02d}.mp4"
        temp = clip.with_suffix(".tmp.mp4")
        
        if not img.exists():
            return False, "Image not found"
        
        try:
            cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-loop", "1", "-i", str(img),
                "-vf", f"scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:force_original_aspect_ratio=decrease,pad={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:(ow-iw)/2:(oh-ih)/2",
                "-c:v", "libx264", "-preset", "veryfast", "-tune", "stillimage",
                "-pix_fmt", "yuv420p", "-crf", "23", "-r", "30",
                "-t", str(duration), str(temp)
            ]
            
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=max(300, duration * 2))
            
            if r.returncode != 0:
                temp.exists() and temp.unlink()
                return False, f"FFmpeg: {r.stderr[-150:]}"
            
            if not temp.exists():
                return False, "No output file"
            
            # Validate before moving
            valid, msg = self.validator.validate(temp, duration)
            if not valid:
                temp.unlink()
                return False, f"Validation: {msg}"
            
            temp.replace(clip)
            return True, "OK"
            
        except subprocess.TimeoutExpired:
            temp.exists() and temp.unlink()
            return False, "Timeout"
        except Exception as e:
            temp.exists() and temp.unlink()
            return False, str(e)[:100]
    
    def process_clips(self) -> int:
        """Process pending clip generation."""
        pending = self.get_pending_clips()
        if not pending:
            return 0
        
        processed = 0
        for data in pending:
            ch, sc = data["chapter_index"], data["scene_index"]
            duration = data.get("duration_sec", 60)

            self.log(f"Generating clip for chapter {ch:04d} scene {sc:02d} ({duration:.1f}s)...")

            success, msg = self.generate_clip(data)

            if success:
                processed += 1
                self.stats.clips_generated += 1
                self.mark_done("clips", f"{ch}_{sc}")
                self.log(f"✓ Clip {ch:04d}:{sc:02d} generated successfully")
            else:
                self.stats.clips_failed += 1
                self.inc_retry(ch, sc)
                self.log(f"✗ Clip {ch:04d}:{sc:02d} failed: {msg}", "ERROR")
        
        return processed
    
    # ==================== Step 4: Videos ====================
    
    def load_chapters(self) -> Dict[int, ChapterInfo]:
        chapters = {}
        for f in sorted(PLAN_DIR.glob("chapter_*.json")):
            try:
                d = json.loads(f.read_text(encoding='utf-8'))
                chapters[d["chapter_index"]] = ChapterInfo(
                    index=d["chapter_index"],
                    title=d["chapter_title"],
                    audio_file=d["audio_file"],
                    audio_duration=d["audio_duration"],
                    scene_count=d["scene_count"],
                    scenes=d.get("scenes", [])
                )
            except:
                pass
        return chapters
    
    def build_segments(self, chapters: Dict[int, ChapterInfo]) -> List[Segment]:
        segments = []
        for vol in VOLUMES:
            vol_chs = [chapters[i] for i in range(vol["start"], vol["end"] + 1) if i in chapters]
            if not vol_chs:
                continue
            
            current, dur, vol_segs = [], 0.0, []
            for ch in vol_chs:
                if dur + ch.audio_duration > MAX_SEGMENT_SECONDS and current:
                    vol_segs.append(current)
                    current, dur = [], 0.0
                current.append(ch)
                dur += ch.audio_duration
            if current:
                vol_segs.append(current)
            
            for i, chs in enumerate(vol_segs):
                segments.append(Segment(
                    volume=vol, part=i + 1, total_parts=len(vol_segs),
                    chapters=chs, total_duration=sum(c.audio_duration for c in chs)
                ))
        return segments
    
    def seg_filename(self, seg: Segment) -> str:
        vol = seg.volume
        base = f"Illumine Lingao - Vol. {vol['vol']:02d} - {vol['name']}" if vol["vol"] > 0 else "Illumine Lingao - Extras"
        if seg.total_parts > 1:
            base += f" (Part {seg.part})"
        return base
    
    def validate_segment_clips(self, seg: Segment) -> Tuple[bool, str]:
        """Validate ALL clips for a segment. Returns (all_valid, error_msg)."""
        for ch in seg.chapters:
            for sc in ch.scenes:
                clip = CLIPS_DIR / f"chapter_{ch.index:04d}_scene_{sc['scene_index']:02d}.mp4"
                expected = sc.get("duration_sec", ch.audio_duration)
                
                valid, msg = self.validator.validate(clip, expected)
                if not valid:
                    return False, f"{clip.name}: {msg}"
        return True, ""
    
    def build_video(self, seg: Segment) -> bool:
        """Build a segment video with full validation."""
        name = self.seg_filename(seg)
        out = VIDEOS_DIR / f"{name}.mp4"
        
        if out.exists():
            return True
        
        self.log(f"Building: {name}")
        
        # CRITICAL: Validate ALL clips before proceeding
        all_valid, error = self.validate_segment_clips(seg)
        if not all_valid:
            self.log(f"  Validation failed: {error}", "ERROR")
            return False
        
        # Create concat lists
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as cf:
                for ch in seg.chapters:
                    for sc in ch.scenes:
                        p = CLIPS_DIR / f"chapter_{ch.index:04d}_scene_{sc['scene_index']:02d}.mp4"
                        cf.write(f"file '{str(p.resolve()).replace(chr(92), '/')}'\n")
                clips_list = cf.name
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as af:
                for ch in seg.chapters:
                    a = CHAPTERS_AUDIO_DIR / ch.audio_file
                    af.write(f"file '{str(a.resolve()).replace(chr(92), '/')}'\n")
                audio_list = af.name
            
            cmd = [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0", "-i", clips_list,
                "-f", "concat", "-safe", "0", "-i", audio_list,
                "-map", "0:v", "-map", "1:a",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
                "-ar", "44100", "-ac", "2",
                "-movflags", "+faststart", "-shortest", str(out)
            ]
            
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
            
            os.unlink(clips_list)
            os.unlink(audio_list)
            
            if r.returncode != 0:
                self.log(f"  FFmpeg error: {r.stderr[-300:]}", "ERROR")
                out.exists() and out.unlink()
                return False
            
            # Validate output
            valid, msg = self.validator.validate(out)
            if not valid:
                self.log(f"  Output validation failed: {msg}", "ERROR")
                out.unlink()
                return False
            
            size_gb = out.stat().st_size / (1024**3)
            self.log(f"  ✓ Complete: {size_gb:.1f} GB")
            return True
            
        except Exception as e:
            self.log(f"  Error: {e}", "ERROR")
            return False
    
    def generate_description(self, seg: Segment) -> str:
        lines = [
            "Illumine Lingao (临高启明) by Blowing Past the Ear (吹牛者)",
            f"Volume {seg.volume['vol']}: {seg.volume['name']}" + (f" — Part {seg.part} of {seg.total_parts}" if seg.total_parts > 1 else ""),
            "",
            "Illumine Lingao is a Chinese alternate history web novel about 500 modern people",
            "who travel back to 1628 Ming Dynasty China. This is a fan-made audiobook of the",
            "English translation, created to make this epic story more accessible.",
            "",
            "I'm just a fan who wanted to contribute in a small way. Enjoy!",
            "",
            "Chapters:"
        ]
        
        t = 0.0
        for ch in seg.chapters:
            h, m, s = int(t//3600), int((t%3600)//60), int(t%60)
            lines.append(f"{h}:{m:02d}:{s:02d} {ch.title}")
            t += ch.audio_duration
        
        lines.extend(["", f"Full playlist: {PLAYLIST_URL}"])
        lines.extend(["", "Read the novel online: https://lingao.entropydrivenmindset.win"])
        return "\n".join(lines)
    
    def process_videos(self) -> int:
        """Build videos for ready segments."""
        chapters = self.load_chapters()
        if not chapters:
            return 0
        
        segments = self.build_segments(chapters)
        built = 0
        
        for seg in segments:
            out = VIDEOS_DIR / f"{self.seg_filename(seg)}.mp4"
            if out.exists():
                continue
            
            # Check if clips are ready
            clips_ready = True
            for ch in seg.chapters:
                for sc in ch.scenes:
                    clip = CLIPS_DIR / f"chapter_{ch.index:04d}_scene_{sc['scene_index']:02d}.mp4"
                    if not clip.exists():
                        clips_ready = False
                        break
                if not clips_ready:
                    break
            
            if not clips_ready:
                continue
            
            if self.build_video(seg):
                built += 1
                self.stats.videos_built += 1
                self.mark_done("videos", self.seg_filename(seg))
                
                # Write description
                desc = DESCRIPTIONS_DIR / f"{self.seg_filename(seg)}.txt"
                desc.write_text(self.generate_description(seg), encoding="utf-8")
            else:
                self.stats.videos_failed += 1
        
        return built
    
    # ==================== Main Loop ====================
    
    def print_status(self):
        p = len(list(PROMPTS_DIR.glob("*.json"))) if PROMPTS_DIR.exists() else 0
        i = len(list(IMAGES_DIR.glob("*.png"))) if IMAGES_DIR.exists() else 0
        c = len(list(CLIPS_DIR.glob("*.mp4"))) if CLIPS_DIR.exists() else 0
        v = len(list(VIDEOS_DIR.glob("*.mp4"))) if VIDEOS_DIR.exists() else 0
        
        print("\n" + "=" * 50)
        print(f"Status (runtime: {self.stats.elapsed()})")
        print("-" * 50)
        print(f"Prompts: {p} | Images: {i} | Clips: {c} | Videos: {v}")
        print(f"Failed: prompts={self.stats.prompts_failed}, images={self.stats.images_failed}, "
              f"clips={self.stats.clips_failed} (retried={self.stats.clips_retried}), videos={self.stats.videos_failed}")
        print("=" * 50 + "\n")
    
    def get_clips_needed_for_segment(self, seg: Segment) -> List[Dict]:
        """Get list of clip data needed for a specific segment."""
        needed = []
        for ch in seg.chapters:
            for sc in ch.scenes:
                clip_path = CLIPS_DIR / f"chapter_{ch.index:04d}_scene_{sc['scene_index']:02d}.mp4"
                expected = sc.get("duration_sec", ch.audio_duration)

                # Validate existing clip
                if clip_path.exists():
                    valid, _ = self.validator.validate(clip_path, expected)
                    if valid:
                        continue
                    # Invalid clip - needs regeneration
                    clip_path.unlink()
                    self.stats.clips_retried += 1

                # Need to generate this clip
                prompt_file = PROMPTS_DIR / f"chapter_{ch.index:04d}_scene_{sc['scene_index']:02d}.json"
                if prompt_file.exists():
                    try:
                        data = json.loads(prompt_file.read_text(encoding='utf-8'))
                        needed.append(data)
                    except:
                        pass
        return needed

    def run_backwards(self):
        """Work backwards: prioritize video building, generate clips as needed."""
        self.log("=" * 50)
        self.log("Integrated Pipeline Starting - BACKWARDS MODE")
        self.log("Priority: Build videos → Generate missing clips → Next video")
        self.log("=" * 50)

        if not self.check_ffmpeg():
            self.log("FFmpeg not found!", "ERROR")
            sys.exit(1)

        if not CHAPTERS_AUDIO_DIR.exists():
            self.log(f"Audio dir not found: {CHAPTERS_AUDIO_DIR}", "ERROR")
            sys.exit(1)

        # Load chapter plans once
        chapters = self.load_chapters()
        if not chapters:
            self.log("No chapter plans found! Run plan_chapters.py first.", "ERROR")
            sys.exit(1)

        segments = self.build_segments(chapters)
        self.log(f"Found {len(segments)} video segments to build")

        try:
            seg_idx = 0
            while seg_idx < len(segments):
                seg = segments[seg_idx]
                seg_name = self.seg_filename(seg)
                output_path = VIDEOS_DIR / f"{seg_name}.mp4"

                # Skip if already done
                if output_path.exists():
                    self.log(f"✓ Video already exists: {seg_name}")
                    seg_idx += 1
                    continue

                self.log(f"\n{'='*50}")
                self.log(f"Processing segment {seg_idx + 1}/{len(segments)}: {seg_name}")
                self.log(f"Chapters: {len(seg.chapters)} | Duration: {seg.total_duration/3600:.1f}h")
                self.log("="*50)

                # Step 1: Check what clips are needed for this video
                needed_clips = self.get_clips_needed_for_segment(seg)

                if needed_clips:
                    self.log(f"Need to generate {len(needed_clips)} clips for this video")

                    # Step 2: Generate the needed clips
                    clips_made = 0
                    for data in needed_clips:
                        ch, sc = data["chapter_index"], data["scene_index"]
                        duration = data.get("duration_sec", 60)

                        self.log(f"Generating clip for chapter {ch:04d} scene {sc:02d} ({duration:.1f}s)...")

                        success, msg = self.generate_clip(data)

                        if success:
                            clips_made += 1
                            self.stats.clips_generated += 1
                            self.mark_done("clips", f"{ch}_{sc}")
                            self.log(f"✓ Clip {ch:04d}:{sc:02d} generated successfully")
                        else:
                            self.stats.clips_failed += 1
                            self.inc_retry(ch, sc)
                            self.log(f"✗ Clip {ch:04d}:{sc:02d} failed: {msg}", "ERROR")

                    self.log(f"Generated {clips_made}/{len(needed_clips)} clips this cycle")

                    # Check if all clips are now ready
                    still_needed = self.get_clips_needed_for_segment(seg)
                    if still_needed:
                        self.log(f"Still missing {len(still_needed)} clips, will retry on next cycle")
                        time.sleep(5)
                        continue  # Try same segment again

                # Step 3: All clips ready - build the video
                self.log(f"All clips ready. Building video: {seg_name}")

                if self.build_video(seg):
                    self.stats.videos_built += 1
                    self.mark_done("videos", seg_name)

                    # Write description
                    desc = DESCRIPTIONS_DIR / f"{seg_name}.txt"
                    desc.write_text(self.generate_description(seg), encoding="utf-8")
                    self.log(f"✓ Video complete: {seg_name}")

                    # Move to next segment
                    seg_idx += 1
                else:
                    self.stats.videos_failed += 1
                    self.log(f"✗ Video build failed: {seg_name}", "ERROR")
                    # Stay on this segment to retry
                    time.sleep(10)

                # Status update every segment
                self.print_status()

            self.log("\n" + "=" * 50)
            self.log("ALL SEGMENTS COMPLETE!")
            self.log("=" * 50)
            self.print_status()

        except KeyboardInterrupt:
            self.log("\nStopped by user")
            self.print_status()

    def run(self):
        """Default to backwards mode (video-first)."""
        self.run_backwards()


def main():
    IntegratedPipeline().run()


if __name__ == "__main__":
    main()
