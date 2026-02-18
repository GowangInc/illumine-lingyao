#!/usr/bin/env python3
"""
Step 0: Plan & Map

Pre-computes the full project structure before generating any assets:
- Parse EPUB to get chapter text
- Use Whisper to generate word-level timestamps for each chapter WAV
- Split chapter text into overlapping chunks
- Map each text chunk to precise audio timestamps via forced alignment
- Extract entities (characters, locations) for consistency tracking

GPU OPTIMIZED: Loads Whisper model once, keeps it in VRAM for all chapters.
"""

import os
import re
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import warnings

# EPUB parsing
try:
    import ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup
except ImportError:
    print("Installing required packages: ebooklib, beautifulsoup4")
    os.system("pip install -q ebooklib beautifulsoup4")
    import ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup

# Whisper for transcription
try:
    import whisper
    import torch
except ImportError:
    print("Installing required package: openai-whisper, torch")
    os.system("pip install -q openai-whisper torch")
    import whisper
    import torch

# Suppress warnings
warnings.filterwarnings("ignore")

# Configuration
CHAPTERS_DIR = Path("Q:/toys/audio-book-maker/outputs/Illumine Lingao (English Translation)/chapters")
EPUB_PATH = Path("Q:/toys/audio-book-maker/inputs/Illumine Lingao (English Translation).epub")
OUTPUT_DIR = Path("plan")
CONSISTENCY_DIR = Path("consistency")
PROGRESS_DIR = Path("progress")

# Chunking parameters
TARGET_TOKENS_PER_CHUNK = 120000
TOKEN_OVERLAP = 10000
TOKENS_PER_CHAR = 0.25

CHUNK_SIZE_CHARS = int(TARGET_TOKENS_PER_CHUNK / TOKENS_PER_CHAR)
OVERLAP_CHARS = int(TOKEN_OVERLAP / TOKENS_PER_CHAR)

# Global model cache for GPU efficiency
_whisper_model = None
_whisper_model_name = None


@dataclass
class Scene:
    scene_index: int
    text_start_char: int
    text_end_char: int
    audio_start_sec: float = 0.0
    audio_end_sec: float = 0.0
    duration_sec: float = 0.0
    entities: List[str] = None
    
    def __post_init__(self):
        if self.entities is None:
            self.entities = []


@dataclass
class ChapterPlan:
    chapter_index: int
    chapter_title: str
    audio_file: str
    audio_duration: float
    scene_count: int
    scenes: List[Scene]
    text_content: str = ""


def get_chapter_audio_files() -> List[Tuple[int, str, Path]]:
    """Discover all chapter audio files."""
    chapters = []
    pattern = re.compile(r'chapter_(\d+)_(.+)\.wav$')
    
    for wav_file in sorted(CHAPTERS_DIR.glob("chapter_*.wav")):
        match = pattern.match(wav_file.name)
        if not match:
            continue
        
        index = int(match.group(1))
        title = match.group(2)
        
        if index >= 2883:
            continue
            
        chapters.append((index, title, wav_file))
    
    return sorted(chapters)


def parse_epub_chapters() -> Dict[int, str]:
    """Parse EPUB and extract text content for each chapter."""
    print(f"Parsing EPUB: {EPUB_PATH}")
    
    if not EPUB_PATH.exists():
        print(f"ERROR: EPUB not found at {EPUB_PATH}")
        return {}
    
    book = epub.read_epub(str(EPUB_PATH))
    chapters_text = {}
    
    chapter_idx = 0
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            content = item.get_content().decode('utf-8', errors='ignore')
            soup = BeautifulSoup(content, 'html.parser')
            
            text = soup.get_text(separator='\n', strip=True)
            text = re.sub(r'\n+', '\n', text)
            text = re.sub(r' +', ' ', text)
            
            if len(text) > 100:
                chapters_text[chapter_idx] = text
                chapter_idx += 1
    
    print(f"Extracted {len(chapters_text)} chapters from EPUB")
    return chapters_text


def load_whisper_model(model_size: str = "tiny"):
    """Load Whisper model once and keep in VRAM."""
    global _whisper_model, _whisper_model_name
    
    if _whisper_model is not None and _whisper_model_name == model_size:
        return _whisper_model
    
    # Unload previous model if different
    if _whisper_model is not None:
        del _whisper_model
        torch.cuda.empty_cache()
    
    print(f"Loading Whisper model ({model_size}) into VRAM...")
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    # Load model to GPU explicitly
    _whisper_model = whisper.load_model(model_size, device=device)
    _whisper_model_name = model_size
    
    # Verify GPU usage
    if torch.cuda.is_available():
        vram_mb = torch.cuda.memory_allocated() / 1024 / 1024
        print(f"  Model loaded: {vram_mb:.0f} MB VRAM used")
    else:
        print("  Model loaded on CPU (CUDA not available)")
    
    return _whisper_model


def transcribe_audio(audio_path: Path, model) -> Dict:
    """Transcribe audio using pre-loaded Whisper model."""
    # Ensure we're using CUDA if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    result = model.transcribe(
        str(audio_path),
        language="en",
        word_timestamps=True,
        verbose=False,
        fp16=(device == "cuda"),  # FP16 only on GPU
        condition_on_previous_text=False,  # Faster, less memory
        initial_prompt="This is a historical fiction audiobook set in 1628 Ming Dynasty China."
    )
    return result


def split_into_chunks(text: str) -> List[Tuple[int, int, str]]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = min(start + CHUNK_SIZE_CHARS, text_len)
        
        if end < text_len:
            search_start = int(start + CHUNK_SIZE_CHARS * 0.8)
            search_end = end
            
            segment = text[search_start:search_end]
            last_period = segment.rfind('. ')
            last_newline = segment.rfind('.\n')
            
            break_point = -1
            if last_period > 0:
                break_point = search_start + last_period + 1
            elif last_newline > 0:
                break_point = search_start + last_newline + 1
            
            if break_point > start:
                end = break_point
        
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append((start, end, chunk_text))
        
        start = end - OVERLAP_CHARS if end < text_len else text_len
        if start >= text_len:
            break
    
    return chunks


def align_text_to_audio(
    text_chunks: List[Tuple[int, int, str]], 
    whisper_result: Dict
) -> List[Tuple[float, float]]:
    """Map text chunks to audio timestamps."""
    words = []
    for segment in whisper_result.get("segments", []):
        for word_info in segment.get("words", []):
            words.append({
                "word": word_info.get("word", "").strip(),
                "start": word_info.get("start", 0.0),
                "end": word_info.get("end", 0.0)
            })
    
    if not words:
        duration = whisper_result.get("segments", [{}])[-1].get("end", 0.0)
        num_chunks = len(text_chunks)
        chunk_duration = duration / num_chunks if num_chunks > 0 else duration
        return [(i * chunk_duration, (i + 1) * chunk_duration) for i in range(num_chunks)]
    
    total_text_chars = sum(len(chunk[2]) for chunk in text_chunks)
    audio_duration = words[-1]["end"] if words else 0.0
    
    word_positions = []
    for word_info in words:
        text_pos = int((word_info["start"] / audio_duration) * total_text_chars) if audio_duration > 0 else 0
        word_positions.append((text_pos, word_info["start"], word_info["end"]))
    
    chunk_times = []
    
    for start_char, end_char, chunk_text in text_chunks:
        start_time = 0.0
        end_time = audio_duration
        
        for i, (pos, w_start, w_end) in enumerate(word_positions):
            if pos >= start_char and start_time == 0.0:
                start_time = w_start
            if pos >= end_char and end_time == audio_duration:
                end_time = w_end
                break
        
        chunk_times.append((start_time, end_time))
    
    return chunk_times


def extract_entities(text: str) -> List[str]:
    """Extract character and location names from text using multi-word patterns."""
    # Look for multi-word capitalized names (much more reliable for character names)
    # e.g. "Xiong Buyou", "Civil Affairs Commissioner Xiao Zishan", "Director Wen"
    multi_word_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b'
    multi_names = re.findall(multi_word_pattern, text[:10000])

    # Also look for Chinese-style names: single capitalized words that appear near
    # dialog attribution ("said X", "X said", "X replied", "X asked")
    dialog_pattern = r'(?:said|replied|asked|exclaimed|shouted|whispered|muttered|declared|announced)\s+([A-Z][a-z]{2,})'
    dialog_names = re.findall(dialog_pattern, text[:10000])
    dialog_pattern2 = r'([A-Z][a-z]{2,})\s+(?:said|replied|asked|exclaimed|shouted|whispered|muttered|declared|announced)'
    dialog_names += re.findall(dialog_pattern2, text[:10000])

    # Common English words to exclude (comprehensive list)
    common_words = {
        # Articles, pronouns, conjunctions
        "The", "A", "An", "This", "That", "These", "Those", "He", "She", "They",
        "We", "You", "I", "It", "And", "But", "Or", "For", "With", "In", "On",
        "At", "To", "From", "Its", "His", "Her", "Their", "Our", "My", "Your",
        # Common sentence starters and adverbs
        "After", "Before", "During", "Since", "Until", "While", "When", "Where",
        "What", "How", "Why", "Who", "Which", "Then", "Now", "Here", "There",
        "Just", "Only", "Also", "Even", "Still", "Yet", "Already", "Soon",
        "Very", "Too", "Much", "More", "Most", "Less", "Least", "Well",
        "Perhaps", "Maybe", "However", "Therefore", "Moreover", "Furthermore",
        "Nevertheless", "Meanwhile", "Otherwise", "Instead", "Besides",
        "Indeed", "Certainly", "Probably", "Apparently", "Obviously",
        "Unfortunately", "Fortunately", "Suddenly", "Finally", "Eventually",
        # Common verbs/adjectives that get capitalized at sentence starts
        "Are", "Is", "Was", "Were", "Has", "Had", "Have", "Does", "Did", "Do",
        "Can", "Could", "Would", "Should", "Will", "May", "Might", "Must",
        "Isn", "Aren", "Wasn", "Weren", "Hasn", "Hadn", "Haven", "Doesn", "Didn",
        "Don", "Won", "Wouldn", "Shouldn", "Couldn",
        "Being", "Having", "Going", "Coming", "Making", "Taking", "Getting",
        "Said", "Asked", "Replied", "Told", "Thought", "Knew", "Saw", "Came",
        "Went", "Made", "Got", "Took", "Gave", "Found", "Left", "Felt",
        "Let", "Put", "Set", "Ran", "Sat", "Stood", "Began", "Kept",
        # Common adjectives/nouns at sentence starts
        "Every", "Each", "All", "Any", "Some", "No", "Not", "Many", "Few",
        "Several", "Both", "Either", "Neither", "Such", "Other", "Another",
        "Great", "Old", "New", "Young", "Long", "Little", "Big", "Small",
        "Good", "Bad", "First", "Second", "Third", "Last", "Next",
        "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten",
        "Once", "Twice", "Half", "Part", "Chapter", "Section", "Book", "Volume",
        # Common abstract nouns
        "Disappointment", "Surprise", "Fear", "Anger", "Joy", "Hope", "Love",
        "Death", "Life", "Time", "Day", "Night", "Morning", "Evening", "Year",
        "People", "Man", "Woman", "Everyone", "Someone", "Anyone", "Nothing",
        "Everything", "Something", "Anything", "Thing", "Things",
        "Way", "Place", "World", "Country", "City", "Town", "Village",
        "House", "Room", "Door", "Road", "Water", "Land", "Sea", "Sky",
        # Discourse markers
        "Yes", "No", "Oh", "Ah", "Well", "So", "Right", "Sure", "Okay",
        "Look", "See", "Think", "Know", "Want", "Need", "Like",
        "If", "Although", "Though", "Because", "Unless", "Whether",
        "Across", "Along", "Among", "Around", "Behind", "Below", "Beneath",
        "Beside", "Between", "Beyond", "Inside", "Outside", "Through",
        "About", "Above", "Over", "Under", "Into", "Out", "Up", "Down",
    }

    entities = []

    # Multi-word names: filter out those starting/ending with common words
    for name in multi_names:
        words = name.split()
        # Skip if ALL words are common
        if all(w in common_words for w in words):
            continue
        # Skip very short results
        if len(name) <= 3:
            continue
        entities.append(name)

    # Dialog-attributed single names (high confidence these are character names)
    for name in dialog_names:
        if name not in common_words and len(name) > 2:
            entities.append(name)

    # Deduplicate and limit
    seen = set()
    unique = []
    for e in entities:
        if e.lower() not in seen:
            seen.add(e.lower())
            unique.append(e)

    return unique[:10]


def is_chapter_completed(chapter_index: int) -> bool:
    """Check if chapter has already been processed."""
    progress_file = PROGRESS_DIR / "plan_completed.txt"
    if not progress_file.exists():
        return False
    
    completed = progress_file.read_text().splitlines()
    return str(chapter_index) in completed


def mark_chapter_completed(chapter_index: int):
    """Mark chapter as completed in progress file."""
    PROGRESS_DIR.mkdir(exist_ok=True)
    progress_file = PROGRESS_DIR / "plan_completed.txt"
    
    with open(progress_file, "a") as f:
        f.write(f"{chapter_index}\n")


def process_chapter(
    chapter_index: int,
    chapter_title: str,
    audio_path: Path,
    epub_text: str,
    whisper_model
) -> Optional[ChapterPlan]:
    """Process a single chapter using pre-loaded Whisper model."""
    
    if is_chapter_completed(chapter_index):
        plan_path = OUTPUT_DIR / f"chapter_{chapter_index:04d}.json"
        if plan_path.exists():
            with open(plan_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                scenes = [Scene(**s) for s in data.get("scenes", [])]
                return ChapterPlan(
                    chapter_index=data["chapter_index"],
                    chapter_title=data["chapter_title"],
                    audio_file=data["audio_file"],
                    audio_duration=data["audio_duration"],
                    scene_count=data["scene_count"],
                    scenes=scenes,
                    text_content=data.get("text_content", "")
                )
        return None
    
    # Transcribe using pre-loaded model (GPU fully utilized)
    try:
        whisper_result = transcribe_audio(audio_path, whisper_model)
    except Exception as e:
        print(f"  ERROR transcribing: {e}")
        return None
    
    audio_duration = whisper_result.get("segments", [{}])[-1].get("end", 0.0)
    
    # Split and align
    text_chunks = split_into_chunks(epub_text)
    chunk_times = align_text_to_audio(text_chunks, whisper_result)
    
    # Create scenes
    scenes = []
    for i, ((start_char, end_char, chunk_text), (start_sec, end_sec)) in enumerate(
        zip(text_chunks, chunk_times)
    ):
        entities = extract_entities(chunk_text)
        
        scene = Scene(
            scene_index=i,
            text_start_char=start_char,
            text_end_char=end_char,
            audio_start_sec=start_sec,
            audio_end_sec=end_sec,
            duration_sec=end_sec - start_sec,
            entities=entities
        )
        scenes.append(scene)
    
    return ChapterPlan(
        chapter_index=chapter_index,
        chapter_title=chapter_title,
        audio_file=audio_path.name,
        audio_duration=audio_duration,
        scene_count=len(scenes),
        scenes=scenes,
        text_content=epub_text
    )


def save_chapter_plan(plan: ChapterPlan):
    """Save chapter plan to JSON file."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    plan_path = OUTPUT_DIR / f"chapter_{plan.chapter_index:04d}.json"
    
    data = {
        "chapter_index": plan.chapter_index,
        "chapter_title": plan.chapter_title,
        "audio_file": plan.audio_file,
        "audio_duration": plan.audio_duration,
        "scene_count": plan.scene_count,
        "scenes": [asdict(s) for s in plan.scenes],
        "text_content": plan.text_content
    }
    
    temp_path = plan_path.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    temp_path.replace(plan_path)


def build_consistency_registry(all_plans: List[ChapterPlan]):
    """Build character and location registries."""
    print("\nBuilding consistency registry...")
    
    char_counts = {}
    for plan in all_plans:
        for scene in plan.scenes:
            for entity in scene.entities:
                char_counts[entity] = char_counts.get(entity, 0) + 1
    
    recurring = {k: v for k, v in char_counts.items() if v >= 3}
    
    characters = {}
    for name, count in recurring.items():
        seed_hint = sum(ord(c) for c in name) % 10000
        
        characters[name] = {
            "description": f"{name} - recurring character in Lingao narrative",
            "first_seen_scene": 0,
            "seed_hint": seed_hint,
            "mentions": count
        }
    
    CONSISTENCY_DIR.mkdir(exist_ok=True)
    
    char_path = CONSISTENCY_DIR / "characters.json"
    with open(char_path, "w", encoding="utf-8") as f:
        json.dump(characters, f, indent=2, ensure_ascii=False)
    
    print(f"  Found {len(characters)} recurring characters")
    
    loc_path = CONSISTENCY_DIR / "locations.json"
    with open(loc_path, "w", encoding="utf-8") as f:
        json.dump({}, f, indent=2)


def print_progress_bar(current: int, total: int, width: int = 40):
    """Print a simple progress bar."""
    pct = current / total if total > 0 else 0
    filled = int(width * pct)
    bar = "█" * filled + "░" * (width - filled)
    print(f"\r[{bar}] {current}/{total} ({pct*100:.1f}%)", end="", flush=True)


def main():
    """Main entry point with full GPU utilization."""
    print("=" * 60)
    print("Step 0: Plan & Map - Illumine Lingao YouTube Project")
    print("GPU Optimized: Model loads once, stays in VRAM")
    print("=" * 60)
    
    if not CHAPTERS_DIR.exists():
        print(f"ERROR: Chapters directory not found: {CHAPTERS_DIR}")
        sys.exit(1)
    
    if not EPUB_PATH.exists():
        print(f"ERROR: EPUB not found: {EPUB_PATH}")
        sys.exit(1)
    
    # Get audio files
    print("\nDiscovering chapter audio files...")
    audio_files = get_chapter_audio_files()
    print(f"Found {len(audio_files)} chapter files")
    
    # Parse EPUB
    epub_chapters = parse_epub_chapters()
    if not epub_chapters:
        print("ERROR: Failed to parse EPUB")
        sys.exit(1)
    
    # Filter to valid chapters
    valid_chapters = []
    for chapter_index, title, audio_path in audio_files:
        epub_text = epub_chapters.get(chapter_index)
        if epub_text:
            valid_chapters.append((chapter_index, title, audio_path, epub_text))
    
    print(f"\n{len(valid_chapters)} chapters have both audio and text")
    
    # Check remaining
    remaining = [(c, t, a, e) for c, t, a, e in valid_chapters if not is_chapter_completed(c)]
    completed_count = len(valid_chapters) - len(remaining)
    
    if completed_count > 0:
        print(f"{completed_count} chapters already processed, {len(remaining)} remaining")
    
    if not remaining:
        print("\nAll chapters already processed!")
        all_plans = []
        for chapter_index, title, audio_path, epub_text in valid_chapters:
            plan_path = OUTPUT_DIR / f"chapter_{chapter_index:04d}.json"
            if plan_path.exists():
                with open(plan_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    scenes = [Scene(**s) for s in data.get("scenes", [])]
                    all_plans.append(ChapterPlan(
                        chapter_index=data["chapter_index"],
                        chapter_title=data["chapter_title"],
                        audio_file=data["audio_file"],
                        audio_duration=data["audio_duration"],
                        scene_count=data["scene_count"],
                        scenes=scenes,
                        text_content=data.get("text_content", "")
                    ))
        if all_plans:
            build_consistency_registry(all_plans)
        return
    
    # Whisper model selection
    whisper_model = os.environ.get("WHISPER_MODEL", "tiny")
    
    # Load model ONCE and keep in VRAM
    model = load_whisper_model(whisper_model)
    
    print(f"\nProcessing {len(remaining)} chapters...")
    print(f"Whisper model: {whisper_model} (loaded in VRAM)")
    print(f"Settings: {CHUNK_SIZE_CHARS} chars/chunk, {OVERLAP_CHARS} chars overlap")
    print("-" * 60)
    
    # Process all chapters with model in VRAM
    all_plans = []
    start_time = time.time()
    
    for idx, (chapter_index, title, audio_path, epub_text) in enumerate(remaining):
        # Progress
        if idx % 10 == 0 or idx == len(remaining) - 1:
            print(f"\n[{idx+1}/{len(remaining)}] Ch {chapter_index}: {title[:50]}...")
        
        # Process (GPU fully utilized)
        plan = process_chapter(chapter_index, title, audio_path, epub_text, model)
        
        if plan:
            save_chapter_plan(plan)
            all_plans.append(plan)
            mark_chapter_completed(chapter_index)
            
            # Print scene info
            if plan.scene_count > 1:
                print(f"  → {plan.scene_count} scenes, {plan.audio_duration/60:.1f} min")
        
        # Progress bar every chapter
        print_progress_bar(idx + 1, len(remaining))
    
    print()  # New line after progress bar
    
    # Cleanup GPU memory
    print("\nUnloading Whisper model from VRAM...")
    global _whisper_model
    del _whisper_model
    torch.cuda.empty_cache()
    
    # Load previous plans for registry
    for chapter_index, title, audio_path, epub_text in valid_chapters:
        if is_chapter_completed(chapter_index) and not any(p.chapter_index == chapter_index for p in all_plans):
            plan_path = OUTPUT_DIR / f"chapter_{chapter_index:04d}.json"
            if plan_path.exists():
                with open(plan_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    scenes = [Scene(**s) for s in data.get("scenes", [])]
                    all_plans.append(ChapterPlan(
                        chapter_index=data["chapter_index"],
                        chapter_title=data["chapter_title"],
                        audio_file=data["audio_file"],
                        audio_duration=data["audio_duration"],
                        scene_count=data["scene_count"],
                        scenes=scenes,
                        text_content=data.get("text_content", "")
                    ))
    
    # Build registry
    if all_plans:
        build_consistency_registry(all_plans)
    
    total_time = time.time() - start_time
    total_scenes = sum(p.scene_count for p in all_plans)
    
    print("\n" + "=" * 60)
    print("Step 0 Complete!")
    print(f"Chapters: {len(all_plans)}")
    print(f"Total scenes: {total_scenes}")
    print(f"Time: {total_time/3600:.1f} hours")
    print(f"Rate: {total_time/len(remaining):.1f}s per chapter")
    print("=" * 60)


if __name__ == "__main__":
    main()
