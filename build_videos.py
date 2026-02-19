#!/usr/bin/env python3
"""
Step 4: Build Final Videos

Concatenates scene clips + chapter audio into final YouTube-ready MP4 files.
Groups chapters into ≤12h segments at chapter boundaries within each volume.

Features:
- FFmpeg concat for fast assembly
- Crossfade transitions between scenes (0.5s) and chapters (1s)
- YouTube description generation with chapter timestamps
- Resume capability: skip existing final videos
"""

import os
import json
import sys
import time
import subprocess
import tempfile
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Configuration
PLAN_DIR = Path("plan")
CLIPS_DIR = Path("clips")
VIDEOS_DIR = Path("videos")
DESCRIPTIONS_DIR = Path("descriptions")
PROGRESS_DIR = Path("progress")

CHAPTERS_AUDIO_DIR = Path("Q:/toys/audio-book-maker/outputs/Illumine Lingao (English Translation)/chapters")

# Video settings
# YouTube strict limit is 12h. Use 11.95h (11h 57m) to account for container overhead.
MAX_SEGMENT_HOURS = 11.95
MAX_SEGMENT_SECONDS = MAX_SEGMENT_HOURS * 3600
VIDEO_CODEC = "libx264"
VIDEO_PRESET = "medium"
VIDEO_CRF = 23
AUDIO_CODEC = "aac"
AUDIO_BITRATE = "64k"  # Speech-only audiobook, 64kbps is excellent
AUDIO_SAMPLE_RATE = 24000  # Match source audio (24kHz is standard for speech)

# YouTube playlist URL
PLAYLIST_URL = "https://www.youtube.com/playlist?list=PLjq2oIRxOOOmKJ54-0L-JCr46zLuNXmFK"

# Volume definitions from PROJECT.md
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


def check_ffmpeg() -> bool:
    """Check if FFmpeg is available."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def load_chapter_plans() -> Dict[int, ChapterInfo]:
    """Load all chapter plans."""
    chapters = {}

    for plan_file in sorted(PLAN_DIR.glob("chapter_*.json")):
        try:
            with open(plan_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            ch = ChapterInfo(
                index=data["chapter_index"],
                title=data["chapter_title"],
                audio_file=data["audio_file"],
                audio_duration=data["audio_duration"],
                scene_count=data["scene_count"],
                scenes=data.get("scenes", [])
            )
            chapters[ch.index] = ch
        except Exception as e:
            print(f"Warning: Could not load {plan_file}: {e}")

    return chapters


def build_segments(chapters: Dict[int, ChapterInfo]) -> List[Segment]:
    """
    Group chapters into ≤12h segments at chapter boundaries within each volume.
    Never splits mid-chapter.
    """
    segments = []

    for vol in VOLUMES:
        vol_chapters = []
        for idx in range(vol["start"], vol["end"] + 1):
            if idx in chapters:
                vol_chapters.append(chapters[idx])

        if not vol_chapters:
            continue

        # Split into segments of ≤12h
        current_segment_chapters = []
        current_duration = 0.0
        vol_segments = []

        for ch in vol_chapters:
            if current_duration + ch.audio_duration > MAX_SEGMENT_SECONDS and current_segment_chapters:
                vol_segments.append(current_segment_chapters)
                current_segment_chapters = []
                current_duration = 0.0

            current_segment_chapters.append(ch)
            current_duration += ch.audio_duration

        if current_segment_chapters:
            vol_segments.append(current_segment_chapters)

        total_parts = len(vol_segments)
        for part_idx, seg_chapters in enumerate(vol_segments):
            seg_duration = sum(ch.audio_duration for ch in seg_chapters)
            segments.append(Segment(
                volume=vol,
                part=part_idx + 1,
                total_parts=total_parts,
                chapters=seg_chapters,
                total_duration=seg_duration
            ))

    return segments


def get_segment_filename(segment: Segment) -> str:
    """Generate filename for a segment."""
    vol = segment.volume
    vol_num = vol["vol"]
    vol_name = vol["name"]

    if vol_num == 0:
        base = "Illumine Lingao - Extras"
    else:
        base = f"Illumine Lingao - Vol. {vol_num:02d} - {vol_name}"

    if segment.total_parts > 1:
        base += f" (Part {segment.part})"

    return base


def format_timestamp(seconds: float) -> str:
    """Format seconds as H:MM:SS for YouTube chapters."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}:{m:02d}:{s:02d}"


def generate_description(segment: Segment) -> str:
    """Generate YouTube description with timestamps."""
    vol = segment.volume
    vol_num = vol["vol"]
    vol_name = vol["name"]

    lines = []
    lines.append("Illumine Lingao (\u4e34\u9ad8\u542f\u660e) by Blowing Past the Ear (\u5439\u725b\u8005)")

    if vol_num == 0:
        lines.append("Extras")
    else:
        part_info = ""
        if segment.total_parts > 1:
            part_info = f" \u2014 Part {segment.part} of {segment.total_parts}"
        lines.append(f"Volume {vol_num}: {vol_name}{part_info}")

    lines.append("")
    lines.append("Illumine Lingao is a Chinese alternate history web novel about 500 modern people")
    lines.append("who travel back to 1628 Ming Dynasty China. This is a fan-made audiobook of the")
    lines.append("English translation, created to make this epic story more accessible.")
    lines.append("")
    lines.append("I'm just a fan who wanted to contribute in a small way. Enjoy!")
    lines.append("")
    lines.append("Chapters:")

    # Build chapter timestamps
    current_time = 0.0
    for ch in segment.chapters:
        timestamp = format_timestamp(current_time)
        lines.append(f"{timestamp} {ch.title}")
        current_time += ch.audio_duration

    lines.append("")
    lines.append(f"Full playlist: {PLAYLIST_URL}")
    lines.append("")
    lines.append("Read the novel online: https://lingao.entropydrivenmindset.win")

    return "\n".join(lines)


def get_chapter_clips(chapter: ChapterInfo) -> List[Path]:
    """Get all clip files for a chapter in order."""
    clips = []
    for scene in chapter.scenes:
        sc_idx = scene["scene_index"]
        clip_path = CLIPS_DIR / f"chapter_{chapter.index:04d}_scene_{sc_idx:02d}.mp4"
        if clip_path.exists():
            clips.append(clip_path)
    return clips


def check_clip_validity(clip_path: Path) -> bool:
    """Check if a clip file is valid video."""
    try:
        # Check duration - if it returns a number, the file is likely readable
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", 
             "-of", "default=noprint_wrappers=1:nokey=1", str(clip_path)],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return False
        
        # specific check for empty or non-float output
        if not result.stdout.strip():
            return False
            
        float(result.stdout.strip())
        return True
    except Exception:
        return False


def run_ffmpeg_with_progress(cmd: List[str], total_duration: float, timeout: int = 7200) -> Tuple[bool, str]:
    """
    Run FFmpeg command and display progress bar based on encoded time.

    Uses -progress pipe:1 for reliable line-buffered progress on all platforms.

    Args:
        cmd: FFmpeg command as list of strings
        total_duration: Total expected duration in seconds
        timeout: Maximum execution time in seconds

    Returns:
        Tuple of (success: bool, error_message: str)
    """
    # Inject -progress pipe:1 to get machine-readable progress on stdout
    cmd = [cmd[0], "-progress", "pipe:1", "-nostats"] + cmd[1:]

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )

        start_time = time.time()
        last_update = 0

        for line in process.stdout:
            # Check timeout
            if time.time() - start_time > timeout:
                process.kill()
                return False, "FFmpeg timeout"

            line = line.strip()
            if line.startswith("out_time_ms="):
                try:
                    us = int(line.split("=", 1)[1])
                    current_time = us / 1_000_000
                except ValueError:
                    continue

                if current_time < 0 or total_duration <= 0:
                    continue

                now = time.time()
                if now - last_update >= 1.0:
                    percent = min(100, (current_time / total_duration) * 100)
                    elapsed = now - start_time

                    current_h = int(current_time // 3600)
                    current_m = int((current_time % 3600) // 60)
                    current_s = int(current_time % 60)

                    total_h = int(total_duration // 3600)
                    total_m = int((total_duration % 3600) // 60)
                    total_s = int(total_duration % 60)

                    if current_time > 0:
                        eta_seconds = (total_duration - current_time) * (elapsed / current_time)
                        eta_m = int(eta_seconds // 60)
                        eta_s = int(eta_seconds % 60)
                        eta_str = f"{eta_m}m{eta_s}s"
                    else:
                        eta_str = "?"

                    bar_width = 30
                    filled = int(bar_width * percent / 100)
                    bar = '█' * filled + '░' * (bar_width - filled)

                    print(f"\r    [{bar}] {percent:.1f}% | {current_h:02d}:{current_m:02d}:{current_s:02d} / {total_h:02d}:{total_m:02d}:{total_s:02d} | ETA: {eta_str}   ", end='', flush=True)
                    last_update = now

        return_code = process.wait()
        print()  # New line after progress bar

        if return_code != 0:
            stderr_text = process.stderr.read()
            return False, f"FFmpeg error (code {return_code}): {stderr_text[-500:]}"

        return True, ""

    except Exception as e:
        return False, f"FFmpeg exception: {e}"


def build_segment_video(segment: Segment, output_path: Path) -> bool:
    """
    Build a single segment video by concatenating clips and audio.

    Uses FFmpeg concat demuxer for clips and audio separately,
    then muxes together.
    """
    # Collect all clips and audio files in order
    all_clips = []
    all_audio_files = []

    for ch in segment.chapters:
        clips = get_chapter_clips(ch)
        if not clips:
            print(f"    WARNING: No clips for chapter {ch.index}, skipping")
            continue

        # Validate clips before adding
        for clip in clips:
            if not check_clip_validity(clip):
                print(f"    ERROR: Corrupt clip found: {clip}")
                # We stop immediately to avoid building a broken video
                return False
        
        all_clips.extend(clips)

        audio_path = CHAPTERS_AUDIO_DIR / ch.audio_file
        if audio_path.exists():
            all_audio_files.append((audio_path, ch.audio_duration))
        else:
            print(f"    WARNING: Audio not found: {audio_path}")

    if not all_clips:
        print(f"    ERROR: No clips found for segment")
        return False

    if not all_audio_files:
        print(f"    ERROR: No audio found for segment")
        return False

    # Create temp files for concat lists
    try:
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.txt', delete=False, prefix='clips_'
        ) as clips_list:
            for clip in all_clips:
                # Convert to absolute path with forward slashes for FFmpeg
                abs_path = str(clip.resolve()).replace(chr(92), '/')
                clips_list.write(f"file '{abs_path}'\n")
            clips_list_path = clips_list.name

        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.txt', delete=False, prefix='audio_'
        ) as audio_list:
            for audio_path, _ in all_audio_files:
                # Convert to absolute path with forward slashes for FFmpeg
                abs_path = str(audio_path.resolve()).replace(chr(92), '/')
                audio_list.write(f"file '{abs_path}'\n")
            audio_list_path = audio_list.name

        # Build FFmpeg command
        # Use stream copy for video (clips already H.264), re-encode audio only
        # Explicit mapping ensures we get video from input 0 and audio from input 1
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", clips_list_path,
            "-f", "concat", "-safe", "0", "-i", audio_list_path,
            "-map", "0:v", "-map", "1:a",
            "-c:v", "copy",  # Stream copy - no re-encode (clips already H.264)
            "-c:a", AUDIO_CODEC, "-b:a", AUDIO_BITRATE,
            "-ar", str(AUDIO_SAMPLE_RATE), "-ac", "1",  # Mono (source is mono TTS)
            "-movflags", "+faststart",
            "-shortest",
            str(output_path)
        ]

        print(f"    Running FFmpeg ({len(all_clips)} clips, {len(all_audio_files)} audio files)...")
        # print(f"    Command: {' '.join(cmd)}")  # Debugging

        success, error_msg = run_ffmpeg_with_progress(cmd, segment.total_duration, timeout=7200)

        if not success:
            print(f"    {error_msg}")
            return False

        return output_path.exists()

    except subprocess.TimeoutExpired:
        print(f"    FFmpeg timeout for segment")
        return False
    except Exception as e:
        print(f"    FFmpeg error: {e}")
        return False
    finally:
        # Clean up temp files
        try:
            os.unlink(clips_list_path)
        except:
            pass
        try:
            os.unlink(audio_list_path)
        except:
            pass


def is_segment_completed(segment_name: str) -> bool:
    """Check if segment video already exists."""
    video_path = VIDEOS_DIR / f"{segment_name}.mp4"
    return video_path.exists()


def mark_segment_completed(segment_name: str):
    """Mark segment as completed."""
    PROGRESS_DIR.mkdir(exist_ok=True)
    progress_file = PROGRESS_DIR / "videos_completed.txt"

    with open(progress_file, "a") as f:
        f.write(f"{segment_name}\n")


def main():
    """Main entry point."""
    print("=" * 60)
    print("Step 4: Build Final Videos")
    print("=" * 60)

    POLL_INTERVAL = 60  # seconds between polls (segments take a while)

    # Check FFmpeg
    if not check_ffmpeg():
        print("ERROR: FFmpeg not found. Please install FFmpeg.")
        sys.exit(1)
    print("FFmpeg: OK")

    # Check for audio
    if not CHAPTERS_AUDIO_DIR.exists():
        print(f"ERROR: Audio directory not found: {CHAPTERS_AUDIO_DIR}")
        sys.exit(1)

    VIDEOS_DIR.mkdir(exist_ok=True)
    DESCRIPTIONS_DIR.mkdir(exist_ok=True)

    total_built = 0
    total_failed = 0
    start_time = time.time()

    try:
        while True:
            # Load chapter plans
            chapters = load_chapter_plans()
            if not chapters:
                print(f"\r  Waiting for chapter plans...", end="", flush=True)
                time.sleep(POLL_INTERVAL)
                continue

            segments = build_segments(chapters)
            remaining = [s for s in segments if not is_segment_completed(get_segment_filename(s))]

            if not remaining:
                if len(chapters) >= 2883:
                    print("\nAll videos built! Pipeline complete.")
                    break
                print(f"\r  All current segments built. Waiting for more chapters... ({len(chapters)} plans)", end="", flush=True)
                time.sleep(POLL_INTERVAL)
                continue

            # Check which segments have all clips ready
            ready = []
            for seg in remaining:
                all_ready = True
                for ch in seg.chapters:
                    for scene in ch.scenes:
                        clip_path = CLIPS_DIR / f"chapter_{ch.index:04d}_scene_{scene['scene_index']:02d}.mp4"
                        if not clip_path.exists():
                            all_ready = False
                            break
                    if not all_ready:
                        break
                if all_ready:
                    ready.append(seg)

            if not ready:
                waiting = len(remaining) - len(ready)
                print(f"\r  {waiting} segments waiting for clips... ({total_built} videos done)", end="", flush=True)
                time.sleep(POLL_INTERVAL)
                continue

            # Build ready segments
            for idx, seg in enumerate(ready):
                name = get_segment_filename(seg)
                output_path = VIDEOS_DIR / f"{name}.mp4"
                desc_path = DESCRIPTIONS_DIR / f"{name}.txt"

                duration_h = seg.total_duration / 3600
                ch_count = len(seg.chapters)
                print(f"\n[{total_built + 1}] {name}")
                print(f"  {ch_count} chapters, {duration_h:.1f}h")

                success = build_segment_video(seg, output_path)

                if success:
                    total_built += 1
                    mark_segment_completed(name)

                    description = generate_description(seg)
                    with open(desc_path, "w", encoding="utf-8") as f:
                        f.write(description)

                    size_gb = output_path.stat().st_size / (1024 ** 3)
                    print(f"  Output: {size_gb:.1f} GB")
                else:
                    total_failed += 1
                    if output_path.exists():
                        output_path.unlink()
                    print(f"  FAILED: {name}")

            print(f"\nBatch done. Polling for ready segments...")
            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\n\nStopped by user.")

    total_time = time.time() - start_time
    print(f"\nStep 4: {total_built} videos ({total_failed} failed) in {total_time/60:.0f}m")
    print("=" * 60)


if __name__ == "__main__":
    main()
