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
import threading
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich import box

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

console = Console()

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


def run_ffmpeg_with_progress(cmd: List[str], total_duration: float, state: dict, timeout: int = 7200) -> Tuple[bool, str]:
    """
    Run FFmpeg command and update state dict with progress for Rich display.

    Uses -progress pipe:1 for reliable line-buffered progress on all platforms.

    Args:
        cmd: FFmpeg command as list of strings
        total_duration: Total expected duration in seconds
        state: Shared dict to update with progress (percent, current_time, eta_str)
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

        # Drain stderr in a background thread to prevent pipe buffer deadlock.
        # Without this, ffmpeg blocks once the OS pipe buffer (~4-8KB) fills with
        # warnings (e.g. timestamp discontinuities at clip boundaries), which stalls
        # stdout progress output and causes the build to grind to a halt.
        stderr_lines = []

        def drain_stderr():
            for line in process.stderr:
                stderr_lines.append(line)

        stderr_thread = threading.Thread(target=drain_stderr, daemon=True)
        stderr_thread.start()

        ffmpeg_start = time.time()

        while True:
            if time.time() - ffmpeg_start > timeout:
                process.kill()
                return False, "FFmpeg timeout"

            line = process.stdout.readline()
            if not line:
                break

            line = line.strip()
            if line.startswith("out_time_ms="):
                try:
                    us = int(line.split("=", 1)[1])
                    current_time = us / 1_000_000
                except ValueError:
                    continue

                if current_time < 0 or total_duration <= 0:
                    continue

                elapsed = time.time() - ffmpeg_start
                percent = min(100, (current_time / total_duration) * 100)

                if current_time > 0:
                    eta_seconds = (total_duration - current_time) * (elapsed / current_time)
                    eta_m = int(eta_seconds // 60)
                    eta_s = int(eta_seconds % 60)
                    eta_str = f"{eta_m}m{eta_s:02d}s"
                else:
                    eta_str = "—"

                state["percent"] = percent
                state["current_time"] = current_time
                state["total_duration"] = total_duration
                state["eta"] = eta_str

        return_code = process.wait()
        stderr_thread.join(timeout=5)

        if return_code != 0:
            stderr_text = "".join(stderr_lines)
            return False, f"FFmpeg error (code {return_code}): {stderr_text[-500:]}"

        return True, ""

    except Exception as e:
        return False, f"FFmpeg exception: {e}"



def build_segment_video(segment: Segment, output_path: Path, state: dict) -> Tuple[bool, str]:
    """
    Build a single segment video by concatenating clips and audio.

    Uses FFmpeg concat demuxer for clips and audio separately,
    then muxes together. Updates state dict for Rich display.

    Returns:
        Tuple of (success, error_message)
    """
    # Collect all clips and audio files in order
    all_clips = []
    all_audio_files = []

    for ch in segment.chapters:
        clips = get_chapter_clips(ch)
        if not clips:
            continue

        # Validate clips before adding
        for clip in clips:
            if not check_clip_validity(clip):
                return False, f"Corrupt clip: {clip}"

        all_clips.extend(clips)

        audio_path = CHAPTERS_AUDIO_DIR / ch.audio_file
        if audio_path.exists():
            all_audio_files.append((audio_path, ch.audio_duration))

    if not all_clips:
        return False, "No clips found for segment"

    if not all_audio_files:
        return False, "No audio found for segment"

    state["clip_count"] = len(all_clips)
    state["audio_count"] = len(all_audio_files)

    # Create temp files for concat lists
    clips_list_path = None
    audio_list_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.txt', delete=False, prefix='clips_',
            encoding='utf-8'
        ) as clips_list:
            for clip in all_clips:
                abs_path = str(clip.resolve()).replace(chr(92), '/')
                clips_list.write(f"file '{abs_path}'\n")
            clips_list_path = clips_list.name

        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.txt', delete=False, prefix='audio_',
            encoding='utf-8'
        ) as audio_list:
            for audio_path, _ in all_audio_files:
                abs_path = str(audio_path.resolve()).replace(chr(92), '/')
                audio_list.write(f"file '{abs_path}'\n")
            audio_list_path = audio_list.name

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", clips_list_path,
            "-f", "concat", "-safe", "0", "-i", audio_list_path,
            "-map", "0:v", "-map", "1:a",
            "-c:v", "copy",
            "-c:a", AUDIO_CODEC, "-b:a", AUDIO_BITRATE,
            "-ar", str(AUDIO_SAMPLE_RATE), "-ac", "1",
            "-movflags", "+faststart",
            "-shortest",
            str(output_path)
        ]

        success, error_msg = run_ffmpeg_with_progress(cmd, segment.total_duration, state, timeout=7200)

        if not success:
            return False, error_msg

        if output_path.exists():
            return True, ""
        return False, "Output file not created"

    except subprocess.TimeoutExpired:
        return False, "FFmpeg timeout"
    except Exception as e:
        return False, f"FFmpeg error: {e}"
    finally:
        for path in (clips_list_path, audio_list_path):
            if path:
                try:
                    os.unlink(path)
                except OSError:
                    pass


def get_video_duration(video_path: Path) -> float:
    """Get duration of a video file in seconds via ffprobe. Returns -1 on error."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0 or not result.stdout.strip():
            return -1
        return float(result.stdout.strip())
    except Exception:
        return -1


def validate_existing_videos():
    """Check all existing videos for corruption or truncation and delete bad ones.

    Checks both file validity (ffprobe can read it) and duration (at least 90%
    of expected segment duration). Truncated videos from killed builds are
    detected and removed so they get rebuilt.
    """
    if not VIDEOS_DIR.exists():
        return

    videos = list(VIDEOS_DIR.glob("*.mp4"))
    if not videos:
        return

    # Build expected duration map from segment data
    expected_durations = {}
    chapters = load_chapter_plans()
    if chapters:
        segments = build_segments(chapters)
        for seg in segments:
            name = get_segment_filename(seg)
            expected_durations[name] = seg.total_duration

    total = len(videos)
    bad_count = 0
    errors = []

    with Live(console=console, refresh_per_second=4) as live:
        for i, video in enumerate(videos):
            pct = (i + 1) / total * 100
            bar_width = 40
            filled = int(bar_width * pct / 100)
            bar = "\u2588" * filled + "\u2591" * (bar_width - filled)

            text = Text()
            text.append("Verifying Videos\n\n", style="bold cyan")
            text.append(f"  [{bar}] {i + 1}/{total}  {pct:.0f}%\n", style="cyan")
            if errors:
                text.append(f"\n  Bad: {bad_count}", style="red")
                text.append(f"  {errors[-1]}\n", style="dim red")
            live.update(Panel(text, box=box.DOUBLE, border_style="cyan"))

            reason = None
            actual_dur = get_video_duration(video)

            if actual_dur < 0:
                reason = "corrupt (ffprobe failed)"
            else:
                # Check against expected duration
                name = video.stem
                expected = expected_durations.get(name)
                if expected and actual_dur < expected * 0.9:
                    actual_h = actual_dur / 3600
                    expected_h = expected / 3600
                    reason = f"truncated ({actual_h:.1f}h / {expected_h:.1f}h expected)"

            if reason:
                errors.append(f"{video.name}: {reason}")
                try:
                    video.unlink()
                    bad_count += 1
                except OSError:
                    pass

    if bad_count > 0:
        console.print(f"Deleted {bad_count} corrupt/truncated videos. They will be rebuilt.")
        for err in errors:
            console.print(f"  {err}")
    else:
        console.print("All existing videos verified successfully.")


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


MAX_DISPLAY_ERRORS = 5  # Show last N errors in the TUI


def render_videos_display(total_segments, total_built, total_failed, waiting_count,
                          current_name, state, last_completed, start_time,
                          waiting_msg="", errors=None) -> Panel:
    """Build a Rich Panel showing video build progress."""
    elapsed = time.time() - start_time
    elapsed_h = int(elapsed // 3600)
    elapsed_m = int((elapsed % 3600) // 60)
    elapsed_s = int(elapsed % 60)

    text = Text()

    # Header
    text.append("  Step 4: Build Videos", style="bold cyan")
    text.append(f"                            {elapsed_h:02d}:{elapsed_m:02d}:{elapsed_s:02d}\n",
                style="dim")

    if waiting_msg:
        text.append(f"\n  {waiting_msg}\n", style="yellow")
    else:
        # Videos progress bar
        pct = (total_built / total_segments * 100) if total_segments > 0 else 0
        bar_width = 30
        filled = int(bar_width * pct / 100)
        bar = "\u2588" * filled + "\u2591" * (bar_width - filled)
        text.append(f"\n  Videos: [{bar}] {total_built}/{total_segments}  built\n",
                    style="white")

        # Current video
        if current_name:
            text.append(f"  Current: {current_name}\n", style="cyan")

            ffmpeg_pct = state.get("percent", 0)
            cur_bar_width = 30
            cur_filled = int(cur_bar_width * ffmpeg_pct / 100)
            cur_bar = "\u2588" * cur_filled + "\u2591" * (cur_bar_width - cur_filled)
            text.append(f"           [{cur_bar}] {ffmpeg_pct:.1f}%\n", style="cyan")

            cur_time = state.get("current_time", 0)
            tot_dur = state.get("total_duration", 0)
            eta = state.get("eta", "—")

            def fmt_hms(s):
                h = int(s // 3600)
                m = int((s % 3600) // 60)
                sec = int(s % 60)
                return f"{h:02d}:{m:02d}:{sec:02d}"

            text.append(f"           {fmt_hms(cur_time)} / {fmt_hms(tot_dur)}  ETA: {eta}\n",
                        style="dim")

        # Stats
        text.append(f"\n  Completed: {total_built}    Failed: {total_failed}    Waiting: {waiting_count}\n",
                    style="white")

        # Last completed
        if last_completed:
            text.append(f"  Last: {last_completed}\n", style="dim")

    # Error log
    if errors:
        text.append(f"\n  Errors ({len(errors)}):\n", style="red bold")
        shown = errors[-MAX_DISPLAY_ERRORS:]
        if len(errors) > MAX_DISPLAY_ERRORS:
            text.append(f"  ... {len(errors) - MAX_DISPLAY_ERRORS} earlier errors hidden\n", style="dim red")
        for err in shown:
            text.append(f"  {err}\n", style="red")

    border = "red" if errors else ("yellow" if waiting_msg else "cyan")
    return Panel(text, box=box.DOUBLE, border_style=border)


def main():
    """Main entry point."""
    # Check FFmpeg
    if not check_ffmpeg():
        console.print("[red bold]ERROR: FFmpeg not found. Please install FFmpeg.[/red bold]")
        sys.exit(1)

    # Check for audio
    if not CHAPTERS_AUDIO_DIR.exists():
        console.print(f"[red bold]ERROR: Audio directory not found: {CHAPTERS_AUDIO_DIR}[/red bold]")
        sys.exit(1)

    VIDEOS_DIR.mkdir(exist_ok=True)
    DESCRIPTIONS_DIR.mkdir(exist_ok=True)

    # Verify existing videos on startup to catch any corruption from killed runs
    validate_existing_videos()

    POLL_INTERVAL = 60
    total_built = 0
    total_failed = 0
    start_time = time.time()
    last_completed = ""
    errors = []
    failed_segments = set()  # Don't retry segments that already failed

    # Shared state for FFmpeg progress (read by display, written by ffmpeg thread)
    ffmpeg_state = {}

    with Live(console=console, refresh_per_second=4) as live:

        def update_display(current_name="", waiting_msg="", total_segments=0, waiting_count=0):
            live.update(render_videos_display(
                total_segments, total_built, total_failed, waiting_count,
                current_name, ffmpeg_state, last_completed, start_time,
                waiting_msg, errors))

        try:
            while True:
                chapters = load_chapter_plans()
                if not chapters:
                    update_display(waiting_msg="Waiting for chapter plans...")
                    time.sleep(POLL_INTERVAL)
                    continue

                segments = build_segments(chapters)
                total_segments = len(segments)
                remaining = [s for s in segments
                            if not is_segment_completed(get_segment_filename(s))
                            and get_segment_filename(s) not in failed_segments]

                if not remaining:
                    if len(chapters) >= 2883:
                        update_display(total_segments=total_segments,
                                       waiting_msg="All videos built! Pipeline complete.")
                        break
                    update_display(total_segments=total_segments,
                                   waiting_msg=f"All current segments built. Waiting for more chapters... ({len(chapters)} plans)")
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
                    update_display(total_segments=total_segments,
                                   waiting_count=len(remaining),
                                   waiting_msg=f"{len(remaining)} segments waiting for clips...")
                    time.sleep(POLL_INTERVAL)
                    continue

                waiting_count = len(remaining) - len(ready)

                # Build ready segments
                for seg in ready:
                    name = get_segment_filename(seg)
                    output_path = VIDEOS_DIR / f"{name}.mp4"
                    desc_path = DESCRIPTIONS_DIR / f"{name}.txt"

                    # Reset ffmpeg state for this segment
                    ffmpeg_state.clear()
                    ffmpeg_state["percent"] = 0
                    ffmpeg_state["current_time"] = 0
                    ffmpeg_state["total_duration"] = seg.total_duration
                    ffmpeg_state["eta"] = "—"

                    # Start a background thread to refresh the display while FFmpeg runs
                    stop_refresh = threading.Event()

                    def refresh_loop():
                        while not stop_refresh.is_set():
                            update_display(current_name=name,
                                           total_segments=total_segments,
                                           waiting_count=waiting_count)
                            stop_refresh.wait(0.25)

                    refresh_thread = threading.Thread(target=refresh_loop, daemon=True)
                    refresh_thread.start()

                    try:
                        success, error_msg = build_segment_video(seg, output_path, ffmpeg_state)
                    finally:
                        stop_refresh.set()
                        refresh_thread.join()

                    if success:
                        total_built += 1
                        mark_segment_completed(name)

                        description = generate_description(seg)
                        with open(desc_path, "w", encoding="utf-8") as f:
                            f.write(description)

                        size_gb = output_path.stat().st_size / (1024 ** 3)
                        last_completed = f"{name}  {size_gb:.1f} GB"
                    else:
                        total_failed += 1
                        failed_segments.add(name)
                        if output_path.exists():
                            output_path.unlink()
                        errors.append(f"{name}: {error_msg}")

                    waiting_count = max(0, waiting_count - 1)
                    update_display(total_segments=total_segments, waiting_count=waiting_count)

                time.sleep(POLL_INTERVAL)

        except KeyboardInterrupt:
            pass

    # Final summary
    total_time = time.time() - start_time
    console.print(f"\nStep 4: {total_built} videos ({total_failed} failed) in {total_time/60:.0f}m")
    if errors:
        console.print(f"\n[red]Errors ({len(errors)}):[/red]")
        for err in errors:
            console.print(f"  [red]{err}[/red]")


if __name__ == "__main__":
    main()
