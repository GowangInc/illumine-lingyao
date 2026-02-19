#!/usr/bin/env python3
"""
Step 3: Generate Video Clips

Creates video clips from chapter title card illustrations.
Each clip is a static image looped for the chapter's audio duration.

Reads from images/ and prompts/ directories, outputs to clips/ directory.

Features:
- Resume capability: skip existing clips
- Continuous polling mode: waits for new images from Step 2
- Each clip matches its scene's exact audio duration
"""

import json
import sys
import time
import subprocess
from pathlib import Path
from typing import Dict, List
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import multiprocessing

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich import box

# Configuration
PROMPTS_DIR = Path("prompts")
IMAGES_DIR = Path("images")
CLIPS_DIR = Path("clips")
PROGRESS_DIR = Path("progress")

# Video settings
OUTPUT_WIDTH = 480
OUTPUT_HEIGHT = 270
FPS = 5
VIDEO_CODEC = "libx264"
PIXEL_FORMAT = "yuv420p"
CRF = 23
PRESET = "veryfast"  # Faster encoding, slightly larger file size
TUNE = "stillimage"  # Optimize for static images

# Parallel processing
# Single worker for older processor to avoid system lag
MAX_WORKERS = 1

console = Console()


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


def validate_existing_clips():
    """Check all existing clips for corruption and delete bad ones."""
    if not CLIPS_DIR.exists():
        return

    clips = list(CLIPS_DIR.glob("*.mp4"))
    if not clips:
        return

    total = len(clips)
    bad_count = 0
    errors = []

    with Live(console=console, refresh_per_second=4) as live:
        for i, clip in enumerate(clips):
            pct = (i + 1) / total * 100
            bar_width = 40
            filled = int(bar_width * pct / 100)
            bar = "█" * filled + "░" * (bar_width - filled)

            text = Text()
            text.append("Verifying Clips\n\n", style="bold cyan")
            text.append(f"  [{bar}] {i + 1}/{total}  {pct:.0f}%\n", style="cyan")
            if errors:
                text.append(f"\n  Corrupt: {bad_count}", style="red")
            live.update(Panel(text, box=box.DOUBLE, border_style="cyan"))

            if not check_clip_validity(clip):
                errors.append(clip.name)
                try:
                    clip.unlink()
                    bad_count += 1
                except OSError:
                    pass

    if bad_count > 0:
        console.print(f"Deleted {bad_count} corrupt clips. They will be regenerated.")
    else:
        console.print("All existing clips verified successfully.")


def generate_static_clip(args: tuple) -> tuple:
    """
    Generate a static clip from an image (no zoom/motion).
    Scales image to output resolution and loops for the given duration.

    Args:
        args: Tuple of (chapter_index, scene_index, image_path_str, output_path_str, duration, active_dict)

    Returns:
        Tuple of (chapter_index, scene_index, success: bool, error_msg: str)
    """
    ch, sc, image_path_str, output_path_str, duration, active_dict = args
    image_path = Path(image_path_str)
    output_path = Path(output_path_str)

    # Use a temp file to ensure atomic write (prevents partial files on interrupt)
    temp_path = output_path.with_suffix(".tmp.mp4")

    # Register as active
    task_id = f"Ch{ch:04d}:Sc{sc:02d}"
    if active_dict is not None:
        active_dict[task_id] = time.time()

    try:
        cmd = [
            "ffmpeg", "-y",
            "-hide_banner", "-loglevel", "error",
            "-loop", "1",
            "-i", str(image_path),
            "-vf", f"scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:force_original_aspect_ratio=decrease,pad={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:(ow-iw)/2:(oh-ih)/2",
            "-c:v", VIDEO_CODEC,
            "-preset", PRESET,
            "-tune", TUNE,
            "-pix_fmt", PIXEL_FORMAT,
            "-crf", str(CRF),
            "-r", str(FPS),
            "-t", str(duration),
            str(temp_path)
        ]

        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=max(300, duration * 2)
        )

        if result.returncode != 0:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass
            error_msg = result.stderr[-200:] if result.stderr else "Unknown error"
            return (ch, sc, False, f"FFmpeg error: {error_msg}")

        if not temp_path.exists():
            return (ch, sc, False, "Output file not created")

        # Atomic rename
        temp_path.replace(output_path)

        return (ch, sc, True, "")
    except subprocess.TimeoutExpired:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        return (ch, sc, False, f"FFmpeg timeout ({duration:.0f}s clip)")
    except Exception as e:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        return (ch, sc, False, f"FFmpeg error: {e}")
    finally:
        # Remove from active list
        if active_dict is not None and task_id in active_dict:
            try:
                del active_dict[task_id]
            except Exception:
                pass


def get_all_scenes() -> List[Dict]:
    """Get all scene data from prompt files, sorted by chapter/scene."""
    if not PROMPTS_DIR.exists():
        return []

    scenes = []
    # Using glob to find all json files
    files = sorted(list(PROMPTS_DIR.glob("chapter_*_scene_*.json")))
    
    for prompt_file in files:
        try:
            with open(prompt_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            scenes.append(data)
        except Exception as e:
            print(f"Warning: Could not load {prompt_file}: {e}")

    return scenes


def is_clip_completed(chapter_index: int, scene_index: int) -> bool:
    """Check if clip already exists."""
    clip_path = CLIPS_DIR / f"chapter_{chapter_index:04d}_scene_{scene_index:02d}.mp4"
    return clip_path.exists()


def mark_scene_completed(chapter_index: int, scene_index: int):
    """Mark scene clip as completed in progress file."""
    PROGRESS_DIR.mkdir(exist_ok=True)
    progress_file = PROGRESS_DIR / "clips_completed.txt"

    with open(progress_file, "a") as f:
        f.write(f"{chapter_index}_{scene_index}\n")



def render_clips_display(total_scenes, batch_total, batch_done, total_processed,
                         total_failed, active_dict, last_completed, start_time,
                         waiting=False) -> Panel:
    """Build a Rich Panel showing clip generation progress."""
    elapsed = time.time() - start_time
    elapsed_h = int(elapsed // 3600)
    elapsed_m = int((elapsed % 3600) // 60)
    elapsed_s = int(elapsed % 60)

    text = Text()

    # Header line with elapsed time
    text.append("  Step 3: Generate Clips", style="bold cyan")
    text.append(f"                          {elapsed_h:02d}:{elapsed_m:02d}:{elapsed_s:02d}\n",
                style="dim")

    if waiting:
        text.append(f"\n  Waiting for images...  ({total_scenes} scenes found, "
                    f"{total_processed} clips created)\n", style="yellow")
    else:
        # Total progress bar
        overall_done = total_processed + total_failed
        pct = (overall_done / total_scenes * 100) if total_scenes > 0 else 0
        bar_width = 30
        filled = int(bar_width * pct / 100)
        bar = "█" * filled + "░" * (bar_width - filled)
        text.append(f"\n  Total:   [{bar}] {overall_done}/{total_scenes}  {pct:.1f}%\n",
                    style="white")

        # Current clip (from active_dict — stores start time)
        try:
            keys = sorted(active_dict.keys())
        except Exception:
            keys = []

        if keys:
            task_id = keys[0]
            try:
                start = active_dict[task_id]
                clip_elapsed = int(time.time() - start)
                text.append(f"  Current: {task_id}  ({clip_elapsed}s)\n", style="cyan")
            except (KeyError, ValueError, TypeError):
                text.append(f"  Current: {task_id}\n", style="cyan")
        else:
            text.append(f"  Current: —\n", style="dim")

        # Stats line
        rate = total_processed / (elapsed / 60) if elapsed > 0 and total_processed > 0 else 0
        text.append(f"\n  Completed: {total_processed}    Failed: {total_failed}", style="white")
        if rate > 0:
            text.append(f"    Rate: {rate:.1f} clips/min", style="white")
        text.append("\n")

        # Last completed
        if last_completed:
            text.append(f"  Last: {last_completed}\n", style="dim")

    border = "yellow" if waiting else "cyan"
    return Panel(text, box=box.DOUBLE, border_style=border)


def main():
    """Main entry point."""
    # Check FFmpeg
    if not check_ffmpeg():
        console.print("[red bold]ERROR: FFmpeg not found. Please install FFmpeg.[/red bold]")
        sys.exit(1)

    CLIPS_DIR.mkdir(exist_ok=True)

    # Verify existing clips on startup to catch any corruption
    validate_existing_clips()

    POLL_INTERVAL = 10
    total_processed = 0
    total_failed = 0
    start_time = time.time()
    last_completed = ""
    errors = []

    # Create manager for shared state
    manager = multiprocessing.Manager()
    active_dict = manager.dict()

    with Live(console=console, refresh_per_second=4) as live:
        try:
            while True:
                all_scenes = get_all_scenes()
                total_scenes = len(all_scenes)
                remaining = []

                for scene_data in all_scenes:
                    ch = scene_data["chapter_index"]
                    sc = scene_data["scene_index"]
                    image_path = IMAGES_DIR / f"chapter_{ch:04d}_scene_{sc:02d}.png"
                    if image_path.exists() and not is_clip_completed(ch, sc):
                        remaining.append(scene_data)

                if not remaining:
                    if total_processed > 0 and total_scenes >= 2883:
                        live.update(render_clips_display(
                            total_scenes, 0, 0, total_processed, total_failed,
                            active_dict, last_completed, start_time))
                        break

                    # Show waiting state
                    live.update(render_clips_display(
                        total_scenes, 0, 0, total_processed, total_failed,
                        active_dict, last_completed, start_time, waiting=True))
                    time.sleep(POLL_INTERVAL)
                    continue

                # Prepare arguments for parallel processing
                clip_args = []
                for scene_data in remaining:
                    ch = scene_data["chapter_index"]
                    sc = scene_data["scene_index"]
                    duration = scene_data.get("duration_sec", 60.0)
                    if duration < 1.0:
                        duration = 1.0

                    image_path = IMAGES_DIR / f"chapter_{ch:04d}_scene_{sc:02d}.png"
                    clip_path = CLIPS_DIR / f"chapter_{ch:04d}_scene_{sc:02d}.mp4"
                    clip_args.append((ch, sc, str(image_path), str(clip_path), duration, active_dict))

                batch_total = len(clip_args)
                batch_done = 0

                executor = ProcessPoolExecutor(max_workers=MAX_WORKERS)
                try:
                    futures = {executor.submit(generate_static_clip, args): args for args in clip_args}
                    pending = set(futures.keys())

                    while pending:
                        done, pending = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)

                        for future in done:
                            ch, sc, success, error_msg = future.result()

                            if success:
                                total_processed += 1
                                last_completed = f"chapter_{ch:04d}_scene_{sc:02d}.mp4"
                                mark_scene_completed(ch, sc)
                            else:
                                total_failed += 1
                                clip_path = CLIPS_DIR / f"chapter_{ch:04d}_scene_{sc:02d}.mp4"
                                if clip_path.exists():
                                    try:
                                        clip_path.unlink()
                                    except OSError:
                                        pass
                                errors.append(f"Ch {ch:04d} Sc {sc:02d}: {error_msg}")

                            batch_done += 1

                        live.update(render_clips_display(
                            total_scenes, batch_total, batch_done, total_processed,
                            total_failed, active_dict, last_completed, start_time))
                finally:
                    executor.shutdown(wait=False, cancel_futures=True)

                time.sleep(POLL_INTERVAL)

        except KeyboardInterrupt:
            pass

    # Final summary
    total_time = time.time() - start_time
    total_clips_done = total_processed + total_failed
    avg_time = total_time / total_clips_done if total_clips_done > 0 else 0
    console.print(f"\nStep 3: {total_processed} clips ({total_failed} failed) in {total_time/3600:.1f}h")
    if total_clips_done > 0:
        console.print(f"Average: {avg_time:.1f}s per clip ({total_clips_done / (total_time / 60):.1f} clips/min)")
    if errors:
        console.print(f"\n[red]Errors ({len(errors)}):[/red]")
        for err in errors[-10:]:
            console.print(f"  [red]{err}[/red]")
        if len(errors) > 10:
            console.print(f"  ... and {len(errors) - 10} more")


if __name__ == "__main__":
    main()
