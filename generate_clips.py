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

import os
import json
import sys
import time
import subprocess
import threading
import re
from pathlib import Path
from typing import Dict, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from tqdm import tqdm

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

    print(f"Verifying integrity of {len(clips)} existing clips...")
    bad_count = 0
    
    # Use tqdm for progress since this can take a while
    for clip in tqdm(clips, desc="Verifying", unit="clip"):
        if not check_clip_validity(clip):
            tqdm.write(f"Found corrupt clip: {clip.name} - Deleting")
            try:
                clip.unlink()
                bad_count += 1
            except OSError as e:
                tqdm.write(f"Error deleting {clip.name}: {e}")
    
    if bad_count > 0:
        print(f"Deleted {bad_count} corrupt clips. They will be regenerated.")
    else:
        print("All existing clips verified successfully.")


def run_ffmpeg_with_progress(cmd: List[str], duration: float, task_id: str, active_dict: dict, timeout: int = 300) -> Tuple[bool, str]:
    """
    Run FFmpeg command and track progress via shared dict for monitor thread.

    Uses -progress pipe:1 for reliable line-buffered progress on all platforms.

    Args:
        cmd: FFmpeg command as list of strings
        duration: Expected clip duration in seconds
        task_id: Task identifier for display
        active_dict: Shared dict to store progress percentage
        timeout: Maximum execution time in seconds

    Returns:
        Tuple of (success: bool, error_message: str)
    """
    # Inject -progress pipe:1 to get machine-readable progress on stdout
    # Insert after "ffmpeg" and before other args
    cmd = [cmd[0], "-progress", "pipe:1", "-nostats"] + cmd[1:]

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )

        # -progress outputs lines like: out_time_ms=5100000
        start_time = time.time()

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
                    if current_time >= 0 and duration > 0:
                        percent = min(100, (current_time / duration) * 100)
                        if active_dict is not None:
                            active_dict[task_id] = percent
                except (ValueError, ZeroDivisionError):
                    pass

        # Wait for process to complete
        return_code = process.wait()

        if return_code != 0:
            stderr_text = process.stderr.read()
            return False, f"FFmpeg error: {stderr_text[-200:]}"

        return True, ""

    except Exception as e:
        return False, f"FFmpeg exception: {e}"


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
    
    # Register as active (0 = starting, will be updated with percent by FFmpeg progress)
    task_id = f"Ch{ch:04d}:Sc{sc:02d}"
    if active_dict is not None:
        active_dict[task_id] = 0

    try:
        # Use faster encoding settings:
        # - preset veryfast: Much faster encoding, ~5-10% larger file size
        # - tune stillimage: Optimizes for static image content
        # - frame lookahead optimizations for static content
        cmd = [
            "ffmpeg", "-y",
            "-hide_banner", "-loglevel", "info",  # Need info level to get progress stats
            "-stats",  # Enable progress statistics
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

        success, error_msg = run_ffmpeg_with_progress(
            cmd, duration, task_id, active_dict, timeout=max(300, duration * 2)
        )

        if not success:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass
            return (ch, sc, False, error_msg)

        if not temp_path.exists():
            return (ch, sc, False, "Output file not created")

        # Atomic rename
        temp_path.replace(output_path)

        return (ch, sc, True, "")
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
            # Use tqdm.write if inside loop, but here print is fine as it's before the bar
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


def monitor_progress(stop_event, active_dict, pbar):
    """Background thread to update the progress bar with active tasks and their progress."""
    while not stop_event.is_set():
        try:
            # Create a local copy of keys to iterate safely
            keys = sorted(active_dict.keys())

            for task_id in keys:
                try:
                    value = active_dict[task_id]
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        if value > 1:
                            # It's a percentage from FFmpeg progress
                            pbar.set_postfix_str(f"{task_id} {value:.0f}%", refresh=True)
                        else:
                            # Starting up, no progress yet
                            pbar.set_postfix_str(f"{task_id} ...", refresh=True)
                except (KeyError, ValueError):
                    pass
        except Exception:
            pass
        time.sleep(0.5)


def main():
    """Main entry point."""
    print("=" * 60)
    print("Step 3: Generate Video Clips (Static Title Cards)")
    print("=" * 60)

    POLL_INTERVAL = 10  # Reduced poll interval for better responsiveness

    # Check FFmpeg
    if not check_ffmpeg():
        print("ERROR: FFmpeg not found. Please install FFmpeg.")
        sys.exit(1)
    print("FFmpeg: OK")

    CLIPS_DIR.mkdir(exist_ok=True)
    
    # Verify existing clips on startup to catch any corruption
    validate_existing_clips()

    total_processed = 0
    total_failed = 0
    start_time = time.time()

    # Create manager for shared state
    manager = multiprocessing.Manager()
    active_dict = manager.dict()

    try:
        while True:
            # Get all scenes and filter to those with images but no clips
            # We move this inside the loop to catch new files
            all_scenes = get_all_scenes()
            remaining = []
            
            # Simple check for 'remaining' work
            # We can't easily parallelize the check without slowing down start, 
            # but it should be fast enough for filesystem checks.
            for scene_data in all_scenes:
                ch = scene_data["chapter_index"]
                sc = scene_data["scene_index"]
                image_path = IMAGES_DIR / f"chapter_{ch:04d}_scene_{sc:02d}.png"
                if image_path.exists() and not is_clip_completed(ch, sc):
                    remaining.append(scene_data)

            if not remaining:
                # If we have processed everything we know about, check if we should exit
                # (Logic from original script: if we have > 0 processed and enough scenes found)
                # But typically this runs as a daemon/watcher.
                if total_processed > 0 and len(all_scenes) >= 2883:
                    print("\nAll clips generated! Pipeline complete.")
                    break
                
                # Visual spinner/waiting message
                sys.stdout.write(f"\rWaiting for images... (Found {len(all_scenes)} scenes, {total_processed} new clips created)   ")
                sys.stdout.flush()
                time.sleep(POLL_INTERVAL)
                continue

            # Clear the "Waiting..." line
            sys.stdout.write("\r" + " " * 80 + "\r")
            sys.stdout.flush()

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
                
                # Pass active_dict to worker
                clip_args.append((ch, sc, str(image_path), str(clip_path), duration, active_dict))
            
            print(f"Processing {len(clip_args)} clips with {MAX_WORKERS} workers...")
            
            # Start background monitoring thread
            stop_monitor = threading.Event()
            
            # Process clips in parallel with tqdm progress bar
            with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                # Submit all tasks
                futures = {executor.submit(generate_static_clip, args): args for args in clip_args}
                
                # tqdm wrapper
                with tqdm(total=len(clip_args), unit="clip", desc="Generating") as pbar:
                    # Start monitor
                    monitor_thread = threading.Thread(target=monitor_progress, args=(stop_monitor, active_dict, pbar))
                    monitor_thread.daemon = True
                    monitor_thread.start()
                    
                    try:
                        for future in as_completed(futures):
                            # Get result
                            ch, sc, success, error_msg = future.result()
                            
                            if success:
                                total_processed += 1
                                mark_scene_completed(ch, sc)
                            else:
                                total_failed += 1
                                clip_path = CLIPS_DIR / f"chapter_{ch:04d}_scene_{sc:02d}.mp4"
                                if clip_path.exists():
                                    try:
                                        clip_path.unlink()
                                    except OSError:
                                        pass
                                tqdm.write(f"FAILED: Ch {ch:04d} Sc {sc:02d} - {error_msg}")
                            
                            pbar.update(1)
                    finally:
                        # Ensure monitor stops even if error occurs
                        stop_monitor.set()
                        monitor_thread.join()

            print(f"Batch done. Polling for new images...")
            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\n\nStopped by user.")

    total_time = time.time() - start_time
    total_clips_done = total_processed + total_failed
    avg_time = total_time / total_clips_done if total_clips_done > 0 else 0
    print(f"\nStep 3: {total_processed} clips ({total_failed} failed) in {total_time/3600:.1f}h")
    if total_clips_done > 0:
        print(f"Average: {avg_time:.1f}s per clip ({total_clips_done / (total_time / 60):.1f} clips/min)")
    print("=" * 60)


if __name__ == "__main__":
    main()
