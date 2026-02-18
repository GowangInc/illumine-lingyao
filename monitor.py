#!/usr/bin/env python3
"""
TUI Progress Monitor for Illumine Lingao YouTube Project

A rich-based terminal UI showing real-time progress across all pipeline steps.
Run anytime to see current status with visual progress bars, GPU stats, and throughput metrics.
"""

import json
import time
import subprocess
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.table import Table
from rich.text import Text
from rich.layout import Layout
from rich import box

# Configuration
console = Console()

PROGRESS_DIR = Path("progress")
PLAN_DIR = Path("plan")
PROMPTS_DIR = Path("prompts")
IMAGES_DIR = Path("images")
CLIPS_DIR = Path("clips")
VIDEOS_DIR = Path("videos")
UPLOAD_MAPPING_FILE = Path("progress/youtube_uploads.json")

TOTAL_CHAPTERS = 2883
ESTIMATED_SCENES = 14400
ESTIMATED_VIDEOS = 58


def get_gpu_info() -> Dict:
    """Get GPU utilization info using nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            if len(parts) >= 4:
                return {
                    "util": int(parts[0]),
                    "mem_used": int(parts[1]),
                    "mem_total": int(parts[2]),
                    "temp": int(parts[3]),
                    "loaded": True
                }
    except:
        pass
    return {"util": 0, "mem_used": 0, "mem_total": 24576, "temp": 0, "loaded": False}


def count_json_files(directory: Path, pattern: str) -> int:
    if not directory.exists():
        return 0
    return len(list(directory.glob(pattern)))

def count_lines(filepath: Path) -> int:
    if not filepath.exists():
        return 0
    try:
        return len(filepath.read_text().splitlines())
    except:
        return 0

def get_latest_file(directory: Path, pattern: str) -> str:
    if not directory.exists():
        return "N/A"
    files = list(directory.glob(pattern))
    if not files:
        return "N/A"
    latest = max(files, key=lambda p: p.stat().st_mtime)
    return latest.name

def get_file_age(filepath: Path) -> str:
    if not filepath.exists():
        return "N/A"
    try:
        age = time.time() - filepath.stat().st_mtime
        if age < 60:
            return f"{int(age)}s"
        elif age < 3600:
            return f"{int(age/60)}m"
        elif age < 86400:
            return f"{int(age/3600)}h"
        else:
            return f"{int(age/86400)}d"
    except:
        return "N/A"

def estimate_rate_and_eta(directory: Path, pattern: str, completed: int, total: int):
    """Estimate processing rate and ETA from the last N files' modification times.

    Looks at the most recent N files (by mtime) in the directory matching the pattern,
    calculates a rolling throughput rate from the time span they cover, and projects
    an ETA for the remaining items.

    Returns (rate_per_min, eta_minutes) or (None, None) if not enough data.
    """
    SAMPLE_SIZE = 10

    if not directory.exists() or completed < 2:
        return None, None

    files = list(directory.glob(pattern))
    if len(files) < 2:
        return None, None

    # Sort by modification time descending, take last N
    files_by_mtime = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)
    sample = files_by_mtime[:SAMPLE_SIZE]

    if len(sample) < 2:
        return None, None

    newest_mtime = sample[0].stat().st_mtime
    oldest_mtime = sample[-1].stat().st_mtime
    time_span_sec = newest_mtime - oldest_mtime

    if time_span_sec <= 0:
        return None, None

    # Number of intervals between the sampled files
    files_in_span = len(sample) - 1
    rate_per_sec = files_in_span / time_span_sec
    rate_per_min = rate_per_sec * 60

    remaining = max(total - completed, 0)
    if remaining == 0 or rate_per_sec == 0:
        return rate_per_min, 0.0

    eta_minutes = remaining / rate_per_min
    return rate_per_min, eta_minutes


def format_eta(eta_minutes) -> str:
    """Format an ETA in minutes into a human-readable string."""
    if eta_minutes is None:
        return ""
    if eta_minutes <= 0:
        return "done"
    if eta_minutes < 1:
        return f"{eta_minutes * 60:.0f}s"
    if eta_minutes < 60:
        return f"{eta_minutes:.0f}m"
    hours = int(eta_minutes // 60)
    mins = int(eta_minutes % 60)
    if hours < 24:
        return f"{hours}h {mins:02d}m"
    days = hours // 24
    rem_hours = hours % 24
    return f"{days}d {rem_hours}h"


def format_rate_and_eta(rate_per_min, eta_minutes, unit: str = "files") -> str:
    """Format rate and ETA into a display string."""
    if rate_per_min is None:
        return ""
    rate_str = f"{rate_per_min:.1f}" if rate_per_min < 100 else f"{rate_per_min:.0f}"
    eta_str = format_eta(eta_minutes)
    result = f"Rate: {rate_str} {unit}/min"
    if eta_str:
        result += f", ETA: {eta_str}"
    return result


def estimate_scenes() -> int:
    if not PLAN_DIR.exists():
        return ESTIMATED_SCENES

    total = 0
    for f in PLAN_DIR.glob("chapter_*.json"):
        try:
            data = json.load(open(f))
            total += data.get("scene_count", 1)
        except:
            pass

    return total if total > 0 else ESTIMATED_SCENES


def count_uploaded_videos() -> int:
    """Count uploaded videos from YouTube mapping file."""
    if not UPLOAD_MAPPING_FILE.exists():
        return 0
    try:
        with open(UPLOAD_MAPPING_FILE, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        return len(mapping)
    except:
        return 0


def create_header() -> Panel:
    """Create header panel with GPU info."""
    gpu = get_gpu_info()
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    text = Text()
    text.append("Illumine Lingao YouTube Pipeline\n", style="bold cyan")
    text.append(f"{current_time}  ", style="dim")
    
    if gpu["loaded"]:
        mem_pct = gpu["mem_used"] / gpu["mem_total"] * 100
        text.append(f"GPU: {gpu['util']:3d}% util  ", style="green" if gpu["util"] > 80 else "yellow")
        text.append(f"VRAM: {gpu['mem_used']}/{gpu['mem_total']} MB ({mem_pct:.0f}%)  ", 
                   style="green" if mem_pct > 70 else "yellow")
        text.append(f"Temp: {gpu['temp']}°C", style="green" if gpu["temp"] < 80 else "red")
    else:
        text.append("GPU: N/A (nvidia-smi not available)", style="dim")
    
    return Panel(text, box=box.DOUBLE)


def create_step_panel(name: str, icon: str, completed: int, total: int, unit: str,
                     latest: str, extra_info: Dict, rate_eta_str: str = "") -> Panel:
    """Create a panel for a pipeline step."""
    
    pct = (completed / total * 100) if total > 0 else 0
    
    # Status
    if pct >= 100:
        status_color = "green"
        status_text = "COMPLETE"
    elif pct > 0:
        status_color = "yellow"
        status_text = "RUNNING"
    else:
        status_color = "dim"
        status_text = "PENDING"
    
    # Content
    text = Text()
    text.append(f"{icon} {name}\n", style=f"bold {status_color}")
    text.append(f"{status_text}\n", style=status_color)
    
    # Progress bar
    bar_width = 25
    filled = int(bar_width * pct / 100)
    bar = "█" * filled + "░" * (bar_width - filled)
    text.append(f"[{bar}] {pct:.1f}%\n", style=status_color)
    text.append(f"{completed:,} / {total:,} {unit}\n", style="dim")

    # Rate and ETA
    if rate_eta_str:
        text.append(f"{rate_eta_str}\n", style="cyan")

    # Extra info
    for key, value in extra_info.items():
        if isinstance(value, int) and value > 1000:
            text.append(f"{key}: {value:,}\n", style="dim")
        elif isinstance(value, str):
            text.append(f"{key}: {value}\n", style="dim")
    
    # Latest file (truncated)
    if latest != "N/A":
        latest_short = latest[:25] + "..." if len(latest) > 25 else latest
        text.append(f"Latest: {latest_short} ({get_file_age(Path())} ago)", style="dim")
    
    return Panel(text, border_style=status_color, box=box.ROUNDED)


def create_steps_grid() -> Table:
    """Create grid layout for all steps."""
    
    # Gather data
    plan_completed = count_lines(PROGRESS_DIR / "plan_completed.txt")
    plan_files = count_json_files(PLAN_DIR, "chapter_*.json")
    
    prompt_completed = count_lines(PROGRESS_DIR / "prompts_completed.txt")
    prompt_files = count_json_files(PROMPTS_DIR, "chapter_*_scene_*.json")
    
    image_completed = count_lines(PROGRESS_DIR / "images_completed.txt")
    image_files = count_json_files(IMAGES_DIR, "chapter_*_scene_*.png")
    
    clip_completed = count_lines(PROGRESS_DIR / "clips_completed.txt")
    clip_files = count_json_files(CLIPS_DIR, "chapter_*_scene_*.mp4")
    
    video_completed = count_lines(PROGRESS_DIR / "videos_completed.txt")
    video_files = count_json_files(VIDEOS_DIR, "*.mp4")
    
    total_scenes = estimate_scenes()
    
    # Create table
    grid = Table(show_header=False, box=None, padding=(0, 1))
    grid.add_column("Step 0-2", width=35)
    grid.add_column("Step 3-4", width=35)
    
    # Step 0 Panel
    step0_extra = {"plan_files": plan_files}
    if plan_files > 0:
        total_plan_scenes = 0
        for f in PLAN_DIR.glob("chapter_*.json"):
            try:
                data = json.load(open(f))
                total_plan_scenes += data.get("scene_count", 0)
            except:
                pass
        step0_extra["total_scenes"] = total_plan_scenes
    
    rate0, eta0 = estimate_rate_and_eta(PLAN_DIR, "chapter_*.json", plan_completed, TOTAL_CHAPTERS)
    step0 = create_step_panel(
        "Step 0: Plan & Map", "🗺️",
        plan_completed, TOTAL_CHAPTERS, "chapters",
        get_latest_file(PLAN_DIR, "chapter_*.json"),
        step0_extra,
        rate_eta_str=format_rate_and_eta(rate0, eta0, "ch")
    )
    
    # Step 1 Panel
    step1_extra = {
        "prompt_files": prompt_files,
        "scenes": total_scenes,
    }
    
    rate1, eta1 = estimate_rate_and_eta(PROMPTS_DIR, "chapter_*_scene_*.json", prompt_completed, TOTAL_CHAPTERS)
    step1 = create_step_panel(
        "Step 1: Prompts", "🎨",
        prompt_completed, TOTAL_CHAPTERS, "chapters",
        get_latest_file(PROMPTS_DIR, "chapter_*_scene_*.json"),
        step1_extra,
        rate_eta_str=format_rate_and_eta(rate1, eta1, "ch")
    )
    
    # Step 2 Panel
    image_size_mb = 0
    if IMAGES_DIR.exists():
        for f in IMAGES_DIR.glob("chapter_*_scene_*.png"):
            try:
                image_size_mb += f.stat().st_size / (1024 * 1024)
            except:
                pass
    
    step2_extra = {
        "images": image_files,
        "size_gb": f"{image_size_mb/1024:.1f} GB" if image_size_mb > 1024 else f"{image_size_mb:.0f} MB"
    }
    
    rate2, eta2 = estimate_rate_and_eta(IMAGES_DIR, "chapter_*_scene_*.png", image_files, total_scenes)
    step2 = create_step_panel(
        "Step 2: Images", "🖼️",
        image_files, total_scenes, "images",
        get_latest_file(IMAGES_DIR, "chapter_*_scene_*.png"),
        step2_extra,
        rate_eta_str=format_rate_and_eta(rate2, eta2, "img")
    )
    
    # Step 3 Panel
    clip_size_mb = 0
    if CLIPS_DIR.exists():
        for f in CLIPS_DIR.glob("chapter_*_scene_*.mp4"):
            try:
                clip_size_mb += f.stat().st_size / (1024 * 1024)
            except:
                pass
    
    step3_extra = {
        "clips": clip_files,
        "size_gb": f"{clip_size_mb/1024:.1f} GB" if clip_size_mb > 1024 else f"{clip_size_mb:.0f} MB"
    }
    
    rate3, eta3 = estimate_rate_and_eta(CLIPS_DIR, "chapter_*_scene_*.mp4", clip_files, total_scenes)
    step3 = create_step_panel(
        "Step 3: Video Clips", "🎬",
        clip_files, total_scenes, "clips",
        get_latest_file(CLIPS_DIR, "chapter_*_scene_*.mp4"),
        step3_extra,
        rate_eta_str=format_rate_and_eta(rate3, eta3, "clip")
    )
    
    # Step 4 Panel
    video_size_gb = 0
    if VIDEOS_DIR.exists():
        for f in VIDEOS_DIR.glob("*.mp4"):
            try:
                video_size_gb += f.stat().st_size / (1024 * 1024 * 1024)
            except:
                pass
    
    step4_extra = {
        "size_gb": f"{video_size_gb:.1f} GB"
    }
    
    rate4, eta4 = estimate_rate_and_eta(VIDEOS_DIR, "*.mp4", video_files, ESTIMATED_VIDEOS)
    step4 = create_step_panel(
        "Step 4: Final Videos", "📹",
        video_files, ESTIMATED_VIDEOS, "videos",
        get_latest_file(VIDEOS_DIR, "*.mp4"),
        step4_extra,
        rate_eta_str=format_rate_and_eta(rate4, eta4, "vid")
    )

    # Step 5 Panel
    uploaded_count = count_uploaded_videos()

    step5_extra = {}
    if uploaded_count > 0:
        # Calculate estimated quota usage
        quota_used_today = uploaded_count % 6  # Assuming ~6 uploads/day max
        step5_extra["quota_today"] = f"{quota_used_today}/6 uploads"

    rate5, eta5 = estimate_rate_and_eta(VIDEOS_DIR, "*.mp4", uploaded_count, ESTIMATED_VIDEOS)
    step5 = create_step_panel(
        "Step 5: YouTube Upload", "📺",
        uploaded_count, ESTIMATED_VIDEOS, "videos",
        "N/A",
        step5_extra,
        rate_eta_str=format_rate_and_eta(rate5, eta5, "vid") if uploaded_count > 0 else ""
    )

    grid.add_row(step0, step3)
    grid.add_row(step1, step4)
    grid.add_row(step2, step5)

    return grid


def create_footer() -> Panel:
    """Create footer with commands."""
    text = Text()
    text.append("Commands: ", style="bold")
    text.append("python pipeline_integrated.py", style="cyan")
    text.append(" (combined) | ")
    text.append("python plan_chapters.py", style="green")
    text.append(" (0) | ")
    text.append("python generate_prompts.py", style="green")
    text.append(" (1) | ")
    text.append("python generate_images.py", style="green")
    text.append(" (2) | ")
    text.append("Ctrl+C", style="red")
    text.append(" to exit")
    return Panel(text, box=box.SIMPLE)


def main():
    """Main TUI loop."""
    console.print("[bold green]Illumine Lingao Pipeline Monitor[/bold green]")
    console.print("[dim]Press Ctrl+C to exit. Run this anytime to check progress.[/dim]\n")
    
    try:
        with Live(refresh_per_second=2, console=console) as live:
            while True:
                layout = Group(
                    create_header(),
                    create_steps_grid(),
                    create_footer()
                )
                live.update(layout)
                time.sleep(0.5)
    except KeyboardInterrupt:
        console.print("\n[dim]Monitor stopped.[/dim]")


if __name__ == "__main__":
    main()
