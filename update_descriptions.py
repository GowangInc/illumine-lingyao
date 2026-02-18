#!/usr/bin/env python3
"""
Update existing YouTube description files with new links.

This script cleans up and standardizes description files to include:
- The correct playlist URL
- The web reader link

Run this if you have existing description files that need updating.
"""

import re
from pathlib import Path

DESCRIPTIONS_DIR = Path("descriptions")
PLAYLIST_URL = "https://www.youtube.com/playlist?list=PLjq2oIRxOOOmKJ54-0L-JCr46zLuNXmFK"
WEBSITE_URL = "https://lingao.entropydrivenmindset.win"


def update_description_file(filepath: Path) -> bool:
    """Update a single description file with new links."""
    content = filepath.read_text(encoding="utf-8")
    
    # Check if already has correct format (both links at the end)
    if content.rstrip().endswith(WEBSITE_URL):
        print(f"OK: {filepath.name}")
        return False
    
    # Remove any existing playlist/website links (to avoid duplicates)
    lines = content.split('\n')
    cleaned_lines = []
    skip_next_empty = False
    
    for line in lines:
        # Skip lines with old links
        if 'Full playlist:' in line or 'Read the novel online:' in line:
            skip_next_empty = True
            continue
        # Skip empty line after removed link
        if skip_next_empty and line.strip() == '':
            skip_next_empty = False
            continue
        skip_next_empty = False
        cleaned_lines.append(line)
    
    # Rebuild content
    content = '\n'.join(cleaned_lines)
    content = content.rstrip()
    
    # Add fresh links
    content += f"\n\nFull playlist: {PLAYLIST_URL}\n\nRead the novel online: {WEBSITE_URL}\n"
    
    filepath.write_text(content, encoding="utf-8")
    print(f"UPDATED: {filepath.name}")
    return True


def main():
    """Update all description files."""
    print("=" * 60)
    print("Updating YouTube Descriptions")
    print("=" * 60)
    
    if not DESCRIPTIONS_DIR.exists():
        print(f"Descriptions directory not found: {DESCRIPTIONS_DIR}")
        return
    
    desc_files = list(DESCRIPTIONS_DIR.glob("*.txt"))
    
    if not desc_files:
        print("No description files found.")
        return
    
    print(f"Found {len(desc_files)} description file(s)")
    print("-" * 60)
    
    updated = 0
    for filepath in sorted(desc_files):
        if update_description_file(filepath):
            updated += 1
    
    print("-" * 60)
    print(f"Updated {updated} file(s)")
    print("=" * 60)


if __name__ == "__main__":
    main()
