#!/usr/bin/env python3
"""
Deploy script for Illumine Lingao website.

Injects AdSense client ID and deploys to Cloudflare Pages (or any static host).

Usage:
    # Set your AdSense ID (Windows)
    set ADSENSE_CLIENT_ID=ca-pub-9165747306481565

    # Set your AdSense ID (Linux/Mac)
    export ADSENSE_CLIENT_ID=ca-pub-9165747306481565

    # Run deploy
    python deploy.py

    # Deploy to Cloudflare
    cd dist && npx wrangler pages deploy .
"""

import os
import shutil
from pathlib import Path

def deploy():
    """Build and prepare the website for deployment."""

    # Get AdSense client ID from environment
    adsense_id = os.environ.get('ADSENSE_CLIENT_ID', '')

    if not adsense_id:
        print("WARNING: ADSENSE_CLIENT_ID not set. Ads will not be included.")
        print("To include ads, set the environment variable:")
        print("  Windows: set ADSENSE_CLIENT_ID=ca-pub-YOUR_ID")
        print("  Linux/Mac: export ADSENSE_CLIENT_ID=ca-pub-YOUR_ID")
        print()

    # Paths
    source_dir = Path('website')
    dist_dir = Path('dist')

    # Clean/create dist folder
    if dist_dir.exists():
        shutil.rmtree(dist_dir)
    dist_dir.mkdir()

    # Copy all files except index.html (which we process)
    for item in source_dir.iterdir():
        if item.name == 'index.html':
            continue
        dest = dist_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)

    # Process index.html - inject AdSense if ID provided
    index_source = source_dir / 'index.html'
    index_dest = dist_dir / 'index.html'

    content = index_source.read_text(encoding='utf-8')

    if adsense_id:
        adsense_script = f'''    <script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client={adsense_id}"
     crossorigin="anonymous"></script>'''
        content = content.replace('    <!-- ADSENSE_PLACEHOLDER -->', adsense_script)
        print(f"[OK] Injected AdSense ID: {adsense_id}")
    else:
        # Remove placeholder entirely if no ID
        content = content.replace('    <!-- ADSENSE_PLACEHOLDER -->\n', '')
        print("[OK] No AdSense ID provided, ads omitted")

    index_dest.write_text(content, encoding='utf-8')

    print(f"\n[OK] Build complete: {dist_dir}/")
    print(f"  Files ready for deployment")

    # Size info
    total_size = sum(f.stat().st_size for f in dist_dir.rglob('*') if f.is_file())
    print(f"  Total size: {total_size / 1024 / 1024:.1f} MB")

    print("\nTo deploy to Cloudflare Pages:")
    print("  cd dist")
    print("  npx wrangler pages deploy .")
    print("\nOr upload the 'dist' folder to any static host.")

if __name__ == '__main__':
    deploy()
