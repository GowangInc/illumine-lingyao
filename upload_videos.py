#!/usr/bin/env python3
"""
Step 5: Upload Videos to YouTube

Uploads finished videos to YouTube with metadata, thumbnails, and playlist organization.
Requires OAuth 2.0 credentials from Google Cloud Console.

Setup:
1. Create project at https://console.cloud.google.com/
2. Enable YouTube Data API v3
3. Create OAuth 2.0 Desktop credentials
4. Download as 'client_secrets.json' in this directory

Features:
- OAuth 2.0 authentication (browser flow on first run, cached after)
- Automatic playlist creation and organization by volume
- Resume capability: tracks uploaded video IDs
- Rate limit handling: respects YouTube API quota (1,600 units per upload)
- Continuous polling mode: uploads as new videos appear
"""

import os
import json
import sys
import time
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Google API libraries
try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaFileUpload
except ImportError:
    print("Installing required packages: google-auth-oauthlib google-api-python-client")
    os.system("pip install -q google-auth-oauthlib google-api-python-client")
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaFileUpload

# Configuration
VIDEOS_DIR = Path("videos")
DESCRIPTIONS_DIR = Path("descriptions")
PROGRESS_DIR = Path("progress")

# OAuth 2.0 scopes
SCOPES = ['https://www.googleapis.com/auth/youtube.upload',
          'https://www.googleapis.com/auth/youtube']

# Credentials files
CLIENT_SECRETS_FILE = "client_secrets.json"
TOKEN_FILE = "youtube_token.pickle"
UPLOAD_MAPPING_FILE = PROGRESS_DIR / "youtube_uploads.json"

# Video settings
VIDEO_CATEGORY = "27"  # Education (22 = People & Blogs, 27 = Education)
DEFAULT_TAGS = [
    "audiobook",
    "Illumine Lingao",
    "临高启明",
    "historical fiction",
    "time travel",
    "Ming Dynasty",
    "AI narration",
    "Kokoro TTS",
    "full audiobook"
]

# Privacy options: "private", "unlisted", "public"
DEFAULT_PRIVACY = "public"  # Videos appear in search/recommendations immediately

# API quota (units per day)
DAILY_QUOTA = 10000
UPLOAD_COST = 1600  # units per video upload
MAX_UPLOADS_PER_DAY = DAILY_QUOTA // UPLOAD_COST  # ~6 videos/day


@dataclass
class VideoMetadata:
    segment_name: str
    video_file: Path
    description_file: Path
    volume_number: int
    volume_name: str
    part_number: Optional[int]
    total_parts: int


def load_upload_mapping() -> Dict[str, str]:
    """Load mapping of segment names to YouTube video IDs."""
    if not UPLOAD_MAPPING_FILE.exists():
        return {}

    with open(UPLOAD_MAPPING_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_upload_mapping(mapping: Dict[str, str]):
    """Save mapping of segment names to YouTube video IDs."""
    PROGRESS_DIR.mkdir(exist_ok=True)
    with open(UPLOAD_MAPPING_FILE, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)


def get_authenticated_service():
    """
    Authenticate with YouTube API using OAuth 2.0.

    First run opens browser for authorization.
    Subsequent runs use cached token.
    """
    creds = None

    # Load cached token if exists
    if Path(TOKEN_FILE).exists():
        with open(TOKEN_FILE, 'rb') as token:
            creds = pickle.load(token)

    # If no valid credentials, authenticate
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("Refreshing access token...")
            creds.refresh(Request())
        else:
            if not Path(CLIENT_SECRETS_FILE).exists():
                print(f"ERROR: {CLIENT_SECRETS_FILE} not found!")
                print("\nSetup instructions:")
                print("1. Go to https://console.cloud.google.com/")
                print("2. Create project and enable YouTube Data API v3")
                print("3. Create OAuth 2.0 Desktop credentials")
                print("4. Download as 'client_secrets.json'")
                sys.exit(1)

            print("First run: Opening browser for OAuth authorization...")
            print("Grant access to upload videos to your YouTube channel.")
            flow = InstalledAppFlow.from_client_secrets_file(
                CLIENT_SECRETS_FILE, SCOPES
            )
            creds = flow.run_local_server(port=0)

        # Save credentials for next run
        with open(TOKEN_FILE, 'wb') as token:
            pickle.dump(creds, token)
        print("Credentials saved.")

    return build('youtube', 'v3', credentials=creds)


def parse_segment_name(segment_name: str) -> VideoMetadata:
    """
    Parse segment filename to extract metadata.

    Format: "Illumine Lingao - Vol. NN - Name" or "... (Part N)"
    """
    video_file = VIDEOS_DIR / f"{segment_name}.mp4"
    description_file = DESCRIPTIONS_DIR / f"{segment_name}.txt"

    # Extract volume number
    vol_match = segment_name.split(" - Vol. ")[1].split(" - ")[0]
    volume_number = int(vol_match)

    # Extract volume name
    parts = segment_name.split(" - ")
    volume_name = parts[2].split(" (Part")[0] if "(Part" in parts[2] else parts[2]

    # Extract part number if exists
    part_number = None
    total_parts = 1
    if "(Part " in segment_name:
        part_str = segment_name.split("(Part ")[1].rstrip(")")
        part_number = int(part_str)
        # Count total parts for this volume
        total_parts = len(list(VIDEOS_DIR.glob(f"*Vol. {volume_number:02d}*.mp4")))

    return VideoMetadata(
        segment_name=segment_name,
        video_file=video_file,
        description_file=description_file,
        volume_number=volume_number,
        volume_name=volume_name,
        part_number=part_number,
        total_parts=total_parts
    )


def upload_video(
    youtube,
    metadata: VideoMetadata,
    privacy: str = DEFAULT_PRIVACY
) -> Optional[str]:
    """
    Upload a video to YouTube.

    Returns video ID on success, None on failure.
    """
    if not metadata.video_file.exists():
        print(f"    ERROR: Video file not found: {metadata.video_file}")
        return None

    if not metadata.description_file.exists():
        print(f"    WARNING: Description file not found: {metadata.description_file}")
        description = ""
    else:
        with open(metadata.description_file, "r", encoding="utf-8") as f:
            description = f.read()

    # Build title
    if metadata.part_number:
        title = f"Illumine Lingao - Vol. {metadata.volume_number:02d} - {metadata.volume_name} (Part {metadata.part_number})"
    else:
        title = f"Illumine Lingao - Vol. {metadata.volume_number:02d} - {metadata.volume_name}"

    # Prepare video metadata
    body = {
        'snippet': {
            'title': title,
            'description': description,
            'tags': DEFAULT_TAGS,
            'categoryId': VIDEO_CATEGORY
        },
        'status': {
            'privacyStatus': privacy,
            'selfDeclaredMadeForKids': False
        }
    }

    # Upload video
    try:
        print(f"    Uploading: {title}")
        print(f"    File: {metadata.video_file.name} ({metadata.video_file.stat().st_size / (1024**3):.1f} GB)")

        media = MediaFileUpload(
            str(metadata.video_file),
            chunksize=10*1024*1024,  # 10MB chunks
            resumable=True
        )

        request = youtube.videos().insert(
            part='snippet,status',
            body=body,
            media_body=media
        )

        response = None
        while response is None:
            status, response = request.next_chunk()
            if status:
                progress = int(status.progress() * 100)
                print(f"\r    Upload progress: {progress}%", end="", flush=True)

        print()  # newline after progress
        video_id = response['id']
        print(f"    ✓ Uploaded: https://youtube.com/watch?v={video_id}")
        return video_id

    except HttpError as e:
        print(f"    ERROR: HTTP {e.resp.status}: {e.content.decode()}")
        return None
    except Exception as e:
        print(f"    ERROR: {e}")
        return None


def get_or_create_playlist(
    youtube,
    playlist_cache: Dict[int, str],
    volume_number: int,
    volume_name: str
) -> Optional[str]:
    """
    Get or create playlist for a volume.

    Returns playlist ID.
    """
    # Check cache
    if volume_number in playlist_cache:
        return playlist_cache[volume_number]

    playlist_title = f"Illumine Lingao - Vol. {volume_number:02d} - {volume_name}"

    try:
        # Search for existing playlist
        request = youtube.playlists().list(
            part='snippet',
            mine=True,
            maxResults=50
        )
        response = request.execute()

        for playlist in response.get('items', []):
            if playlist['snippet']['title'] == playlist_title:
                playlist_id = playlist['id']
                print(f"    Found existing playlist: {playlist_title}")
                playlist_cache[volume_number] = playlist_id
                return playlist_id

        # Create new playlist
        print(f"    Creating playlist: {playlist_title}")
        request = youtube.playlists().insert(
            part='snippet,status',
            body={
                'snippet': {
                    'title': playlist_title,
                    'description': f"Illumine Lingao (临高启明) Volume {volume_number} - English audiobook"
                },
                'status': {
                    'privacyStatus': DEFAULT_PRIVACY
                }
            }
        )
        response = request.execute()
        playlist_id = response['id']
        playlist_cache[volume_number] = playlist_id
        print(f"    ✓ Created playlist: https://youtube.com/playlist?list={playlist_id}")
        return playlist_id

    except HttpError as e:
        print(f"    ERROR creating playlist: {e}")
        return None


def add_to_playlist(
    youtube,
    playlist_id: str,
    video_id: str,
    position: int
) -> bool:
    """Add video to playlist at specific position."""
    try:
        request = youtube.playlistItems().insert(
            part='snippet',
            body={
                'snippet': {
                    'playlistId': playlist_id,
                    'resourceId': {
                        'kind': 'youtube#video',
                        'videoId': video_id
                    },
                    'position': position
                }
            }
        )
        request.execute()
        return True
    except HttpError as e:
        print(f"    WARNING: Could not add to playlist: {e}")
        return False


def main():
    """Main entry point."""
    print("=" * 60)
    print("Step 5: Upload Videos to YouTube")
    print("=" * 60)

    POLL_INTERVAL = 300  # 5 minutes between polls (videos take a while)

    # Authenticate
    print("\nAuthenticating with YouTube...")
    try:
        youtube = get_authenticated_service()
        print("✓ Authenticated")
    except Exception as e:
        print(f"ERROR: Authentication failed: {e}")
        sys.exit(1)

    # Load upload mapping
    upload_mapping = load_upload_mapping()
    playlist_cache = {}  # volume_number -> playlist_id

    total_uploaded = 0
    total_failed = 0
    start_time = time.time()
    uploads_today = 0
    day_start = time.time()

    try:
        while True:
            # Find videos to upload
            video_files = sorted(VIDEOS_DIR.glob("*.mp4"))
            remaining = [
                f.stem for f in video_files
                if f.stem not in upload_mapping
            ]

            if not remaining:
                if total_uploaded > 0:
                    print("\nAll videos uploaded! Pipeline complete.")
                    break
                print(f"\r  Waiting for videos... ({len(video_files)} total, {total_uploaded} uploaded)", end="", flush=True)
                time.sleep(POLL_INTERVAL)
                continue

            # Check daily quota
            if time.time() - day_start > 86400:
                # Reset daily counter
                uploads_today = 0
                day_start = time.time()

            if uploads_today >= MAX_UPLOADS_PER_DAY:
                wait_time = 86400 - (time.time() - day_start)
                print(f"\r  Daily quota reached ({uploads_today}/{MAX_UPLOADS_PER_DAY}). Waiting {wait_time/3600:.1f}h...", end="", flush=True)
                time.sleep(POLL_INTERVAL)
                continue

            # Upload available videos
            for segment_name in remaining:
                if uploads_today >= MAX_UPLOADS_PER_DAY:
                    break

                print(f"\n[{total_uploaded + 1}] {segment_name}")

                # Parse metadata
                metadata = parse_segment_name(segment_name)

                # Upload video
                video_id = upload_video(youtube, metadata)

                if video_id:
                    total_uploaded += 1
                    uploads_today += 1

                    # Save mapping
                    upload_mapping[segment_name] = video_id
                    save_upload_mapping(upload_mapping)

                    # Add to playlist
                    playlist_id = get_or_create_playlist(
                        youtube,
                        playlist_cache,
                        metadata.volume_number,
                        metadata.volume_name
                    )

                    if playlist_id:
                        # Calculate position in playlist (0-indexed)
                        position = metadata.part_number - 1 if metadata.part_number else 0
                        add_to_playlist(youtube, playlist_id, video_id, position)

                    # Show quota status
                    print(f"    Quota used today: {uploads_today}/{MAX_UPLOADS_PER_DAY} uploads")
                else:
                    total_failed += 1
                    print(f"    FAILED: {segment_name}")

            print(f"\nBatch done. Polling for new videos...")
            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\n\nStopped by user.")

    total_time = time.time() - start_time
    print(f"\nStep 5: {total_uploaded} uploaded ({total_failed} failed) in {total_time/3600:.1f}h")
    print("=" * 60)


if __name__ == "__main__":
    main()
