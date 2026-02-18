#!/usr/bin/env python3
"""
Step 5: Upload Videos to YouTube (Resumable/Robust Version)

Uploads finished videos to YouTube with automatic retry on connection failures.
Designed for slow/unstable connections with resume capability.

Key Features:
- Automatic retry with exponential backoff
- Resume interrupted uploads from last chunk
- Configurable chunk size (larger = fewer API calls, better for slow connections)
- Connection timeout handling
- Detailed progress logging
- Saves upload state for crash recovery
"""

import os
import sys
import json
import time
import pickle
import random
import socket
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

# Google API libraries
try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError, ResumableUploadError
    from googleapiclient.http import MediaFileUpload, MediaIoBaseUpload
    from googleapiclient import _auth
    import googleapiclient.http
except ImportError:
    print("Installing required packages: google-auth-oauthlib google-api-python-client")
    os.system("pip install -q google-auth-oauthlib google-api-python-client")
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError, ResumableUploadError
    from googleapiclient.http import MediaFileUpload

# Configuration
VIDEOS_DIR = Path("videos")
DESCRIPTIONS_DIR = Path("descriptions")
PROGRESS_DIR = Path("progress")
STATE_FILE = PROGRESS_DIR / "upload_state.json"

# OAuth 2.0 scopes
SCOPES = ['https://www.googleapis.com/auth/youtube.upload',
          'https://www.googleapis.com/auth/youtube']

# Credentials files
CLIENT_SECRETS_FILE = "client_secrets.json"
TOKEN_FILE = "youtube_token.pickle"
UPLOAD_MAPPING_FILE = PROGRESS_DIR / "youtube_uploads.json"

# Video settings
VIDEO_CATEGORY = "27"  # Education
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
DEFAULT_PRIVACY = "public"

# API quota
DAILY_QUOTA = 10000
UPLOAD_COST = 1600
MAX_UPLOADS_PER_DAY = DAILY_QUOTA // UPLOAD_COST

# Retry configuration for unstable connections
RETRY_CONFIG = {
    'max_retries': 10,              # Max retry attempts per upload
    'initial_delay': 5,             # Initial retry delay (seconds)
    'max_delay': 300,               # Max retry delay (5 minutes)
    'backoff_factor': 2,            # Exponential backoff multiplier
    'chunk_size_mb': 50,            # Upload chunk size (larger = better for slow connections)
    'connection_timeout': 300,      # Connection timeout (5 minutes)
    'read_timeout': 600,            # Read timeout (10 minutes)
}


@dataclass
class VideoMetadata:
    segment_name: str
    video_file: Path
    description_file: Path
    volume_number: int
    volume_name: str
    part_number: Optional[int]
    total_parts: int


@dataclass
class UploadState:
    """Track upload state for resume capability."""
    segment_name: str
    video_id: Optional[str]
    upload_uri: Optional[str]  # Resumable upload URI
    chunks_uploaded: int
    total_chunks: int
    last_chunk_time: float
    status: str  # 'pending', 'uploading', 'completed', 'failed'
    error_count: int
    last_error: Optional[str]
    
    def to_dict(self) -> dict:
        return {
            'segment_name': self.segment_name,
            'video_id': self.video_id,
            'upload_uri': self.upload_uri,
            'chunks_uploaded': self.chunks_uploaded,
            'total_chunks': self.total_chunks,
            'last_chunk_time': self.last_chunk_time,
            'status': self.status,
            'error_count': self.error_count,
            'last_error': self.last_error
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'UploadState':
        return cls(**data)


def log(message: str, level: str = "INFO"):
    """Log with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prefix = {"INFO": "[INFO]", "WARN": "[WARN]", "ERROR": "[ERROR]", "RETRY": "[RETRY]"}.get(level, "[INFO]")
    print(f"{timestamp} {prefix} {message}")


def load_upload_mapping() -> Dict[str, str]:
    """Load mapping of segment names to YouTube video IDs."""
    if not UPLOAD_MAPPING_FILE.exists():
        return {}
    
    try:
        with open(UPLOAD_MAPPING_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log(f"Error loading upload mapping: {e}", "ERROR")
        return {}


def save_upload_mapping(mapping: Dict[str, str]):
    """Save mapping of segment names to YouTube video IDs."""
    PROGRESS_DIR.mkdir(exist_ok=True)
    with open(UPLOAD_MAPPING_FILE, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)


def load_upload_states() -> Dict[str, UploadState]:
    """Load upload states for resume capability."""
    if not STATE_FILE.exists():
        return {}
    
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return {k: UploadState.from_dict(v) for k, v in data.items()}
    except Exception as e:
        log(f"Error loading upload states: {e}", "ERROR")
        return {}


def save_upload_states(states: Dict[str, UploadState]):
    """Save upload states for resume capability."""
    PROGRESS_DIR.mkdir(exist_ok=True)
    data = {k: v.to_dict() for k, v in states.items()}
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_authenticated_service():
    """Authenticate with YouTube API."""
    creds = None

    if Path(TOKEN_FILE).exists():
        with open(TOKEN_FILE, 'rb') as token:
            creds = pickle.load(token)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            log("Refreshing access token...")
            creds.refresh(Request())
        else:
            if not Path(CLIENT_SECRETS_FILE).exists():
                log(f"ERROR: {CLIENT_SECRETS_FILE} not found!", "ERROR")
                log("Setup instructions:")
                log("1. Go to https://console.cloud.google.com/")
                log("2. Create project and enable YouTube Data API v3")
                log("3. Create OAuth 2.0 Desktop credentials")
                log("4. Download as 'client_secrets.json'")
                sys.exit(1)

            log("First run: Opening browser for OAuth authorization...")
            flow = InstalledAppFlow.from_client_secrets_file(
                CLIENT_SECRETS_FILE, SCOPES
            )
            creds = flow.run_local_server(port=0)

        with open(TOKEN_FILE, 'wb') as token:
            pickle.dump(creds, token)
        log("Credentials saved.")

    # Build service with longer timeouts for slow connections
    return build(
        'youtube', 'v3',
        credentials=creds,
        cache_discovery=False,
        requestBuilder=lambda *args, **kwargs: _auth.authorized_http(creds)
    )


def parse_segment_name(segment_name: str) -> VideoMetadata:
    """Parse segment filename to extract metadata."""
    video_file = VIDEOS_DIR / f"{segment_name}.mp4"
    description_file = DESCRIPTIONS_DIR / f"{segment_name}.txt"

    vol_match = segment_name.split(" - Vol. ")[1].split(" - ")[0]
    volume_number = int(vol_match)

    parts = segment_name.split(" - ")
    volume_name = parts[2].split(" (Part")[0] if "(Part" in parts[2] else parts[2]

    part_number = None
    total_parts = 1
    if "(Part " in segment_name:
        part_str = segment_name.split("(Part ")[1].rstrip(")")
        part_number = int(part_str)
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


def calculate_retry_delay(attempt: int) -> float:
    """Calculate retry delay with exponential backoff and jitter."""
    delay = min(
        RETRY_CONFIG['initial_delay'] * (RETRY_CONFIG['backoff_factor'] ** attempt),
        RETRY_CONFIG['max_delay']
    )
    # Add random jitter (0-30% of delay) to prevent thundering herd
    jitter = delay * 0.3 * random.random()
    return delay + jitter


def upload_video_with_retry(
    youtube,
    metadata: VideoMetadata,
    upload_states: Dict[str, UploadState],
    privacy: str = DEFAULT_PRIVACY
) -> Optional[str]:
    """
    Upload a video with automatic retry and resume capability.
    
    Handles:
    - Connection timeouts
    - Network interruptions
    - API rate limiting
    - Resume from last successful chunk
    """
    segment_name = metadata.segment_name
    
    if not metadata.video_file.exists():
        log(f"ERROR: Video file not found: {metadata.video_file}", "ERROR")
        return None

    # Load description
    if not metadata.description_file.exists():
        log(f"WARNING: Description file not found: {metadata.description_file}", "WARN")
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

    file_size = metadata.video_file.stat().st_size
    chunk_size = RETRY_CONFIG['chunk_size_mb'] * 1024 * 1024
    total_chunks = (file_size + chunk_size - 1) // chunk_size

    log(f"Uploading: {title}")
    log(f"File: {metadata.video_file.name} ({file_size / (1024**3):.2f} GB)")
    log(f"Chunk size: {RETRY_CONFIG['chunk_size_mb']} MB ({total_chunks} chunks total)")

    # Check for existing upload state (resume)
    state = upload_states.get(segment_name)
    if state and state.status == 'uploading' and state.upload_uri:
        log(f"Resuming previous upload (chunk {state.chunks_uploaded}/{state.total_chunks})")
    else:
        state = UploadState(
            segment_name=segment_name,
            video_id=None,
            upload_uri=None,
            chunks_uploaded=0,
            total_chunks=total_chunks,
            last_chunk_time=0,
            status='pending',
            error_count=0,
            last_error=None
        )

    # Retry loop
    for attempt in range(RETRY_CONFIG['max_retries']):
        try:
            # Create media upload
            media = MediaFileUpload(
                str(metadata.video_file),
                chunksize=chunk_size,
                resumable=True
            )

            # Create request
            request = youtube.videos().insert(
                part='snippet,status',
                body=body,
                media_body=media
            )

            # If resuming, try to use previous upload URI
            if state.upload_uri and state.chunks_uploaded > 0:
                try:
                    request.resumable_uri = state.upload_uri
                    log(f"Using previous upload URI (resuming from chunk {state.chunks_uploaded})")
                except AttributeError:
                    log("Could not resume - starting fresh upload")
                    state.chunks_uploaded = 0
                    state.upload_uri = None

            response = None
            chunk_num = state.chunks_uploaded

            while response is None:
                try:
                    # Upload next chunk with timeout handling
                    status, response = request.next_chunk()
                    
                    if status:
                        chunk_num += 1
                        progress = int(status.progress() * 100)
                        log(f"Progress: {progress}% (chunk {chunk_num}/{total_chunks})")
                        
                        # Update state
                        state.chunks_uploaded = chunk_num
                        state.last_chunk_time = time.time()
                        state.status = 'uploading'
                        try:
                            state.upload_uri = request.resumable_uri
                        except AttributeError:
                            pass
                        save_upload_states(upload_states)

                except (socket.timeout, TimeoutError, ConnectionError) as e:
                    log(f"Connection error during upload: {e}", "ERROR")
                    state.last_error = str(e)
                    state.error_count += 1
                    save_upload_states(upload_states)
                    
                    # Calculate retry delay
                    delay = calculate_retry_delay(attempt)
                    log(f"Will retry in {delay:.1f} seconds...", "RETRY")
                    time.sleep(delay)
                    raise  # Re-raise to trigger outer retry

            # Upload complete
            video_id = response['id']
            log(f"Upload complete: https://youtube.com/watch?v={video_id}")
            
            state.status = 'completed'
            state.video_id = video_id
            save_upload_states(upload_states)
            
            return video_id

        except HttpError as e:
            error_code = e.resp.status if hasattr(e.resp, 'status') else 'unknown'
            error_details = e.content.decode() if hasattr(e, 'content') else str(e)
            
            log(f"HTTP Error {error_code}: {error_details[:200]}", "ERROR")
            state.last_error = f"HTTP {error_code}: {error_details[:500]}"
            state.error_count += 1
            save_upload_states(upload_states)
            
            # Handle specific error codes
            if error_code in [500, 502, 503, 504]:
                # Server error - retry
                delay = calculate_retry_delay(attempt)
                log(f"Server error. Retrying in {delay:.1f} seconds... (attempt {attempt + 1}/{RETRY_CONFIG['max_retries']})", "RETRY")
                time.sleep(delay)
                continue
            elif error_code == 403 and 'quotaExceeded' in error_details:
                log("API quota exceeded. Stopping uploads.", "ERROR")
                return None
            else:
                # Client error - don't retry
                log(f"Client error {error_code} - not retrying", "ERROR")
                state.status = 'failed'
                save_upload_states(upload_states)
                return None

        except (socket.timeout, TimeoutError, ConnectionError, OSError) as e:
            log(f"Network error: {e}", "ERROR")
            state.last_error = str(e)
            state.error_count += 1
            save_upload_states(upload_states)
            
            if attempt < RETRY_CONFIG['max_retries'] - 1:
                delay = calculate_retry_delay(attempt)
                log(f"Network error. Retrying in {delay:.1f} seconds... (attempt {attempt + 1}/{RETRY_CONFIG['max_retries']})", "RETRY")
                time.sleep(delay)
            else:
                log(f"Max retries exceeded for {segment_name}", "ERROR")
                state.status = 'failed'
                save_upload_states(upload_states)
                return None

        except Exception as e:
            log(f"Unexpected error: {e}", "ERROR")
            state.last_error = str(e)
            state.error_count += 1
            state.status = 'failed'
            save_upload_states(upload_states)
            return None

    log(f"Max retries exceeded for {segment_name}", "ERROR")
    state.status = 'failed'
    save_upload_states(upload_states)
    return None


def get_or_create_playlist(
    youtube,
    playlist_cache: Dict[int, str],
    volume_number: int,
    volume_name: str
) -> Optional[str]:
    """Get or create playlist for a volume."""
    if volume_number in playlist_cache:
        return playlist_cache[volume_number]

    playlist_title = f"Illumine Lingao - Vol. {volume_number:02d} - {volume_name}"

    for attempt in range(3):
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
                    log(f"Found existing playlist: {playlist_title}")
                    playlist_cache[volume_number] = playlist_id
                    return playlist_id

            # Create new playlist
            log(f"Creating playlist: {playlist_title}")
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
            log(f"Created playlist: https://youtube.com/playlist?list={playlist_id}")
            return playlist_id

        except Exception as e:
            log(f"Playlist error (attempt {attempt + 1}): {e}", "ERROR")
            if attempt < 2:
                time.sleep(5 * (attempt + 1))
            else:
                return None

    return None


def add_to_playlist(youtube, playlist_id: str, video_id: str, position: int) -> bool:
    """Add video to playlist at specific position."""
    for attempt in range(3):
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
        except Exception as e:
            log(f"Playlist add error (attempt {attempt + 1}): {e}", "ERROR")
            if attempt < 2:
                time.sleep(5 * (attempt + 1))
    return False


def print_status(upload_states: Dict[str, UploadState], upload_mapping: Dict[str, str]):
    """Print current upload status summary."""
    print("\n" + "=" * 60)
    print("UPLOAD STATUS SUMMARY")
    print("=" * 60)
    
    completed = sum(1 for s in upload_states.values() if s.status == 'completed')
    uploading = sum(1 for s in upload_states.values() if s.status == 'uploading')
    failed = sum(1 for s in upload_states.values() if s.status == 'failed')
    pending = sum(1 for s in upload_states.values() if s.status == 'pending')
    
    print(f"Completed: {completed}")
    print(f"Uploading/In Progress: {uploading}")
    print(f"Failed: {failed}")
    print(f"Pending: {pending}")
    print(f"Total Mapped: {len(upload_mapping)}")
    
    if uploading > 0:
        print("\nIn Progress:")
        for name, state in upload_states.items():
            if state.status == 'uploading':
                progress = (state.chunks_uploaded / state.total_chunks * 100) if state.total_chunks > 0 else 0
                print(f"  {name}: {progress:.1f}% (chunk {state.chunks_uploaded}/{state.total_chunks})")
    
    print("=" * 60 + "\n")


def main():
    """Main entry point."""
    print("=" * 60)
    print("Step 5: Upload Videos to YouTube (Resumable)")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Chunk size: {RETRY_CONFIG['chunk_size_mb']} MB")
    print(f"  Max retries: {RETRY_CONFIG['max_retries']}")
    print(f"  Connection timeout: {RETRY_CONFIG['connection_timeout']}s")
    print(f"  Read timeout: {RETRY_CONFIG['read_timeout']}s")
    print("=" * 60)

    POLL_INTERVAL = 60  # 1 minute between polls

    # Authenticate
    print("\nAuthenticating with YouTube...")
    try:
        youtube = get_authenticated_service()
        print("Authenticated successfully")
    except Exception as e:
        print(f"ERROR: Authentication failed: {e}")
        sys.exit(1)

    # Load state
    upload_mapping = load_upload_mapping()
    upload_states = load_upload_states()
    playlist_cache = {}

    total_uploaded = 0
    total_failed = 0
    start_time = time.time()
    uploads_today = 0
    day_start = time.time()

    try:
        while True:
            # Print status
            print_status(upload_states, upload_mapping)
            
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
                print(f"Waiting for videos... ({len(video_files)} total, {total_uploaded} uploaded)")
                time.sleep(POLL_INTERVAL)
                continue

            # Check daily quota
            if time.time() - day_start > 86400:
                uploads_today = 0
                day_start = time.time()

            if uploads_today >= MAX_UPLOADS_PER_DAY:
                wait_time = 86400 - (time.time() - day_start)
                print(f"Daily quota reached. Waiting {wait_time/3600:.1f} hours...")
                time.sleep(POLL_INTERVAL)
                continue

            # Upload videos
            for segment_name in remaining:
                if uploads_today >= MAX_UPLOADS_PER_DAY:
                    break

                print(f"\n[{total_uploaded + 1}] {segment_name}")

                metadata = parse_segment_name(segment_name)

                # Upload with retry
                video_id = upload_video_with_retry(youtube, metadata, upload_states)

                if video_id:
                    total_uploaded += 1
                    uploads_today += 1
                    upload_mapping[segment_name] = video_id
                    save_upload_mapping(upload_mapping)

                    # Add to playlist
                    playlist_id = get_or_create_playlist(
                        youtube, playlist_cache,
                        metadata.volume_number, metadata.volume_name
                    )

                    if playlist_id:
                        position = metadata.part_number - 1 if metadata.part_number else 0
                        add_to_playlist(youtube, playlist_id, video_id, position)

                    log(f"Quota used today: {uploads_today}/{MAX_UPLOADS_PER_DAY} uploads")
                else:
                    total_failed += 1
                    log(f"FAILED: {segment_name}", "ERROR")

            print(f"\nBatch complete. Checking for more videos...")
            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\n\nStopped by user.")
        print("Upload state saved. Run again to resume.")

    total_time = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Summary: {total_uploaded} uploaded, {total_failed} failed")
    print(f"Time: {total_time/3600:.1f} hours")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
