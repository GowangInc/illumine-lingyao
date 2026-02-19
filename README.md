# Illumine Lingao YouTube Audiobook

An AI-powered pipeline for creating illustrated audiobook videos of [Illumine Lingao](https://lingao.entropydrivenmindset.win) (临高启明), a Chinese alternate history web novel about 500 modern people who travel back to 1628 Ming Dynasty China.

**YouTube:** [Volume 1 - Setting Sail](https://www.youtube.com/playlist?list=PLjq2oIRxOOOmKJ54-0L-JCr46zLuNXmFK)
**Web Reader:** [lingao.entropydrivenmindset.win](https://lingao.entropydrivenmindset.win)

## Overview

This project generates YouTube-ready audiobook videos with:
- **AI-narrated audio** (Kokoro TTS)
- **AI-generated illustrations** (Flux.2 Klein) in Ming Dynasty ink wash painting style
- **Chapter timestamps** in video descriptions
- **Visual consistency** across chapters via an entity registry with deterministic seeds
- **~58 videos** covering the complete 617-hour, 2,883-scene translation

## Video Pipeline

The pipeline consists of 6 steps, each resumable and runnable independently:

| Step | Script | Description |
|------|--------|-------------|
| 0 | `plan_chapters.py` | Whisper transcription + scene mapping |
| 1 | `generate_prompts.py` | Generate image prompts (Qwen3-4B, local) |
| 2 | `generate_images.py` | Generate illustrations (Flux.2 Klein) |
| 3 | `generate_clips.py` | Create video clips from images (FFmpeg, 480x270 @ 5fps) |
| 4 | `build_videos.py` | Concatenate clips + audio into final MP4s |
| 5 | `upload_videos.py` | Upload to YouTube with playlist organization |

### Quick Start

```powershell
# Run all steps with integrated pipeline
.venv\Scripts\python pipeline_integrated.py

# Or run individual steps
.venv\Scripts\python plan_chapters.py      # Step 0: Planning
.venv\Scripts\python generate_prompts.py   # Step 1: Prompts
.venv\Scripts\python generate_images.py    # Step 2: Images
.venv\Scripts\python generate_clips.py     # Step 3: Clips
.venv\Scripts\python build_videos.py       # Step 4: Videos
```

## Project Structure

```
├── pipeline_integrated.py      # Main integrated pipeline (recommended)
├── pipeline_orchestrator.py    # Alternative orchestrator version
├── plan_chapters.py            # Step 0: Audio analysis & planning
├── generate_prompts.py         # Step 1: AI prompt generation
├── generate_images.py          # Step 2: Image generation
├── generate_clips.py           # Step 3: Video clip creation
├── build_videos.py             # Step 4: Final video assembly
├── upload_videos.py            # Step 5: YouTube upload
├── upload_videos_resumable.py  # Resumable upload for unstable connections
├── update_descriptions.py      # Update YouTube descriptions post-upload
├── monitor.py                  # TUI progress monitor
├── deploy.py                   # Website build & deployment
├── consistency/                # Character/location entity registry
├── website/                    # Web reader (deploy to Cloudflare/Netlify)
│   ├── index.html
│   ├── css/
│   ├── js/
│   └── content/
├── .gitignore
└── README.md
```

Generated content (not tracked in git):
- `plan/` - Chapter plans with text-to-audio mappings
- `prompts/` - Generated image prompts (JSON)
- `images/` - AI-generated chapter illustrations (1024x1024)
- `clips/` - Individual scene video clips (480x270 @ 5fps)
- `videos/` - Final YouTube-ready MP4s
- `descriptions/` - YouTube description files with timestamps
- `progress/` - Resume tracking files

## Requirements

### Hardware
- GPU: NVIDIA GPU with 16GB+ VRAM (RTX 3090 recommended)
- RAM: 32GB+ recommended
- Storage: ~500GB for generated content

### Software
- Python 3.10+
- FFmpeg (must be on PATH)
- CUDA-compatible PyTorch

### Python Dependencies

```bash
pip install transformers torch accelerate bitsandbytes
pip install diffusers
pip install ebooklib beautifulsoup4
pip install openai-whisper
pip install rich
pip install google-auth-oauthlib google-api-python-client
```

## YouTube Upload Setup

To upload videos to YouTube:

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a project and enable YouTube Data API v3
3. Create OAuth 2.0 Desktop credentials
4. Download `client_secrets.json` to the project folder
5. Run `upload_videos.py` and authenticate on first run

**Note:** The free API tier allows ~6 video uploads per day (10,000 quota units).

## Web Reader

The `website/` folder contains a web-based reader for the full English translation, deployable to Cloudflare Pages, Netlify, or any static host.

**Live reader:** https://lingao.entropydrivenmindset.win

### Deployment

The AdSense client ID is injected at build time via environment variable (kept out of git):

```powershell
# 1. Copy the example env file and edit with your ID
copy .env.example .env
# Edit .env and set ADSENSE_CLIENT_ID=ca-pub-YOUR_ID

# 2. Build
python deploy.py

# 3. Deploy to Cloudflare
cd dist
npx wrangler pages deploy .
```

Without the `ADSENSE_CLIENT_ID` set, the site deploys without ads.

## Art Style

Images are generated in Ming Dynasty ink wash painting style (文人画), 1628:
- Muted earth tones with cobalt and vermillion accents
- Reference artists: Xiang Shengmo, Zhang Ruitu, Zhang Feng
- Jingdezhen blue and white porcelain motifs

Character and location consistency is maintained across chapters using an entity registry with deterministic seeds, so the same person looks roughly the same across hundreds of chapters.

## Novel Information

- **Title:** Illumine Lingao (临高启明)
- **Author:** Blowing Past the Ear (吹牛者)
- **Genre:** Time-travel historical fiction (穿越种田流)
- **Setting:** 1628 AD, late Ming Dynasty, Lingao County, Hainan Island
- **Premise:** 500+ modern people travel back in time to build an industrial society

## License

MIT
