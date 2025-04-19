# Video Transcription Tool

A Python-based tool for transcribing videos using the faster-whisper library.

## Features

- Transcribe video files to text with timestamps
- Support for multiple Whisper model sizes (tiny, base, small, medium, large)
- Automatic language detection
- Progress tracking for long transcriptions
- Optimized for Apple Silicon (M1/M2) Macs

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/youtube-dl.git
   cd youtube-dl
   ```
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Install FFmpeg (required for audio extraction):
   1. Mac
    ```brew install ffmpeg```
   2. Linux (Debian/Ubuntu)
    ```sudo apt-get install ffmpeg```
   3. Windows
    ```choco install ffmpeg```

## Usage

Basic usage:
```bash
python transcribe.py /path/to/video.mp4
```

With options:
```bash
python transcribe.py /path/to/video.mp4 -m medium -t -v
```

## Command Line Options

| Option               | Description                                                                  |
| -------------------- | ---------------------------------------------------------------------------- |
| `-o, --output`       | Path to save the transcript (default: same as video with .txt extension)     |
| `-m, --model`        | Whisper model size (tiny, base, small, medium, large-v1, large-v2, large-v3) |
| `-l, --language`     | Language code (optional, will auto-detect if not provided)                   |
| `-t, --timestamps`   | Include timestamps in the output                                             |
| `-d, --device`       | Device to use for inference (cpu, cuda)                                      |
| `-c, --compute-type` | Compute type for the model (int8, int8_float16, float16, float32)            |
| `-k, --keep-audio`   | Keep the extracted audio file                                                |
| `-v, --verbose`      | Print progress information                                                   |

## Examples
Transcribe a video with timestamps and verbose output:

```bash
python transcribe.py my-video.mp4 -t -v
```

Use a larger model for better accuracy:
```bash
python transcribe.py my-video.mp4 -m medium -t -v
```

Specify output location:
```bash
python transcribe.py my-video.mp4 -o my-transcript.txt
```

## Requirements
- Python 3.9+
- FFmpeg
- faster-whisper
- torch
- numpy
- ffmpeg-python

## License
Apache License 2.0
