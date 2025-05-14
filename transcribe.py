#!/usr/bin/env python3
"""
Transcribe video files to text using faster-whisper.
"""

import os
import sys
import time
import argparse
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, Union, List, Tuple, Callable

try:
    from faster_whisper import WhisperModel
except ImportError:
    print("Error: faster-whisper is not installed.")
    print("Please install it with: pip install faster-whisper")
    sys.exit(1)


def extract_audio(video_path: str, output_path: Optional[str] = None) -> str:
    """
    Extract audio from video file using ffmpeg.

    Args:
        video_path: Path to the video file
        output_path: Path to save the extracted audio (optional)

    Returns:
        Path to the extracted audio file
    """
    if output_path is None:
        # Create a temporary file with .wav extension
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        output_path = temp_file.name
        temp_file.close()

    # Run ffmpeg to extract audio
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vn",  # No video
        "-acodec", "pcm_s16le",  # PCM 16-bit little-endian format
        "-ar", "16000",  # 16kHz sample rate
        "-ac", "1",  # Mono
        "-y",  # Overwrite output file if it exists
        output_path
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"Error extracting audio: {e}")
        if os.path.exists(output_path):
            os.unlink(output_path)
        sys.exit(1)

    return output_path


def transcribe_audio(
    audio_path: str,
    model_size: str = "base",
    language: Optional[str] = None,
    device: str = "cpu",
    compute_type: str = "float16",
    verbose: bool = False
) -> List[Tuple[str, float, float]]:
    """
    Transcribe audio file using faster-whisper.

    Args:
        audio_path: Path to the audio file
        model_size: Size of the Whisper model to use
        language: Language code (optional, will auto-detect if not provided)
        device: Device to use for inference ("cpu" or "cuda")
        compute_type: Compute type for the model
        verbose: Whether to print progress information

    Returns:
        List of tuples containing (text, start_time, end_time)
    """
    if verbose:
        print(f"Loading Whisper model ({model_size})...")

    # Check if device is mps and convert to cpu since faster-whisper doesn't support mps directly
    if device == "mps":
        if verbose:
            print(
                "MPS device is not directly supported by faster-whisper, falling back to CPU")
        device = "cpu"

    # Try to load the model with the requested compute type
    try:
        model = WhisperModel(model_size, device=device,
                             compute_type=compute_type)
    except RuntimeError as e:
        # If float16 fails, fall back to int8
        if compute_type == "float16" and "float16 compute type" in str(e):
            if verbose:
                print(f"Warning: {e}")
                print("Falling back to int8 compute type")
            compute_type = "int8"
            model = WhisperModel(model_size, device=device,
                                 compute_type=compute_type)
        else:
            # Re-raise if it's a different error
            raise

    if verbose:
        print(f"Transcribing audio file: {audio_path}")
        print(f"Using device: {device}, compute type: {compute_type}")

    # Get audio duration for progress reporting
    try:
        import ffmpeg
        probe = ffmpeg.probe(audio_path)
        audio_duration = float(probe['format']['duration'])
        if verbose:
            print(
                f"Audio duration: {format_timestamp(audio_duration)} (HH:MM:SS)")
    except Exception as e:
        if verbose:
            print(f"Could not determine audio duration: {e}")
        audio_duration = None

    # Try to transcribe with progress_callback first, if it fails, try without it
    try:
        # Create a progress callback
        def progress_callback(current_time: float, total_time: float):
            if verbose and audio_duration:
                percent = min(100, current_time / audio_duration * 100)
                current_time_str = format_timestamp(current_time)
                total_time_str = format_timestamp(audio_duration)
                print(
                    f"Progress: {current_time_str}/{total_time_str} ({percent:.1f}%)", end="\r")

        # Transcribe the audio with progress callback
        segments, info = model.transcribe(
            audio_path,
            language=language,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            progress_callback=progress_callback
        )
    except TypeError as e:
        if "unexpected keyword argument 'progress_callback'" in str(e):
            if verbose:
                print(
                    "\nYour version of faster-whisper doesn't support progress callbacks.")
                print("Transcribing without progress updates...")

            # Transcribe without progress callback
            segments, info = model.transcribe(
                audio_path,
                language=language,
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )
        else:
            # Re-raise if it's a different error
            raise

    if verbose:
        print("\nTranscription complete!")
        print(
            f"Detected language: {info.language} (probability: {info.language_probability:.2f})")

    # Collect all segments
    results = []
    segment_count = 0
    for segment in segments:
        results.append((segment.text, segment.start, segment.end))
        segment_count += 1
        if verbose and segment_count % 50 == 0:
            print(f"Processed {segment_count} segments...")

    if verbose:
        print(f"Total segments: {segment_count}")

    return results


def save_transcript(
    transcript: List[Tuple[str, float, float]],
    output_path: str,
    include_timestamps: bool = False
) -> None:
    """
    Save transcript to a file.

    Args:
        transcript: List of tuples containing (text, start_time, end_time)
        output_path: Path to save the transcript
        include_timestamps: Whether to include timestamps in the output
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for text, start, end in transcript:
            if include_timestamps:
                # Format timestamps as [HH:MM:SS.mmm]
                start_str = format_timestamp(start)
                end_str = format_timestamp(end)
                f.write(f"[{start_str} --> {end_str}] {text.strip()}\n")
            else:
                f.write(f"{text.strip()}\n")


def format_timestamp(seconds: float) -> str:
    """
    Format seconds as HH:MM:SS.mmm

    Args:
        seconds: Time in seconds

    Returns:
        Formatted timestamp
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"


def format_time_duration(seconds: float) -> str:
    """
    Format a duration in seconds to a human-readable string.

    Args:
        seconds: Time duration in seconds

    Returns:
        Formatted duration string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def transcribe_video(
    video_path: str,
    output_path: Optional[str] = None,
    model_size: str = "base",
    language: Optional[str] = None,
    include_timestamps: bool = False,
    device: str = "cpu",  # Updated to match transcribe_audio
    compute_type: str = "float16",
    keep_audio: bool = False,
    verbose: bool = False
) -> str:
    """
    Transcribe a video or audio file to text.

    Args:
        video_path: Path to the video or audio file
        output_path: Path to save the transcript (optional)
        model_size: Size of the Whisper model to use
        language: Language code (optional, will auto-detect if not provided)
        include_timestamps: Whether to include timestamps in the output
        device: Device to use for inference ("cpu", "cuda", or "mps")
        compute_type: Compute type for the model
        keep_audio: Whether to keep the extracted audio file
        verbose: Whether to print progress information

    Returns:
        Path to the saved transcript
    """
    # Validate input file
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video or audio file not found: {video_path}")

    # Supported audio file extensions
    audio_extensions = {'.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg'}

    # Get file extension
    file_ext = Path(video_path).suffix.lower()

    # Get media information
    if verbose:
        try:
            import ffmpeg
            probe = ffmpeg.probe(video_path)
            duration = float(probe['format']['duration'])
            print(f"Media duration: {format_timestamp(duration)} (HH:MM:SS)")
            print(f"Media format: {probe['format']['format_long_name']}")
            print(
                f"Media size: {int(probe['format']['size']) / (1024*1024):.2f} MB")
        except Exception as e:
            print(f"Could not get media information: {e}")

    # Determine output path if not provided
    if output_path is None:
        video_path_obj = Path(video_path)
        output_path = str(video_path_obj.with_suffix(".txt"))

    # If input is audio, use it directly; otherwise, extract audio
    if file_ext in audio_extensions:
        if verbose:
            print(
                f"Input file is an audio file ({file_ext}), skipping audio extraction.")
        audio_path = video_path
        extracted = False
    else:
        if verbose:
            print(f"Extracting audio from: {video_path}")
            start_time = time.time()
        audio_path = None
        if keep_audio:
            audio_path = str(Path(video_path).with_suffix(".wav"))
        audio_path = extract_audio(video_path, audio_path)
        extracted = True
        if verbose:
            extraction_time = time.time() - start_time
            print(
                f"Audio extraction completed in {extraction_time:.2f} seconds")

    try:
        # Transcribe the audio
        if verbose:
            print(f"Starting transcription with model: {model_size}")
            start_time = time.time()

        transcript = transcribe_audio(
            audio_path,
            model_size=model_size,
            language=language,
            device=device,
            compute_type=compute_type,
            verbose=verbose
        )

        if verbose:
            transcription_time = time.time() - start_time
            print(
                f"Transcription completed in {format_time_duration(transcription_time)}")

        # Save the transcript
        if verbose:
            print(f"Saving transcript to: {output_path}")

        save_transcript(transcript, output_path, include_timestamps)

        if verbose:
            print(f"Transcript saved successfully to: {output_path}")
            print(f"Total segments: {len(transcript)}")
            if transcript:
                total_duration = transcript[-1][2] - transcript[0][1]
                print(
                    f"Transcribed content duration: {format_time_duration(total_duration)}")

        return output_path

    finally:
        # Clean up the temporary audio file if it was extracted
        if extracted and not keep_audio and audio_path and os.path.exists(audio_path) and audio_path != video_path:
            if verbose:
                print(f"Cleaning up temporary audio file: {audio_path}")
            os.unlink(audio_path)


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe video files to text using faster-whisper")
    parser.add_argument("video_path", help="Path to the video file")
    parser.add_argument(
        "-o", "--output", help="Path to save the transcript (default: same as video with .txt extension)")
    parser.add_argument("-m", "--model", default="base", choices=["tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3"],
                        help="Whisper model size (default: base)")
    parser.add_argument(
        "-l", "--language", help="Language code (optional, will auto-detect if not provided)")
    parser.add_argument("-t", "--timestamps", action="store_true",
                        help="Include timestamps in the output")
    parser.add_argument("-d", "--device", default="cpu", choices=["cpu", "cuda"],
                        help="Device to use for inference (default: cpu)")
    parser.add_argument("-c", "--compute-type", default="int8",
                        choices=["int8", "int8_float16", "float16", "float32"],
                        help="Compute type for the model (default: float16, falls back to int8 if not supported)")
    parser.add_argument("-k", "--keep-audio", action="store_true",
                        help="Keep the extracted audio file")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print progress information")

    args = parser.parse_args()

    try:
        transcribe_video(
            args.video_path,
            args.output,
            args.model,
            args.language,
            args.timestamps,
            args.device,
            args.compute_type,
            args.keep_audio,
            args.verbose
        )
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()