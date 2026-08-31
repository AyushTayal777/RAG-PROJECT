import os
import re

from pydub import AudioSegment
from youtube_transcript_api import YouTubeTranscriptApi

def extract_video_id(url: str) -> str:
    """Extract YouTube video ID from URL."""

    patterns = [
        r"(?:youtube\.com/watch\?v=)([^&]+)",
        r"(?:youtu\.be/)([^?&]+)",
        r"(?:youtube\.com/shorts/)([^?&]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)

        if match:
            return match.group(1)

    raise ValueError("Invalid YouTube URL")


def get_youtube_transcript(url: str) -> str:
    """Fetch YouTube transcript without downloading the video."""

    video_id = extract_video_id(url)

    print("Fetching YouTube transcript...")

    api = YouTubeTranscriptApi()

    transcript = api.fetch(
        video_id,
        languages=["en", "hi"]
    )

    text = " ".join(
        snippet.text for snippet in transcript
    )

    if not text.strip():
        raise ValueError("YouTube transcript is empty.")

    print("YouTube transcript fetched successfully.")

    return text


def convert_to_wav(input_path: str) -> str:
    """Convert any audio/video file to WAV format using pydub."""

    output_path = os.path.splitext(input_path)[0] + "_converted.wav"

    audio = AudioSegment.from_file(input_path)

    audio = audio.set_channels(1).set_frame_rate(16000)

    audio.export(output_path, format="wav")

    return output_path


def chunk_audio(wav_path: str, chunk_minutes: int = 5) -> list:
    """Split WAV audio into chunks."""

    audio = AudioSegment.from_wav(wav_path)

    chunk_ms = chunk_minutes * 60 * 1000

    chunks = []

    for i, start in enumerate(range(0, len(audio), chunk_ms)):

        chunk = audio[start:start + chunk_ms]

        chunk_path = f"{wav_path}_chunk_{i}.wav"

        chunk.export(chunk_path, format="wav")

        chunks.append(chunk_path)

    return chunks