#!/usr/bin/env python3
"""
Russian Audio Transcription CLI
Converts Russian audio files to text using Whisper Large V3 Russian model.
"""

import argparse
import json
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import numpy as np
import torch
import torchaudio
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from tqdm import tqdm


# Model configuration
DEFAULT_MODEL = "antony66/whisper-large-v3-russian"
DEFAULT_CHUNK_SIZE = 30  # seconds
DEFAULT_OVERLAP = 5  # seconds


@dataclass
class AudioInfo:
    """Audio file information."""
    path: Path
    format: str
    codec: str
    duration: float
    sample_rate: int
    channels: int
    bitrate: Optional[int]

    def summary(self) -> str:
        """Human-readable summary."""
        duration_str = format_duration(self.duration)
        return f"{self.format.upper()} | {self.codec} | {duration_str} | {self.sample_rate}Hz | {self.channels}ch"


def format_duration(seconds: float) -> str:
    """Format seconds as human-readable duration."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}ч {minutes}м {secs}с"
    elif minutes > 0:
        return f"{minutes}м {secs}с"
    else:
        return f"{secs}с"


def log_info(msg: str) -> None:
    print(f"[INFO] {msg}", file=sys.stderr)


def log_error(msg: str) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)


def log_step(step: int, total: int, msg: str) -> None:
    print(f"[{step}/{total}] {msg}", file=sys.stderr)


def check_dependencies() -> bool:
    """Check if required system dependencies are available."""
    missing = []
    for cmd in ["ffmpeg", "ffprobe", "sox"]:
        if shutil.which(cmd) is None:
            missing.append(cmd)
    if missing:
        log_error(f"Не установлены: {', '.join(missing)}")
        log_error("Установи: brew install ffmpeg sox")
        return False
    return True


def probe_audio(path: Path) -> Optional[AudioInfo]:
    """Get detailed audio file information using ffprobe."""
    try:
        result = subprocess.run([
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format", "-show_streams",
            str(path)
        ], capture_output=True, text=True, check=True)

        data = json.loads(result.stdout)

        # Find audio stream
        audio_stream = None
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "audio":
                audio_stream = stream
                break

        if not audio_stream:
            return None

        fmt = data.get("format", {})

        return AudioInfo(
            path=path,
            format=fmt.get("format_name", "unknown").split(",")[0],
            codec=audio_stream.get("codec_name", "unknown"),
            duration=float(fmt.get("duration", 0)),
            sample_rate=int(audio_stream.get("sample_rate", 0)),
            channels=int(audio_stream.get("channels", 0)),
            bitrate=int(fmt.get("bit_rate")) if fmt.get("bit_rate") else None
        )
    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError, ValueError):
        return None


def is_audio_file(path: Path) -> tuple[bool, str]:
    """Check if file is a valid audio file. Returns (is_valid, message)."""
    if not path.exists():
        return False, f"Файл не найден: {path}"

    if not path.is_file():
        return False, f"Не файл: {path}"

    info = probe_audio(path)
    if info is None:
        return False, f"Не удалось прочитать как аудио: {path.name}"

    if info.duration < 0.5:
        return False, f"Слишком короткий файл: {format_duration(info.duration)}"

    return True, info.summary()


def convert_audio(input_path: Path, output_path: Path) -> None:
    """Convert audio to 16kHz mono WAV with normalization."""
    # First pass: convert to 48kHz with ffmpeg
    tmp_48k = output_path.with_suffix(".48k.wav")

    ffmpeg_cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-ar", "48000", "-ac", "1",
        "-loglevel", "error",
        str(tmp_48k)
    ]
    subprocess.run(ffmpeg_cmd, check=True)

    # Second pass: normalize and resample to 16kHz with sox
    sox_cmd = [
        "sox", str(tmp_48k), "-r", "16000", "-c", "1", str(output_path),
        "compand", "0.3,1", "6:-70,-60,-20", "-5", "-90", "0.2",
        "norm", "-0.5"
    ]
    subprocess.run(sox_cmd, capture_output=True, check=True)

    tmp_48k.unlink()


def chunk_audio_fixed(
    input_wav: Path,
    chunk_dir: Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP
) -> list[Path]:
    """Split audio into fixed-size chunks with overlap."""
    from scipy.io import wavfile

    sample_rate, audio_data = wavfile.read(str(input_wav))

    # Convert to float32
    if audio_data.dtype == np.int16:
        audio_float = audio_data.astype(np.float32) / 32767.0
    else:
        audio_float = audio_data.astype(np.float32)

    # Ensure mono
    if len(audio_float.shape) > 1:
        audio_float = audio_float.mean(axis=1)

    total_samples = len(audio_float)
    samples_per_chunk = chunk_size * sample_rate
    samples_overlap = overlap * sample_rate
    stride = samples_per_chunk - samples_overlap

    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk_paths = []

    start = 0
    idx = 0
    while start < total_samples:
        end = min(start + samples_per_chunk, total_samples)
        chunk = audio_float[start:end]

        # Save as int16 WAV
        chunk_int16 = (chunk * 32767).astype(np.int16)
        chunk_path = chunk_dir / f"chunk_{idx:05d}.wav"
        wavfile.write(str(chunk_path), sample_rate, chunk_int16)
        chunk_paths.append(chunk_path)

        start += stride
        idx += 1

    return chunk_paths


def chunk_audio_vad(
    input_wav: Path,
    chunk_dir: Path,
    min_silence_duration: float = 0.5,
    max_chunk_duration: float = 30.0
) -> list[Path]:
    """Split audio using Voice Activity Detection for natural breaks."""
    import webrtcvad
    from scipy.io import wavfile

    # Load WAV with scipy (no torchcodec dependency)
    sample_rate, audio_data = wavfile.read(str(input_wav))

    # Handle stereo -> mono
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)

    # Normalize to float [-1, 1] for resampling if needed
    if audio_data.dtype == np.int16:
        audio_float = audio_data.astype(np.float32) / 32767.0
    elif audio_data.dtype == np.int32:
        audio_float = audio_data.astype(np.float32) / 2147483647.0
    else:
        audio_float = audio_data.astype(np.float32)

    # Resample to 16kHz if needed
    if sample_rate != 16000:
        from scipy import signal
        num_samples = int(len(audio_float) * 16000 / sample_rate)
        audio_float = signal.resample(audio_float, num_samples)
        sample_rate = 16000

    # Convert to 16-bit PCM bytes for webrtcvad
    pcm_data = (audio_float * 32767).astype(np.int16).tobytes()

    # Initialize VAD (aggressiveness 0-3, higher = more aggressive filtering)
    vad = webrtcvad.Vad(2)

    # Process in 30ms frames (480 samples at 16kHz)
    frame_duration_ms = 30
    frame_size = int(sample_rate * frame_duration_ms / 1000)
    frame_bytes = frame_size * 2  # 16-bit = 2 bytes per sample

    # Detect speech segments
    speech_segments = []
    is_speech = False
    segment_start = 0.0
    silence_frames = 0
    min_silence_frames = int(min_silence_duration * 1000 / frame_duration_ms)

    for i in range(0, len(pcm_data) - frame_bytes, frame_bytes):
        frame = pcm_data[i:i + frame_bytes]
        frame_time = i / 2 / sample_rate  # Convert byte offset to seconds

        try:
            frame_is_speech = vad.is_speech(frame, sample_rate)
        except Exception:
            frame_is_speech = True  # Assume speech on error

        if frame_is_speech:
            if not is_speech:
                # Start of speech
                is_speech = True
                segment_start = frame_time
            silence_frames = 0
        else:
            if is_speech:
                silence_frames += 1
                if silence_frames >= min_silence_frames:
                    # End of speech segment
                    segment_end = frame_time - (silence_frames * frame_duration_ms / 1000)
                    if segment_end > segment_start:
                        speech_segments.append({'start': segment_start, 'end': segment_end})
                    is_speech = False
                    silence_frames = 0

    # Handle last segment
    if is_speech:
        segment_end = len(pcm_data) / 2 / sample_rate
        if segment_end > segment_start:
            speech_segments.append({'start': segment_start, 'end': segment_end})

    if not speech_segments:
        raise ValueError("VAD не обнаружил речь в аудио")

    # Group speech segments into chunks respecting max_chunk_duration
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk_paths = []

    current_start = 0.0
    current_segments = []

    for seg in speech_segments:
        seg_end = seg['end']
        chunk_duration = seg_end - current_start

        # If adding this segment would exceed max duration, save current chunk
        if current_segments and chunk_duration > max_chunk_duration:
            chunk_path = _save_chunk_np(
                audio_float, sample_rate, chunk_dir,
                current_start, current_segments[-1]['end'],
                len(chunk_paths)
            )
            chunk_paths.append(chunk_path)
            current_start = seg['start']
            current_segments = []

        current_segments.append(seg)

    # Save remaining segments
    if current_segments:
        chunk_path = _save_chunk_np(
            audio_float, sample_rate, chunk_dir,
            current_start, current_segments[-1]['end'],
            len(chunk_paths)
        )
        chunk_paths.append(chunk_path)

    return chunk_paths


def _save_chunk_np(
    audio: np.ndarray,
    sample_rate: int,
    chunk_dir: Path,
    start_sec: float,
    end_sec: float,
    idx: int
) -> Path:
    """Save a chunk of audio from numpy array."""
    from scipy.io import wavfile

    start_sample = int(start_sec * sample_rate)
    end_sample = int(end_sec * sample_rate)
    chunk = audio[start_sample:end_sample]

    # Convert to int16 for WAV
    chunk_int16 = (chunk * 32767).astype(np.int16)

    chunk_path = chunk_dir / f"chunk_{idx:05d}.wav"
    wavfile.write(str(chunk_path), sample_rate, chunk_int16)
    return chunk_path


def _save_chunk(
    waveform: torch.Tensor,
    sample_rate: int,
    chunk_dir: Path,
    start_sec: float,
    end_sec: float,
    idx: int
) -> Path:
    """Helper to save a chunk of audio (torch version)."""
    start_sample = int(start_sec * sample_rate)
    end_sample = int(end_sec * sample_rate)
    chunk = waveform[:, start_sample:end_sample]

    chunk_path = chunk_dir / f"chunk_{idx:05d}.wav"
    torchaudio.save(str(chunk_path), chunk, sample_rate)
    return chunk_path


def load_model(model_id: str = DEFAULT_MODEL):
    """Load Whisper model and processor."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        use_safetensors=True
    ).to(device)

    return model, processor, device


def transcribe_chunks(
    model,
    processor,
    device: str,
    chunk_paths: list[Path],
    show_progress: bool = True
) -> str:
    """Transcribe all audio chunks and combine results."""
    from scipy.io import wavfile

    texts = []
    iterator = tqdm(chunk_paths, desc="Транскрипция", unit="chunk") if show_progress else chunk_paths

    for chunk_path in iterator:
        # Load with scipy (no torchcodec)
        sample_rate, audio_data = wavfile.read(str(chunk_path))

        # Convert to float32
        if audio_data.dtype == np.int16:
            audio_float = audio_data.astype(np.float32) / 32767.0
        else:
            audio_float = audio_data.astype(np.float32)

        # Ensure mono
        if len(audio_float.shape) > 1:
            audio_float = audio_float.mean(axis=1)

        # Process audio
        input_features = processor(
            audio_float,
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features.to(device)

        # Generate transcription
        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                language="ru",
                task="transcribe"
            )

        text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        texts.append(text.strip())

    return " ".join(texts)


def format_output(text: str, format_type: str, source_file: Optional[Path] = None) -> str:
    """Format transcription output."""
    if format_type == "txt":
        return text
    elif format_type == "md":
        header = f"# Транскрипция: {source_file.name}\n\n" if source_file else "# Транскрипция\n\n"
        return header + text
    elif format_type == "json":
        data = {"transcription": text}
        if source_file:
            data["source"] = source_file.name
        return json.dumps(data, ensure_ascii=False, indent=2)
    elif format_type == "srt":
        lines = text.split(". ")
        srt_lines = []
        for i, line in enumerate(lines, 1):
            if line.strip():
                srt_lines.append(f"{i}\n00:00:00,000 --> 00:00:00,000\n{line.strip()}.\n")
        return "\n".join(srt_lines)
    return text


def transcribe_file(
    input_path: Path,
    output_path: Optional[Path] = None,
    model_id: str = DEFAULT_MODEL,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    use_vad: bool = True,
    output_format: str = "txt",
    keep_temp: bool = False,
    quiet: bool = False
) -> str:
    """Main transcription function."""

    # Validate input
    is_valid, message = is_audio_file(input_path)
    if not is_valid:
        raise ValueError(message)

    if not check_dependencies():
        sys.exit(1)

    # Get audio info
    info = probe_audio(input_path)
    if not quiet and info:
        log_info(f"Файл: {input_path.name}")
        log_info(f"Формат: {info.summary()}")

    # Estimate work
    total_steps = 4
    current_step = 0

    # Create temp directory for intermediate files
    temp_dir = Path(tempfile.mkdtemp(prefix="transcribe_"))

    try:
        # Step 1: Convert audio
        current_step += 1
        if not quiet:
            log_step(current_step, total_steps, "Конвертация аудио в 16kHz WAV...")
        converted_wav = temp_dir / "audio_16k.wav"
        convert_audio(input_path, converted_wav)

        # Step 2: Chunk audio
        current_step += 1
        chunk_dir = temp_dir / "chunks"
        if use_vad:
            if not quiet:
                log_step(current_step, total_steps, "Разбиение по паузам (VAD)...")
            chunk_paths = chunk_audio_vad(converted_wav, chunk_dir)
        else:
            if not quiet:
                log_step(current_step, total_steps, f"Разбиение на чанки ({chunk_size}с)...")
            chunk_paths = chunk_audio_fixed(
                converted_wav, chunk_dir,
                chunk_size=chunk_size, overlap=overlap
            )

        if not quiet:
            log_info(f"Создано {len(chunk_paths)} фрагментов")

        # Step 3: Load model
        current_step += 1
        if not quiet:
            log_step(current_step, total_steps, "Загрузка модели Whisper...")
        model, processor, device = load_model(model_id)
        if not quiet:
            log_info(f"Устройство: {device.upper()}")

        # Step 4: Transcribe
        current_step += 1
        if not quiet:
            log_step(current_step, total_steps, "Распознавание речи...")
        text = transcribe_chunks(model, processor, device, chunk_paths, show_progress=not quiet)

        # Format output
        formatted = format_output(text, output_format, input_path)

        # Save or return
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(formatted, encoding="utf-8")
            if not quiet:
                log_info(f"Сохранено: {output_path}")

        return formatted

    finally:
        if not keep_temp:
            shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description="Транскрипция русскоязычного аудио в текст",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  %(prog)s audio.mp3                    # Транскрипция (VAD по умолчанию)
  %(prog)s audio.wav -o result.txt      # Сохранить в файл
  %(prog)s audio.m4a -f md              # Вывод в markdown
  %(prog)s audio.ogg --no-vad           # Фиксированные чанки без VAD

Поддерживаемые форматы: wav, mp3, ogg, m4a, flac, aac, wma, opus и другие
        """
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Аудио файл для транскрипции"
    )

    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Путь для сохранения (по умолчанию: имя_файла.txt)"
    )

    parser.add_argument(
        "-f", "--format",
        choices=["txt", "md", "json", "srt"],
        default="txt",
        help="Формат вывода (по умолчанию: txt)"
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Модель Whisper (по умолчанию: {DEFAULT_MODEL})"
    )

    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="Отключить VAD, использовать фиксированные чанки"
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Размер чанка в секундах (по умолчанию: {DEFAULT_CHUNK_SIZE})"
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_OVERLAP,
        help=f"Перекрытие чанков в секундах (по умолчанию: {DEFAULT_OVERLAP})"
    )

    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Сохранить временные файлы"
    )

    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Тихий режим (только результат)"
    )

    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 1.0.0"
    )

    args = parser.parse_args()

    # Determine output path
    output_path = args.output
    if output_path is None:
        ext = args.format if args.format != "srt" else "srt"
        output_path = args.input.with_suffix(f".{ext}")

    try:
        result = transcribe_file(
            input_path=args.input,
            output_path=output_path,
            model_id=args.model,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            use_vad=not args.no_vad,
            output_format=args.format,
            keep_temp=args.keep_temp,
            quiet=args.quiet
        )

        if args.quiet:
            print(result)

    except KeyboardInterrupt:
        print("\nПрервано", file=sys.stderr)
        sys.exit(130)
    except ValueError as e:
        log_error(str(e))
        sys.exit(1)
    except Exception as e:
        log_error(f"Ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
