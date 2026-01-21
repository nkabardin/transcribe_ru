#!/usr/bin/env python3
"""
Batch transcription of multiple audio files.
"""

import argparse
import sys
from pathlib import Path

from transcribe import transcribe_file, probe_audio, log_info, log_error, format_duration


def find_audio_files(directory: Path, recursive: bool = False) -> list[Path]:
    """Find all audio files in directory using ffprobe detection."""
    if recursive:
        candidates = directory.rglob("*")
    else:
        candidates = directory.glob("*")

    audio_files = []
    for f in candidates:
        if f.is_file() and probe_audio(f) is not None:
            audio_files.append(f)

    return sorted(audio_files)


def main():
    parser = argparse.ArgumentParser(
        description="Batch транскрипция аудиофайлов",
        epilog="""
Примеры:
  %(prog)s ./recordings              # Транскрибировать все аудио в папке
  %(prog)s ./recordings -r           # Рекурсивно
  %(prog)s ./recordings --skip       # Пропустить уже обработанные
        """
    )

    parser.add_argument(
        "input_dir",
        type=Path,
        help="Директория с аудиофайлами"
    )

    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        help="Директория для результатов (по умолчанию: та же)"
    )

    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Искать рекурсивно"
    )

    parser.add_argument(
        "--skip", "--skip-existing",
        action="store_true",
        dest="skip_existing",
        help="Пропустить файлы с существующей транскрипцией"
    )

    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="Отключить VAD"
    )

    args = parser.parse_args()

    if not args.input_dir.is_dir():
        log_error(f"Не директория: {args.input_dir}")
        sys.exit(1)

    output_dir = args.output_dir or args.input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    log_info(f"Поиск аудиофайлов в {args.input_dir}...")
    audio_files = find_audio_files(args.input_dir, args.recursive)

    if not audio_files:
        log_info("Аудиофайлы не найдены")
        return

    # Calculate total duration
    total_duration = 0.0
    for f in audio_files:
        info = probe_audio(f)
        if info:
            total_duration += info.duration

    log_info(f"Найдено {len(audio_files)} файлов ({format_duration(total_duration)})")
    print()

    success_count = 0
    skip_count = 0
    fail_count = 0

    for i, audio_path in enumerate(audio_files, 1):
        output_path = output_dir / f"{audio_path.stem}.txt"

        if args.skip_existing and output_path.exists():
            print(f"[SKIP] {audio_path.name}")
            skip_count += 1
            continue

        print(f"[{i}/{len(audio_files)}] {audio_path.name}")

        try:
            transcribe_file(
                input_path=audio_path,
                output_path=output_path,
                use_vad=not args.no_vad,
                quiet=False
            )
            success_count += 1
            print()
        except Exception as e:
            log_error(f"{audio_path.name}: {e}")
            fail_count += 1
            print()

    print()
    log_info(f"Готово: {success_count} обработано, {skip_count} пропущено, {fail_count} ошибок")


if __name__ == "__main__":
    main()
