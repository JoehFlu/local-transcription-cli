# source ~/gigaam-env/bin/activate
# python3 transcribe.py audio.mp4 --model rnnt --segment 20

import subprocess
import sys
import re
import shutil
from pathlib import Path
import tempfile
import argparse
from datetime import timedelta

def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("segment должен быть больше 0")
    return parsed


def ensure_dependency(command: str) -> None:
    if shutil.which(command):
        return
    print(f"❌ Команда не найдена: {command}")
    sys.exit(1)


def run_command(command: list[str], capture_output: bool = False) -> subprocess.CompletedProcess:
    try:
        kwargs = {
            "check": True,
            "text": True,
        }
        if capture_output:
            kwargs["capture_output"] = True
        else:
            kwargs["stdout"] = subprocess.DEVNULL
            kwargs["stderr"] = subprocess.DEVNULL

        return subprocess.run(command, **kwargs)
    except FileNotFoundError:
        print(f"❌ Команда не найдена: {command[0]}")
        sys.exit(1)
    except subprocess.CalledProcessError as exc:
        print(f"❌ Ошибка при выполнении команды: {' '.join(command)}")
        if capture_output and exc.stderr:
            print(exc.stderr.strip())
        sys.exit(exc.returncode or 1)


def clean_text(text: str) -> str:
    """Минимальная очистка — только самое необходимое"""
    if not text:
        return ""

    # Убираем Audio: 20. 0s
    text = re.sub(r'Audio:\s*\d+\.?\s*0?s?', '', text, flags=re.IGNORECASE)

    # Убираем обрезанные слова в конце строки (типа "на пос-", "кото-")
    text = re.sub(r'\s+[a-zA-Zа-яА-ЯёЁ-]{2,7}-\s*$', '', text, flags=re.MULTILINE)

    # Убираем таймкоды и мусор
    text = re.sub(r'\[\d{2}:\d{2}:\d{2},\d{3} -> \d{2}:\d{2}:\d{2},\d{3}\]', '', text)
    text = re.sub(r'Loading audio:.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Split into \d+ chunks', '', text)
    text = re.sub(r'Transcribed in .*', '', text)
    text = re.sub(r'Saved: .*', '', text)
    text = re.sub(r'tmp\w+_\w+', '', text)

    # Собираем строки и убираем пустые
    lines = [line.strip() for line in text.split('\n') if line.strip()]

    text = ' '.join(lines)

    # Самая простая финальная чистка
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s+([.,!?])', r'\1', text)
    text = re.sub(r'([.!?])\s*', r'\1 ', text)

    return text.strip()


def needs_conversion(file_path: str) -> bool:
    return Path(file_path).suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4a", ".mp3"}


def convert_to_wav(input_file: str) -> str:
    wav_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    print("🎥 Конвертирую в wav 16kHz...")
    run_command([
        "ffmpeg", "-i", input_file, "-vn", "-acodec", "pcm_s16le",
        "-ac", "1", "-ar", "16000", "-y", wav_file
    ])
    print("✅ wav готов")
    return wav_file


def split_wav(wav_file: str, segment_duration: int = 20) -> list:
    print(f"✂️ Разбиваю на сегменты по {segment_duration} сек...")
    base = Path(wav_file).with_suffix('')
    pattern = str(base) + "_seg%03d.wav"

    run_command([
        "ffmpeg", "-i", wav_file, "-f", "segment",
        "-segment_time", str(segment_duration), "-c", "copy", pattern
    ])

    segments = sorted(Path(base.parent).glob(f"{base.name}_seg*.wav"))
    print(f"✅ Создано {len(segments)} сегментов")
    return [str(s) for s in segments]


def transcribe_segment(segment_file: str, model_type: str = "rnnt") -> str:
    result = run_command([
        "gigaam-mlx", segment_file, "--model-type", model_type
    ], capture_output=True)
    return result.stdout.strip()


def format_timestamp(seconds: int) -> str:
    return str(timedelta(seconds=seconds)).split('.')[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Путь к аудио/видео файлу")
    parser.add_argument("--model", choices=["rnnt", "ctc"], default="rnnt")
    parser.add_argument("--segment", type=positive_int, default=20)
    args = parser.parse_args()
    ensure_dependency("ffmpeg")
    ensure_dependency("gigaam-mlx")

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"❌ Файл не найден: {args.input_file}")
        sys.exit(1)

    wav_file = None
    segments = []

    try:
        if needs_conversion(args.input_file):
            wav_file = convert_to_wav(args.input_file)
        else:
            wav_file = args.input_file

        segments = split_wav(wav_file, args.segment)

        output_file = input_path.stem + "_транскрипция.txt"

        print(f"\n📝 Транскрипция ({args.model.upper()})...\n")

        with open(output_file, "w", encoding="utf-8") as f:
            f.write("ТРАНСКРИПЦИЯ\n")
            f.write(f"Модель: {args.model.upper()} | Сегмент: {args.segment} сек\n")
            f.write("=" * 70 + "\n\n")

            current_time = 0
            for seg in segments:
                ts = format_timestamp(current_time)
                print(f"[{ts}] ", end="", flush=True)

                raw = transcribe_segment(seg, args.model)
                cleaned = clean_text(raw)

                if cleaned:
                    print(cleaned[:120] + "..." if len(cleaned) > 120 else cleaned)
                    f.write(f"[{ts}]\n{cleaned}\n\n")
                else:
                    print("(пусто)")

                current_time += args.segment

        print(f"\n✅ Готово! Результат сохранён в: {output_file}")

    finally:
        if wav_file and wav_file != args.input_file:
            Path(wav_file).unlink(missing_ok=True)
        
        for seg in segments:
            Path(seg).unlink(missing_ok=True)
            Path(seg).with_suffix(".txt").unlink(missing_ok=True)

        for f in Path(".").glob("*_seg*.wav"):
            f.unlink(missing_ok=True)
        for f in Path(".").glob("*_seg*.txt"):
            f.unlink(missing_ok=True)

        print("🧹 Все временные файлы удалены")


if __name__ == "__main__":
    main()
