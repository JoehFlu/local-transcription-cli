import subprocess
import sys
import re
import shutil
from pathlib import Path
import tempfile
import argparse
import warnings
from datetime import timedelta

AUDIO_VIDEO_FORMATS_TO_CONVERT = {
    ".mp4", ".mov", ".avi", ".mkv", ".webm",
    ".m4a", ".mp3", ".flac", ".aac", ".ogg",
}

MODEL_CACHE_DIR = Path(".cache/gigaam")
MAX_SHORTFORM_SEGMENT_SECONDS = 25

warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message="The given buffer is not writable, and PyTorch does not support non-writable tensors.*",
    category=UserWarning,
)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("segment должен быть больше 0")
    if parsed > MAX_SHORTFORM_SEGMENT_SECONDS:
        raise argparse.ArgumentTypeError(
            f"segment должен быть не больше {MAX_SHORTFORM_SEGMENT_SECONDS} секунд "
            "для short-form транскрибации через gigaam"
        )
    return parsed


def ensure_dependency(command: str) -> None:
    if shutil.which(command):
        return
    print(f"❌ Команда не найдена: {command}")
    sys.exit(1)


def ensure_python_dependency(module_name: str, package_name: str) -> None:
    try:
        __import__(module_name)
    except ImportError:
        print(f"❌ Python-модуль не найден: {module_name}")
        print(f"Установите пакет: pip install {package_name}")
        sys.exit(1)


def run_command(
    command: list[str],
    capture_output: bool = False,
    input_text: str | None = None,
) -> subprocess.CompletedProcess:
    try:
        kwargs = {
            "check": True,
            "text": True,
        }
        if input_text is not None:
            kwargs["input"] = input_text
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

    text = re.sub(r'Audio:\s*\d+\.?\s*0?s?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+[a-zA-Zа-яА-ЯёЁ-]{2,7}-\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\[\d{2}:\d{2}:\d{2},\d{3} -> \d{2}:\d{2}:\d{2},\d{3}\]', '', text)
    text = re.sub(r'Loading audio:.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Split into \d+ chunks', '', text)
    text = re.sub(r'Transcribed in .*', '', text)
    text = re.sub(r'Saved: .*', '', text)
    text = re.sub(r'tmp\w+_\w+', '', text)

    lines = [line.strip() for line in text.split('\n') if line.strip()]
    text = ' '.join(lines)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s+([.,!?])', r'\1', text)
    text = re.sub(r'([.!?])\s*', r'\1 ', text)

    return text.strip()


def collapse_repeated_words(text: str) -> str:
    words = text.split()
    if not words:
        return ""

    collapsed = [words[0]]
    for word in words[1:]:
        if word.casefold() == collapsed[-1].casefold():
            continue
        collapsed.append(word)

    deduped_pairs: list[str] = []
    index = 0
    while index < len(collapsed):
        if (
            index + 3 < len(collapsed)
            and collapsed[index].casefold() == collapsed[index + 2].casefold()
            and collapsed[index + 1].casefold() == collapsed[index + 3].casefold()
        ):
            deduped_pairs.extend(collapsed[index:index + 2])
            index += 4
            continue

        deduped_pairs.append(collapsed[index])
        index += 1

    return " ".join(deduped_pairs)


def looks_like_fragment(text: str) -> bool:
    if len(text) <= 30:
        return True

    words = text.split()
    if len(words) <= 4:
        return True

    last_word = words[-1]
    return last_word.isalpha() and len(last_word) <= 7


def normalize_transcript_chunks(
    transcript_chunks: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    normalized: list[tuple[str, str]] = []

    for timestamp, chunk_text in transcript_chunks:
        text = collapse_repeated_words(chunk_text)
        if not text:
            continue

        if normalized and len(text) <= 40 and looks_like_fragment(text):
            prev_timestamp, prev_text = normalized[-1]
            normalized[-1] = (prev_timestamp, f"{prev_text} {text}".strip())
            continue

        normalized.append((timestamp, text))

    return normalized


def write_transcript_file(
    output_file: str,
    model_name: str,
    segment_duration: int,
    transcript_chunks: list[tuple[str, str]],
) -> None:
    normalized_chunks = normalize_transcript_chunks(transcript_chunks)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("ТРАНСКРИПЦИЯ\n")
        f.write(f"Модель: {model_name} | Сегмент: {segment_duration} сек\n")
        f.write("=" * 70 + "\n\n")

        for ts, chunk_text in normalized_chunks:
            f.write(f"[{ts}]\n{chunk_text}\n\n")


def needs_conversion(file_path: str) -> bool:
    return Path(file_path).suffix.lower() in AUDIO_VIDEO_FORMATS_TO_CONVERT


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
    if not segments:
        print("❌ ffmpeg не создал ни одного сегмента")
        sys.exit(1)
    print(f"✅ Создано {len(segments)} сегментов")
    return [str(s) for s in segments]


def load_asr_model(model_type: str):
    import gigaam

    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return gigaam.load_model(model_type, download_root=str(MODEL_CACHE_DIR))


def transcribe_segment(segment_file: str, model) -> str:
    return model.transcribe(segment_file).strip()


def format_timestamp(seconds: int) -> str:
    return str(timedelta(seconds=seconds)).split('.')[0]


def build_ollama_prompt(transcript: str) -> str:
    return (
        "Сделай краткую и понятную сводку этого транскрипта на русском языке.\n"
        "Используй только информацию из транскрипта и не додумывай факты.\n"
        "Если в разговоре есть договорённости или следующие шаги, кратко выдели их отдельно.\n\n"
        f"Транскрипт:\n{transcript}"
    )


def run_ollama_postprocess(transcript: str, model: str) -> str:
    prompt = build_ollama_prompt(transcript)
    result = run_command(
        ["ollama", "run", model],
        capture_output=True,
        input_text=prompt,
    )
    return sanitize_ollama_output(result.stdout)


def sanitize_ollama_output(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return ""

    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = cleaned.replace("...done thinking.", "").strip()
    cleaned = re.sub(
        r"(?is)^thinking\.\.\..*?(?=^#\s+|^\*\*|^итоги\b|^сводка\b|^резюме\b|^ключевые тезисы\b|^action items\b)",
        "",
        cleaned,
        flags=re.MULTILINE,
    ).strip()
    cleaned = re.sub(
        r"(?is)^thinking process:.*?(?=^#\s+|^\*\*|^итоги\b|^сводка\b|^резюме\b|^ключевые тезисы\b|^action items\b)",
        "",
        cleaned,
        flags=re.MULTILINE,
    ).strip()

    heading_match = re.search(r"(?m)^#\s+.+", cleaned)
    if heading_match:
        return cleaned[heading_match.start():].strip()

    bold_heading_match = re.search(r"(?m)^\*\*[^*\n]+\*\*", cleaned)
    if bold_heading_match:
        return cleaned[bold_heading_match.start():].strip()

    summary_markers = [
        "Итоги",
        "Сводка",
        "Резюме",
        "Ключевые тезисы",
        "Action Items",
    ]
    for marker in summary_markers:
        marker_match = re.search(rf"(?m)^{re.escape(marker)}.*", cleaned)
        if marker_match:
            return cleaned[marker_match.start():].strip()

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    filtered_lines = []
    skip_prefixes = (
        "thinking",
        "thinking process",
        "analyze the request",
        "analyze the transcript",
        "key facts extracted",
        "wait,",
        "okay,",
        "let's",
        "i need to",
        "the transcript says",
        "candidate is",
        "company name:",
        "project names:",
        "team structure:",
        "salary:",
    )

    for line in lines:
        normalized = line.casefold()
        if normalized.startswith(skip_prefixes):
            continue
        if re.match(r"^\d+\.\s+\*\*.*\*\*:?$", line):
            continue
        if line.startswith("*   **") or line.startswith("**Wait"):
            continue
        filtered_lines.append(line)

    cleaned = "\n".join(filtered_lines).strip()
    if cleaned:
        return cleaned

    return cleaned


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Путь к аудио/видео файлу")
    parser.add_argument("--model", choices=["rnnt", "ctc"], default="rnnt")
    parser.add_argument("--segment", type=positive_int, default=15)
    parser.add_argument("--output", help="Путь к выходному txt-файлу")
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Дополнительно обработать итоговый транскрипт через Ollama и сохранить *_summary.txt",
    )
    parser.add_argument(
        "--ollama-model",
        default="ministral-3:8b",
        help="Локальная модель Ollama для постобработки, например ministral-3:8b",
    )
    args = parser.parse_args()
    ensure_dependency("ffmpeg")
    ensure_python_dependency("gigaam", "gigaam")
    if args.summary:
        ensure_dependency("ollama")

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"❌ Файл не найден: {args.input_file}")
        sys.exit(1)

    wav_file = None
    segments = []
    asr_model = None

    try:
        if needs_conversion(args.input_file):
            wav_file = convert_to_wav(args.input_file)
        else:
            wav_file = args.input_file

        segments = split_wav(wav_file, args.segment)
        print(f"📦 Загружаю модель GigaAM ({args.model.upper()})...")
        asr_model = load_asr_model(args.model)
        print("✅ Модель загружена")

        output_file = args.output or (input_path.stem + "_транскрипция.txt")
        transcript_chunks: list[tuple[str, str]] = []

        print(f"\n📝 Транскрипция ({args.model.upper()})...\n")
        write_transcript_file(output_file, args.model.upper(), args.segment, transcript_chunks)

        current_time = 0
        for seg in segments:
            ts = format_timestamp(current_time)
            print(f"[{ts}] ", end="", flush=True)

            raw = transcribe_segment(seg, asr_model)
            cleaned = clean_text(raw)

            if cleaned:
                print(cleaned[:120] + "..." if len(cleaned) > 120 else cleaned)
                transcript_chunks.append((ts, cleaned))
                write_transcript_file(
                    output_file,
                    args.model.upper(),
                    args.segment,
                    transcript_chunks,
                )
            else:
                print("(пусто)")

            current_time += args.segment

        transcript_chunks = normalize_transcript_chunks(transcript_chunks)

        print(f"\n✅ Готово! Результат сохранён в: {output_file}")

        if args.summary and transcript_chunks:
            print(f"🧠 Постобработка через Ollama ({args.ollama_model})...")
            llm_result = run_ollama_postprocess(
                transcript="\n\n".join(
                    f"[{ts}]\n{chunk_text}" for ts, chunk_text in transcript_chunks
                ),
                model=args.ollama_model,
            )
            llm_output_path = input_path.with_name(f"{input_path.stem}_summary.txt")
            with open(llm_output_path, "w", encoding="utf-8") as f:
                f.write("LLM POST-PROCESSING\n")
                f.write(
                    f"Модель распознавания: {args.model.upper()} | "
                    f"Ollama: {args.ollama_model}\n"
                )
                f.write("=" * 70 + "\n\n")
                f.write(llm_result + "\n")

            print(f"✅ LLM-результат сохранён в: {llm_output_path}")

    finally:
        if wav_file and wav_file != args.input_file:
            Path(wav_file).unlink(missing_ok=True)
        
        for seg in segments:
            Path(seg).unlink(missing_ok=True)
            Path(seg).with_suffix(".txt").unlink(missing_ok=True)

        print("🧹 Все временные файлы удалены")


if __name__ == "__main__":
    main()
