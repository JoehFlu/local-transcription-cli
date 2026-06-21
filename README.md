# Local Transcription CLI

Локальная транскрибация аудио и видео через `ffmpeg` и `GigaAM` с опциональной
локальной сводкой через `Ollama`. Версия предназначена для macOS на Apple
Silicon.

## Возможности

- `mp4`, `mov`, `avi`, `mkv`, `webm`, `m4a`, `mp3`, `wav`, `flac`, `aac`, `ogg`;
- модели GigaAM `rnnt` и `ctc`;
- автоматический поиск единственного медиафайла в текущей папке;
- настраиваемая длина сегмента и текстовый результат с таймкодами;
- локальная сводка и action items через Ollama по флагу `--summary`;
- локальный кэш весов GigaAM в `.cache/gigaam`.

Распознавание речи выполняет GigaAM. Ollama не отправляет транскрипт в облако и
используется только для дополнительной сводки.

## Требования для macOS

- macOS на Apple Silicon;
- Python 3.11;
- Homebrew;
- `ffmpeg`;
- Ollama — только если нужна сводка.

Python 3.14 пока лучше не использовать: зависимости GigaAM/MLX могут быть с ним
несовместимы.

## Установка на macOS

```bash
brew install python@3.11 ffmpeg
brew install --cask ollama

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

Проверка:

```bash
python -c "import gigaam; print('gigaam ok')"
ffmpeg -version
```

Для локальной сводки запустите приложение Ollama и загрузите модель:

```bash
ollama pull ministral-3:3b
ollama list
```

## Использование

Если в текущей папке ровно один медиафайл:

```bash
source .venv/bin/activate
python transcribe.py
```

Или укажите файл явно:

```bash
python transcribe.py interview.m4a
python transcribe.py interview.mkv --model rnnt --segment 15
python transcribe.py interview.mp3 --output result.txt
```

Транскрибация и локальная сводка через Ollama:

```bash
python transcribe.py interview.m4a --summary
python transcribe.py interview.m4a --summary --ollama-model ministral-3:3b
```

По умолчанию создаются:

- `<имя>_transcript.txt`;
- `<имя>_summary.txt`, если указан `--summary`.

## Модели GigaAM

При первом запуске веса скачиваются в `.cache/gigaam`. Если автоматическая
загрузка недоступна, скачайте модель вручную:

- RNNT: `https://cdn.chatwm.opensmodel.sberdevices.ru/GigaAM/v2_rnnt.ckpt`
- CTC: `https://cdn.chatwm.opensmodel.sberdevices.ru/GigaAM/v2_ctc.ckpt`

Файл нужно сохранить как `.cache/gigaam/v2_rnnt.ckpt` или
`.cache/gigaam/v2_ctc.ckpt`.

## Ограничения

- Для short-form GigaAM сегмент ограничен 25 секундами; обычно хорошо работает
  значение 15 секунд.
- Таймкоды считаются по размеру сегмента.
- Скорость зависит от длительности записи, модели и доступных ресурсов.
- Ollama должна быть запущена, а выбранная модель — заранее загружена локально.

## License

MIT. Сторонние зависимости, `ffmpeg` и веса моделей распространяются на своих
условиях.
