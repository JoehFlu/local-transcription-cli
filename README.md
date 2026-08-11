# Local Transcription CLI

Единая локальная транскрибация аудио и видео для macOS на Apple Silicon и
Windows через WSL 2 с Ubuntu. Распознавание выполняют `ffmpeg` и `GigaAM`, а
опциональную локальную сводку — `Ollama`.

Транскрипты и сводки обрабатываются локально и не отправляются в облако.

## Возможности

- `mp4`, `mov`, `avi`, `mkv`, `webm`, `m4a`, `mp3`, `wav`, `flac`, `aac`, `ogg`;
- модели GigaAM `rnnt` и `ctc`;
- автоматическая последовательная обработка всех медиафайлов в папке;
- обработка одного или нескольких явно указанных файлов;
- настраиваемая длина сегмента и текстовый результат с таймкодами;
- локальная сводка и action items через Ollama по флагу `--summary`;
- локальный кэш весов GigaAM в `.cache/gigaam`.

## Поддерживаемые платформы

- macOS на Apple Silicon;
- Windows 10/11 через WSL 2 с Ubuntu;
- обычный Linux использует тот же путь установки, что и WSL.

Нативный запуск из PowerShell или `cmd` не поддерживается: в Windows команды
проекта нужно выполнять внутри терминала WSL.

Рекомендуется Python 3.11. Python 3.14 пока лучше не использовать из-за
возможной несовместимости зависимостей GigaAM/MLX.

## Клонирование

```bash
git clone https://github.com/JoehFlu/local-transcription-cli.git
cd local-transcription-cli
```

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

## Установка в Windows через WSL 2

Сначала откройте PowerShell от имени администратора и установите Ubuntu:

```powershell
wsl --install -d Ubuntu
```

После перезагрузки откройте Ubuntu из меню Windows. Все следующие команды
выполняются уже внутри WSL:

```bash
sudo apt update
sudo apt install -y ffmpeg python3.11 python3.11-venv python3-pip curl

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

Если `python3.11` отсутствует в репозитории вашей версии Ubuntu, установите
Python 3.11 привычным для дистрибутива способом и повторите создание `.venv`.

Проверка:

```bash
python -c "import gigaam; print('gigaam ok')"
ffmpeg -version
```

Для лучшей производительности храните проект и медиафайлы в файловой системе
WSL, например в `~/local-transcription-cli`, а не в `/mnt/c/...`.

## Локальная сводка через Ollama

Ollama нужна только для флага `--summary`.

На macOS запустите установленное приложение Ollama. В WSL установите Linux-
версию и запустите сервис:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve
```

В другом терминале загрузите модель и проверьте установку:

```bash
ollama pull ministral-3:3b
ollama list
```

## Использование

Активируйте окружение:

```bash
source .venv/bin/activate
```

Если в текущей папке лежат медиафайлы, скрипт обработает их все по очереди:

```bash
python transcribe.py
```

Или укажите один или несколько файлов явно:

```bash
python transcribe.py interview.m4a
python transcribe.py video1.mp4 video2.mp4 video3.mp4
python transcribe.py interview.mkv --model rnnt --segment 15
python transcribe.py interview.mp3 --output result.txt
```

Транскрибация и локальная сводка через Ollama:

```bash
python transcribe.py interview.m4a --summary
python transcribe.py interview.m4a --summary --ollama-model ministral-3:3b
python transcribe.py video1.mp4 video2.mp4 --summary
```

По умолчанию создаются:

- `<имя>_transcript.txt`;
- `<имя>_summary.txt`, если указан `--summary`.

Флаг `--output` можно использовать только при обработке одного файла, чтобы
несколько транскриптов случайно не записались в один путь.

## Модели GigaAM

При первом запуске веса скачиваются в `.cache/gigaam`. Если автоматическая
загрузка недоступна, скачайте модель вручную:

- RNNT: `https://cdn.chatwm.opensmodel.sberdevices.ru/GigaAM/v2_rnnt.ckpt`;
- CTC: `https://cdn.chatwm.opensmodel.sberdevices.ru/GigaAM/v2_ctc.ckpt`.

Файл нужно сохранить как `.cache/gigaam/v2_rnnt.ckpt` или
`.cache/gigaam/v2_ctc.ckpt`.

## Проверка проекта

Быстрые тесты не загружают модели и не требуют медиафайлов:

```bash
python -m unittest discover -s tests -v
```

Та же проверка автоматически запускается в GitHub Actions на macOS и Ubuntu.

## Ограничения

- Для short-form GigaAM сегмент ограничен 25 секундами; обычно хорошо работает
  значение 15 секунд.
- Таймкоды считаются по размеру сегмента.
- Скорость зависит от длительности записи, модели и доступных ресурсов.
- Ollama должна быть запущена, а выбранная модель — заранее загружена локально.

## License

MIT. Сторонние зависимости, `ffmpeg` и веса моделей распространяются на своих
условиях.
