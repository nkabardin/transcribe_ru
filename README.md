# transcribe

CLI tool for transcribing Russian audio files to text using Whisper Large V3 Russian model.

[Русская версия ниже](#русский)

## Features

- **Zero setup**: Just run `./transcribe` — it creates venv, installs dependencies, offers to add to PATH
- **Any audio format**: MP3, WAV, OGG, M4A, FLAC, AAC, OPUS, etc. (anything ffmpeg can read)
- **Smart chunking**: Uses Voice Activity Detection (VAD) to split audio at natural pauses
- **Russian-optimized**: Uses [antony66/whisper-large-v3-russian](https://huggingface.co/antony66/whisper-large-v3-russian) model (WER 6.39 vs 9.84 for base Whisper)
- **Multiple output formats**: txt, markdown, json, srt

## Requirements

- Python 3.10+
- ffmpeg and sox (`brew install ffmpeg sox`)

## Usage

```bash
# First run — auto-setup
./transcribe

# Transcribe a file (VAD enabled by default)
./transcribe recording.mp3

# Specify output file
./transcribe audio.m4a -o result.txt

# Output as markdown
./transcribe podcast.ogg -f md

# Disable VAD, use fixed 30s chunks
./transcribe long_file.wav --no-vad

# Batch process a directory
python src/batch.py ./recordings --skip
```

## How It Works

### 1. Audio Conversion
```
Input (any format) → ffmpeg → 48kHz WAV → sox (normalize) → 16kHz mono WAV
```
Sox applies companding and normalization to improve speech clarity.

### 2. Voice Activity Detection (VAD)
Uses [webrtcvad](https://github.com/wiseman/py-webrtcvad) to detect speech segments:
- Processes audio in 30ms frames
- Groups speech segments into chunks (max 30 seconds)
- Splits at natural pauses (silence > 0.5s)

This produces better transcription than fixed-size chunks because Whisper gets complete phrases.

### 3. Transcription
Each chunk is processed by Whisper Large V3 Russian:
- Loads audio as float32 numpy array
- Extracts mel spectrogram features
- Generates text tokens autoregressively
- Decodes to Russian text

### 4. Output
Results are concatenated and saved in the requested format.

## CLI Options

```
./transcribe <file> [options]

Options:
  -o, --output PATH      Output file path
  -f, --format FORMAT    Output format: txt, md, json, srt (default: txt)
  --no-vad               Disable VAD, use fixed chunks
  --chunk-size N         Chunk size in seconds (default: 30)
  --overlap N            Chunk overlap in seconds (default: 5)
  --model MODEL          Whisper model ID
  -q, --quiet            Suppress progress output
  --keep-temp            Keep temporary files
  --version              Show version
```

## Wrapper Commands

```bash
./transcribe --setup    # Full setup (venv, deps, PATH)
./transcribe --update   # Update dependencies
./transcribe --path     # Configure PATH
```

---

# Русский

CLI для транскрипции русскоязычных аудиофайлов в текст с использованием Whisper Large V3 Russian.

## Возможности

- **Без настройки**: Просто запусти `./transcribe` — создаст venv, установит зависимости, предложит добавить в PATH
- **Любой формат**: MP3, WAV, OGG, M4A, FLAC, AAC, OPUS и др. (всё что читает ffmpeg)
- **Умное разбиение**: Использует VAD для разбивки аудио по паузам
- **Оптимизирован для русского**: Модель [antony66/whisper-large-v3-russian](https://huggingface.co/antony66/whisper-large-v3-russian) (WER 6.39 против 9.84 у базового Whisper)
- **Форматы вывода**: txt, markdown, json, srt

## Требования

- Python 3.10+
- ffmpeg и sox (`brew install ffmpeg sox`)

## Использование

```bash
# Первый запуск — автонастройка
./transcribe

# Транскрибировать файл (VAD включён по умолчанию)
./transcribe recording.mp3

# Указать выходной файл
./transcribe audio.m4a -o result.txt

# Вывод в markdown
./transcribe podcast.ogg -f md

# Отключить VAD, использовать фиксированные чанки по 30с
./transcribe long_file.wav --no-vad

# Batch обработка директории
python src/batch.py ./recordings --skip
```

## Как это работает

### 1. Конвертация аудио
```
Вход (любой формат) → ffmpeg → 48kHz WAV → sox (нормализация) → 16kHz mono WAV
```
Sox применяет компандирование и нормализацию для улучшения разборчивости речи.

### 2. Детекция голосовой активности (VAD)
Использует [webrtcvad](https://github.com/wiseman/py-webrtcvad) для определения сегментов речи:
- Обрабатывает аудио фреймами по 30мс
- Группирует сегменты речи в чанки (макс. 30 секунд)
- Разбивает по естественным паузам (тишина > 0.5с)

Это даёт лучшую транскрипцию чем фиксированные чанки, потому что Whisper получает законченные фразы.

### 3. Транскрипция
Каждый чанк обрабатывается Whisper Large V3 Russian:
- Загружает аудио как float32 numpy массив
- Извлекает mel-спектрограмму
- Генерирует текстовые токены авторегрессивно
- Декодирует в русский текст

### 4. Вывод
Результаты объединяются и сохраняются в запрошенном формате.

## Опции CLI

```
./transcribe <файл> [опции]

Опции:
  -o, --output PATH      Путь выходного файла
  -f, --format FORMAT    Формат: txt, md, json, srt (по умолчанию: txt)
  --no-vad               Отключить VAD, фиксированные чанки
  --chunk-size N         Размер чанка в секундах (по умолчанию: 30)
  --overlap N            Перекрытие чанков в секундах (по умолчанию: 5)
  --model MODEL          ID модели Whisper
  -q, --quiet            Тихий режим
  --keep-temp            Сохранить временные файлы
  --version              Показать версию
```

## Команды обёртки

```bash
./transcribe --setup    # Полная настройка (venv, зависимости, PATH)
./transcribe --update   # Обновить зависимости
./transcribe --path     # Настроить PATH
```

## License

MIT
