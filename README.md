[English](#english) · [中文](#chinese) · [日本語](#japanese)

<a id="english"></a>

# Video Knowledge Agent

Turn video or audio into searchable knowledge with timestamped evidence. The pipeline transcribes speech, corrects the transcript, groups it into semantic chapters, extracts structured knowledge, and builds a searchable index. It can inspect short video clips and run OCR when an answer depends on the picture. Groq and DeepSeek API modes and local model modes are available.

## Architecture

```text
Video or audio file -> FFmpeg audio extraction -> Groq or local ASR
                    -> DeepSeek or local text -> semantic chapters
                    -> structured knowledge and timestamped Moments
                    -> publication and SQLite FTS5 search
Video source + relevant Moment -> short clip -> frames/OCR (on demand)
```

The `process` command accepts a video or audio file path. A separate acquisition workflow can supply that file when starting from a URL; URL fetching is not part of this repository. Without provenance metadata, `source_url` is an internal `local-video://` or `local-audio://` timestamp reference. A neighboring `.info.json` sidecar can preserve an original webpage citation.

## Installation and usage

Requires Python 3.11+ and either FFmpeg or `imageio-ffmpeg`, which supplies an FFmpeg executable. The following PowerShell commands run independently of Codex from the repository root. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) first, or use an existing Python 3.11+ environment with `python -m pip install -e ".[groq,openai,media,ocr]"`. A fresh clone contains no video or knowledge data; process your own file before asking questions.

```powershell
uv venv .venv --python 3.11
uv pip install --python .venv\Scripts\python.exe -e ".[groq,openai,media,ocr]"
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
# Edit .env: choose providers and set the keys required by API modes.
.\.venv\Scripts\video-knowledge.exe process "D:\videos\my-video.mp4"
.\.venv\Scripts\video-knowledge.exe answer "Where is the shop?" --answer-language en
```

Copy `.env.example` to the project-root `.env` and choose the model providers there. Groq and DeepSeek keys are required only for their respective API modes. The code reads `.env` directly; an explicit API-key argument takes precedence, and process environment variables remain a fallback. Never commit real keys. `process` builds the searchable index, so `answer` can run immediately afterward. The answer command also works on previously indexed videos without reprocessing them.

Set `ASR_PROVIDER=groq|local` and `TEXT_PROVIDER=deepseek|local` independently in `.env`. The API defaults use Groq Whisper Turbo and DeepSeek Flash. Local ASR uses faster-whisper (`LOCAL_ASR_MODEL`, `LOCAL_ASR_DEVICE`, `LOCAL_ASR_COMPUTE_TYPE`); local text uses an Ollama model named in `LOCAL_TEXT_MODEL` through the loopback `LOCAL_TEXT_BASE_URL`. Install the selected dependencies and pull/start the local models before processing. For example, set both providers to `local` and run `ollama pull <your-model>` followed by `ollama serve`. Local text mode never falls back to DeepSeek if its server or model is missing. Switching providers or models invalidates the affected cached stages. The extra Groq review of uncertain Japanese ASR segments runs only with Groq ASR.

`process` supports `--language zh|en|ja`, `--output-language source`, repeated `--term` options, and `--force-stage`. Running it again on the same file reuses completed transcription, proofreading, chapter, and knowledge artifacts. If the source changes, its fingerprint produces a new `video_id`, preventing stale results from being reused. To normalize audio only:

```powershell
.\.venv\Scripts\video-knowledge.exe prepare-audio "D:\videos\my-video.mp4"
```

Run data is stored under `data/raw/`, `data/work/`, `data/outputs/`, and `data/index/`. Video, audio, databases, and environment files are excluded by `.gitignore`. Before publishing to a public Git repository, inspect `git status` and `git diff --cached`; do not force-add media or keys.

## Retrieval and visual evidence

```powershell
.\.venv\Scripts\video-knowledge.exe search "your question"          # Free local full-text search
.\.venv\Scripts\video-knowledge.exe answer "Where is the shop?" --answer-language en
.\.venv\Scripts\video-knowledge.exe answer "What is shown on the shop screen?" --answer-language en --visual-mode auto
```

Chapters, structured knowledge, and timestamped Moments are indexed with SQLite FTS5. `search` needs no model or API. `answer` retrieves nearby corrected speech and uses the selected text provider to write an answer with checked evidence links (Chinese by default; use `--answer-language en|ja` for English or Japanese). For visual questions it extracts only the matching video window, adds time-aligned OCR, and can send up to two selected frames to DeepSeek Flash when the question requires direct visual judgment. Use `--visual-mode off|ocr|frames` to override automatic selection and `--format json` for programmatic output and usage estimates. Each answer call may incur API charges in DeepSeek mode.

PowerShell prints image paths rather than displaying images inline. To open the screenshots used in an answer:

```powershell
$result = .\.venv\Scripts\video-knowledge.exe answer "What is shown on the shop screen?" --answer-language en --visual-mode auto --format json | ConvertFrom-Json
$result.answer_markdown
$result.evidence | Where-Object image_path | ForEach-Object { Invoke-Item $_.image_path }
```

If the indexed source contains only audio, no screenshot can be produced. A local video can provide screenshots; an audio-only URL acquisition cannot. Internal `local-video://` links need a player or an upstream agent to resolve them for playback. For low-dependency text-only Q&A against an existing index, install `.[openai]` and add `--visual-mode off`; Groq, media, and OCR extras are needed only for their respective stages. Audio transcription uses 120-second core chunks with about five seconds of context at each end. Stored knowledge remains in the source language.

## Usage notice and disclaimer

Use this tool only with media you are authorized to process. You are responsible for copyright, platform terms, privacy, and API charges. In API mode, audio is sent to Groq for transcription and text to DeepSeek for analysis. In local mode, the configured faster-whisper and loopback Ollama models process them on your computer. Check the selected providers before processing sensitive content. Do not use it to bypass DRM or access restrictions, and do not redistribute source media, clips, or substantial transcript text without the necessary rights. ASR and model-generated summaries may be inaccurate; verify important claims against the cited source and timestamp. The software is provided as-is, without legal advice or a guarantee of accuracy or availability. Keep `.env` private, and rotate any key that is exposed.

For visual answers, up to two selected frames may also be sent to DeepSeek Flash. OCR text and generated answers can be wrong; check the linked frame and source when accuracy matters.

## Verification

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

---

<a id="chinese"></a>

# Video Knowledge Agent

把视频或音频转成可检索、可追溯时间位置的知识。流程包括语音转写、文本校对、按语义划分章节、提取结构化知识和建立检索索引；当答案依赖画面时，还能截取相关片段并运行 OCR。模型处理可选 Groq 与 DeepSeek API，也可选本机模型。

## 分层

```text
视频或音频文件 → FFmpeg 抽取音频 → Groq 或本机 ASR
               → DeepSeek 或本机文字处理 → 语义章节
               → 结构化知识与带时间戳的 Moment
               → 发布与 SQLite FTS5 检索
视频源 + 命中的 Moment → 截取短片 → 关键帧/OCR（按需）
```

`process` 命令接收视频或音频文件路径。若从 URL 开始，可由独立的获取流程先提供文件；本仓库不包含 URL 获取功能。没有来源元数据时，`source_url` 是内部的 `local-video://` 或 `local-audio://` 时间标识；相邻的 `.info.json` 文件可保留原网页引用。

## 安装与使用

要求 Python 3.11+，安装 FFmpeg 或安装提供 FFmpeg 可执行文件的 `imageio-ffmpeg`。以下 PowerShell 命令可在仓库根目录独立运行，无需 Codex。先安装 [uv](https://docs.astral.sh/uv/getting-started/installation/)；已有 Python 3.11+ 环境也可使用 `python -m pip install -e ".[groq,openai,media,ocr]"`。新克隆的仓库不含视频和知识数据，须先处理自己的文件，再进行问答。

```powershell
uv venv .venv --python 3.11
uv pip install --python .venv\Scripts\python.exe -e ".[groq,openai,media,ocr]"
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
# 在本机编辑 .env：选择模式，并填写所用 API 模式的密钥。
.\.venv\Scripts\video-knowledge.exe process "D:\videos\my-video.mp4"
.\.venv\Scripts\video-knowledge.exe answer "商店在哪条路？"
```

将 `.env.example` 复制为项目根目录的 `.env`，在其中选择模型来源；只有使用对应 API 模式时才需要填写 Groq 或 DeepSeek 密钥。程序会直接读取该文件；显式传入的密钥优先，进程环境变量作为后备。切勿提交真实密钥。`process` 会建立检索索引，随后即可运行 `answer`；已有索引的视频无需重新处理。

在 `.env` 中分别设置 `ASR_PROVIDER=groq|local` 和 `TEXT_PROVIDER=deepseek|local`，两项可以混用。默认 API 模式使用 Groq Whisper Turbo 与 DeepSeek Flash。本地 ASR 使用 faster-whisper，并由 `LOCAL_ASR_MODEL`、`LOCAL_ASR_DEVICE`、`LOCAL_ASR_COMPUTE_TYPE` 控制；本地文字处理通过 `LOCAL_TEXT_BASE_URL` 连接本机 Ollama，模型名填写在 `LOCAL_TEXT_MODEL`。使用前需安装相应依赖、下载模型并启动服务，例如 `ollama pull <模型名>`、`ollama serve`。本地服务或模型不可用时会报错，不会自动回退到 DeepSeek。切换模型或来源会重跑受影响的缓存阶段。日语 ASR 的额外 Groq 复核仅在 Groq ASR 模式运行。

`process` 支持 `--language zh|en|ja`、`--output-language source`、重复的 `--term` 和 `--force-stage`。重跑同一文件会复用已完成的转写、校对、章节与知识文件。源文件如果改变，文件指纹产生新的 `video_id`，避免把旧结果误当成新视频。仅统一音频格式可执行：

```powershell
.\.venv\Scripts\video-knowledge.exe prepare-audio "D:\videos\my-video.mp4"
```

运行数据保存在 `data/raw/`、`data/work/`、`data/outputs/` 和 `data/index/`；视频文件、音频、数据库、环境文件均被 `.gitignore` 排除。发布到公开 Git 前应检查 `git status` 和 `git diff --cached`，不要强行加入媒体或密钥。

## 检索与画面证据

```powershell
.\.venv\Scripts\video-knowledge.exe search "要查的问题"          # 免费的本地全文检索
.\.venv\Scripts\video-knowledge.exe answer "商店在哪条路？"
.\.venv\Scripts\video-knowledge.exe answer "商店画面显示了什么？" --visual-mode auto
```

章节、结构化知识及带时间戳的 Moment 进入 SQLite FTS5 索引。`search` 不调用模型或 API。`answer` 检索校对后的语音上下文，用当前文字模型生成中文回答，时间链接由程序核对生成。画面问题只截取命中的时间窗，加入对齐的 OCR；确实要判断画面时，可向 DeepSeek Flash 发送最多两张关键帧。可用 `--visual-mode off|ocr|frames` 覆盖自动判断，`--format json` 返回结构化结果和用量估算。DeepSeek 模式下每次回答可能产生 API 费用。

PowerShell 不会在终端内直接显示截图。以下命令可打开答案使用的截图：

```powershell
$result = .\.venv\Scripts\video-knowledge.exe answer "商店画面显示了什么？" --visual-mode auto --format json | ConvertFrom-Json
$result.answer_markdown
$result.evidence | Where-Object image_path | ForEach-Object { Invoke-Item $_.image_path }
```

只有音频的来源无法生成截图；本地视频可以提供画面，而仅获取音频的 URL 流程不能。内部 `local-video://` 时间引用需要播放器或上层 Agent 解析才能跳转。若已有索引且只需文字问答，可仅安装 `.[openai]` 并加 `--visual-mode off`；Groq、媒体和 OCR 依赖仅在使用相应阶段时需要。音频转写默认切成 120 秒核心片段，两端各留约 5 秒上下文；知识文件默认保持源语言。

## 使用声明与免责声明

仅处理你有权使用的素材；版权、平台条款、隐私和 API 费用由使用者自行核对与承担。API 模式会将音频发送给 Groq 转写、文字发送给 DeepSeek 分析；本地模式使用本机的 faster-whisper 和回环地址上的 Ollama。处理敏感内容前应确认所选模式。请勿用它绕过 DRM 或访问限制；未取得相应权利时，不要再分发原媒体、片段或大段转写文本。ASR 与模型总结可能出错，重要结论须回看原始素材及所附时间戳。本软件按现状提供，不构成法律意见，也不保证准确性或服务可用性。请妥善保管 `.env`；密钥泄露后应及时轮换。

画面问答还可能向 DeepSeek Flash 发送最多两张关键帧。OCR 和生成的回答都可能出错，重要结论请核对所附截图与来源。

## 验证

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

---

<a id="japanese"></a>

# Video Knowledge Agent

動画や音声を、時刻付きの根拠をたどれる検索可能な知識に変換します。音声の書き起こし、テキストの校正、内容に基づく章分け、構造化知識の抽出、検索インデックスの作成までを行います。映像に答えの根拠がある場合は、該当部分を切り出して OCR で確認できます。Groq・DeepSeek の API とローカルモデルを選択できます。

## 構成

```text
動画または音声ファイル → FFmpeg による音声抽出 → Groq またはローカル ASR
                     → DeepSeek またはローカルのテキスト処理 → 内容に基づく章分け
                     → 構造化知識と時刻付き Moment → 公開用データ・SQLite FTS5 検索
動画の入力元 + 該当 Moment → 短いクリップ → フレーム/OCR（必要な場合のみ）
```

`process` コマンドは動画または音声ファイルのパスを受け取ります。URL から始める場合は、別の取得処理でファイルを用意できます。このリポジトリには URL からの取得機能は含まれません。出典情報がない場合、`source_url` には `local-video://` または `local-audio://` の内部参照を使います。隣接する `.info.json` ファイルがあれば、元の Web ページへの参照を保持できます。

## インストールと使い方

Python 3.11 以降、および FFmpeg か FFmpeg 実行ファイルを提供する `imageio-ffmpeg` が必要です。以下の PowerShell コマンドはリポジトリのルートから Codex なしで実行できます。まず [uv](https://docs.astral.sh/uv/getting-started/installation/) を導入してください。Python 3.11 以降がすでにある場合は `python -m pip install -e ".[groq,openai,media,ocr]"` でもインストールできます。クローン直後は動画も知識データも含まれないため、質問する前に自分のファイルを処理してください。

```powershell
uv venv .venv --python 3.11
uv pip install --python .venv\Scripts\python.exe -e ".[groq,openai,media,ocr]"
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
# .env で実行先を選び、API モードで必要なキーを設定します。
.\.venv\Scripts\video-knowledge.exe process "D:\videos\my-video.mp4"
.\.venv\Scripts\video-knowledge.exe answer "店はどの通りにありますか？" --answer-language ja
```

`.env.example` をプロジェクト直下の `.env` にコピーし、モデルの実行先を選びます。Groq と DeepSeek のキーは、それぞれの API モードを使う場合だけ必要です。コードは `.env` を直接読み込みます。明示的に渡したキーが優先され、プロセス環境変数はフォールバックとして使われます。実際のキーを Git にコミットしないでください。`process` が検索インデックスを作成するので、続けて `answer` を実行できます。インデックス登録済みの動画を再処理する必要はありません。

`.env` の `ASR_PROVIDER=groq|local` と `TEXT_PROVIDER=deepseek|local` は別々に選べます。標準の API モードは Groq Whisper Turbo と DeepSeek Flash を使用します。ローカル ASR は faster-whisper を使い、`LOCAL_ASR_MODEL`・`LOCAL_ASR_DEVICE`・`LOCAL_ASR_COMPUTE_TYPE` で設定します。ローカルの文章処理は `LOCAL_TEXT_BASE_URL` のループバック Ollama サーバーを使い、`LOCAL_TEXT_MODEL` にモデル名を指定します。必要な依存関係とモデルを導入し、`ollama pull <モデル名>`、`ollama serve` などで準備してください。ローカルモデルが使えない場合はエラーになり、DeepSeek へ自動的に切り替わりません。実行先やモデルの変更時は影響を受けるキャッシュ工程を再実行します。日本語 ASR の追加 Groq 確認は Groq ASR モードでのみ実行します。

`process` では `--language zh|en|ja`、`--output-language source`、複数の `--term`、`--force-stage` を指定できます。同じファイルを再実行すると、完了済みの成果物を再利用します。元ファイルが変わると新しい `video_id` を生成し、古い結果の誤用を防ぎます。音声形式だけを統一する場合：

```powershell
.\.venv\Scripts\video-knowledge.exe prepare-audio "D:\videos\my-video.mp4"
```

実行データは `data/raw/`、`data/work/`、`data/outputs/`、`data/index/` に保存されます。動画、音声、データベース、環境設定ファイルは `.gitignore` で除外します。公開 Git リポジトリへアップロードする前に `git status` と `git diff --cached` を確認し、メディアやキーを強制追加しないでください。

## 検索と映像の根拠

```powershell
.\.venv\Scripts\video-knowledge.exe search "質問文"          # API 不要のローカル全文検索
.\.venv\Scripts\video-knowledge.exe answer "店はどの通りにありますか？" --answer-language ja
.\.venv\Scripts\video-knowledge.exe answer "店の画面には何が表示されていますか？" --answer-language ja --visual-mode auto
```

章、構造化知識、時刻付き Moment は SQLite FTS5 に登録されます。`search` はモデルも API も使いません。`answer` は校正済み音声の前後関係を検索し、選択したテキストモデルで根拠リンク付きの回答を作成します（既定は中国語。英語・日本語は `--answer-language en|ja`）。映像についての質問では該当時間帯だけを切り出して OCR を照合し、画面そのものの判断が必要なら DeepSeek Flash に最大 2 枚のキーフレームを送れます。`--visual-mode off|ocr|frames` で自動判定を変更でき、`--format json` で構造化結果と使用量の推定を確認できます。DeepSeek モードの回答ごとに API 料金が発生する場合があります。

PowerShell は画像をターミナル内に表示しません。回答で使ったスクリーンショットを開くには：

```powershell
$result = .\.venv\Scripts\video-knowledge.exe answer "店の画面には何が表示されていますか？" --answer-language ja --visual-mode auto --format json | ConvertFrom-Json
$result.answer_markdown
$result.evidence | Where-Object image_path | ForEach-Object { Invoke-Item $_.image_path }
```

音声のみを取得した入力からはスクリーンショットを生成できません。ローカル動画には映像が含まれますが、音声だけを取得する URL 処理には映像がありません。内部の `local-video://` 時刻参照を再生するには、プレーヤーまたは上位のエージェントによる解決が必要です。登録済みデータに対するテキスト問答だけなら、`.[openai]` のみを導入し、`--visual-mode off` を指定できます。Groq・メディア・OCR の依存関係は、それぞれの工程を使う場合に必要です。音声は既定で 120 秒の主要区間に分け、各区間の前後に約 5 秒の文脈を付けてアップロードします。保存する知識は既定で原語のまま保持します。

## 利用上の注意と免責事項

権利を持つ素材だけを処理してください。著作権、プラットフォームの規約、プライバシー、API 料金の確認と負担は利用者の責任です。API モードでは音声を Groq、テキストを DeepSeek に送信します。ローカルモードでは faster-whisper とループバック接続の Ollama を使います。機密情報を処理する前に選択したモードを確認してください。DRM やアクセス制限の回避に使わず、必要な権利なしに元メディア、クリップ、長い書き起こしを再配布しないでください。ASR やモデルの要約には誤りがあり得るため、重要な内容は元資料とタイムスタンプで確認してください。本ソフトウェアは現状有姿で提供され、法律上の助言、正確性、サービスの継続性を保証しません。`.env` を非公開に保ち、キーが漏えいした場合は更新してください。

映像についての回答では、選択したキーフレームを最大 2 枚まで DeepSeek Flash に送信する場合があります。OCR と生成された回答には誤りがあり得るため、重要な内容は添付画像と元資料で確認してください。

## 検証

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m unittest discover -s tests
```
