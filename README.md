[English](#english) · [中文](#chinese) · [日本語](#japanese)

<a id="english"></a>

# Video Knowledge Agent — local-video-first core

This repository processes only **media files already available on your computer that you are authorized to use**. Its input can be a local video or audio file path; it does not resolve web pages, search for videos, or download them automatically. The core can still make network requests to the Groq ASR and DeepSeek text APIs. “Local” describes the media source, not fully offline inference.

## Architecture

```text
Optional external acquisition tool (not in this repository) -> local audio file
                                                            |
                                                process <file path>
                                                            |
                            FFmpeg audio extraction -> Groq or local ASR
                              -> DeepSeek or local text -> chapters/knowledge
                              -> Moments -> publication/FTS5 index
                                                            |
                 With a local video only: extract a short clip -> frames/OCR
```

The interface between an external acquisition tool and the core is **a local file path**. The same core can process your own recordings, licensed material, videos produced by another project, or online material you have obtained lawfully. Keeping this interface does not cause the core to download anything. Without provenance metadata, `source_url` is an internal `local-video://` or `local-audio://` timestamp reference, not a browser-openable URL. A neighboring `.info.json` sidecar can preserve an original webpage citation without causing any download.

## Installation and usage

Requires Python 3.11+ and either FFmpeg or `imageio-ffmpeg`, which supplies an FFmpeg executable. Copy `.env.example` to the project-root `.env` and choose the model providers there. Groq and DeepSeek keys are required only for their respective API modes. The code reads `.env` directly; an explicit API-key argument takes precedence, and process environment variables remain a fallback. Never commit real keys.

```powershell
python -m pip install -e ".[groq,openai,media,ocr]"
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
# Edit .env: choose providers and set the keys required by API modes.
python -m video_knowledge_agent.cli process "D:\videos\my-video.mp4"
```

Set `ASR_PROVIDER=groq|local` and `TEXT_PROVIDER=deepseek|local` independently in `.env`. The API defaults use Groq Whisper Turbo and DeepSeek Flash. Local ASR uses faster-whisper (`LOCAL_ASR_MODEL`, `LOCAL_ASR_DEVICE`, `LOCAL_ASR_COMPUTE_TYPE`); local text uses an Ollama model named in `LOCAL_TEXT_MODEL` through the loopback `LOCAL_TEXT_BASE_URL`. Install the selected dependencies and pull/start the local models before processing. For example, set both providers to `local` and run `ollama pull <your-model>` followed by `ollama serve`. Local text mode never falls back to DeepSeek if its server or model is missing. Switching providers or models invalidates the affected cached stages. The extra Groq review of uncertain Japanese ASR segments runs only with Groq ASR.

`process` accepts a local video or audio file. It supports `--language zh|en|ja`, `--output-language source`, repeated `--term` options, and `--force-stage`. Running it again on the same file reuses completed transcription, proofreading, chapter, and knowledge artifacts. If the source changes, its fingerprint produces a new `video_id`, preventing stale results from being reused. To normalize audio only:

```powershell
python -m video_knowledge_agent.cli prepare-audio "D:\videos\my-video.mp4"
```

Private run data is stored under `data/raw/`, `data/work/`, `data/outputs/`, and `data/index/`. Video, audio, databases, and environment files are excluded by `.gitignore`. Before publishing to a public Git repository, still inspect `git status` and `git diff --cached`; in particular, do not use `git add -f` to force-add media or keys. Git exclusion is not a copyright license and does not replace checking source rights, platform terms, or API service terms.

## Retrieval and visual evidence

```powershell
python -m video_knowledge_agent.cli search "your question"
python -m video_knowledge_agent.cli clip-moment-video <video_id> --moment-id <moment_id>
python -m video_knowledge_agent.cli ocr-moment <clip-file-path-from-previous-command>
```

Chapters, structured knowledge, and timestamped Moments are indexed locally with SQLite FTS5. When a local video is available and visual evidence is needed, the system extracts only the matching time window instead of processing every frame of the full video. Audio-only inputs cannot provide visual clips or OCR. By default, audio is uploaded in 120-second core chunks with about five seconds of context at each end. Output remains in the source language by default. A local player or an upstream agent must resolve internal timestamp references before playback.

## Private network acquisition tool

This machine has a separate optional batch acquisition tool under `local-tools/network-downloader/`. That directory and `data/imports/` are ignored by Git and are **not part of the public core**. The tool downloads the best audio stream only, validates it, and hands the completed local audio path to `process`; locally, `--process` can run both steps in sequence. Its README explains usage, installation, and authorization considerations. If it needs version control, put it in a separate private repository.

The repository retains a `media_loader` injection point for testing or integrating external adapters, but the default implementation accepts local files only. The public core has no website downloading, access-control bypass, or batch crawling code. Even when batch network downloading is technically possible, assess copyright, site terms, and user authorization before any public deployment.

## Usage notice and disclaimer

Use this tool only with media you are authorized to process. You are responsible for copyright, platform terms, privacy, and API charges. In API mode, audio is sent to Groq for transcription and text to DeepSeek for analysis. In local mode, the configured faster-whisper and loopback Ollama models process them on your computer. Check the selected providers before processing sensitive content. Do not use it to bypass DRM or access restrictions, and do not redistribute source media, clips, or substantial transcript text without the necessary rights. ASR and model-generated summaries may be inaccurate; verify important claims against the cited source and timestamp. The software is provided as-is, without legal advice or a guarantee of accuracy or availability. Keep `.env` private, and rotate any key that is exposed.

## Verification

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests
```

---

<a id="chinese"></a>

# Video Knowledge Agent — local-video-first core

本仓库只处理**已经位于本机、且你有权处理的媒体文件**。输入可以是本地视频或音频文件路径，不会解析网页、搜索视频或自动下载。核心仍可联网调用 Groq ASR 与 DeepSeek 文本 API；“本地”指媒体来源，不代表全离线推理。

## 分层

```text
可选的外部获取器（不在本仓库） ──> 本地音频文件
                                       │
                            本仓库的 process <文件路径>
                                       │
               FFmpeg 抽音频 → Groq 或本地 ASR → DeepSeek 或本地文字处理
                     → 章节/结构化知识 → Moment → 发布/FTS5 索引
                                       │
                有本地视频且需画面时：截短片 → 关键帧/OCR
```

外部获取器与核心之间只有**本地文件路径**这一接口。自己拍摄、获得许可、其他项目生成的音视频，以及合法取得的在线素材，均可由同一个核心处理。代码不会因为保留这个接口而自动联网下载。若无来源元数据，`source_url` 是内部 `local-video://` 或 `local-audio://` 时间标识，不能在浏览器打开；相邻的 `.info.json` 可保留原网页引用，但不会触发下载。

## 安装与使用

要求 Python 3.11+，安装 FFmpeg 或安装提供 FFmpeg 可执行文件的 `imageio-ffmpeg`。将 `.env.example` 复制为项目根目录的 `.env`，在其中选择模型来源；只有使用对应 API 模式时才需要填写 Groq 或 DeepSeek 密钥。程序会直接读取该文件；显式传入的密钥优先，进程环境变量作为后备。切勿提交真实密钥。

```powershell
python -m pip install -e ".[groq,openai,media,ocr]"
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
# 在本机编辑 .env：选择模式，并填写所用 API 模式的密钥。
python -m video_knowledge_agent.cli process "D:\videos\my-video.mp4"
```

在 `.env` 中分别设置 `ASR_PROVIDER=groq|local` 和 `TEXT_PROVIDER=deepseek|local`，两项可以混用。默认 API 模式使用 Groq Whisper Turbo 与 DeepSeek Flash。本地 ASR 使用 faster-whisper，并由 `LOCAL_ASR_MODEL`、`LOCAL_ASR_DEVICE`、`LOCAL_ASR_COMPUTE_TYPE` 控制；本地文字处理通过 `LOCAL_TEXT_BASE_URL` 连接本机 Ollama，模型名填写在 `LOCAL_TEXT_MODEL`。使用前需安装相应依赖、下载模型并启动服务，例如 `ollama pull <模型名>`、`ollama serve`。本地服务或模型不可用时会报错，不会自动回退到 DeepSeek。切换模型或来源会重跑受影响的缓存阶段。日语 ASR 的额外 Groq 复核仅在 Groq ASR 模式运行。

`process` 接受本地视频或音频文件，并支持 `--language zh|en|ja`、`--output-language source`、重复的 `--term` 和 `--force-stage`。重跑同一文件会复用已完成的转写、校对、章节与知识文件。源文件如果改变，文件指纹产生新的 `video_id`，避免把旧结果误当成新视频。仅统一音频格式可执行：

```powershell
python -m video_knowledge_agent.cli prepare-audio "D:\videos\my-video.mp4"
```

运行后的私有数据保存在 `data/raw/`、`data/work/`、`data/outputs/` 和 `data/index/`；视频文件、音频、数据库、环境文件均被 `.gitignore` 排除。发布到公开 Git 前仍应检查 `git status` 和 `git diff --cached`，尤其不要 `git add -f` 强行加入媒体或密钥。Git 忽略不是版权许可，也不能替代对素材来源、平台条款与 API 服务条款的判断。

## 检索与画面证据

```powershell
python -m video_knowledge_agent.cli search "要查的问题"
python -m video_knowledge_agent.cli clip-moment-video <video_id> --moment-id <moment_id>
python -m video_knowledge_agent.cli ocr-moment <上一步返回的clip文件路径>
```

章、结构化知识及带时间戳 Moment 进入本地 SQLite FTS5 索引。有本地原视频时，画面处理只在需要时截取命中时间窗，不逐帧处理整部视频；只有音频输入时不能截画面或 OCR。音频默认切成 120 秒核心片段并在两端各保留约 5 秒上下文上传；默认输出保持源语言。内部时间引用由本地播放器或上层 Agent 解析后再跳转。

## 私有联网获取器

当前机器上提供一个独立的可选批量获取器：`local-tools/network-downloader/`。它和 `data/imports/` 被 Git 忽略，**不是公开核心的一部分**。获取器只下载最佳音频流并校验完整性，再把音频文件路径交给 `process`；在本机可加 `--process` 连续运行。它的用途、安装和授权提醒见该目录内 README。若要单独版本管理，应放入私人仓库。

本仓库保留 `media_loader` 注入点供外部适配器测试或集成，但默认实现只接受本地文件。没有对网站的下载、绕过访问控制或批量抓取代码。网络批量下载即便技术上可行，公开部署前仍须评估版权、网站使用条款和用户授权。

## 使用声明与免责声明

仅处理你有权使用的素材；版权、平台条款、隐私和 API 费用由使用者自行核对与承担。API 模式会将音频发送给 Groq 转写、文字发送给 DeepSeek 分析；本地模式使用本机的 faster-whisper 和回环地址上的 Ollama。处理敏感内容前应确认所选模式。请勿用它绕过 DRM 或访问限制；未取得相应权利时，不要再分发原媒体、片段或大段转写文本。ASR 与模型总结可能出错，重要结论须回看原始素材及所附时间戳。本软件按现状提供，不构成法律意见，也不保证准确性或服务可用性。请妥善保管 `.env`；密钥泄露后应及时轮换。

## 验证

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests
```

---

<a id="japanese"></a>

# Video Knowledge Agent — ローカル動画を入力とするコア

このリポジトリが処理するのは、**手元にあり、利用する権利を持つメディアファイル**だけです。入力はローカルの動画または音声ファイルのパスであり、Web ページの解析、動画検索、自動ダウンロードは行いません。一方、コアは Groq の ASR API と DeepSeek のテキスト API にネットワーク経由でアクセスできます。「ローカル」はメディアの入力元を指し、推論まで完全にオフラインという意味ではありません。

## 構成

```text
任意の外部取得ツール（本リポジトリには含まれない）→ ローカル音声ファイル
                                                   │
                                         process <ファイルパス>
                                                   │
                 FFmpeg による音声抽出 → Groq またはローカル ASR
                   → DeepSeek またはローカル文章処理 → 章・構造化知識の抽出
                   → Moment → 公開用データ・FTS5 インデックス
                                                   │
               ローカル動画がある場合のみ：該当部分を切り出し → フレーム/OCR
```

外部取得ツールとコアの接点は、**ローカルファイルのパス**だけです。自分で撮影した動画、利用許諾を得た素材、別のプロジェクトで生成した音声・動画、適法に取得したオンライン素材を同じコアで処理できます。この接点を残しても、コアが自動で動画をダウンロードすることはありません。元情報がない場合、`source_url` はブラウザーで開けない `local-video://` または `local-audio://` の内部参照です。隣接する `.info.json` があれば元ページへの参照を保持できますが、コアからのダウンロードは発生しません。

## インストールと使い方

Python 3.11 以降、および FFmpeg か FFmpeg 実行ファイルを提供する `imageio-ffmpeg` が必要です。`.env.example` をプロジェクト直下の `.env` にコピーし、モデルの実行先を選びます。Groq と DeepSeek のキーは、それぞれの API モードを使う場合だけ必要です。コードは `.env` を直接読み込みます。明示的に渡したキーが優先され、プロセス環境変数はフォールバックとして使われます。実際のキーを Git にコミットしないでください。

```powershell
python -m pip install -e ".[groq,openai,media,ocr]"
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
# .env で実行先を選び、API モードで必要なキーを設定します。
python -m video_knowledge_agent.cli process "D:\videos\my-video.mp4"
```

`.env` の `ASR_PROVIDER=groq|local` と `TEXT_PROVIDER=deepseek|local` は別々に選べます。標準の API モードは Groq Whisper Turbo と DeepSeek Flash を使用します。ローカル ASR は faster-whisper を使い、`LOCAL_ASR_MODEL`・`LOCAL_ASR_DEVICE`・`LOCAL_ASR_COMPUTE_TYPE` で設定します。ローカルの文章処理は `LOCAL_TEXT_BASE_URL` のループバック Ollama サーバーを使い、`LOCAL_TEXT_MODEL` にモデル名を指定します。必要な依存関係とモデルを導入し、`ollama pull <モデル名>`、`ollama serve` などで準備してください。ローカルモデルが使えない場合はエラーになり、DeepSeek へ自動的に切り替わりません。実行先やモデルの変更時は影響を受けるキャッシュ工程を再実行します。日本語 ASR の追加 Groq 確認は Groq ASR モードでのみ実行します。

`process` はローカルの動画または音声ファイルを受け取ります。`--language zh|en|ja`、`--output-language source`、複数の `--term`、`--force-stage` を指定できます。同じファイルを再実行すると、完了済みの成果物を再利用します。元ファイルが変わると新しい `video_id` を生成し、古い結果の誤用を防ぎます。音声形式だけを統一する場合：

```powershell
python -m video_knowledge_agent.cli prepare-audio "D:\videos\my-video.mp4"
```

実行時の非公開データは `data/raw/`、`data/work/`、`data/outputs/`、`data/index/` に保存されます。動画、音声、データベース、環境設定ファイルは `.gitignore` で除外します。公開 Git リポジトリへアップロードする前には、必ず `git status` と `git diff --cached` を確認し、特に `git add -f` でメディアやキーを強制追加しないでください。Git での除外は著作権上の許諾ではなく、素材の権利、プラットフォーム規約、API の利用規約の確認に代わるものではありません。

## 検索と映像の根拠

```powershell
python -m video_knowledge_agent.cli search "質問文"
python -m video_knowledge_agent.cli clip-moment-video <video_id> --moment-id <moment_id>
python -m video_knowledge_agent.cli ocr-moment <前のコマンドで返されたクリップのパス>
```

章、構造化知識、時刻付き Moment はローカルの SQLite FTS5 に登録されます。ローカル元動画がある場合に限り、映像の根拠が必要な時間帯だけを切り出します。音声だけの入力では映像クリップや OCR を利用できません。音声は既定で 120 秒の主要区間に分け、各区間の前後に約 5 秒の文脈を付けてアップロードします。出力は既定で原語のまま保持します。内部時刻参照の解決はローカルプレーヤーまたは上位のエージェントが担当します。

## 非公開のネットワーク取得ツール

このマシンには、独立した任意の一括取得ツールが `local-tools/network-downloader/` にあります。このディレクトリと `data/imports/` は Git の対象外で、**公開コアには含まれません**。ツールは最適な音声ストリームだけを取得・検証し、ローカル音声のパスを `process` に渡します。手元では `--process` を付けて連続実行できます。用途、インストール方法、権利確認上の注意は、そのディレクトリの README を参照してください。

このリポジトリには外部アダプターのテストや統合に使う `media_loader` 注入ポイントを残していますが、既定の実装はローカルファイルのみを受け付けます。公開コアには Web サイトからのダウンロード、アクセス制御の回避、一括クロールのコードはありません。ネットワーク経由の一括ダウンロードが技術的に可能でも、公開運用の前に著作権、サイトの利用規約、ユーザーの権限を確認してください。

## 利用上の注意と免責事項

権利を持つ素材だけを処理してください。著作権、プラットフォームの規約、プライバシー、API 料金の確認と負担は利用者の責任です。API モードでは音声を Groq、テキストを DeepSeek に送信します。ローカルモードでは faster-whisper とループバック接続の Ollama を使います。機密情報を処理する前に選択したモードを確認してください。DRM やアクセス制限の回避に使わず、必要な権利なしに元メディア、クリップ、長い書き起こしを再配布しないでください。ASR やモデルの要約には誤りがあり得るため、重要な内容は元資料とタイムスタンプで確認してください。本ソフトウェアは現状有姿で提供され、法律上の助言、正確性、サービスの継続性を保証しません。`.env` を非公開に保ち、キーが漏えいした場合は更新してください。

## 検証

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests
```
