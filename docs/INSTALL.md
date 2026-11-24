# 安装与运行 / Installation

[English installation and upgrade instructions](#installation-and-updates-in-english)

## 支持范围

目前安装脚本支持 Windows 10/11 x64、NVIDIA 显卡、Python 3.12.10。通过 uv 在项目内安装 Python 和两个独立环境，避免人声分离依赖与网页/训练依赖冲突。参考运行机器有 16GB 显存、32GB 内存；不要把这一配置当成严格最低要求，也不要把其他平台视为已验证。

准备近期 NVIDIA 驱动、可用的网络和至少 20GB 空间。首次下载约 3.02GB 模型及工具，PyTorch 等依赖另外占数 GB。完整训练还会生成切片、特征和检查点。

## 安装

安装 [uv](https://docs.astral.sh/uv/getting-started/installation/)。从[最新发行版](https://github.com/dh666i/AIcoversong/releases/latest)下载 `AIcoversong-source.zip` 和 `AIcoversong-source.zip.sha256`。在下载目录运行以下命令，两处显示的 SHA-256 应相同（忽略大小写）。

```powershell
Get-FileHash .\AIcoversong-source.zip -Algorithm SHA256
Get-Content .\AIcoversong-source.zip.sha256
```

解压源码包，进入包含 `启动翻唱.bat` 的文件夹，在 PowerShell 中运行安装脚本。使用 Git 的读者也可以克隆仓库并在项目根目录执行同一命令。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools/setup.ps1 -DownloadModels
```

安装器使用官方 Python/PyTorch/PyPI 来源，不写入系统 Python，不预置 Clash 地址。`-DownloadModels` 明确启用模型下载；如果暂时不下载可省略，之后执行：

```powershell
.\env\Scripts\python.exe tools/download_assets.py --list
.\env\Scripts\python.exe tools/download_assets.py
```

下载器支持断点续传、SHA-256 校验，已校验的文件不会重复下载。模型来源、单文件体积和许可说明见 [MODELS.md](MODELS.md)。如果某个固定资源上游已变更，校验会失败，请报告对应文件；不要绕过校验。

**可选的男女对唱模型**约 527 MB。在「翻唱范围」中选择从歌曲自动分离男女声时需要此模型；仅制作普通翻唱或使用已有男女分轨，可跳过这一步。

```powershell
.\env\Scripts\python.exe tools/download_assets.py --duet
.\env\Scripts\python.exe tools/download_assets.py --duet --check
```

### 国内网络

下载器默认遵循系统代理设置，不依赖某台电脑的代理端口。只想直连可用 `--direct`；需要替换 Hugging Face 的下载域名，可明确指定：

```powershell
.\env\Scripts\python.exe tools/download_assets.py --hf-endpoint https://hf-mirror.com
```

替换域名不会改变固定路径或 SHA-256。请自行信任所选镜像。Python 包索引可由 uv 的标准环境变量配置，脚本不会写入长期个人配置。

## 启动与关闭

双击 `启动翻唱.bat`，或运行 `powershell -NoProfile -ExecutionPolicy Bypass -File tools/start.ps1`。页面默认仅监听 `127.0.0.1:6969`。双击 `关闭翻唱.bat` 可关闭由启动器创建的服务。

首次运行会根据上游模板生成 `Applio/assets/config.json`；此文件属于本机设置，不要提交到 Git。首次创建的配置不代表你已经同意任何条款，训练前仍由你在页面确认。

安装完成后，正常本地录音/翻唱不需要远端推理 API。不要把没有身份验证的本地网页直接开放到公网。

## 更新已有安装

1. 等当前翻唱或训练结束，保存需要的录音，然后双击 `关闭翻唱.bat`。
2. **Git 安装**：在项目根目录运行 `git pull --ff-only`。**ZIP 安装**：将[最新版源码包](https://github.com/dh666i/AIcoversong/releases/latest/download/AIcoversong-source.zip)解压到临时目录，把解出的 `AIcoversong` 文件夹里的内容合并到原安装目录，替换同名源码。保留原目录中的录音、作品、训练模型、环境和本机配置。
3. 双击原安装目录中的 `启动翻唱.bat`。如果浏览器里还开着旧标签页，刷新 `http://127.0.0.1:6969`。启动器会复用已经运行的服务，所以更新源码后需要重新启动服务才能载入新界面。

升级后可以继续使用已有的个人音色。自 v0.4.0 起，旧作品可以直接下载为 MP3、FLAC 等格式。要导出其中一条音轨，在「作品库」打开「管理 → 下载分轨」。

要让旧作品应用辅音蜂鸣修复，在「作品库」里从已保存的分轨创建新混音。要修复自动分轨时漏掉的合唱声部，在「创作工作台」重新选择原曲并开始翻唱，工作台会在换声前重新分离主唱与和声。两种操作都会生成新作品。

## 录音和音色

网页的「我的声音」提供朗读与清唱稿。录完后试听并保存，累计足够干净的录音再开始训练。训练会使用当前已保存的数据，不会附送作者个人音色。

支持使用自己有权使用的 RVC 音色：把导出的 `.pth` 和对应 `.index` 放入 `Applio/logs/<你的模型目录>/`，然后刷新音色列表。不要选择训练用的 `G_*.pth` / `D_*.pth`，也不要加载不可信来源的 Python/PyTorch 文件。

## 常见问题

| 现象 | 检查方法 |
|---|---|
| `uv` 找不到 | 安装后重新打开终端，运行 `uv --version` |
| CUDA 不可用 | 在 `env` 内检查 `torch.cuda.is_available()`，确认 NVIDIA 驱动和官方 cu128 依赖安装成功 |
| 模型缺失 | 运行 `env\Scripts\python.exe tools/download_assets.py --check`，按输出补齐 |
| 麦克风不可用 | 打开本机页面，按「我的声音」里的提示检查设备连接、Windows 输入和麦克风隐私设置、浏览器站点权限，或关闭占用设备的程序。点「重新检测」后再录音 |
| 端口被占用 | 关闭旧服务或检查 6969 上的进程，不要反复启动多个实例 |
| Windows 页面文件不足 | 关闭其他占内存程序并检查系统页面文件；录音预处理已使用保守并发 |
| 效果有电音、错字 | 检查输入和声/混响、分离主唱与干净录音覆盖；训练更多轮并不保证更好 |

## 验证边界

固定依赖版本取自当前可工作的参考环境，资产 SHA-256 来自本地已用文件并尽可能对照上游清单。本仓库的 CI 不进行 GPU 训练或完整歌曲转换，不能代替一台新机器上的完整安装和音频验收。遇到安装问题请附系统、显卡、Python/uv 版本与去除私人路径的报错，不要上传个人音色、原始录音或未授权歌曲。

## Installation and updates in English

**Requirements.** Use a Windows 10/11 x64 PC with an NVIDIA GPU, a current NVIDIA driver and [uv](https://docs.astral.sh/uv/getting-started/installation/). Reserve at least 20 GB for setup and extra space for recordings, training and saved covers. The setup script installs Python 3.12.10 and manages two local environments. Base models and tools total about 3.02 GB, in addition to Python and PyTorch dependencies.

**Install.** Download `AIcoversong-source.zip` and its matching `.sha256` file from the [latest release](https://github.com/dh666i/AIcoversong/releases/latest). Compare the output of `Get-FileHash .\AIcoversong-source.zip -Algorithm SHA256` against the digest in the checksum file. Extract the ZIP, open PowerShell in the folder containing `启动翻唱.bat`, and run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools/setup.ps1 -DownloadModels
```

For automatic duet separation, run `.\env\Scripts\python.exe tools/download_assets.py --duet` to download the optional model (about 527 MB). Supplying your own aligned male and female stems lets you skip this download. Run `启动翻唱.bat` to open the studio at `http://127.0.0.1:6969`. [Workspace guide in English](../README.en.md#using-the-workspace)

**Update.** Let any active job finish and run `关闭翻唱.bat`. For a Git installation on `main`, run `git pull --ff-only`. For a ZIP installation, extract the verified release to a temporary folder, then merge its source files into the existing installation. Keep your local recordings (`recordings/`), covers (`output/`), voice models (`Applio/logs/`), settings (`Applio/assets/config.json`) and installed environments. Restart with `启动翻唱.bat`.

Existing voice models work after upgrading. Starting with v0.4.0, you can export existing covers as MP3, FLAC or another supported format. Open **管理 → 下载分轨** in the library to export an individual stem.

To reduce consonant buzz in a saved cover, use the library to make a new mix from its saved stems. For a missed chorus part, start a new cover from the original song so the separator can reassess the two singers before conversion. Both operations create a new library entry.
