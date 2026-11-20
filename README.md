<div align="center">

# AIcoversong

**用自己的声音，唱喜欢的歌。**

本地录音 · 音色训练 · AI 翻唱

[English](README.en.md) · [安装指南](docs/INSTALL.md) · [模型与授权](docs/MODELS.md) · [反馈问题](https://github.com/dh666i/AIcoversong/issues)

[![CI](https://github.com/dh666i/AIcoversong/actions/workflows/check.yml/badge.svg)](https://github.com/dh666i/AIcoversong/actions/workflows/check.yml)
[![License: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%20%2B%20NVIDIA-76b900.svg)](docs/INSTALL.md)

</div>

AIcoversong 把录音、训练自己的音色、上传歌曲、分离人声、转换与混音放在一个中文网页里。你不需要自己串联多个音频工具。

项目基于 [Applio](https://github.com/IAHispano/Applio)、RVC 与 [audio-separator](https://github.com/nomadkaraoke/python-audio-separator) 构建，保留上游许可与致谢。模型在本地运行，录音与歌曲无需上传到第三方推理服务。

![AIcoversong 创作工作台](docs/images/workspace.jpg)

<details>
<summary>查看录音与训练页面</summary>

![录音稿与麦克风工作区](docs/images/voice.jpg)

</details>

截图来自隔离的界面演示，使用合成音频与演示音色名称。

## 可以做什么

- **录下自己的声音**：直接使用浏览器麦克风，按清楚区分的朗读和清唱稿录制、试听、保存。
- **训练个人音色**：把已保存的录音整理成训练集，查看训练状态，训练完成后选用对应音色。
- **一首歌，一次操作**：上传歌曲后，依次分离人声和伴奏、提取主唱并去混响、转换音色、处理和混音。
- **按你的习惯调整**：选择和声保留方式、音域处理方式、人声与伴奏音量。
- **保留作品**：试听完整翻唱或单独的人声、伴奏，下载结果；在作品库搜索、排序、改名、回收/恢复，并从已保存分轨创建新的混音版本。

翻唱会跟随输入歌曲的旋律。它不是“任何素材都不会跑调”的保证：分离质量、音色训练覆盖、原唱和声与唱法都会影响结果。

## 开始使用

当前正式安装路径面向 **Windows 10/11 x64 + NVIDIA CUDA**。参考开发环境是 Windows 11、RTX 5060 Ti 16GB、32GB 内存；其他显存容量未逐一验证。macOS、AMD 和 Linux 的完整流程暂未验证。

1. 安装近期 NVIDIA 驱动和 [uv](https://docs.astral.sh/uv/getting-started/installation/)；Windows 可用 `winget install --id astral-sh.uv --exact`。
2. 下载本仓库 ZIP 并解压，或执行：

   ```powershell
   git clone https://github.com/dh666i/AIcoversong.git
   cd AIcoversong
   ```

3. 阅读 [模型来源与各自许可](docs/MODELS.md)，然后运行：

   ```powershell
   powershell -NoProfile -ExecutionPolicy Bypass -File tools/setup.ps1 -DownloadModels
   ```

4. 双击 `启动翻唱.bat`，浏览器打开 `http://127.0.0.1:6969`。先在「我的声音」录音并训练，再在翻唱页上传歌曲。

首次安装需要联网下载依赖与约 **3.02GB** 基础模型/工具文件，另有数 GB 的 PyTorch 和 Python 依赖，建议至少预留 **20GB**，训练和作品会继续占用空间。仓库不包含作者的个人音色、录音、歌曲或付费服务凭据。

更详细的要求、国内下载选项与故障处理见 [安装指南](docs/INSTALL.md)。安装器不预设个人代理地址，可使用你的标准网络环境。

## 项目结构

```text
AIcoversong/
├─ Applio/             固定上游版本的引擎源码、独立网页与 CoverStudio
├─ tools/              翻唱、录音、训练、安装及发布检查脚本
├─ requirements/       Windows / Python 3.12 的固定依赖版本
├─ assets/             下载清单与上游资源校验值
├─ docs/               安装、模型授权和隐私说明
└─ tests/              不依赖私人录音的回归检查
```

使用后会生成 `recordings/`、`output/`、`cache/`、`logs/` 和模型目录；这些内容都被 Git 排除。请备份自己的录音和训练模型，更新源码时不要删除它们。

## 开发与反馈

欢迎提交可复现的问题和改进。[贡献指南](CONTRIBUTING.md) 说明了开发检查、数据边界与提交方式；漏洞请遵循 [安全政策](SECURITY.md)。

```powershell
python tools/verify_package.py --tracked
python -m unittest discover -s tests -v
```

CI 检查源码和发布内容，不会下载模型或在 GitHub 上训练。GPU 音质与干净机器上的完整安装验证须单独记录，不能由 CI 通过推定。

## 许可与声音权利

AIcoversong 原创代码使用 [MIT License](LICENSE)；Applio 及其他组件保留其原始许可。**代码许可不授予任何歌曲、音色或模型权重的权利**，也不意味着所有下载模型都可以任意商用。

请使用自己的声音或已取得同意的声音，以及拥有使用权的歌曲。发布或商用翻唱时，需自行确认歌曲、录音、声音人格权及模型条款。[第三方声明](THIRD_PARTY_NOTICES.md) 与 [模型说明](docs/MODELS.md) 列出了边界。
