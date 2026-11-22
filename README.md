<div align="center">

# AIcoversong

**用自己的声音，唱喜欢的歌。**

本地录音 · 音色训练 · AI 翻唱

[下载最新版源码](https://github.com/dh666i/AIcoversong/releases/latest/download/AIcoversong-source.zip) · [安装与更新](docs/INSTALL.md) · [更新日志](CHANGELOG.md) · [English](README.en.md)

[![CI](https://github.com/dh666i/AIcoversong/actions/workflows/check.yml/badge.svg)](https://github.com/dh666i/AIcoversong/actions/workflows/check.yml)
[![License: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%20%2B%20NVIDIA-76b900.svg)](docs/INSTALL.md)

</div>

AIcoversong 是运行在 Windows 电脑上的中文翻唱工作台。可以用麦克风录下自己的声音并训练音色，再上传一首歌，在本机完成分离、转换、混音和试听。

项目基于 [Applio](https://github.com/IAHispano/Applio)、RVC 与 [audio-separator](https://github.com/nomadkaraoke/python-audio-separator) 构建，保留上游许可与致谢。模型在本地运行，录音与歌曲无需上传到第三方推理服务。

![创作工作台的原曲上传、个人音色选择和最近作品试听](docs/images/workspace.jpg)

<details>
<summary>查看展开的声音设置和混音参数</summary>

![音域与和声选项、混音滑块、数值输入框和重置按钮](docs/images/settings.jpg)

</details>

<details>
<summary>查看录音与训练页面</summary>

![录音稿与麦克风工作区](docs/images/voice.jpg)

</details>

截图来自隔离的界面演示，使用合成音频与演示音色名称。

## 工作台用法

| 标签页 | 操作 |
|---|---|
| **我的声音** | 按稿件朗读或清唱，试听后保存。累计至少 5 分钟录音并勾选使用声明，即可训练个人音色。遇到设备或权限问题，按麦克风旁的提示处理后点「重新检测」。 |
| **创作工作台** | 拖入原曲，选好个人音色，点「开始翻唱」。右侧显示制作进度，完成后可试听成品或展开「分开听」试听分轨。空闲时可查看最近作品的波形和时长，直接播放或下载。 |
| **作品库** | 按歌名或原文件名搜索，切换排序。点「管理」可以改名、删除作品，或从已有分轨创建新的混音版本。 |

需要调整声音时，展开「声音设置」选择音域适配、和声处理和人声清晰度。「混音与精细调节」里还有混响、人声与伴奏音量、精细处理和音色检索强度。滑块旁可直接输入数值，也可以单独恢复该项默认值。

### 男女对唱

在「翻唱范围」选择只换男声、只换女声或男女声都换。仅换一方时，另一方保留原唱，合唱段也按两个声部处理；两边都换时可以共用一个音色，也可以分别指定。

- **从歌曲自动分离**会先估计男女声，再分别检查主唱与和声。发现分错声部的和声时，工作台会在换声前将它移回对应声轨。这个复核默认开启，使用本机已安装的模型。完成后可展开「分开听」核对结果。整首角色相反时，可以勾选「交换自动分出的男、女声部」重新制作。同音齐唱或混响较重时仍可能串音。[模型安装与来源](docs/MODELS.md)
- **使用已有分轨**：同时上传男声和女声的 WAV 或 FLAC；两轨应与原曲从同一时间起点导出，保持完整曲长，保留休止处的静音。此模式也适用于已有录音工程的对唱。

### 批量整理作品

在「作品库」搜索要整理的歌曲，展开「批量管理」，逐首勾选或全选当前筛选结果，可打包下载所选成品 WAV，也可批量删除。确认删除后，所选作品的成品、分轨及本次制作记录会从磁盘移除。旧版本已移入 `output/.trash/` 的作品仍保存在该目录。

翻唱会跟随输入歌曲的旋律。它不是“任何素材都不会跑调”的保证：分离质量、音色训练覆盖、原唱和声与唱法都会影响结果。

## 开始使用

当前正式安装路径面向 **Windows 10/11 x64 + NVIDIA CUDA**。参考开发环境是 Windows 11、RTX 5060 Ti 16GB、32GB 内存；其他显存容量未逐一验证。macOS、AMD 和 Linux 的完整流程暂未验证。

1. 安装近期 NVIDIA 驱动和 [uv](https://docs.astral.sh/uv/getting-started/installation/)；Windows 可用 `winget install --id astral-sh.uv --exact`。
2. 下载并解压[最新发布的源码包](https://github.com/dh666i/AIcoversong/releases/latest/download/AIcoversong-source.zip)，或克隆仓库：

   ```powershell
   git clone https://github.com/dh666i/AIcoversong.git
   cd AIcoversong
   ```

3. 阅读 [模型来源与各自许可](docs/MODELS.md)，然后运行：

   ```powershell
   powershell -NoProfile -ExecutionPolicy Bypass -File tools/setup.ps1 -DownloadModels
   ```

4. 双击 `启动翻唱.bat`，浏览器打开 `http://127.0.0.1:6969`。首次使用可先到「我的声音」录音和训练，已有个人音色可直接打开「创作工作台」。

首次安装需要联网下载依赖与约 **3.02GB** 基础模型/工具文件，另有数 GB 的 PyTorch 和 Python 依赖，建议至少预留 **20GB**，训练和作品会继续占用空间。仓库不包含作者的个人音色、录音、歌曲或付费服务凭据。

已有安装请在任务结束后关闭旧服务，更新源码，再双击 `启动翻唱.bat`。具体步骤及国内下载选项见[安装与更新指南](docs/INSTALL.md)。

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
