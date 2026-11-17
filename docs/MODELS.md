# 模型、工具与授权 / Models and terms

仓库只分发源码与下载清单，不包含下列权重和可执行程序。首次下载总计约 **3.02GB（十进制）**。`assets/model-manifest.json` 是文件名、字节数、SHA-256 和下载地址的完整清单。

| 用途 | 资源 | 大小（约） | 来源和授权边界 |
|---|---|---:|---|
| 语音内容特征 | ContentVec `pytorch_model.bin` + config | 378MB | [Applio 资源仓库](https://huggingface.co/IAHispano/Applio/tree/3ff370b7023086f57e200ef86a9a251ddf4da472)；仓库模型卡标记 MIT，ContentVec/HuBERT 本身的来源及上游条款仍适用 |
| 音高 | RMVPE | 181MB | 同一 Applio 资源仓库；[RMVPE](https://github.com/Dream-High/RMVPE) |
| 备用音高 | FCPE、SwiftF0 | 43.5MB | Applio 提供的资源；保留各自模型条款，不由本仓库重新许可 |
| 音色训练起点 | RVC v2 / HiFi-GAN 40k G、D | 216MB | Applio 资源仓库；不含任何个人训练音色 |
| 人声与伴奏分离 | Kim Mel-Band Roformer FT / unwa | 913MB | [作者模型仓库](https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT/tree/e42d447e5026e85492945b73438b7a98b9314347)；检查时模型卡未声明明确 license 字段，不能据此推定无限制商用 |
| 主唱与和声分离 | BS Roformer Karaoke / frazer-becruily | 204MB | [作者模型仓库](https://huggingface.co/becruily/BS-Roformer-Karaoke/tree/0d66103c6b79c0dd0e9efd38ca70db7042f6f695)；检查时模型卡未声明明确 license 字段，商业/再分发需核对作者说明 |
| 去混响 | Mel-Band Roformer / anvuew | 913MB | [作者模型仓库](https://huggingface.co/anvuew/dereverb_mel_band_roformer/tree/cef05ad2b5b3145ea5c149d3ad5d1f8439b34d06)，模型卡声明 GPL-3.0；相关条款适用于该模型，不能改标 MIT |
| 解码与封装 | FFmpeg / FFprobe | 175MB | Applio 提供的 Windows 构建，FFmpeg 自述 LGPL-3.0-or-later；[FFmpeg legal](https://ffmpeg.org/legal.html)；未随源码仓库分发 |

模型卡信息核对日期为 2026-11-17。分离模型和 YAML 从 [audio-separator model-configs release](https://github.com/nomadkaraoke/python-audio-separator/releases/tag/model-configs) 下载，哈希对照其 0.48.0 的 `model-checksums.json`。这是下载来源与一致性依据，**不是该项目替模型作者授予许可**。

Applio 资源 URL 固定到仓库修订；部分分离资源/公共目录 JSON 使用 release 或 upstream URL，文件内容仍由 SHA-256 固定。旧内容不再可下载时应更新清单并复核兼容性，而不是跳过哈希检查。

## 权利分别确认

- AIcoversong 的 MIT 许可适用于原创代码，不覆盖第三方权重、歌曲、原唱录音或个人声纹。
- 对自己的声音训练通常不涉及他人声音许可；使用别人的声音应先取得明确同意。
- 开源工具可用，不代表所得翻唱能随意发布、营利或再分发。
- 若要制作含依赖、FFmpeg 和模型的整合包，需要另行履行对应的署名、许可证和可能的对应源代码义务。本仓库未制作这种二进制整合包。

## 仅核对已安装文件

```powershell
.\env\Scripts\python.exe tools/download_assets.py --check
```

该命令只检查大小和哈希，不下载、不加载模型、不训练。
