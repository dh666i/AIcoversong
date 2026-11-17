<div align="center">

# AIcoversong

**Your voice. The songs you love.**

Local recording · Voice training · AI song covers

[简体中文](README.md) · [Installation](docs/INSTALL.md) · [Model licenses](docs/MODELS.md) · [Issues](https://github.com/dh666i/AIcoversong/issues)

</div>

AIcoversong brings microphone recording, voice training, vocal separation, voice conversion and mixing into one Chinese-language web interface. Audio processing runs locally; your recordings and songs do not need a third-party inference service.

It builds on [Applio](https://github.com/IAHispano/Applio), RVC and [audio-separator](https://github.com/nomadkaraoke/python-audio-separator), with upstream attribution and licenses retained.

![AIcoversong workspace](docs/images/workspace.jpg)

Screenshots show an isolated UI demo with synthetic audio and a placeholder voice name.

## Features

- Record and preview your own reading and singing samples in the browser.
- Train a personal voice model from saved recordings, with visible task status.
- Upload a song and run vocal separation, lead/backing separation, dereverberation, conversion and mixing.
- Choose how to handle backing vocals, pitch range and vocal/instrumental levels.
- Listen to and download the finished cover and separate stems; revisit previous results.

The converted voice follows the source melody. Quality still depends on separation, training coverage, harmony and singing style; the app does not guarantee perfect pronunciation or pitch.

## Quick start

The documented setup targets **Windows 10/11 x64 with an NVIDIA GPU**. The reference machine uses Windows 11, an RTX 5060 Ti 16GB and 32GB RAM. Other GPU capacities, macOS, AMD and Linux have not been validated end to end.

1. Install a recent NVIDIA driver and [uv](https://docs.astral.sh/uv/getting-started/installation/).
2. Clone or extract this repository.
3. Read [model-specific terms](docs/MODELS.md), then run:

   ```powershell
   powershell -NoProfile -ExecutionPolicy Bypass -File tools/setup.ps1 -DownloadModels
   ```

4. Run `启动翻唱.bat` (or `tools/start.ps1`). Open `http://127.0.0.1:6969`, record and train under the voice tab, then upload a song on the cover tab.

Initial setup downloads roughly **3.02GB** of model/tool assets, plus Python and several GB of dependencies. Reserve at least **20GB** initially; training and saved covers use more. No personal voice model, recordings, songs or credentials are distributed with the source.

See [installation details](docs/INSTALL.md), [privacy](docs/PRIVACY.md), [contributing](CONTRIBUTING.md) and [security reporting](SECURITY.md). English documentation is provided; the application UI currently prioritizes Chinese users.

## Development

```powershell
python tools/verify_package.py
python -m unittest discover -s tests -v
```

CI validates source/package hygiene without downloading models or performing training. Passing CI does not establish GPU sound quality or clean-machine installation success.

## License

Original AIcoversong code is [MIT licensed](LICENSE). Vendored Applio code retains its own MIT notice. Other dependencies, fonts and externally downloaded model weights retain their own terms; see [third-party notices](THIRD_PARTY_NOTICES.md).

Use your own voice or voices you have permission to use, and audio you are entitled to process. The code license does not grant rights to songs, recordings, voices or every model weight. Check those rights before publishing or commercial use.
