# Third-party notices

AIcoversong is an independent application built using the following projects. It is not an official Applio release and does not claim upstream endorsement. Original notices must be retained when redistributing the relevant components.

| Component | Use | License / source |
|---|---|---|
| Applio | Vendored RVC engine, training and interface infrastructure | MIT; full notice at [Applio/LICENSE](Applio/LICENSE), upstream [commit 6f69726b140c615c8cc6831a7db8b7c5e48ba8af](https://github.com/IAHispano/Applio/tree/6f69726b140c615c8cc6831a7db8b7c5e48ba8af) |
| RVC ecosystem | Voice conversion architecture and underlying methods through Applio | Retain embedded source notices; [RVC project](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) |
| audio-separator 0.48.0 | Model execution for vocal, lead/backing and reverb separation | MIT package metadata; [source](https://github.com/nomadkaraoke/python-audio-separator/tree/v0.48.0). Its dependencies and model weights have separate licenses |
| Gradio 6.20.0 | Local web interface | Apache-2.0; [source](https://github.com/gradio-app/gradio) |
| PyTorch / torchaudio / torchvision | Neural inference and training | BSD-style notices and bundled dependency terms; [PyTorch](https://github.com/pytorch/pytorch) |
| ContentVec / HuBERT, RMVPE, FCPE, SwiftF0 | Speech features and pitch extraction | Model-specific terms; [model inventory](docs/MODELS.md) |
| NumPy, SciPy, SoundFile, librosa, soxr, FAISS | Signal processing and feature retrieval | Their distributed licenses (including BSD, ISC, LGPL-related native components, Apache/MIT as applicable); installed package metadata is authoritative |
| FFmpeg / FFprobe | Audio decoding and conversion subprocesses | Not vendored. The pinned local FFmpeg build reports LGPL-3.0-or-later; different builds may be GPL. [License and source obligations](https://ffmpeg.org/legal.html) |
| Syne variable font | Upstream Applio UI font | SIL Open Font License 1.1; [notice](licenses/Syne-OFL.txt) |

The App's original code license does not replace these terms. In particular, `diffq-fixed` in the separate audio-separator environment includes GPL-3.0 license text. Dependencies are installed from their upstream packages, not bundled in this source distribution; a binary redistribution requires a separate review of corresponding-source and notice obligations.

## Upstream provenance and local changes

The `Applio/` snapshot derives from the fixed commit above. AIcoversong adds the standalone `cover_app.py`, `assets/ui/cover.css`, the CoverStudio plugin and the `tools/` workflow. There are also retained local adjustments in Applio's app shell, plugin and download integration. The inference/training engine's source notices and [Applio usage terms](Applio/TERMS_OF_USE.md) remain available.

Any bundled stock files under `Applio/logs/mute*` or `Applio/logs/reference` are drawn from that upstream commit and are checked byte-for-byte against `assets/upstream-fixtures.json`. They are upstream training fixtures, not the developer's voice recordings. No other log, dataset or model directory belongs in the repository.

## Model weights

No neural weights are redistributed here. `assets/model-manifest.json` supplies upstream locations, sizes and SHA-256 digests. Those digests establish file identity, **not permission to use a model**. Read [docs/MODELS.md](docs/MODELS.md) before downloading or redistributing models. Some model cards do not clearly declare a license; this repository does not invent one or label those weights MIT.
