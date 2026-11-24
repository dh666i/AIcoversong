# Changelog

## 0.4.0 (2026-11-24)

- Songs and aligned duet stems now accept the same 31 file extensions, including WavPack, APE, CAF, MKA and WebM. Containers with multiple audio streams use the first audio stream.
- Download a finished mix or a saved stem as WAV, FLAC, MP3, M4A (AAC), OGG (Vorbis), Opus or AIFF. WAV downloads preserve the original file. Other formats are encoded locally.
- Choose the audio format for mixes inside a batch ZIP. Older covers can be exported from their saved tracks without rerunning conversion. Open **管理 → 下载分轨** in the library to see which stems a work has.

[Compare v0.3.0 and v0.4.0](https://github.com/dh666i/AIcoversong/compare/v0.3.0...v0.4.0)

## 0.3.0 (2026-11-22)

### Duet covers

- Replace the male part, female part, or both. When replacing both, select one personal voice for both singers or assign a different voice to each. Supply aligned WAV or FLAC vocal stems, or install the optional local separator (approximately 527 MB).
- Automatic separation checks the lead and backing vocals independently before conversion. It can recover backing lines assigned to the wrong singer in the initial split. This adds three local separation passes. Similar voices singing in unison can still be difficult to separate.
- Voice processing reduces short periodic buzzes in converted consonants and suppresses duplicated vocals when a louder singer bleeds into a quiet, automatically separated part.

### Desktop studio and library

- Select individual works or all works in a filtered search, export their finished mixes in one ZIP, or delete the selection after confirmation. **Deletion is permanent.** Items previously moved to `output/.trash/` remain in that directory.
- The desktop studio adds motion and clearer feedback when switching views, changing controls and following a cover in progress.

### Upgrading

Restart the studio after updating the source. Existing voices can be reused without retraining. To apply the new harmony check to an existing duet, create a new cover from the original song. A remix reuses its saved stems. [Installation and upgrade guide](docs/INSTALL.md)

[Compare v0.2.0 and v0.3.0](https://github.com/dh666i/AIcoversong/compare/v0.2.0...v0.3.0)

## 0.2.0 (2026-11-21)

- The desktop cover workspace now enables **开始翻唱** when both a song and a personal voice are selected. The monitor shows job readiness and the latest finished cover with a waveform sampled from its WAV file, duration, playback and download.
- Sound settings are grouped into basic voice controls and advanced mixing. The summary follows the selected pitch and harmony modes. Numeric slider inputs and reset buttons stay visible in expanded panels, including the remix inspector.
- Recording feedback identifies missing, blocked or busy microphones and offers a device recheck. Audio players show loading and error states and provide a retry action.
- The library supports search, sorting, renaming, reversible trash and new mixes from saved stems. Finished covers have a foldout for listening to and downloading individual stems.
- The Chinese and English READMEs now include desktop screenshots and instructions for upgrading an existing installation.

## 0.1.0

Initial public source release of AIcoversong.

- Local recording and guided reading/singing prompts.
- Personal RVC voice training with retained recordings.
- Vocal/instrumental separation, lead-vocal refinement, voice conversion and mixing.
- Chinese cover workspace, task feedback and listening history.
- Windows NVIDIA setup with isolated Python environments and verified asset downloads.
- Bilingual project documentation and explicit third-party/model licensing boundaries.

This release distributes source code, not personal voice models, songs or a bundled GPU runtime.
