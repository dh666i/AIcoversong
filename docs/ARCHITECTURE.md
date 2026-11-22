# Architecture

AIcoversong keeps the UI separate from GPU work. Importing the guided workflow does not load neural models. The default entry point is `Applio/cover_app.py`; it loads the `CoverStudio` plugin and calls the helpers in `tools/`.

```mermaid
flowchart LR
  A[Song upload] --> B[Vocals and instrumental]
  B -->|Solo| C[Full vocal or refined lead]
  B -->|Automatic duet| D[Estimate roles and review harmonies]
  U[Uploaded duet stems] --> R[Select male or female part or both]
  D --> R
  C --> E[Voice conversion]
  R --> E
  F[Selected voice models] --> E
  E --> G[Vocal treatment and final mix]
  B -->|Instrumental| G
  R -->|Preserved audio| G
  S[Saved stems from an existing work] -->|Remix| G
  G --> H[Preview, download and library]
```

`tools/cover_pipeline.py` manages stages and stores each result in a unique local run folder. Separation uses `separator-env`; conversion and the web app use `env`. Stage processes release GPU resources between steps. `vocal_engine.py` handles cover-specific conversion and segment alignment. `vocal_mix.py` performs vocal treatment and final mixing.

For a solo cover, refinement optionally separates the lead from the backing vocals and dereverberates the lead before conversion. For an automatic duet, `duet_separator.py` first estimates the two singers, then separates lead and backing vocals and estimates both singers again in each layer. `duet_harmony.py` uses those independent estimates to move identifiable harmony between the original role stems before conversion, preserving their sum. Uploaded, time-aligned male and female stems bypass this automatic separation and review.

Only the selected roles enter voice conversion. In a duet, `duet_mix.py` combines the treated conversions with the instrumental, the unselected singer and any unassigned audio from automatic separation. Vocal treatment checks for periodic buzz in converted consonants. The duet mixer also checks automatically separated parts for amplified leakage from the other singer.

Library remixes read the saved stems and write a new work. They can apply updated vocal treatment, but they reuse the original role assignments. To apply a newer harmony separation model or routing algorithm, start a new cover from the song.

The UI's `/cover` action starts a background worker and returns an opaque task token. A timer calls `/cover_status` to obtain progress and terminal results. The browser keeps only this token for refresh recovery while the server is running; restarting the server clears in-memory tasks. Completed audio remains in the local works library. Cancelling requests worker cleanup before enabling another task. Covers and training share a lock to avoid concurrent GPU jobs. A completed or failed task leaves the current upload available for another attempt.

`voice_recording.py` provides prompts and stores saved microphone takes. `voice_training.py` creates a training snapshot and launches Applio preprocessing, feature extraction, training and indexing. Trained weights and indexes remain in the local model directory and are excluded from publication.

The full upstream Applio interface is retained for advanced use, but its optional features are not all covered by the guided UI's installation/validation scope. Model loading, network features and third-party dependencies must not be assumed safe for an untrusted public multiuser deployment.
