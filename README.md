# Audio Clean

A private, local tool for cleaning up spoken-word recordings: podcasts, talks, radio segments and interviews. Drop in a file, pick a cleanup mode, compare before and after, and download.

Everything runs on this Mac. Recordings never leave the machine.

## Start

```sh
./run.sh
```

Opens http://127.0.0.1:8765. The first run downloads model weights (about 700 MB) into the Hugging Face cache.

## Cleanup modes

| Mode | Engine | Best for | Speed (Apple Silicon) |
|---|---|---|---|
| Fast noise removal | DeepFilterNet 3 | Fans, hiss, hum, traffic | ~0.1× clip length |
| Voice isolation | Resemble Enhance denoiser | Busier rooms, background voices | ~0.5× clip length |
| Studio restore | Resemble Enhance enhancer | Thin or muffled voices | ~4× clip length (CPU) |

Options:

- **Strength**: blends the cleaned signal over the original. Below 100% keeps some room tone and sounds more natural.
- **Podcast loudness**: levels to -16 LUFS integrated, -1.5 dBTP peak.
- **Output**: 24-bit WAV or 192 kbps MP3.

Any format ffmpeg can read is accepted, including video (the audio track is extracted).

## Layout

- `app/engines.py`: decoding, the three engines, blending and loudness levelling.
- `app/main.py`: FastAPI server with a single-worker job queue.
- `static/index.html`: the page.
- `data/jobs/<id>/`: uploads and results (git-ignored; safe to delete).

Uploads are kept in each job's `input/` directory under generated filenames.
Intermediate audio and results are kept separately in `work/`; downloads retain
the recording's original name with `_clean` added.

Saved jobs remain available through their existing job URLs after a restart.
Interrupted jobs are marked as errors and must be submitted again. If a status
check loses its connection, use **Retry status check** to reconnect to that job.
File selection and submission stay locked while that job is active.

## Regression checks

With the project environment installed:

```sh
.venv/bin/python -m unittest discover -s tests -v
node tests/frontend.cjs
```

These checks cover upload isolation, job recovery and interface error handling.
They use processing and browser stubs, so they do not assess audio quality.

## Notes on dependencies

`resemble-enhance` pins an old training stack. `pyproject.toml` uses uv overrides to keep torch 2.1.2 and NumPy 1.x, and drops `deepspeed` (training only, no macOS build) and `gradio`. `app/engines.py` registers a stand-in `deepspeed` module so the inference imports succeed, and fetches weights with `huggingface_hub` instead of git-lfs.

On torch 2.1, the Studio restore stage aborts on Apple's GPU backend, so it runs on the CPU. The other modes use the GPU where possible.

## Licence

MIT, see [LICENSE](LICENSE). The models are separate projects under their own licences: [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) (MIT or Apache-2.0) and [Resemble Enhance](https://github.com/resemble-ai/resemble-enhance) (MIT).
