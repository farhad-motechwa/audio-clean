"""Audio cleanup engines: DeepFilterNet (fast denoise) and Resemble Enhance (studio restore)."""

import os
import subprocess
import sys
import types
from functools import cache
from pathlib import Path

# Torch 2.1's Apple GPU backend lacks a few ops Resemble uses (e.g. weight_norm);
# let those run on CPU. Must be set before torch is imported.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import soundfile as sf
import torch

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def _install_deepspeed_shim():
    """Resemble Enhance imports deepspeed from its training modules even at inference.

    deepspeed cannot be installed cleanly on macOS, and none of it runs during
    inference, so we register empty stand-in modules that satisfy the imports.
    """
    if "deepspeed" in sys.modules:
        return

    class _Stub:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            return _Stub()

        def __getattr__(self, name):
            return _Stub()

    def module(name):
        m = types.ModuleType(name)
        m.__getattr__ = lambda attr: _Stub
        sys.modules[name] = m
        return m

    for name in ("deepspeed", "deepspeed.accelerator", "deepspeed.runtime",
                 "deepspeed.runtime.engine", "deepspeed.runtime.utils"):
        module(name)


def decode(src: Path, dst: Path, sr: int):
    """Decode any audio or video file to mono float WAV at the given sample rate."""
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(src), "-vn", "-ac", "1", "-ar", str(sr),
         "-c:a", "pcm_f32le", str(dst)],
        check=True,
    )


def encode(src: Path, dst: Path, loudnorm: bool):
    """Encode the final file, optionally levelling to podcast loudness (-16 LUFS, -1.5 dBTP)."""
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(src)]
    if loudnorm:
        cmd += ["-af", "loudnorm=I=-16:TP=-1.5:LRA=11", "-ar", "48000"]
    if dst.suffix == ".mp3":
        cmd += ["-c:a", "libmp3lame", "-b:a", "192k"]
    else:
        cmd += ["-c:a", "pcm_s24le"]
    subprocess.run(cmd + [str(dst)], check=True)


# --- DeepFilterNet -----------------------------------------------------------

@cache
def _deepfilter():
    from df.enhance import init_df
    model, state, _ = init_df(log_level="ERROR")
    return model, state


def run_deepfilter(wav: np.ndarray) -> np.ndarray:
    from df.enhance import enhance
    model, state = _deepfilter()
    audio = torch.from_numpy(wav).float().unsqueeze(0)
    return enhance(model, state, audio).squeeze(0).numpy()


# --- Resemble Enhance --------------------------------------------------------

@cache
def _resemble_run_dir() -> Path:
    """Fetch the enhancer weights into the Hugging Face cache (no git-lfs needed)."""
    from huggingface_hub import snapshot_download
    repo = snapshot_download("ResembleAI/resemble-enhance", allow_patterns=["enhancer_stage2/*"])
    return Path(repo) / "enhancer_stage2"


def run_resemble(wav: np.ndarray, sr: int, mode: str) -> tuple[np.ndarray, int]:
    _install_deepspeed_shim()
    from resemble_enhance.enhancer.inference import denoise, enhance
    dwav = torch.from_numpy(wav).float()
    run_dir = _resemble_run_dir()
    if mode == "denoise":
        out, out_sr = denoise(dwav, sr, DEVICE, run_dir=run_dir)
    else:
        # The enhancer's diffusion stage aborts on MPS under torch 2.1, so it runs on CPU.
        out, out_sr = enhance(dwav, sr, "cpu", nfe=64, solver="midpoint", lambd=0.9, tau=0.5,
                              run_dir=run_dir)
    return out.cpu().numpy(), out_sr


# --- Pipeline ----------------------------------------------------------------

ENGINES = {
    "deepfilter": "Fast noise removal (DeepFilterNet)",
    "resemble-denoise": "Voice isolation (Resemble denoiser)",
    "resemble-enhance": "Studio restore (Resemble Enhance, slow)",
}


def process(src: Path, workdir: Path, engine: str, strength: float, loudnorm: bool,
            out_format: str) -> Path:
    """Clean one file. strength 0..1 blends the processed signal over the original."""
    if engine not in ENGINES:
        raise ValueError(f"Unknown engine: {engine}")

    sr = 48000 if engine == "deepfilter" else 44100
    decoded = workdir / "decoded.wav"
    decode(src, decoded, sr)
    original, _ = sf.read(decoded, dtype="float32")

    if engine == "deepfilter":
        cleaned, out_sr = run_deepfilter(original), sr
    else:
        cleaned, out_sr = run_resemble(original, sr, engine.removeprefix("resemble-"))

    if out_sr != sr:
        # Bring the original to the output rate so the two can be blended.
        resampled = workdir / "original_resampled.wav"
        decode(src, resampled, out_sr)
        original, _ = sf.read(resampled, dtype="float32")

    n = min(len(original), len(cleaned))
    mixed = strength * cleaned[:n] + (1.0 - strength) * original[:n]

    mixed_path = workdir / "mixed.wav"
    sf.write(mixed_path, mixed, out_sr, subtype="FLOAT")

    out = workdir / f"{src.stem}_clean.{out_format}"
    encode(mixed_path, out, loudnorm)
    return out
