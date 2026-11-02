"""Native Apple GPU smoke test; saves evidence without changing the app or its dependencies."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'output' / 'playwright'
OUT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')


def worker():
    import numpy as np
    import torch
    from app import engines
    engines._install_deepspeed_shim()
    from resemble_enhance.enhancer import inference
    seen = []
    original = inference.inference_chunk
    def observed(model, audio, sr, device, *args, **kwargs):
        def before(module, inputs):
            seen.append({'parameters': str(next(module.parameters()).device),
                         'inputs': [str(v.device) for v in inputs if isinstance(v, torch.Tensor)]})
        handle = model.register_forward_pre_hook(before)
        try:
            return original(model, audio, sr, device, *args, **kwargs)
        finally:
            handle.remove()
    inference.inference_chunk = observed
    sr = 44100
    wav = (.1 * np.sin(2 * np.pi * 220 * np.arange(sr * 3) / sr)).astype('float32')
    started = time.monotonic()
    audio, rate = engines.run_resemble(wav, sr, 'denoise')
    torch.mps.synchronize()
    passed = (engines.DEVICE == 'mps' and bool(seen)
              and all(d['parameters'].startswith('mps') and d['inputs']
                      and all(v.startswith('mps') for v in d['inputs']) for d in seen)
              and bool(np.isfinite(audio).all()) and len(audio) > 0)
    result = {'passed': passed, 'selected_device': engines.DEVICE, 'forward_devices': seen,
              'duration_seconds': len(audio) / rate, 'sample_rate': rate,
              'elapsed_seconds': time.monotonic() - started,
              'mps_allocated_bytes': torch.mps.current_allocated_memory(),
              'note': 'MPS CPU fallback remains enabled; this verifies model/input placement, not that every operator executes on GPU.'}
    print('GPU_RESULT=' + json.dumps(result), flush=True)
    return 0 if passed else 1


def main():
    import torch
    result = {'torch': torch.__version__, 'mps_built': torch.backends.mps.is_built(),
              'mps_available': torch.backends.mps.is_available(),
              'studio_restore': 'CPU by application design',
              'deepfilter': 'No explicit MPS placement in application; not counted as GPU inference'}
    code = 2
    if not result['mps_available']:
        result.update(status='blocked', reason='MPS is unavailable in this Python execution session')
    else:
        try:
            value = (torch.ones(4, device='mps') * 2).sum().item()
            torch.mps.synchronize()
            result['tensor_check_passed'] = value == 8
            proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker'],
                                  cwd=ROOT, capture_output=True, text=True, timeout=180)
            (OUT / 'gpu-worker.log').write_text(proc.stdout + '\n' + proc.stderr)
            lines = [line for line in proc.stdout.splitlines() if line.startswith('GPU_RESULT=')]
            result['worker_exit_code'] = proc.returncode
            if lines:
                result['denoiser'] = json.loads(lines[-1].split('=', 1)[1])
            passed = result['tensor_check_passed'] and proc.returncode == 0 and result.get('denoiser', {}).get('passed', False)
            result['status'] = 'passed' if passed else 'failed'
            code = 0 if passed else 1
        except Exception as exc:
            result.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            code = 1
    (OUT / 'gpu-results.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return code


if __name__ == '__main__':
    raise SystemExit(worker() if '--worker' in sys.argv else main())
