"""Regression checks for upload isolation and job recovery, without model downloads."""
import asyncio
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import UploadFile
from app import main


class ImmediateExecutor:
    def submit(self, function, *args):
        function(*args)


def fake_process(src, workdir, engine, strength, loudnorm, out_format):
    # Exercise the filenames that previously overwrote or collided with uploads.
    (workdir / 'decoded.wav').write_bytes(b'decoded')
    (workdir / 'mixed.wav').write_bytes(b'mixed')
    out = workdir / f'{src.stem}_clean.{out_format}'
    out.write_bytes(b'processed')
    return out


class JobTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        for patcher in [patch.object(main, 'JOBS_DIR', self.root),
                        patch.dict(main.jobs, clear=True),
                        patch.object(main, 'executor', ImmediateExecutor()),
                        patch.object(main, '_duration', return_value=3.0),
                        patch.object(main.engines, 'process', side_effect=fake_process)]:
            patcher.start()
            self.addCleanup(patcher.stop)

    def upload(self, name='speech.wav'):
        upload = UploadFile(filename=name, file=io.BytesIO(b'original audio'))
        return asyncio.run(main.create_job(upload, 'deepfilter', .9, True, 'wav'))

    def test_reserved_filenames_preserve_original(self):
        for name in ['decoded.wav', 'mixed.wav', 'job.json', 'original_resampled.wav']:
            with self.subTest(name=name):
                job = self.upload(name)
                folder = self.root / job['id']
                self.assertEqual(job['status'], 'done')
                self.assertEqual((folder / job['input']).read_bytes(), b'original audio')
                self.assertEqual(Path(job['input']).parts[0], 'input')
                self.assertEqual(Path(job['output']).parts[0], 'work')
                self.assertEqual(json.loads((folder / 'job.json').read_text())['name'], name)

    def test_download_retains_display_filename(self):
        job = self.upload('My recording.wav')
        response = main.result(job['id'], download=True)
        self.assertIn("My%20recording_clean.wav", response.headers['content-disposition'])

    def test_completed_job_restored(self):
        job = self.upload()
        main.jobs.clear()
        main._restore_jobs()
        self.assertEqual(main.get_job(job['id'])['status'], 'done')
        self.assertTrue(Path(main.original(job['id']).path).is_file())
        self.assertTrue(Path(main.result(job['id']).path).is_file())

    def test_lifespan_restores_records(self):
        job = self.upload()
        main.jobs.clear()
        async def startup():
            async with main.lifespan(main.app):
                self.assertIn(job['id'], main.jobs)
        asyncio.run(startup())

    def test_interrupted_jobs_are_not_requeued(self):
        for status in ['queued', 'processing']:
            with self.subTest(status=status):
                job = self.upload()
                job['status'] = status
                main._save(job)
                main.jobs.clear()
                main._restore_jobs()
                recovered = main.get_job(job['id'])
                self.assertEqual(recovered['status'], 'error')
                self.assertIn('interrupted', recovered['error'])
                saved = json.loads((self.root / job['id'] / 'job.json').read_text())
                self.assertEqual(saved['status'], 'error')

    def test_legacy_flat_paths_restore(self):
        job = self.upload()
        folder = self.root / job['id']
        (folder / job['input']).rename(folder / 'legacy.wav')
        (folder / job['output']).rename(folder / 'legacy_clean.wav')
        job.update(input='legacy.wav', output='legacy_clean.wav')
        job.pop('name')
        main._save(job)
        main.jobs.clear()
        main._restore_jobs()
        self.assertEqual(main.get_job(job['id'])['status'], 'done')
        self.assertIn('legacy_clean.wav', main.result(job['id'], True).headers['content-disposition'])

    def test_missing_output_becomes_error(self):
        job = self.upload()
        (self.root / job['id'] / job['output']).unlink()
        main.jobs.clear()
        main._restore_jobs()
        self.assertEqual(main.get_job(job['id'])['status'], 'error')
        self.assertIn('missing', main.jobs[job['id']]['error'])

    def test_corrupt_record_does_not_prevent_other_recovery(self):
        good = self.upload()
        corrupt = self.root / 'corrupt'
        corrupt.mkdir()
        (corrupt / 'job.json').write_text('{broken')
        main.jobs.clear()
        with self.assertLogs(main.__name__, level='WARNING'):
            main._restore_jobs()
        self.assertIn(good['id'], main.jobs)
        self.assertNotIn('corrupt', main.jobs)

    def test_atomic_metadata_write_leaves_no_temporary_file(self):
        job = self.upload()
        self.assertFalse((self.root / job['id'] / 'job.tmp').exists())
        self.assertEqual(json.loads((self.root / job['id'] / 'job.json').read_text())['status'], 'done')


if __name__ == '__main__':
    unittest.main()
