"""Disposable HDF5 checks for validation before destination truncation."""
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

try:
    import h5py
except ModuleNotFoundError as exc:
    if exc.name != 'h5py':
        raise
    h5py = None

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('split_session', ROOT / 'train/split_session.py')
split = importlib.util.module_from_spec(SPEC)
if h5py is not None:
    SPEC.loader.exec_module(split)


@unittest.skipIf(h5py is None, 'Optional HDF5 pipeline dependency h5py is unavailable')
class SplitPreservationTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.src = self.root / 'session.h5'
        self.dst = self.root / 'result.h5'

    def source(self, label_rows=10):
        with h5py.File(self.src, 'w') as h:
            h.attrs['session'] = 'synthetic'
            samples = h.create_group('samples')
            samples.attrs['F'] = 10
            samples.create_dataset('t_ns', data=np.arange(10, dtype=np.int64) * 10**9)
            samples.create_dataset('X', data=np.arange(20, dtype=np.float16).reshape(10, 2))
            # Existing soft/hard source links must remain readable and unchanged.
            samples['X_soft'] = h5py.SoftLink('/samples/X')
            samples['X_hard'] = samples['X']
            labels = h.create_group('labels')
            labels.attrs['F'] = label_rows
            labels.create_dataset('joints', data=np.ones((label_rows, 18, 2), dtype=np.float32))
            h.create_dataset('links/retained', data=[7])

    def existing(self, path=None):
        with h5py.File(path or self.dst, 'w') as h:
            h.attrs['marker'] = 'preserve on error'

    def write(self):
        return split.write_split(self.src, self.dst, np.arange(10) < 8,
                                 role='train', frac=.8, gap_s=0, cut_t_ns=8*10**9)

    def test_invalid_rows_preserve_existing_input_and_both_outputs(self):
        self.source(label_rows=9)
        out = self.root / 'out'
        out.mkdir()
        outputs = [out / 'session-train.h5', out / 'session-val.h5']
        for p in outputs:
            self.existing(p)
        before = {p: p.read_bytes() for p in [self.src, *outputs]}
        with self.assertRaises(SystemExit) as exc:
            split.main(['--h5', str(self.src), '--out-dir', str(out), '--gap-s', '0'])
        self.assertEqual(str(exc.exception), 'labels/joints: 행 정렬 아님 — 분할 불가')
        for p, original in before.items():
            self.assertEqual(p.read_bytes(), original, str(p))

    def test_invalid_rows_do_not_create_destination(self):
        self.source(label_rows=9)
        before = self.src.read_bytes()
        with self.assertRaises(SystemExit):
            self.write()
        self.assertFalse(self.dst.exists())
        self.assertEqual(self.src.read_bytes(), before)

    def test_non_dataset_error_preserves_output(self):
        self.source()
        with h5py.File(self.src, 'a') as h:
            h.create_group('labels/invalid_group')
        self.existing()
        before = self.dst.read_bytes()
        with self.assertRaises(SystemExit) as exc:
            self.write()
        self.assertEqual(str(exc.exception), 'labels/invalid_group: 행 정렬 아님 — 분할 불가')
        self.assertEqual(self.dst.read_bytes(), before)

    def test_valid_split_overwrites_and_preserves_values_attrs_and_source(self):
        self.source()
        self.existing()
        before = self.src.read_bytes()
        self.assertEqual(self.write(), 8)
        self.assertEqual(self.src.read_bytes(), before)
        with h5py.File(self.dst, 'r') as h:
            self.assertEqual(set(h), {'samples', 'labels'})
            self.assertNotIn('marker', h.attrs)
            self.assertEqual(h.attrs['session'], 'synthetic')
            self.assertEqual(h.attrs['split_source'], str(self.src))
            self.assertEqual(h.attrs['split_role'], 'train')
            self.assertEqual(h.attrs['split_frac'], .8)
            self.assertEqual(h.attrs['split_gap_s'], 0)
            self.assertEqual(h.attrs['split_cut_t_ns'], 8*10**9)
            for name in ('X', 'X_soft', 'X_hard'):
                np.testing.assert_array_equal(h['samples/'+name], np.arange(20).reshape(10, 2)[:8])
                self.assertEqual(h['samples/'+name].dtype, np.dtype('float16'))
                self.assertIsInstance(h['samples'].get(name, getlink=True), h5py.HardLink)
            np.testing.assert_array_equal(h['samples/t_ns'], np.arange(8) * 10**9)
            np.testing.assert_array_equal(h['labels/joints'], np.ones((8, 18, 2), dtype=np.float32))
            self.assertEqual(h['samples'].attrs['F'], 8)
            self.assertEqual(h['labels'].attrs['F'], 8)

    def test_destination_symlink_and_hardlink_keep_existing_write_through(self):
        self.source()
        for kind in ('symlink', 'hardlink'):
            with self.subTest(kind=kind):
                target = self.root / (kind + '.h5')
                self.existing(target)
                if kind == 'symlink':
                    self.dst.symlink_to(target)
                else:
                    os.link(target, self.dst)
                inode = target.stat().st_ino
                self.assertEqual(self.write(), 8)
                self.assertEqual(target.stat().st_ino, inode)
                self.assertEqual(self.dst.stat().st_ino, inode)
                self.assertEqual(self.dst.is_symlink(), kind == 'symlink')
                with h5py.File(target, 'r') as h:
                    self.assertEqual(h['samples/X'].shape, (8, 2))
                self.dst.unlink()


if __name__ == '__main__':
    unittest.main()
