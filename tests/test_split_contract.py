"""Additional tiny HDF5 contracts for the validation/copy boundary."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
try:
    import h5py
except ModuleNotFoundError as error:
    if error.name != 'h5py':
        raise
    h5py = None

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('split_contract_source', ROOT / 'train/split_session.py')
split = importlib.util.module_from_spec(SPEC)
if h5py is not None:
    SPEC.loader.exec_module(split)


@unittest.skipIf(h5py is None, 'Optional HDF5 pipeline dependency h5py is unavailable')
class SplitContractTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(prefix='poseborne-split-contract-')
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.src, self.dst = self.root / 'session.h5', self.root / 'result.h5'

    def source(self):
        with h5py.File(self.src, 'w') as h:
            h.attrs['origin'] = 'independent synthetic fixture'
            samples = h.create_group('samples')
            samples.attrs['F'] = 10
            samples.create_dataset('t_ns', data=np.arange(10, dtype=np.uint64) * 10**9)
            samples.create_dataset('X', data=np.arange(20, dtype=np.float16).reshape(10, 2))
            labels = h.create_group('labels')
            labels.attrs.update(F=10, W=64, H=48)
            labels.create_dataset('pose18', data=np.arange(540, dtype=np.float32).reshape(10, 18, 3))
            labels.create_dataset('det_score', data=np.linspace(0, 1, 10, dtype=np.float32))
            labels.create_dataset('status', data=np.arange(10, dtype=np.uint8))
            labels['pose18'].attrs['not_copied_by_legacy_splitter'] = 1
            h.create_dataset('video/t_ns', data=[1, 2])
            h.create_dataset('links/raw', data=[7])

    def marker(self, path):
        with h5py.File(path, 'w') as h:
            h.attrs['marker'] = 'existing output'

    def write(self):
        return split.write_split(self.src, self.dst, np.arange(10) < 6,
                                 role='train', frac=.6, gap_s=0, cut_t_ns=6*10**9)

    def test_late_label_mismatch_and_scalar_keep_existing_error_types_and_bytes(self):
        for scalar in (False, True):
            with self.subTest(scalar=scalar):
                self.source()
                with h5py.File(self.src, 'a') as h:
                    del h['labels/status']
                    h['labels'].create_dataset('status', data=3 if scalar else np.arange(9))
                self.marker(self.dst)
                before = {p: p.read_bytes() for p in (self.src, self.dst)}
                with self.assertRaises(IndexError if scalar else SystemExit) as error:
                    self.write()
                if not scalar:
                    self.assertEqual(str(error.exception), 'labels/status: 행 정렬 아님 — 분할 불가')
                for path, data in before.items():
                    self.assertEqual(path.read_bytes(), data)

    def test_external_dataset_materializes_and_dangling_link_preserves_destination(self):
        # The public writer accepts an arbitrary HDF5 file and dereferences child links.
        # Exercise this narrow accepted input without introducing any new link policy.
        self.source()
        payload = self.root / 'payload.h5'
        with h5py.File(payload, 'w') as h:
            h.create_dataset('rows', data=np.arange(30, dtype=np.int16).reshape(10, 3))
        with h5py.File(self.src, 'a') as h:
            h['labels/extra'] = h5py.ExternalLink('payload.h5', '/rows')
        original = {p: p.read_bytes() for p in (self.src, payload)}
        self.assertEqual(self.write(), 6)
        with h5py.File(self.dst, 'r') as h:
            np.testing.assert_array_equal(h['labels/extra'], np.arange(30).reshape(10, 3)[:6])
            self.assertIsInstance(h['labels'].get('extra', getlink=True), h5py.HardLink)
            self.assertEqual(h['labels/extra'].dtype, np.dtype('int16'))
        for path, data in original.items():
            self.assertEqual(path.read_bytes(), data)
        with h5py.File(self.src, 'a') as h:
            h['labels/zz_missing'] = h5py.ExternalLink('does-not-exist.h5', '/missing')
        before = self.dst.read_bytes()
        with self.assertRaises(SystemExit) as error:
            self.write()
        self.assertEqual(str(error.exception), 'labels/zz_missing: 행 정렬 아님 — 분할 불가')
        self.assertEqual(self.dst.read_bytes(), before)

    def test_main_nonzero_gap_keeps_output_selection_attributes_and_overwrite(self):
        self.source()
        out = self.root / 'out'
        out.mkdir()
        for role in ('train', 'val'):
            self.marker(out / f'session-{role}.h5')
        before = self.src.read_bytes()
        self.assertEqual(split.main(['--h5', str(self.src), '--out-dir', str(out),
                                    '--frac', '.6', '--gap-s', '1']), 0)
        self.assertEqual(self.src.read_bytes(), before)
        for role, rows in [('train', np.arange(5)), ('val', np.arange(6, 10))]:
            with h5py.File(out / f'session-{role}.h5', 'r') as h:
                self.assertEqual(set(h), {'samples', 'labels'})
                self.assertEqual(h.attrs['origin'], 'independent synthetic fixture')
                self.assertNotIn('marker', h.attrs)
                self.assertEqual(h.attrs['split_role'], role)
                self.assertEqual(h.attrs['split_gap_s'], 1)
                self.assertEqual(h.attrs['split_cut_t_ns'], 6*10**9)
                self.assertEqual(h['labels'].attrs['F'], len(rows))
                self.assertEqual(h['samples'].attrs['F'], len(rows))
                self.assertEqual((h['labels'].attrs['W'], h['labels'].attrs['H']), (64, 48))
                np.testing.assert_array_equal(h['samples/t_ns'], rows * 10**9)
                np.testing.assert_array_equal(h['samples/X'], np.arange(20).reshape(10, 2)[rows])
                np.testing.assert_array_equal(h['labels/status'], rows)
                np.testing.assert_array_equal(h['labels/pose18'], np.arange(540).reshape(10, 18, 3)[rows])
                self.assertEqual(h['samples/X'].dtype, np.dtype('float16'))
                self.assertEqual(h['labels/pose18'].dtype, np.dtype('float32'))
                self.assertEqual(h['labels/status'].dtype, np.dtype('uint8'))
                self.assertEqual(dict(h['labels/pose18'].attrs), {})

    def test_upstream_invalid_hdf5_and_timestamps_leave_outputs_untouched(self):
        self.marker(self.dst)
        before = self.dst.read_bytes()
        self.src.write_bytes(b'not an HDF5 file')
        with self.assertRaises(OSError):
            self.write()
        self.assertEqual(self.dst.read_bytes(), before)
        self.source()
        with h5py.File(self.src, 'a') as h:
            h['samples/t_ns'][2] = 0
        out = self.root / 'out'
        out.mkdir()
        for role in ('train', 'val'):
            self.marker(out / f'session-{role}.h5')
        original = {p: p.read_bytes() for p in out.iterdir()}
        with self.assertRaises(SystemExit) as error:
            split.main(['--h5', str(self.src), '--out-dir', str(out)])
        self.assertEqual(str(error.exception), 'samples/t_ns 비단조 — 레코더 규약 위반, 분할 불가')
        for path, data in original.items():
            self.assertEqual(path.read_bytes(), data)


if __name__ == '__main__':
    unittest.main()
