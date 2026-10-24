"""Actual sample builds on tiny disposable sessions, without devices/models."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
try:
    import h5py
except ModuleNotFoundError as error:
    if error.name != 'h5py':
        raise
    h5py = None

if h5py is not None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'host'))
    from csi_pipe.samples import build
    from csi_pipe.store import SessionWriter

MISSING = ('영상 앵커 세션 — 보정값 필요: configs/pairing.json 또는 '
           '--csi-corr-ms/--cam-corr-ms (무보정 빌드는 --no-correction)')
START = 1_800_000_000_000_000_000


def session(path, video):
    writer = SessionWriter(path, meta={'session': 'synthetic preservation fixture'}, chunk=100)
    for i in range(401):
        for rx in range(3):
            for tx in range(3):
                writer.append(rx_id=rx, tx_idx=tx, t_ns=START+i*10_000_000,
                              esp_us=1_000_000+i*10_000,
                              iq=np.full((56, 2), 20+i % 3, dtype=np.int8),
                              seq=i, rssi=-40, noise=-90, boot_id=7)
    writer.close()
    if video:
        with h5py.File(path, 'r+') as h:
            for key, values in [('t_ns', [START+1_000_000_000, START+2_000_000_000]),
                                ('frame_idx', [0, 1])]:
                h['video/'+key].resize(len(values), axis=0)
                h['video/'+key][:] = values


def snapshot(path, groups):
    result = {}
    with h5py.File(path, 'r') as h:
        def visit(name, obj):
            item = {'attrs': {k: np.asarray(v).tolist() for k, v in obj.attrs.items()}}
            if isinstance(obj, h5py.Dataset):
                item.update(shape=list(obj.shape), dtype=str(obj.dtype),
                            chunks=obj.chunks, compression=obj.compression,
                            sha256=hashlib.sha256(obj[...].tobytes()).hexdigest())
            result[name] = item
        for group in groups:
            visit(group, h[group])
            h[group].visititems(lambda name, obj, g=group: visit(g+'/'+name, obj))
    return result


@unittest.skipIf(h5py is None, 'Optional HDF5 pipeline dependency h5py is unavailable')
class BuildPreservationTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / 'synthetic.h5'

    def test_missing_corrections_force_keeps_previous_file_bytes(self):
        session(self.path, video=True)
        build(self.path, fit_window_s=2, corrections={'csi_ms': 0, 'cam_ms': 0})
        with h5py.File(self.path, 'r+') as h:
            for group in ('grid', 'samples'):
                h[group].attrs['prior_valid_marker'] = 'preserve'
        before = self.path.read_bytes()
        with self.assertRaises(SystemExit) as error:
            build(self.path, fit_window_s=2, force=True, corrections=None)
        self.assertEqual(str(error.exception), MISSING)
        self.assertEqual(self.path.read_bytes(), before)

    def test_existing_groups_without_force_keeps_refusal_and_bytes(self):
        session(self.path, video=True)
        build(self.path, fit_window_s=2, corrections={'csi_ms': 0, 'cam_ms': 0})
        before = self.path.read_bytes()
        with self.assertRaises(SystemExit) as error:
            build(self.path, fit_window_s=2, corrections=None)
        self.assertEqual(str(error.exception), '기존 /grid·/samples 존재 — --force로 재빌드')
        self.assertEqual(self.path.read_bytes(), before)

    def test_valid_force_rebuild_preserves_values_schema_raw_and_metadata(self):
        for video, corrections in [(False, None),
                                   (True, {'csi_ms': 0, 'cam_ms': 0}),
                                   (True, {'csi_ms': 2, 'cam_ms': 7, 'source': 'fixture'})]:
            with self.subTest(video=video, corrections=corrections):
                path = self.path.with_name('case'+str(video)+str(corrections is None)+str(len(corrections or {}))+'.h5')
                session(path, video)
                report = build(path, fit_window_s=2, corrections=corrections)
                before = snapshot(path, ('grid', 'samples', 'meta', 'links', 'video'))
                with h5py.File(path, 'r+') as h:
                    h['grid'].attrs['replace_me'] = 1
                    h['samples'].attrs['replace_me'] = 1
                rebuilt = build(path, fit_window_s=2, force=True, corrections=corrections)
                self.assertEqual(rebuilt, report)
                self.assertEqual(snapshot(path, ('grid', 'samples', 'meta', 'links', 'video')), before)
                with h5py.File(path, 'r') as h:
                    meta = json.loads(h['meta'].attrs['build'])
                    self.assertEqual(meta['anchors'], 'video' if video else 'synthetic@20.0Hz')
                    if video:
                        np.testing.assert_array_equal(h['samples/t_ns'], h['video/t_ns'])


if __name__ == '__main__':
    unittest.main()
