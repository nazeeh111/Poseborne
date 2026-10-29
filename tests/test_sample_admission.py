"""Actual raw-session -> nine-link sample/label admission, without devices/models."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
try:
    import h5py
except ImportError:
    h5py = None

ROOT = Path(__file__).resolve().parents[1]
AVAILABLE = h5py is not None and importlib.util.find_spec('yaml') is not None
if AVAILABLE:
    for folder in ('host', 'teacher', 'train'):
        sys.path.insert(0, str(ROOT / folder))
    from csi_pipe.samples import build
    from csi_pipe.store import SessionWriter
    from csi_teacher.labels import LabelResult, write_h5
    from csi_teacher.pam import build_pam
    from csi_train.data import build_splits, load_role

START = 1_800_000_000_000_000_000
STEP = 10_000_000


def recording(path, missing=(), changed_link=None):
    writer = SessionWriter(path, chunk=100)
    for packet in range(401):
        for rx in range(3):
            for tx in range(3):
                if (rx, tx) in missing:
                    continue
                real = 10 + 3 * rx + tx + packet % 11
                if (rx, tx) == changed_link:
                    real += 7
                iq = np.column_stack((np.full(56, real), np.arange(56) % 3)).astype(np.int8)
                writer.append(rx_id=rx, tx_idx=tx, t_ns=START+packet*STEP,
                              esp_us=1_000_000+packet*10_000, iq=iq, seq=packet,
                              rssi=-40-rx*3-tx, noise=-90, boot_id=7)
    # Arrival order differs from frame order. Samples must sort timestamps,
    # while the teacher join must use each timestamp's original frame index.
    for offset, frame in ((200, 2), (100, 0), (150, 1)):
        writer.append_video(START+offset*STEP, frame)
    writer.close()


def add_labels(path):
    poses = np.zeros((3, 18, 3), np.float32)
    for frame in range(3):
        poses[frame, :, 0] = 8+frame
        poses[frame, :, 1] = 12+2*frame
        poses[frame, :, 2] = .9
    res = LabelResult(poses, np.zeros(3, np.uint8), np.ones(3, np.float32), 64, 48)
    write_h5(path, res)
    build_pam(path, say=lambda _: None)


def group_snapshot(group):
    result = {}
    def record(name, obj):
        value = {'attrs': {k: np.asarray(v).tolist() for k, v in obj.attrs.items()}}
        if isinstance(obj, h5py.Dataset):
            value.update(shape=obj.shape, dtype=str(obj.dtype), chunks=obj.chunks,
                         compression=obj.compression,
                         sha256=hashlib.sha256(obj[...].tobytes()).hexdigest())
        result[name] = value
    record('', group)
    group.visititems(record)
    return result


def snapshot(path):
    with h5py.File(path, 'r') as h:
        return {key: group_snapshot(h[key]) for key in h if not key.startswith('_poseborne_build_')}


@unittest.skipUnless(AVAILABLE, 'Raw-session integration requires h5py and PyYAML')
class SampleAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='poseborne-admission-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root/'session.h5'

    def build(self, path=None, **kwargs):
        return build(path or self.path, fit_window_s=2,
                     corrections={'csi_ms': 0, 'cam_ms': 0}, **kwargs)

    def test_missing_link_never_becomes_a_valid_nine_link_training_window(self):
        recording(self.path, missing={(2, 2)})
        result = self.build()
        self.assertEqual(result['N'], 3)
        self.assertEqual(result['valid_ratio'], 0)
        with h5py.File(self.path, 'r') as h:
            self.assertTrue(h['grid/mask'][:, 2, 2].all())
            self.assertFalse(h['grid/mask'][:, 0, 0].any())
            self.assertFalse(h['samples/valid'][...].any())
            np.testing.assert_array_equal(h['samples/X'][:, :, 2, 2], 0)
            metadata = json.loads(h['meta'].attrs['build'])
            self.assertEqual(metadata['links']['missing'], ['22'])
            self.assertEqual(len(metadata['links']['present']), 8)
        add_labels(self.path)
        with self.assertRaisesRegex(SystemExit, 'no valid labeled CSI samples'):
            load_role({'sessions': [{'h5': str(self.path), 'role': 'train'}]}, 'train')

    def test_link_packet_carrier_axes_and_teacher_timestamp_join(self):
        recording(self.path)
        result = self.build()
        self.assertEqual(result['valid_ratio'], 1)
        add_labels(self.path)
        with h5py.File(self.path, 'r') as h:
            np.testing.assert_array_equal(h['samples/t_ns'], START+np.array([100, 150, 200])*STEP)
            actual = h['samples/X'][...].reshape(3, 5, 56, 3, 3)
            # Independent raw packet/carrier formula, not a call to the alignment
            # helpers. Grid interpolation roundoff is below float16 resolution.
            expected = np.empty_like(actual)
            for row, end in enumerate((100, 150, 200)):
                for past, packet in enumerate(range(end-5, end)):
                    for carrier in range(56):
                        for rx in range(3):
                            for tx in range(3):
                                real = 10+3*rx+tx+packet%11
                                expected[row, past, carrier, rx, tx] = np.sqrt(real*real+(carrier%3)**2)
            np.testing.assert_array_equal(actual, expected)
            diagonal = np.arange(18)
            for frame in range(3):
                expected_xy = np.array([(8+frame)/64, (12+2*frame)/48], np.float16)
                np.testing.assert_array_equal(h['samples/Y'][...][frame, :2, diagonal, diagonal].T,
                                              np.broadcast_to(expected_xy[:, None], (2, 18)))
            self.assertEqual(json.loads(h['meta'].attrs['build'])['links']['missing'], [])
        rows = load_role({'sessions': [{'h5': str(self.path), 'role': 'val'}]}, 'val')
        self.assertEqual(rows.X.shape, (3, 280, 3, 3))

    def test_changed_link_changes_only_its_tensor_coordinate(self):
        original, changed = self.path, self.root/'changed.h5'
        recording(original)
        recording(changed, changed_link=(1, 2))
        self.build(original)
        self.build(changed)
        with h5py.File(original, 'r') as a, h5py.File(changed, 'r') as b:
            before, after = a['samples/X'][...], b['samples/X'][...]
            difference = np.any(before != after, axis=(0, 1))
            expected = np.zeros((3, 3), bool); expected[1, 2] = True
            np.testing.assert_array_equal(difference, expected)
            np.testing.assert_array_equal(a['samples/valid'], b['samples/valid'])

    def test_bad_link_schema_and_names_preserve_prior_result_on_force(self):
        for defect in ('name', 'shape', 'row_count', 'too_short', 'iq_overflow', 'rssi_overflow'):
            with self.subTest(defect=defect):
                path = self.root/f'{defect}.h5'
                recording(path)
                self.build(path)
                with h5py.File(path, 'r+') as h:
                    if defect == 'name':
                        h.move('links/22', 'links/22_alias')
                    elif defect == 'shape':
                        del h['links/22/iq']
                        h['links/22/iq'] = np.zeros((401, 55, 2), np.int8)
                    elif defect == 'row_count':
                        h['links/22/seq'].resize(400, axis=0)
                    elif defect == 'iq_overflow':
                        del h['links/22/iq']
                        h['links/22/iq'] = np.full((401, 56, 2), 100000, np.int32)
                    elif defect == 'rssi_overflow':
                        del h['links/22/rssi']
                        h['links/22/rssi'] = np.full(401, 100000, np.int32)
                    else:
                        for ds in h['links/22'].values(): ds.resize(1, axis=0)
                before = path.read_bytes()
                with self.assertRaises(SystemExit): self.build(path, force=True)
                self.assertEqual(path.read_bytes(), before)

    def test_mid_computation_failures_preserve_existing_results_and_labels(self):
        for trigger in ('그리드 ', '샘플 '):
            with self.subTest(stage=trigger):
                path = self.root/('grid-failure.h5' if trigger.startswith('그') else 'sample-failure.h5')
                recording(path)
                self.build(path)
                add_labels(path)
                with h5py.File(path, 'r+') as h:
                    h['grid'].attrs['previous_result'] = 'preserve'
                    h['samples'].attrs['previous_result'] = 'preserve'
                before = snapshot(path)
                def fail(message):
                    if message.startswith(trigger):
                        raise RuntimeError('injected ordinary progress callback failure')
                with self.assertRaisesRegex(RuntimeError, 'ordinary progress'):
                    self.build(path, force=True, block=100, progress=fail)
                self.assertEqual(snapshot(path), before)
                with h5py.File(path, 'r') as h:
                    self.assertFalse(any(k.startswith('_poseborne_build_') for k in h))

    def test_first_build_failure_leaves_only_original_capture(self):
        recording(self.path)
        before = snapshot(self.path)
        def fail(message):
            if message.startswith('샘플 '):
                raise RuntimeError('injected first-build callback failure')
        with self.assertRaisesRegex(RuntimeError, 'first-build callback'):
            self.build(progress=fail)
        self.assertEqual(snapshot(self.path), before)
        with h5py.File(self.path, 'r') as h:
            self.assertEqual(set(h), {'meta', 'links', 'video'})

    def test_link_or_flush_promotion_failure_restores_previous_groups_and_metadata(self):
        for operation in ('link', 'flush'):
            with self.subTest(operation=operation):
                path = self.root/f'{operation}-failure.h5'
                recording(path)
                self.build(path)
                add_labels(path)
                before = snapshot(path)
                fired = []
                original_set = h5py.Group.__setitem__
                original_flush = h5py.File.flush
                def fail_link(group, key, value):
                    if operation == 'link' and group.name == '/' and key == 'samples' and not fired:
                        fired.append(True)
                        raise RuntimeError('injected promotion link failure')
                    return original_set(group, key, value)
                def fail_flush(file):
                    if operation == 'flush' and not fired:
                        fired.append(True)
                        raise RuntimeError('injected promotion flush failure')
                    return original_flush(file)
                with patch.object(h5py.Group, '__setitem__', fail_link), patch.object(h5py.File, 'flush', fail_flush):
                    with self.assertRaisesRegex(RuntimeError, 'promotion'):
                        self.build(path, force=True)
                self.assertTrue(fired)
                self.assertEqual(snapshot(path), before)
                with h5py.File(path, 'r') as h:
                    self.assertFalse(any(k.startswith('_poseborne_build_') for k in h))

    def test_failed_recovery_keeps_backup_data_reachable(self):
        recording(self.path)
        self.build()
        add_labels(self.path)
        before = snapshot(self.path)
        original_set = h5py.Group.__setitem__
        def fail_samples_link(group, key, value):
            if group.name == '/' and key == 'samples':
                raise RuntimeError('injected persistent sample-link failure')
            return original_set(group, key, value)
        with patch.object(h5py.Group, '__setitem__', fail_samples_link):
            with self.assertRaisesRegex(RuntimeError, 'previous data retained at'):
                self.build(force=True)
        with h5py.File(self.path, 'r') as h:
            stages = [k for k in h if k.startswith('_poseborne_build_')]
            self.assertEqual(len(stages), 1)
            previous = h[stages[0]+'/previous']
            self.assertEqual(group_snapshot(previous['grid']), before['grid'])
            self.assertEqual(group_snapshot(previous['samples']), before['samples'])
            self.assertEqual(str(previous.attrs['build']), before['meta']['']['attrs']['build'])
            self.assertEqual(group_snapshot(h['links']), before['links'])
            self.assertEqual(group_snapshot(h['video']), before['video'])

    def test_numpy_integer_and_float_controls_build_successfully(self):
        recording(self.path)
        result = build(self.path, fit_window_s=np.float32(2), block=np.int64(100),
                       anchor_rate=np.float32(20), max_gap_run=np.int64(2),
                       corrections={'csi_ms': 0, 'cam_ms': 0})
        self.assertEqual(result['valid_ratio'], 1)
        with h5py.File(self.path, 'r') as h:
            self.assertEqual(json.loads(h['meta'].attrs['build'])['fit_window_s'], 2.0)
            self.assertFalse(any(k.startswith('_poseborne_build_') for k in h))

    def test_invalid_numeric_controls_preserve_previous_result(self):
        recording(self.path)
        self.build()
        before = self.path.read_bytes()
        cases = [('block', 0), ('block', -1), ('block', 1.5), ('block', True),
                 ('max_gap_run', -1), ('max_gap_run', 1.5),
                 ('anchor_rate', 0), ('anchor_rate', np.nan), ('anchor_rate', np.inf),
                 ('anchor_rate', 2e9), ('fit_window_s', 0), ('fit_window_s', np.inf)]
        for name, value in cases:
            with self.subTest(parameter=name, value=value), self.assertRaises(SystemExit):
                kwargs = dict(fit_window_s=2, corrections={'csi_ms': 0, 'cam_ms': 0}, force=True)
                kwargs[name] = value
                build(self.path, **kwargs)
            self.assertEqual(self.path.read_bytes(), before)

    def test_same_session_train_validation_aliases_are_rejected(self):
        import os
        recording(self.path)
        self.build()
        add_labels(self.path)
        symlink, hardlink = self.root/'alias.h5', self.root/'hardlink.h5'
        symlink.symlink_to(self.path)
        os.link(self.path, hardlink)
        before = self.path.read_bytes()
        for validation in (self.path, symlink, hardlink):
            for roles in (('train', 'val'), ('val', 'train')):
                manifest = {'sessions': [{'h5': str(self.path), 'role': roles[0]},
                                         {'h5': str(validation), 'role': roles[1]}]}
                with self.subTest(alias=validation, roles=roles), self.assertRaisesRegex(SystemExit, 'train/val overlap'):
                    build_splits(manifest)
        self.assertEqual(self.path.read_bytes(), before)

    def test_distinct_session_splits_keep_train_only_statistics(self):
        validation = self.root/'validation.h5'
        recording(self.path)
        recording(validation, changed_link=(1, 2))
        for path in (self.path, validation):
            self.build(path)
            add_labels(path)
        manifest = {'sessions': [{'h5': str(self.path), 'role': 'train'},
                                 {'h5': str(validation), 'role': 'val'}]}
        splits = build_splits(manifest)
        with h5py.File(self.path, 'r') as h:
            raw = h['samples/X'][...].astype(np.float32).reshape(-1, 5, 56, 3, 3)
        normalized = raw / np.sqrt(np.einsum('npkij,npkij->npij', raw, raw))[:, :, None]
        flattened = normalized.reshape(-1, 280, 3, 3)
        np.testing.assert_allclose(splits['mu'], flattened.mean(axis=(0, 1)), atol=2e-8, rtol=1e-6)
        np.testing.assert_allclose(splits['sigma'], flattened.std(axis=(0, 1)), atol=2e-8, rtol=1e-6)
        self.assertEqual(splits['train'].X.dtype, np.float16)
        self.assertFalse(np.array_equal(splits['train'].X, splits['val'].X))

    def test_bad_clock_or_anchor_preflight_preserves_previous_derived_groups(self):
        for defect in ('clock', 'anchor', 'correction'):
            with self.subTest(defect=defect):
                path = self.root/f'{defect}.h5'
                recording(path)
                self.build(path)
                with h5py.File(path, 'r+') as h:
                    if defect == 'clock':
                        h['links/22/esp_us'][:] = 1
                    elif defect == 'anchor':
                        h['video/t_ns'][:] = START-10*STEP
                before = path.read_bytes()
                with self.assertRaises(SystemExit):
                    if defect == 'correction':
                        build(path, fit_window_s=2, force=True,
                              corrections={'csi_ms': np.inf, 'cam_ms': 0})
                    else:
                        self.build(path, force=True)
                self.assertEqual(path.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
