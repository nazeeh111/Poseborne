"""Real WiSPPN forwards on synthetic CSI test contracts, never pose accuracy."""
import importlib.util
import json
import subprocess
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
AVAILABLE = all(importlib.util.find_spec(n) for n in ('torch', 'h5py', 'yaml'))
if AVAILABLE:
    import torch
    sys.path[:0] = [str(ROOT / 'train'), str(ROOT / 'rt')]
    from csi_train.data import CHANNEL_CONVENTION, normalize_inputs
    from csi_train.fit import load_ckpt, save_ckpt
    from csi_train.model import WiSPPN
    from csi_rt.infer import PoseEstimator


def reference(raw, mu, sigma, phase=None, rssi=None, mu_phase=None, sigma_phase=None):
    """Independent scalar packet/link implementation with the training f16 boundary."""
    out = np.empty((280, 3, 3), np.float32)
    for rx in range(3):
        for tx in range(3):
            for p in range(5):
                sl = slice(p * 56, (p + 1) * 56)
                v = raw[sl, rx, tx].astype(np.float32)
                n = np.sqrt(np.sum(v * v))
                v = v / (n if n else 1)
                if rssi is not None:
                    v *= np.float32(10 ** (np.float32(rssi[p, rx, tx]) / np.float32(20)))
                out[sl, rx, tx] = (v - mu[rx, tx]) / sigma[rx, tx]
    out = out.astype(np.float16)
    if phase is not None:
        out = np.concatenate([out, ((phase.astype(np.float32) - mu_phase) / sigma_phase).astype(np.float16)])
    return out


@unittest.skipUnless(AVAILABLE, 'Install torch, h5py and PyYAML for inference contracts')
class InferenceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='poseborne-random-contract-')
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'random-contract.pt'
        rng = np.random.default_rng(21)
        self.raw = rng.uniform(1, 50, (280, 3, 3)).astype(np.float16)
        self.raw[:56, 0, 0] = 0
        self.phase = rng.uniform(-.2, .2, self.raw.shape).astype(np.float16)
        self.rssi = rng.uniform(-65, -25, (5, 3, 3)).astype(np.float16)
        self.mu = np.linspace(-.1, .1, 9, dtype=np.float32).reshape(3, 3)
        self.sigma = np.linspace(.5, 1.5, 9, dtype=np.float32).reshape(3, 3)
        self.mu_phase = self.mu - .17
        self.sigma_phase = self.sigma * .7

    def checkpoint(self, features=(), vector=False):
        torch.manual_seed(13)
        channels = 560 if 'phase' in features else 280
        model = WiSPPN(in_ch=channels, vector_head=vector).eval()
        save_ckpt(self.path, model, mu=self.mu, sigma=self.sigma,
                  config={'in_ch': channels, 'vector_head': vector, 'features': list(features)},
                  kappa=.5, best={'epoch': -1, 'purpose': 'random contract test'},
                  mu_phase=self.mu_phase if 'phase' in features else None,
                  sigma_phase=self.sigma_phase if 'phase' in features else None)
        return model

    def test_real_models_and_changed_input_match_independent_reference(self):
        for features, vector in [((), False), (('rssi',), False), (('phase', 'rssi'), True)]:
            with self.subTest(features=features, vector=vector):
                model = self.checkpoint(features, vector)
                estimator = PoseEstimator(self.path, device='cpu')
                optional = {}
                if 'phase' in features:
                    optional['phase'] = self.phase
                if 'rssi' in features:
                    optional['rssi'] = self.rssi
                for raw in (self.raw, np.roll(self.raw, 17, axis=0)):
                    manual = reference(raw, self.mu, self.sigma, **optional,
                                       mu_phase=self.mu_phase, sigma_phase=self.sigma_phase)
                    batch = {k: v[None] for k, v in optional.items()}
                    actual = normalize_inputs(raw[None], features, self.mu, self.sigma,
                                              mu_phase=self.mu_phase, sigma_phase=self.sigma_phase, **batch)[0]
                    np.testing.assert_allclose(actual, manual, rtol=1e-3, atol=2e-4)
                    with torch.no_grad():
                        full = model(torch.from_numpy(manual[None].astype(np.float32)))[0].numpy()
                    d = np.arange(18)
                    expected = full if vector else full[:, d, d]
                    xy, confidence = estimator(raw, **optional)
                    np.testing.assert_allclose(xy, expected[:2].T, rtol=2e-5, atol=2e-7)
                    np.testing.assert_allclose(confidence, expected[2], rtol=2e-5, atol=2e-7)
                    self.assertEqual(xy.shape, (18, 2))
                    self.assertTrue(np.isfinite(xy).all())

    def test_built_hdf5_evaluation_cli_computes_pixel_joint_error(self):
        import h5py
        model = self.checkpoint(('phase', 'rssi'), vector=True)
        session = Path(self.temp.name) / 'generated.h5'
        config = Path(self.temp.name) / 'sessions.yaml'
        # Deliberately generated labels, not measurements or trained quality.
        xy = np.stack([np.linspace(.1, .7, 18), np.linspace(.2, .8, 18)], axis=1).astype(np.float32)
        target = np.zeros((1, 4, 18, 18), np.float32)
        d = np.arange(18)
        target[0, 0, d, d], target[0, 1, d, d] = xy[:, 0], xy[:, 1]
        target[0, 2, d, d] = .8
        with h5py.File(session, 'w') as h:
            for name, value in {'X': self.raw[None], 'X_phase': self.phase[None],
                    'rssi': self.rssi[None], 'Y': target, 'valid': [True],
                    'label_ok': [True], 'presence': [True]}.items():
                h.create_dataset('samples/' + name, data=value)
            h.create_group('labels').attrs.update(W=640, H=480)
        config.write_text('sessions:\n  - {h5: "' + str(session) + '", role: test}\n')
        result = subprocess.run([sys.executable, str(ROOT / 'poseborne.py'), 'train', 'eval',
            '--config', str(config), '--ckpt', str(self.path), '--split', 'test',
            '--device', 'cpu'], capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads((self.path.parent / 'report-test.json').read_text())
        manual = reference(self.raw, self.mu, self.sigma, phase=self.phase, rssi=self.rssi,
                           mu_phase=self.mu_phase, sigma_phase=self.sigma_phase)
        with torch.no_grad():
            predicted = model(torch.from_numpy(manual[None].astype(np.float32)))[0, :2].T.numpy()
        expected = np.linalg.norm((predicted - xy) * [640, 480], axis=1).mean()
        self.assertAlmostEqual(report['mpjpe_px'], expected, places=3)
        self.assertEqual(report['n_eval'], 1)

    def test_packet_path_and_missing_features_fail_before_forward(self):
        for features in [('rssi',), ('phase',)]:
            with self.subTest(features=features):
                self.checkpoint(features)
                with self.assertRaisesRegex(ValueError, 'amplitude only'):
                    PoseEstimator(self.path, packet_only=True, device='cpu')
                estimator = PoseEstimator(self.path, device='cpu')
                with self.assertRaisesRegex(ValueError, 'required'):
                    estimator(self.raw)

    def test_invalid_metadata_is_rejected_before_model_allocation(self):
        valid = {'config': {'in_ch': 280, 'vector_head': False, 'features': []},
                 'channel_convention': CHANNEL_CONVENTION,
                 'mu': torch.zeros(3, 3), 'sigma': torch.ones(3, 3)}
        cases = [({'channel_convention': 'tx then rx'}, 'channel_convention'),
                 ({'sigma': torch.zeros(3, 3)}, 'positive'),
                 ({'mu': torch.ones(1)}, '3,3'),
                 ({'mu': torch.full((3, 3), float('nan'))}, 'finite'),
                 ({'config': {'in_ch': 560, 'vector_head': False}}, 'in_ch'),
                 ({'config': {'in_ch': 560, 'vector_head': False, 'features': ['phase']}}, 'mu_phase')]
        for replacement, message in cases:
            with self.subTest(message=message):
                torch.save(valid | replacement, self.path)
                with self.assertRaisesRegex(ValueError, message):
                    load_ckpt(self.path)

    def test_legacy_amplitude_and_undeclared_features(self):
        self.checkpoint()
        ck = torch.load(self.path, weights_only=True)
        del ck['config']['features']
        torch.save(ck, self.path)
        estimator = PoseEstimator(self.path, packet_only=True, device='cpu')
        xy, _ = estimator(self.raw)
        self.assertTrue(np.isfinite(xy).all())
        with self.assertRaisesRegex(ValueError, 'not declared'):
            estimator(self.raw, rssi=self.rssi)

    def test_input_shape_and_finiteness_fail_closed(self):
        for raw in [self.raw[:279], np.full_like(self.raw, np.nan)]:
            with self.assertRaises(ValueError):
                normalize_inputs(raw[None], (), self.mu, self.sigma)
        with self.assertRaisesRegex(ValueError, 'shape'):
            normalize_inputs(self.raw[None], ('rssi',), self.mu, self.sigma, rssi=self.rssi[None, :4])


if __name__ == '__main__':
    unittest.main()
