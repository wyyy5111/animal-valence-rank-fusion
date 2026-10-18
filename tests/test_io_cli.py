"""Cache integrity and phase-separation checks using temporary synthetic data."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from animal_valence.io import FeatureCache, check_split, load_features, read_json, write_json
from animal_valence.cli import main


class CacheTests(unittest.TestCase):
    def test_join_by_id_and_nan_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / 'a.npz', Path(directory) / 'b.npz'
            np.savez(a, ids=np.array(['b', 'a']), X=np.array([[2., 4.], [1., np.nan]]))
            np.savez(b, ids=np.array(['a', 'b']), X=np.array([[7.], [8.]]))
            output = load_features([a, b], [{'id': 'a'}, {'id': 'b'}])
            np.testing.assert_allclose(output, [[1., np.nan, 7.], [2., 4., 8.]], equal_nan=True)

    def test_invalid_offsets_and_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.npz'
            np.savez(path, ids=np.array(['a', 'b']), X_cat=np.ones((3, 2)), offsets=np.array([0, 0, 3]))
            with self.assertRaises(ValueError):
                FeatureCache(path)
            np.savez(path, ids=np.array(['a', 'a']), X=np.ones((2, 2)))
            with self.assertRaises(ValueError):
                FeatureCache(path)

    def test_dual_stream_requires_aligned_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / 'a.npz', Path(directory) / 'b.npz'
            np.savez(a, ids=np.array(['a']), X_cat=np.ones((2, 2)), offsets=np.array([0, 2]))
            np.savez(b, ids=np.array(['a']), X_cat=np.ones((3, 2)), offsets=np.array([0, 3]))
            with self.assertRaises(ValueError):
                load_features([a, b], [{'id': 'a'}])

    def test_cross_cache_hash_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / f'{i}.npz' for i in range(2)]
            for i, path in enumerate(paths):
                np.savez(path, ids=np.array(['a']), X=np.ones((1, 2)), audio_sha256=np.array([str(i) * 64]))
            with self.assertRaises(ValueError):
                load_features(paths, [{'id': 'a'}])

    def test_id_group_hash_split_guards(self):
        a = {'id': 'a', 'domain': 'd', 'y_true': 0, 'group': 'g1', 'audio_sha256': '0' * 64}
        b = {'id': 'b', 'domain': 'd', 'y_true': 1, 'group': 'g2', 'audio_sha256': '1' * 64}
        check_split([a], [b])
        for key in ['id', 'group', 'audio_sha256']:
            with self.subTest(key=key), self.assertRaises(ValueError):
                check_split([a], [dict(b, **{key: a[key]})])


class CliTests(unittest.TestCase):
    def test_target_labels_cannot_change_fusion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [{'id': str(i), 'domain': 'd', 'y_true': 'UNREADABLE', 'probabilities': [i / 10, 1 - i / 10]}
                    for i in range(1, 6)]
            path = root / 'member.json'
            write_json(path, {'member': 'm', 'label_names': ['Negative', 'Positive'], 'predictions': rows})
            out = root / 'fused.json'
            main(['fuse', '--predictions', str(path), '--output', str(out)])
            first = read_json(out)
            self.assertTrue(all('y_true' not in r for r in first['predictions']))
            for row in rows:
                row['y_true'] = {'invalid': 'label'}
            write_json(path, {'member': 'm', 'label_names': ['Negative', 'Positive'], 'predictions': rows})
            main(['fuse', '--predictions', str(path), '--output', str(out)])
            self.assertEqual(first, read_json(out))

    def test_fuse_rejects_mismatched_class_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for i in range(2):
                path = root / f'{i}.json'; paths.append(str(path))
                write_json(path, {'member': str(i), 'label_names': ['a', 'b'][::1 if i == 0 else -1],
                                 'predictions': [{'id': 'a', 'domain': 'd', 'probabilities': [.2, .8]},
                                                 {'id': 'b', 'domain': 'd', 'probabilities': [.8, .2]}]})
            with self.assertRaises(ValueError):
                main(['fuse', '--predictions', *paths, '--output', str(root / 'out.json')])

    def test_help(self):
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as raised:
            main(['--help'])
        self.assertEqual(raised.exception.code, 0)

    def test_score_columns_must_match_class_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'bad.json'
            write_json(path, {'member': 'm', 'label_names': ['Negative', 'Positive'],
                             'predictions': [{'id': 'a', 'domain': 'd', 'probabilities': [.1, .2, .7]},
                                             {'id': 'b', 'domain': 'd', 'probabilities': [.7, .1, .2]}]})
            with self.assertRaisesRegex(ValueError, 'Score columns'):
                main(['fuse', '--predictions', str(path), '--output', str(root / 'out.json')])


if __name__ == '__main__':
    unittest.main()
