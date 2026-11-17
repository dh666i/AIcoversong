"""Publication boundary tests: only standard-library dependencies."""
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'tools' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PublicationTests(unittest.TestCase):
    def test_private_audio_is_rejected(self):
        checker = load('verify_package')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / 'recordings' / 'my.wav'
            path.parent.mkdir()
            path.write_bytes(b'not-for-publication')
            self.assertTrue(checker.inspect(root, [path], {}))

    def test_fixture_must_match_upstream(self):
        checker = load('verify_package')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / 'sample.wav'
            path.write_bytes(b'public fixture')
            approved = {'sample.wav': hashlib.sha256(path.read_bytes()).hexdigest()}
            self.assertEqual(checker.inspect(root, [path], approved), [])
            path.write_bytes(b'private recording replacing fixture')
            self.assertTrue(checker.inspect(root, [path], approved))

    def test_asset_path_cannot_escape(self):
        downloader = load('download_assets')
        item = {'dest': '../secret.bin', 'url': 'https://example.com/model', 'size': 1, 'sha256': 'a'*64}
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                downloader.destination(Path(folder), item)

    def test_hash_is_required(self):
        downloader = load('download_assets')
        item = {'dest': 'assets/model.bin', 'url': 'https://example.com/model', 'size': 1}
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                downloader.destination(Path(folder), item)


if __name__ == '__main__':
    unittest.main()
