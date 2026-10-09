import tempfile
import unittest
from pathlib import Path

try:
    from src.train import load_config
except ModuleNotFoundError:
    load_config = None


@unittest.skipIf(load_config is None, "PyTorch and PyYAML are not installed")
class ConfigTests(unittest.TestCase):
    def test_yaml_values_and_overrides_are_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.yaml"
            config_path.write_text("epochs: 3\nbatch_size: 8\ndownload: false\n")
            config = load_config(config_path, {"epochs": 5, "device": "cpu"})
        self.assertEqual(config.epochs, 5)
        self.assertEqual(config.batch_size, 8)
        self.assertFalse(config.download)
        self.assertEqual(config.device, "cpu")
