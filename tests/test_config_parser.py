import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


PARSER_PATH = (
    Path(__file__).resolve().parents[1] / 'tree_learn' / 'util' / 'parser.py')


class FakeMunch(dict):

    __getattr__ = dict.__getitem__

    @classmethod
    def fromDict(cls, value):
        if isinstance(value, dict):
            return cls({key: cls.fromDict(item) for key, item in value.items()})
        if isinstance(value, list):
            return [cls.fromDict(item) for item in value]
        return value


def load_parser_with_json_yaml():
    yaml_module = types.ModuleType('yaml')
    yaml_module.safe_load = json.load
    munch_module = types.ModuleType('munch')
    munch_module.Munch = FakeMunch
    spec = importlib.util.spec_from_file_location(
        'treelearn_parser_under_test', PARSER_PATH)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(
            sys.modules, {'yaml': yaml_module, 'munch': munch_module}):
        spec.loader.exec_module(module)
    return module


class ConfigParserTests(unittest.TestCase):

    def test_top_level_scalar_and_nested_dict_overrides(self):
        parser = load_parser_with_json_yaml()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            defaults = root / 'defaults.json'
            defaults.write_text(json.dumps({
                'tile_generation': True,
                'epochs': 100,
                'nested': {'kept': 1, 'overridden': 2},
            }), encoding='utf-8')
            config = root / 'config.json'
            config.write_text(json.dumps({
                'default_args': [str(defaults)],
                'tile_generation': False,
                'epochs': 50,
                'nested': {'overridden': 3},
            }), encoding='utf-8')

            result = parser.get_config(str(config))

        self.assertIs(result.tile_generation, False)
        self.assertEqual(result.epochs, 50)
        self.assertEqual(result.nested.kept, 1)
        self.assertEqual(result.nested.overridden, 3)


if __name__ == '__main__':
    unittest.main()