"""Load task-local SGLang modules with FA4 prefill graph compatibility fixes."""
import importlib.abc
import importlib.util
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent / 'modules'
_MODULES = {
    'sglang.srt.model_executor.model_runner': _ROOT / 'model_runner.py',
    'sglang.srt.layers.radix_attention': _ROOT / 'radix_attention.py',
    'sglang.srt.model_executor.runner.prefill_cuda_graph_runner': _ROOT / 'prefill_cuda_graph_runner.py',
}

class _GraphMetadataOverlay(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        source = _MODULES.get(fullname)
        if source is not None:
            return importlib.util.spec_from_file_location(fullname, source)
        return None

sys.meta_path.insert(0, _GraphMetadataOverlay())
