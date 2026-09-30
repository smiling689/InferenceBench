from __future__ import annotations

import ast
import functools
import inspect
import textwrap


class SamplingFastpaths(ast.NodeTransformer):
    def __init__(self):
        self.replacements = {'top_k_renorm_prob': 0, 'top_p_renorm_prob': 0}

    def visit_Assign(self, node):
        node = self.generic_visit(node)
        if not isinstance(node.value, ast.Call) or not isinstance(
            node.value.func, ast.Name
        ):
            return node
        operation = node.value.func.id
        if operation not in self.replacements:
            return node
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            return node
        if node.targets[0].id != 'target_probs':
            return node
        self.replacements[operation] += 1
        predicate = (
            'need_top_k_sampling'
            if operation == 'top_k_renorm_prob'
            else 'need_top_p_sampling'
        )
        conditional = ast.If(
            test=ast.Attribute(
                value=ast.Name(id='sampling_info', ctx=ast.Load()),
                attr=predicate,
                ctx=ast.Load(),
            ),
            body=[node],
            orelse=[],
        )
        return ast.copy_location(conditional, node)


def register():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    from sglang.srt.speculative import eagle_utils

    original = eagle_utils.eagle_sample
    source = textwrap.dedent(inspect.getsource(original))
    syntax_tree = ast.parse('from __future__ import annotations\n' + source)
    transform = SamplingFastpaths()
    syntax_tree = transform.visit(syntax_tree)
    if transform.replacements != {'top_k_renorm_prob': 1, 'top_p_renorm_prob': 1}:
        raise RuntimeError(
            f'Unexpected speculative sampling implementation: {transform.replacements}'
        )
    ast.fix_missing_locations(syntax_tree)
    namespace = original.__globals__.copy()
    exec(compile(syntax_tree, original.__code__.co_filename, 'exec'), namespace)
    replacement = functools.update_wrapper(namespace[original.__name__], original)
    HookRegistry.register(
        'sglang.srt.speculative.eagle_utils.eagle_sample',
        replacement,
        HookType.REPLACE,
    )
