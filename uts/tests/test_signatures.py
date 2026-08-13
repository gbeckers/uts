"""
Tests on the source itself rather than on behaviour.

"""
import ast
import pathlib
from unittest import TestCase

import uts


def _sourcefiles():
    root = pathlib.Path(uts.__file__).parent
    for path in sorted(root.rglob('*.py')):
        yield path, path.relative_to(root.parent), ast.parse(path.read_text())


def _bare_excepts():
    """Yields (path, lineno, funcname) for every `except:` with no exception
    type."""
    for _, relpath, tree in _sourcefiles():
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                yield relpath, node.lineno


class TestNoBareExcepts(TestCase):

    def test_no_bare_except_clauses(self):
        "no except clause catches everything unconditionally"
        # A bare `except:` also catches KeyboardInterrupt and SystemExit, so it
        # swallows an interrupt as though it were a normal error. It also hides
        # what actually went wrong: two of the ones that used to be here were
        # covering python 2 leftovers that had been broken since the port.
        #
        # Catching broadly on purpose is fine, but it should say so by catching
        # Exception, as uts.processing.parallel does for its worker threads.
        found = list(_bare_excepts())
        message = '\n'.join(f'{p}:{line}' for p, line in found)
        assert not found, 'bare except clauses found:\n' + message


def _mutable_defaults():
    """Yields (path, lineno, funcname, argname) for every function argument
    whose default is a mutable literal."""
    root = pathlib.Path(uts.__file__).parent
    for path in sorted(root.rglob('*.py')):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            args = node.args.args + node.args.kwonlyargs
            defaults = ([None] * (len(node.args.args) - len(node.args.defaults))
                        + list(node.args.defaults) + list(node.args.kw_defaults))
            for arg, default in zip(args, defaults):
                if isinstance(default, (ast.Dict, ast.List, ast.Set)):
                    yield (path.relative_to(root.parent), node.lineno,
                           node.name, arg.arg)


class TestNoMutableDefaults(TestCase):

    def test_no_mutable_default_arguments(self):
        "no function has a mutable default argument"
        # A mutable default is created once and shared by every call, so a
        # function that writes into it leaks state between calls. saveh5 used
        # to reach _save_epoched(attrs={}) and write origintime and duration
        # into that shared dict, which then stayed polluted for the life of
        # the process.
        found = list(_mutable_defaults())
        message = '\n'.join(f'{p}:{line} {func}({arg}=...)'
                            for p, line, func, arg in found)
        assert not found, 'mutable default arguments found:\n' + message
