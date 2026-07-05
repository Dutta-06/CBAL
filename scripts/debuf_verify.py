"""
generate_requirements.py
=========================
Scans all .py files in a project folder, extracts import statements,
filters out stdlib and relative imports, and writes a requirements.txt
without version specifications.

Usage:
    python generate_requirements.py                  # scans current directory
    python generate_requirements.py --dir /path/to/project
    python generate_requirements.py --dir . --out requirements.txt
"""

import ast
import os
import sys
import argparse
import sysconfig

# ── Standard library module names (Python 3) ─────────────────────────────────
# We exclude these from requirements.txt since they ship with Python.

def get_stdlib_modules():
    """Returns a set of stdlib module names for the current Python version."""
    # sys.stdlib_module_names is available in Python 3.10+
    if hasattr(sys, 'stdlib_module_names'):
        return set(sys.stdlib_module_names)

    # Fallback for Python < 3.10: use a known list + sysconfig paths
    import pkgutil
    stdlib_path = sysconfig.get_paths()['stdlib']
    stdlib_modules = set()
    for importer, modname, ispkg in pkgutil.iter_modules([stdlib_path]):
        stdlib_modules.add(modname)

    # Add commonly missed builtins
    extras = {
        'os', 'sys', 're', 'io', 'abc', 'ast', 'copy', 'json', 'math',
        'time', 'enum', 'uuid', 'hmac', 'zlib', 'gzip', 'csv', 'xml',
        'html', 'http', 'urllib', 'email', 'socket', 'struct', 'queue',
        'array', 'heapq', 'bisect', 'random', 'string', 'textwrap',
        'hashlib', 'base64', 'codecs', 'locale', 'gettext', 'argparse',
        'logging', 'warnings', 'traceback', 'inspect', 'typing', 'types',
        'functools', 'itertools', 'operator', 'contextlib', 'dataclasses',
        'pathlib', 'shutil', 'tempfile', 'glob', 'fnmatch', 'stat',
        'platform', 'subprocess', 'threading', 'multiprocessing', 'signal',
        'gc', 'weakref', 'collections', 'datetime', 'calendar', 'decimal',
        'fractions', 'statistics', 'pprint', 'reprlib', 'numbers', 'cmath',
        'pickle', 'shelve', 'sqlite3', 'configparser', 'tomllib',
        'unittest', 'doctest', 'pdb', 'profile', 'timeit', 'token',
        'tokenize', 'keyword', 'dis', 'compileall', 'zipfile', 'tarfile',
        'sysconfig', 'pkgutil', 'importlib', 'runpy', 'builtins',
        '_thread', '__future__',
    }
    return stdlib_modules | extras


# ── Import name → PyPI package name mapping ───────────────────────────────────
# Some packages are imported under a different name than their PyPI package.

IMPORT_TO_PYPI = {
    'cv2':            'opencv-python',
    'PIL':            'Pillow',
    'sklearn':        'scikit-learn',
    'skimage':        'scikit-image',
    'bs4':            'beautifulsoup4',
    'yaml':           'PyYAML',
    'dotenv':         'python-dotenv',
    'google.cloud':   'google-cloud',
    'dateutil':       'python-dateutil',
    'attr':           'attrs',
    'pkg_resources':  'setuptools',
    'wx':             'wxPython',
    'gi':             'PyGObject',
    'usb':            'pyusb',
    'serial':         'pyserial',
    'OpenGL':         'PyOpenGL',
    'Crypto':         'pycryptodome',
    'jwt':            'PyJWT',
    'pyannote':       'pyannote.audio',
    'librosa':        'librosa',
    'transformers':   'transformers',
    'torch':          'torch',
    'torchaudio':     'torchaudio',
    'scipy':          'scipy',
    'numpy':          'numpy',
    'pandas':         'pandas',
    'tqdm':           'tqdm',
    'matplotlib':     'matplotlib',
}


def extract_imports_from_file(filepath):
    """Parse a .py file and return top-level import names."""
    imports = set()
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            source = f.read()
        tree = ast.parse(source, filename=filepath)
    except SyntaxError:
        print(f"  [!] Syntax error, skipping: {filepath}")
        return imports

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                # Take only the top-level name: 'os.path' → 'os'
                imports.add(alias.name.split('.')[0])

        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:
                continue  # skip relative imports (from . import x)
            if node.module:
                imports.add(node.module.split('.')[0])

    return imports


def scan_project(root_dir):
    """Walk all .py files in root_dir and collect import names."""
    all_imports = set()
    py_files = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Skip common non-project directories
        dirnames[:] = [
            d for d in dirnames
            if d not in {'.git', '__pycache__', '.venv', 'venv',
                         'env', 'node_modules', '.tox', 'dist', 'build',
                         '.eggs', '*.egg-info'}
        ]
        for fname in filenames:
            if fname.endswith('.py'):
                fpath = os.path.join(dirpath, fname)
                py_files.append(fpath)
                file_imports = extract_imports_from_file(fpath)
                all_imports |= file_imports

    print(f"  Scanned {len(py_files)} .py files")
    return all_imports


def resolve_pypi_name(import_name):
    """Map import name to PyPI package name where they differ."""
    return IMPORT_TO_PYPI.get(import_name, import_name)


def main():
    parser = argparse.ArgumentParser(
        description="Generate requirements.txt from project imports (no versions)"
    )
    parser.add_argument('--dir', default='.', help='Project root directory (default: .)')
    parser.add_argument('--out', default='requirements.txt', help='Output file (default: requirements.txt)')
    args = parser.parse_args()

    root_dir = os.path.abspath(args.dir)
    print(f"\nScanning: {root_dir}")

    stdlib = get_stdlib_modules()
    all_imports = scan_project(root_dir)

    # Filter: remove stdlib, private names, and the project's own local modules
    local_modules = {
        os.path.splitext(f)[0]
        for f in os.listdir(root_dir)
        if f.endswith('.py')
    }
    # Also collect sub-package names (directories with __init__.py)
    local_packages = {
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
        and os.path.exists(os.path.join(root_dir, d, '__init__.py'))
    }
    local_all = local_modules | local_packages

    third_party = set()
    for name in all_imports:
        if name.startswith('_'):
            continue
        if name in stdlib:
            continue
        if name in local_all:
            continue
        third_party.add(resolve_pypi_name(name))

    # Sort and write
    sorted_deps = sorted(third_party, key=str.lower)

    out_path = os.path.join(root_dir, args.out)
    with open(out_path, 'w') as f:
        for dep in sorted_deps:
            f.write(dep + '\n')

    print(f"\n  Found {len(sorted_deps)} third-party dependencies:")
    for dep in sorted_deps:
        print(f"    {dep}")
    print(f"\n  Written to: {out_path}")


if __name__ == '__main__':
    main()