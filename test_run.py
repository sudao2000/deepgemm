#!/usr/bin/env python3
"""Run all cases of every tests/test*_one.py file, one case per subprocess.

Each test*_one.py supports listing its cases (dry run) and running a single
case by index, so this runner:
  1. discovers all test*_one.py files in tests/,
  2. imports each module to read its SUITES dict (files without SUITES have
     a single anonymous suite, e.g. test_hyperconnection_one.py),
  3. lists the cases of every suite via a dry run,
  4. runs each case in its own subprocess, so a crash or assertion failure
     in one case does not affect the others.

Usage:
    python test_run.py                 # run all cases of all files
    python test_run.py bf16 einsum     # only files whose name contains a given substring
    python test_run.py --dry-run       # list the per-case commands without running them
"""

import importlib
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
TESTS_DIR = REPO_ROOT / 'tests'

CASE_LINE = re.compile(r'^\s*\[\s*\d+\]')


def run_cmd(args: list) -> subprocess.CompletedProcess:
    # Make the repo root importable (for `import deep_gemm`) in the child
    env = dict(os.environ)
    env['PYTHONPATH'] = str(REPO_ROOT) + os.pathsep + env.get('PYTHONPATH', '')
    return subprocess.run(
        [sys.executable, *args],
        cwd=TESTS_DIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def get_suites(module) -> list:
    # Suite name, or None for files with a single anonymous suite
    return list(module.SUITES) if hasattr(module, 'SUITES') else [None]


def case_args(file_name: str, suite, index) -> list:
    args = [file_name]
    if suite is not None:
        args.append(suite)
    if index is not None:
        args.append(str(index))
    return args


def count_cases(file_name: str, suite) -> int:
    # Dry run: count the " [idx] ..." lines in the output
    proc = run_cmd(case_args(file_name, suite, 'dry' if suite is None else None))
    assert proc.returncode == 0, f'Dry run failed: {file_name} {suite}\n{proc.stdout}'
    return sum(1 for line in proc.stdout.splitlines() if CASE_LINE.match(line))


def main() -> None:
    dry_run = '--dry-run' in sys.argv
    filters = [a for a in sys.argv[1:] if a != '--dry-run']
    files = sorted(TESTS_DIR.glob('test*_one.py'))
    if filters:
        files = [f for f in files if any(s in f.name for s in filters)]
    assert files, f'No test*_one.py files matched: {filters}'

    sys.path.insert(0, str(TESTS_DIR))
    sys.path.insert(0, str(REPO_ROOT))

    passed, failed = [], []
    for file in files:
        module = importlib.import_module(file.stem)
        for suite in get_suites(module):
            tag = f'{file.stem}:{suite}' if suite is not None else file.stem
            num_cases = count_cases(file.name, suite)
            print(f'== {tag}: {num_cases} cases')
            for i in range(num_cases):
                args = case_args(file.name, suite, i)
                if dry_run:
                    print(f'  python tests/{args[0]} {" ".join(args[1:])}')
                    continue
                t = time.time()
                proc = run_cmd(args)
                elapsed = time.time() - t
                status = 'PASS' if proc.returncode == 0 else 'FAIL'
                print(f'  [{status}] {tag} #{i} ({elapsed:.1f}s)')
                (passed if proc.returncode == 0 else failed).append((tag, i, proc.stdout))

    if dry_run:
        return

    print(f'\n{len(passed)} passed, {len(failed)} failed')
    for tag, i, output in failed:
        print(f'\n{"=" * 80}\nFAILED: {tag} #{i}\n{"=" * 80}')
        # Show only the tail of the output, where the error is
        print('\n'.join(output.splitlines()[-30:]))
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
