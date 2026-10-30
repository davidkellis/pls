#!/usr/bin/env python3
"""Verify the copied website code, run both suites, and compare actual Scala/TS fits."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(*args, cwd=ROOT):
    subprocess.run(list(map(str, args)), cwd=cwd, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coursier', default='cs')
    parser.add_argument('--java-home', type=Path)
    parser.add_argument('--website', type=Path, help='Also verify byte identity with the website checkout')
    parser.add_argument('--report', type=Path, default=ROOT / 'target/parity/report.json')
    args = parser.parse_args()
    work = ROOT / 'target/parity'
    work.mkdir(parents=True, exist_ok=True)
    report = args.report.resolve()
    report.parent.mkdir(parents=True, exist_ok=True)
    # Remove stale success output before starting a verification run.
    report.unlink(missing_ok=True)
    run('npm', 'ci', '--ignore-scripts', '--no-audit', '--no-fund', cwd=ROOT / 'typescript')
    run('npm', 'run', 'check', cwd=ROOT / 'typescript')
    run('npm', 'test', cwd=ROOT / 'typescript')
    prepare = ['node', ROOT / 'typescript/test/parity.mjs', 'prepare', work / 'input.json']
    if args.website:
        prepare += [args.website.resolve()]
    run(*prepare)
    scala = [sys.executable, ROOT / 'scripts/test.py', '--coursier', args.coursier,
             '--parity-input', work / 'input.json', '--parity-output', work / 'scala.json']
    if args.java_home:
        scala += ['--java-home', args.java_home.resolve()]
    run(*scala)
    run('node', ROOT / 'typescript/test/parity.mjs', 'compare', work / 'input.json', work / 'scala.json', report)


if __name__ == '__main__':
    main()
