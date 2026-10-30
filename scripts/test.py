#!/usr/bin/env python3
"""Compile and run the full ScalaTest suite without the retired sbt/Bintray plugins.

Requires Python 3 and Coursier (cs). Uses the project's Scala/Breeze versions
and JDK 8, downloading missing artifacts through Coursier's normal cache.
"""
import argparse
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coursier', default='cs', help='Coursier executable')
    parser.add_argument('--java-home', type=Path, help='Existing JDK 8 home; otherwise provision Temurin 8')
    args = parser.parse_args()
    classpath = subprocess.check_output([
        args.coursier, 'fetch', '--classpath',
        'org.scala-lang:scala-compiler:2.11.7',
        'org.scalanlp:breeze_2.11:0.11.2',
        'org.scalatest:scalatest_2.11:2.2.4',
        'org.scala-lang.modules:scala-parser-combinators_2.11:1.0.4',
        'com.opencsv:opencsv:3.6',
    ], text=True).strip()
    java_home = args.java_home or Path(subprocess.check_output([
        args.coursier, 'java-home', '--jvm', 'temurin:8'
    ], text=True).strip())
    java = str(java_home / 'bin/java')
    classes = ROOT / 'target/manual-tests'
    if classes.exists():
        shutil.rmtree(classes)
    classes.mkdir(parents=True)
    sources = sorted((ROOT / 'src/main/scala').rglob('*.scala')) + sorted((ROOT / 'src/test/scala').rglob('*.scala'))
    subprocess.run([java, '-Xmx1g', '-cp', classpath, 'scala.tools.nsc.Main', '-usejavacp',
                    '-cp', classpath, '-d', str(classes), *map(str, sources)], cwd=ROOT, check=True)
    runtime = str(classes) + ':' + str(ROOT / 'src/test/resources') + ':' + classpath
    subprocess.run([java, '-Xmx1g', '-cp', runtime, 'org.scalatest.tools.Runner',
                    '-R', str(classes), '-o'], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
