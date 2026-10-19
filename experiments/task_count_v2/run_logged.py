"""Run a command once, preserving its UTF-8 output and exit status.

Example: python run_logged.py --log execution.log -- python train.py train
Existing logs are never overwritten. This utility changes no study settings.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")
    args.log.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    with args.log.open("xb") as capture:
        process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, env=environment)
        assert process.stdout is not None
        for line in process.stdout:
            capture.write(line)
            capture.flush()
            sys.stdout.buffer.write(line)
            sys.stdout.buffer.flush()
        return process.wait()


if __name__ == "__main__":
    raise SystemExit(main())
