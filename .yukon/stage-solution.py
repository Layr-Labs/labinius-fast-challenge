#!/usr/bin/env python3
"""Copy an untrusted solution tree without following repository-controlled links."""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path
from typing import NoReturn

MAX_FILES = 100_000
MAX_BYTES = 2 * 1024 * 1024 * 1024
COPY_CHUNK = 1024 * 1024


def fail(message: str) -> NoReturn:
    raise SystemExit(message)


def checked_relative(path: Path, root: Path) -> Path:
    try:
        relative = path.relative_to(root)
    except ValueError:
        fail(f"path escapes solution root: {path}")
    if relative.is_absolute() or ".." in relative.parts:
        fail(f"invalid solution path: {relative}")
    if ".git" in relative.parts:
        fail(f"git control data is not accepted: {relative}")
    return relative


def copy_regular(source: Path, destination: Path, mode: int) -> None:
    source_fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        destination_fd = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o755 if mode & 0o111 else 0o644,
        )
        try:
            while chunk := os.read(source_fd, COPY_CHUNK):
                view = memoryview(chunk)
                while view:
                    written = os.write(destination_fd, view)
                    view = view[written:]
        finally:
            os.close(destination_fd)
    finally:
        os.close(source_fd)


def stage(source_arg: str, destination_arg: str) -> None:
    source = Path(source_arg).absolute()
    destination = Path(destination_arg).absolute()

    source_info = os.lstat(source)
    if not stat.S_ISDIR(source_info.st_mode) or stat.S_ISLNK(source_info.st_mode):
        fail("solution source must be a real directory")
    if destination.exists() or destination.is_symlink():
        fail("solution destination must not already exist")

    destination.mkdir(parents=True, mode=0o700)
    file_count = 0
    byte_count = 0

    def unreadable(_error: OSError) -> NoReturn:
        fail("solution tree is not fully readable")

    for current, directories, files in os.walk(
        source, topdown=True, onerror=unreadable, followlinks=False
    ):
        current_path = Path(current)
        current_relative = checked_relative(current_path, source)
        destination_current = destination / current_relative

        for directory_name in list(directories):
            child = current_path / directory_name
            relative = checked_relative(child, source)
            info = os.lstat(child)
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                fail(f"only real directories are accepted: {relative}")
            (destination / relative).mkdir(mode=0o755)

        for file_name in files:
            child = current_path / file_name
            relative = checked_relative(child, source)
            info = os.lstat(child)
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                fail(f"only regular files are accepted: {relative}")
            file_count += 1
            byte_count += info.st_size
            if file_count > MAX_FILES:
                fail(f"solution exceeds {MAX_FILES} files")
            if byte_count > MAX_BYTES:
                fail(f"solution exceeds {MAX_BYTES} bytes")
            copy_regular(child, destination_current / file_name, info.st_mode)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        fail("usage: stage-solution.py SOURCE_SOLUTION DESTINATION_SOLUTION")
    stage(sys.argv[1], sys.argv[2])
