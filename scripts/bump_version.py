from __future__ import annotations

import argparse
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
VERSION_FILE = ROOT / "VERSION"
PACKAGE_VERSION_FILE = ROOT / "neuroswift" / "_version.py"


SEMVER_RE = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:[-+][0-9A-Za-z.-]+)?$")


def read_version() -> str:
    return VERSION_FILE.read_text(encoding="utf-8").strip()


def validate_version(version: str) -> str:
    if not SEMVER_RE.match(version):
        raise ValueError(
            f"Invalid version '{version}'. Expected semantic version format like 0.1.0"
        )
    return version


def write_version(version: str) -> None:
    VERSION_FILE.write_text(version + "\n", encoding="utf-8")
    PACKAGE_VERSION_FILE.write_text(
        '__all__ = ["__version__"]\n\n__version__ = "' + version + '"\n',
        encoding="utf-8",
    )


def bump(version: str, part: str) -> str:
    base = version.split("-", 1)[0].split("+", 1)[0]
    major, minor, patch = [int(x) for x in base.split(".")]
    if part == "major":
        return f"{major + 1}.0.0"
    if part == "minor":
        return f"{major}.{minor + 1}.0"
    if part == "patch":
        return f"{major}.{minor}.{patch + 1}"
    raise ValueError(f"Unsupported bump part: {part}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NeuroSwift version controller")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("show", help="Print the current version")

    bump_parser = subparsers.add_parser("bump", help="Bump semantic version")
    bump_parser.add_argument("part", choices=["major", "minor", "patch"])

    set_parser = subparsers.add_parser("set", help="Set an explicit semantic version")
    set_parser.add_argument("version")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    current = validate_version(read_version())

    if args.command == "show":
        print(current)
        return

    if args.command == "bump":
        new_version = bump(current, args.part)
    else:
        new_version = validate_version(args.version)

    write_version(new_version)
    print(new_version)
    print(f"Suggested git tag: v{new_version}")


if __name__ == "__main__":
    main()
