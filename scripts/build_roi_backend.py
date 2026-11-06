"""Build a byte-bound viewport extension without changing the native package."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from msv2.roi import backend_path, ensure_backend


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--backend-dir", type=Path)
    args = parser.parse_args()
    receipt = ensure_backend(args.repo, args.backend_dir)
    print(
        json.dumps(
            dict(
                state=receipt["state"],
                module=receipt["module"],
                directory=str(backend_path(args.repo, args.backend_dir)),
                adapter_sha256=receipt["adapter_sha256"],
                patch_sha256=receipt["patch_sha256"],
                extension_sha256=receipt["extension_sha256"],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
