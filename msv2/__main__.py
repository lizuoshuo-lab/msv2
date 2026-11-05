import sys

from .cli import main

try:
    raise SystemExit(main())
except KeyboardInterrupt:
    print("MSV2_INTERRUPTED", file=sys.stderr)
    raise SystemExit(130)
except Exception as error:
    print(f"MSV2_FAILED: {error}", file=sys.stderr)
    raise SystemExit(1)
