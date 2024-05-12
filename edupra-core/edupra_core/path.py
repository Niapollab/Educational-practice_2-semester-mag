import os
import sys


def ensure_exists(path: str) -> None:
    if not os.path.exists(path):
        print("The path {} doesn't exists".format(path), file=sys.stderr)
        sys.exit()
