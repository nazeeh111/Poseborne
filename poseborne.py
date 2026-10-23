"""Poseborne command facade. The original wipose.py entry point remains compatible."""
from wipose import main


if __name__ == "__main__":
    raise SystemExit(main())
