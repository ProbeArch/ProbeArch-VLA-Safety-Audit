"""Allow ``python -m probearch`` to use the same CLI as the console script."""

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
