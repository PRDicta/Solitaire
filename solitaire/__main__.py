"""Allow running solitaire as: python -m solitaire"""
from solitaire.platform_utils import ensure_utf8
ensure_utf8()

from solitaire.cli import main
main()