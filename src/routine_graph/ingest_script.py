"""Compatibility launcher: python -m routine_graph.ingest_script <JSON inputs>."""

import sys

from routine_graph.cli import main

if __name__ == "__main__":

    main(["ingest", *sys.argv[1:]])
