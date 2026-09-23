#!/usr/bin/env python3
"""Executable script for running bilateral mirror statistics & quality layer (M2)."""

import sys

from tr_ru_forecast.data.quality.cli import main

if __name__ == "__main__":
    sys.exit(main())
