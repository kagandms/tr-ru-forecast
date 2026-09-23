#!/usr/bin/env python3
"""Executable script for running UN Comtrade trade data ingestion."""

import sys

from tr_ru_forecast.data.ingestion.cli import main

if __name__ == "__main__":
    sys.exit(main())
