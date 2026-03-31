#!/usr/bin/env python3
"""Diagnose why FRED may not be used for market data."""

import os
from pathlib import Path

# Mimic market_data's load_dotenv
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(_env_path)
    print(f"Loaded .env from: {_env_path}")
    print(f"  File exists: {_env_path.exists()}")
except ImportError:
    print("python-dotenv not installed")
    _env_path = None

key = (os.environ.get("FRED_API_KEY") or "").strip()
print(f"\nFRED_API_KEY in env: {'yes' if key else 'no'}")
if key:
    print(f"  Length: {len(key)} chars")
    print(f"  First 4 chars: {key[:4]}...")

try:
    from fredapi import Fred
    print("\nfredapi: installed")
except ImportError as e:
    print(f"\nfredapi: NOT INSTALLED ({e})")
    print("  Run: pip install fredapi")
    exit(1)

if key:
    print("\nTesting FRED API call...")
    try:
        fred = Fred(api_key=key)
        s = fred.get_series("VIXCLS", observation_start="2024-01-01", observation_end="2024-01-31")
        print(f"  OK: got {len(s)} observations")
    except Exception as e:
        print(f"  FAILED: {e}")
