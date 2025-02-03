#!/bin/sh
pip install beartype pytest
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install fair-esm==2.0.0 --no-deps
