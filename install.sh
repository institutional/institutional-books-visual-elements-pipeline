#!/bin/bash

# DEBIAN & CO | System-level dependencies (partial)
if command -v apt >/dev/null 2>&1; then
    sudo apt install libgl1 libjpeg-dev libtiff-dev libpng-dev
fi

uv python install 3.12;
uv sync;

cp .env.example .env;