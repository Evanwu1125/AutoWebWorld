#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

export https_proxy=http://127.0.0.1:1087
export http_proxy=http://127.0.0.1:1087

THEME="traveling-airbnb"

python -m src.agent.fsm --theme "$THEME"
