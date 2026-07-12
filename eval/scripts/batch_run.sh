#!/bin/bash
# Convenience script for batch running GUI-agent evaluation tasks.
# Supports both WebVoyager and Mind2Web datasets — configure paths in batch_config.yaml.

set -e

# Get the directory where this script resides (scripts/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Change to project root directory
cd "$PROJECT_ROOT"

# ========================================
# Set required environment variables
# ========================================

# Set EasyAgent config file path (specifies available models)
export EA_DEFAULT_CONFIG="${EA_DEFAULT_CONFIG:-$PROJECT_ROOT/global_configs/model_config.yaml}"
echo "EA_DEFAULT_CONFIG: $EA_DEFAULT_CONFIG"

# Default config file
CONFIG="${CONFIG:-scripts/batch_config.yaml}"

# Parse arguments
DRY_RUN=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN="--dry-run"
            shift
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dry-run] [--config CONFIG_FILE]"
            exit 1
            ;;
    esac
done

echo "========================================="
echo "Batch GUI-Agent Task Runner"
echo "========================================="
echo "Config: $CONFIG"
echo "Project Root: $PROJECT_ROOT"
echo "========================================="
echo ""

# Run batch tasks
python scripts/run_batch_web_tasks.py --config "$CONFIG" $DRY_RUN

echo ""
echo "========================================="
echo "Batch run completed!"
echo "========================================="
