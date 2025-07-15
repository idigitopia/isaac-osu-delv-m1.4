#!/bin/bash

# --------------------[ Settings ]--------------------

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# --------------------[ Functions ]--------------------

print_info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# --------------------[ Script Start ]--------------------

# Check for target argument (file or folder)
if [ -z "$1" ]; then
    print_error "No target file or directory specified. Usage: ./run_formatter.sh <path to file or directory>"
    exit 1
fi
TARGET="$1"

# Store current PYTHONPATH if in a Conda environment
if [ -n "${CONDA_DEFAULT_ENV}" ]; then
    CACHE_PYTHONPATH=${PYTHONPATH}
    export PYTHONPATH=""
    print_info "Cleared PYTHONPATH temporarily for Conda environment."
fi

# Ensure pre-commit is installed
if ! command -v pre-commit >/dev/null 2>&1; then
    print_info "pre-commit not found. Installing..."
    pip install --quiet pre-commit
    if [ $? -ne 0 ]; then
        print_error "Failed to install pre-commit. Aborting."
        exit 1
    fi
fi

# Prepare list of files
if [ -d "$TARGET" ]; then
    print_info "Target is a folder: ${TARGET}"
    FILES=$(git ls-files "$TARGET")
elif [ -f "$TARGET" ]; then
    print_info "Target is a file: ${TARGET}"
    FILES="$TARGET"
else
    print_error "Target '$TARGET' does not exist."
    exit 1
fi

# Check if there are any files to run pre-commit on
if [ -z "$FILES" ]; then
    print_error "No git-tracked files found in target '$TARGET'. Nothing to run."
    exit 0
fi

# Run pre-commit
print_info "Running pre-commit hooks..."
pre-commit run --files $FILES
RESULT=$?

# Restore PYTHONPATH if previously cleared
if [ -n "${CONDA_DEFAULT_ENV}" ]; then
    export PYTHONPATH=${CACHE_PYTHONPATH}
    print_info "Restored PYTHONPATH."
fi

# Final message
if [ $RESULT -eq 0 ]; then
    print_success "pre-commit completed successfully."
else
    print_error "pre-commit completed with errors."
    exit $RESULT
fi
