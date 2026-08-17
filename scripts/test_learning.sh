#!/usr/bin/env bash
# Test the mangahue-learning crate
set -e
cd "$(dirname "$0")/.."
echo "=== Building mangahue-learning ==="
cargo build -p mangahue-learning 2>&1 | tail -5
echo ""
echo "=== Running mangahue-learning tests ==="
cargo test -p mangahue-learning 2>&1 | tail -20
echo ""
echo "=== Done ==="
