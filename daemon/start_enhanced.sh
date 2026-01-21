#!/bin/bash
# NovaSR Enhanced Audio - Start Script

# Use virtual environment python
PYTHON="/home/benbi/.local/share/nova-sr-enhancer/bin/python"
DAEMON="/home/benbi/.local/share/nova-sr-enhancer/daemon/novasr_enhanced_daemon.py"

# Ensure we're using the venv with NovaSR installed
cd "/home/benbi/.local/share/nova-sr-enhancer"
exec "$PYTHON" "$DAEMON"
