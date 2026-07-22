#!/usr/bin/env bash
set -euo pipefail
# MangaHue Python Bot - VPS Setup Script
# Run: sudo bash setup.sh

echo "=== Setting up MangaHue Python Bot ==="

# 1. Install system deps
apt-get update -qq
apt-get install -y -qq python3-venv python3-pip

# 2. Create bot directory
mkdir -p /opt/mangahue_bot/data/usage
cp bot.py colorize.py downloader.py postprocess.py requirements.txt .env mangahue-bot.service /opt/mangahue_bot/

# 3. Create virtual environment
python3 -m venv /opt/mangahue_bot/venv
/opt/mangahue_bot/venv/bin/pip install --upgrade pip
/opt/mangahue_bot/venv/bin/pip install -r /opt/mangahue_bot/requirements.txt

# 4. Install systemd service
cp /opt/mangahue_bot/mangahue-bot.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable mangahue-bot

# 5. Start
echo ""
echo "Edit /opt/mangahue_bot/.env and set your REPLICATE_API_TOKEN"
echo "Then: systemctl start mangahue-bot"
echo "      journalctl -u mangahue-bot -f"
