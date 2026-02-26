#!/bin/bash
set -e

echo "Install nano"
apt-get update
apt-get install -y nano

echo "Install tmux"
apt-get install -y tmux

echo "✅ Pre install concluded successfully!"