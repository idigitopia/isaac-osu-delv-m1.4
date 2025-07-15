#!/bin/bash

FOLDER_NAME=$1
SOURCE_FILE=".vscode/launch.${FOLDER_NAME}.json"
TARGET_FILE=".vscode/launch.json"

if [ -z "$FOLDER_NAME" ]; then
  echo "❌ Usage: ./activate_launch.sh <folder_name>"
  echo "   Example: ./activate_launch.sh core"
  exit 1
fi

if [ ! -f "$SOURCE_FILE" ]; then
  echo "❌ Launch config '$SOURCE_FILE' does not exist."
  exit 1
fi

ln -sf "$(basename "$SOURCE_FILE")" "$TARGET_FILE"

echo ""
echo "🔁 Switching launch configuration"
echo "──────────────────────────────────────────────"
echo "📂 Source : $SOURCE_FILE"
echo "🔗 Target : $TARGET_FILE"
echo "✅ Now using: launch.$FOLDER_NAME.json"
echo "──────────────────────────────────────────────"