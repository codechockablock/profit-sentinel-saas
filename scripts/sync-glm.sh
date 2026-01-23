#!/bin/bash
# Sync GLM core from private repo

SOURCE=~/profit-sentinel-core
DEST=~/profit-sentinel-saas/packages/sentinel-engine/src/sentinel_engine

cp "$SOURCE/core.py" "$DEST/core.py"
cp "$SOURCE/context.py" "$DEST/context.py"

echo "✅ Synced core.py and context.py from profit-sentinel-core"
echo "   Don't forget to commit!"
