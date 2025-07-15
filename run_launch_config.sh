#!/bin/bash

CONFIG_NAME="$1"
LAUNCH_FILE=".vscode/launch.json"
WORKSPACE=$(pwd)

echo -e "🔧  Launching config: \033[1;34m$CONFIG_NAME\033[0m"
echo -e "📂  Project root:     \033[1;32m$WORKSPACE\033[0m"
echo -e "📝  Config file:      \033[1;33m$LAUNCH_FILE\033[0m"
echo ""

if [ -z "$CONFIG_NAME" ]; then
  echo "❌  Usage: $0 <config_name>"
  exit 1
fi

if [ ! -f "$LAUNCH_FILE" ]; then
  echo "❌  launch.json not found at $LAUNCH_FILE"
  exit 1
fi

# Parse config using Python with json5 for relaxed JSON parsing
readarray -t CMD <<< $(python3 - "$CONFIG_NAME" "$LAUNCH_FILE" <<'EOF'
import sys, os

try:
    import json5
except ImportError:
    print("ERR: Please run 'pip install json5' to use this script.", file=sys.stderr)
    sys.exit(1)

config_name, path = sys.argv[1], sys.argv[2]
data = json5.load(open(path))

configs = data.get("configurations", [])
cfg = next((c for c in configs if c.get("name") == config_name), None)

if not cfg:
    print(f"ERR: No config found with name '{config_name}'", file=sys.stderr)
    sys.exit(1)

program = cfg.get("program", "")
args = cfg.get("args", [])
env = cfg.get("env", {})

cwd = os.getcwd()
program = program.replace("${workspaceFolder}", cwd).replace("${workspaceRoot}", cwd)

for k, v in env.items():
    print(f"export {k}='{v}'")

print("RUN_CMD_START")
print(program)
for a in args:
    print(a)
EOF
)

# Parse Python output
env_exports=()
run_cmd=()
in_cmd=0

for line in "${CMD[@]}"; do
  if [[ "$line" == ERR:* ]]; then
    echo -e "❌  $line"
    exit 1
  fi
  if [[ "$line" == "RUN_CMD_START" ]]; then
    in_cmd=1
    continue
  fi
  if [[ $in_cmd -eq 0 ]]; then
    env_exports+=("$line")
  else
    run_cmd+=("$line")
  fi
done

# Export environment variables
if [ ${#env_exports[@]} -gt 0 ]; then
  echo "🌿  Setting environment variables:"
  for e in "${env_exports[@]}"; do
    echo "    → $e"
    eval "$e"
  done
  echo ""
fi

# Show command being run
echo "🚀  Running command:"
echo -e "    \033[1m${run_cmd[*]}\033[0m"
echo ""

# Execute the command
if [[ "${run_cmd[0]}" == *.py ]]; then
  python "${run_cmd[@]}"
else
  "${run_cmd[@]}"
fi
