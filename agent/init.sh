#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

set -euo pipefail

CANNBOT_URL="https://gitcode.com/cann/cannbot-skills.git"
CANN_SAMPLES_URL="https://gitcode.com/cann/cann-samples.git"
ASC_DEVKIT_URL="https://gitcode.com/cann/asc-devkit.git"

if [ -t 1 ]; then
    GREEN='\033[0;32m'; YELLOW='\033[0;33m'; RED='\033[0;31m'
    CYAN='\033[0;36m'; BOLD='\033[1m'; DIM='\033[2m'; NC='\033[0m'
else
    GREEN=''; YELLOW=''; RED=''; CYAN=''; BOLD=''; DIM=''; NC=''
fi

ok()   { echo -e "  ${GREEN}✓${NC}${DIM} $*${NC}"; }
warn() { echo -e "  ${YELLOW}⚠${NC}${DIM} $*${NC}"; }
err()  { echo -e "  ${RED}✗${NC}${DIM} $*${NC}"; }
info() { echo -e "  ${CYAN}→${NC}${DIM} $*${NC}"; }
step() { echo -e "${DIM}$*${NC}"; }

VERSION="1.0.0"

show_help() {
    cat << EOF
ops-sparse Agent 初始化脚本

Usage: bash init.sh <claude|opencode> [options]

Arguments:
  claude                    Target: Claude Code
  opencode                  Target: OpenCode

Options:
  --help, -h                Show this help message
  --clean                   Remove existing config directories before init
  --cannbot <path>          Path to cannbot-skills directory (default: clone from official)
  --samples <path>          Path to cann-samples directory (default: clone from official)
  --asc <path>              Path to asc-devkit directory (default: clone from official)

Official URL:
  cannbot-skills: ${CANNBOT_URL}
  cann-samples:   ${CANN_SAMPLES_URL}
  asc-devkit:     ${ASC_DEVKIT_URL}

Examples:
  bash init.sh claude
  bash init.sh opencode
  bash init.sh claude --clean
  bash init.sh --clean
  bash init.sh claude --cannbot /path/to/cannbot-skills
  bash init.sh claude --samples /path/to/cann-samples --asc /path/to/asc-devkit
EOF
}

if [[ $# -lt 1 ]]; then
    echo -e "${RED}Error: Missing required argument <claude|opencode>${NC}"
    echo ""
    show_help
    exit 1
fi

case "$1" in
    --help|-h)
        show_help
        exit 0
        ;;
    --clean)
        TARGET_ENV=""
        CLEAN_MODE=true
        shift
        ;;
    claude|opencode)
        TARGET_ENV="$1"
        CLEAN_MODE=false
        shift
        ;;
    *)
        echo -e "${RED}Error: First argument must be 'claude' or 'opencode'${NC}"
        echo ""
        show_help
        exit 1
        ;;
esac

CANNBOT_PATH=""
SAMPLES_PATH=""
ASC_PATH=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --help|-h)
            show_help
            exit 0
            ;;
        --clean)
            CLEAN_MODE=true
            shift
            ;;
        --cannbot)
            if [ -z "$2" ] || [[ "$2" == --* ]]; then
                echo -e "${RED}Error: --cannbot requires a path argument${NC}"
                exit 1
            fi
            CANNBOT_PATH="$2"
            shift 2
            ;;
        --samples)
            if [ -z "$2" ] || [[ "$2" == --* ]]; then
                echo -e "${RED}Error: --samples requires a path argument${NC}"
                exit 1
            fi
            SAMPLES_PATH="$2"
            shift 2
            ;;
        --asc)
            if [ -z "$2" ] || [[ "$2" == --* ]]; then
                echo -e "${RED}Error: --asc requires a path argument${NC}"
                exit 1
            fi
            ASC_PATH="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}Error: Unknown argument '$1'${NC}"
            echo ""
            show_help
            exit 1
            ;;
    esac
done

SCRIPT_DIR=$(dirname "$(realpath "$0")")
AGENT_DIR="$SCRIPT_DIR"
OPS_SPARSE_DIR=$(realpath "$SCRIPT_DIR/..")

if [ "$CLEAN_MODE" = true ] && [ -z "$TARGET_ENV" ]; then
    echo -e "  ${BOLD}Cleaning up...${NC}"
    for dir_name in .claude .opencode .agent; do
        target="$OPS_SPARSE_DIR/$dir_name"
        if [ -d "$target" ] || [ -L "$target" ]; then
            rm -rf "$target"
            ok "$dir_name/ removed"
        else
            info "$dir_name/ not found, skipping"
        fi
    done
    echo ""
    echo -e "  ${GREEN}${BOLD}✓ Cleanup completed!${NC}"
    echo ""
    exit 0
fi

if [ "$TARGET_ENV" = "claude" ]; then
    CONFIG_DIR_NAME=".claude"
    GUIDE_DST_NAME="CLAUDE.md"
    QUICK_START_CMD="cd $OPS_SPARSE_DIR && claude"
else
    CONFIG_DIR_NAME=".opencode"
    GUIDE_DST_NAME="AGENTS.md"
    QUICK_START_CMD="cd $OPS_SPARSE_DIR && opencode"
fi

CONFIG_DIR="$OPS_SPARSE_DIR/$CONFIG_DIR_NAME"

if [ -n "$CANNBOT_PATH" ]; then
    CANNBOT_PATH=$(realpath "$CANNBOT_PATH" 2>/dev/null || echo "$CANNBOT_PATH")
    SKILLS_REPO="$CANNBOT_PATH"
else
    SKILLS_REPO="$CONFIG_DIR/ref-repos/cannbot-skills"
fi

if [ -n "$SAMPLES_PATH" ]; then
    SAMPLES_PATH=$(realpath "$SAMPLES_PATH" 2>/dev/null || echo "$SAMPLES_PATH")
fi

if [ -n "$ASC_PATH" ]; then
    ASC_PATH=$(realpath "$ASC_PATH" 2>/dev/null || echo "$ASC_PATH")
fi

echo -e "  ${BOLD}Configuration:${NC}"
echo -e "  target env:   ${CYAN}$TARGET_ENV${NC} (config dir: $CONFIG_DIR_NAME/)"
echo -e "  ops-sparse:   ${CYAN}$OPS_SPARSE_DIR${NC}"
if [ -n "$CANNBOT_PATH" ]; then
    echo -e "  cannbot:      ${CYAN}$CANNBOT_PATH${NC} (local)"
else
    echo -e "  cannbot:      ${CYAN}clone from $CANNBOT_URL${NC}"
fi
if [ -n "$SAMPLES_PATH" ]; then
    echo -e "  cann-samples: ${CYAN}$SAMPLES_PATH${NC} (local)"
else
    echo -e "  cann-samples: ${CYAN}clone from $CANN_SAMPLES_URL${NC}"
fi
if [ -n "$ASC_PATH" ]; then
    echo -e "  asc-devkit:   ${CYAN}$ASC_PATH${NC} (local)"
else
    echo -e "  asc-devkit:   ${CYAN}clone from $ASC_DEVKIT_URL${NC}"
fi
echo ""

if [ -n "$CANNBOT_PATH" ] && [ ! -d "$CANNBOT_PATH" ]; then
    err "cannbot-skills directory not found: $CANNBOT_PATH"
    exit 1
fi
if [ -n "$SAMPLES_PATH" ] && [ ! -d "$SAMPLES_PATH" ]; then
    err "cann-samples directory not found: $SAMPLES_PATH"
    exit 1
fi
if [ -n "$ASC_PATH" ] && [ ! -d "$ASC_PATH" ]; then
    err "asc-devkit directory not found: $ASC_PATH"
    exit 1
fi
if [ ! -d "$OPS_SPARSE_DIR" ]; then
    err "ops-sparse directory not found: $OPS_SPARSE_DIR"
    exit 1
fi

cd "$OPS_SPARSE_DIR"

step "[1/8] Creating $CONFIG_DIR_NAME directory and .agent/dev-docs..."
mkdir -p "$CONFIG_DIR"
ok "$CONFIG_DIR_NAME/ created"
mkdir -p "$OPS_SPARSE_DIR/.agent/dev-docs"
ok ".agent/dev-docs/ created"

step "[2/8] Linking agent configuration..."
agent_md="$AGENT_DIR/AGENT.md"
if [ -f "$agent_md" ]; then
    dst="$CONFIG_DIR/$GUIDE_DST_NAME"
    if [ -L "$dst" ] || [ -e "$dst" ]; then
        rm -f "$dst"
    fi
    ln -sf "$agent_md" "$dst"
    ok "$GUIDE_DST_NAME -> agent/AGENT.md"
else
    warn "agent/AGENT.md not found, skipping"
fi

step "[3/8] Linking agents..."
mkdir -p "$CONFIG_DIR/agents"
local_agents="$AGENT_DIR/agents"
agent_count=0

if [ -d "$local_agents" ]; then
    for agent_file in "$local_agents"/*.md; do
        [ -f "$agent_file" ] || continue
        agent_name=$(basename "$agent_file")
        agent_dst="$CONFIG_DIR/agents/$agent_name"
        if [ -L "$agent_dst" ] || [ -e "$agent_dst" ]; then
            rm -f "$agent_dst"
        fi
        ln -sf "$agent_file" "$agent_dst"
        agent_count=$((agent_count + 1))
        ok "agent: $agent_name"
    done
    [ "$agent_count" -eq 0 ] && warn "No agents found in agent/agents/"
else
    warn "agent/agents/ directory not found, skipping"
fi

step "[4/8] Setting up cannbot-skills..."

if [ -n "$CANNBOT_PATH" ]; then
    ok "Using local cannbot-skills: $CANNBOT_PATH"
else
    if [ -d "$SKILLS_REPO/.git" ]; then
        info "cannbot-skills already exists, updating..."
        cd "$SKILLS_REPO"
        pull_err=$(git pull --quiet 2>&1) || warn "git pull failed: $pull_err"
        cd "$OPS_SPARSE_DIR"
        ok "cannbot-skills updated"
    else
        info "Cloning cannbot-skills from $CANNBOT_URL ..."
        clone_err=$(git clone --quiet "$CANNBOT_URL" "$SKILLS_REPO" 2>&1) || {
            err "Failed to clone cannbot-skills from $CANNBOT_URL: $clone_err"
            exit 1
        }
        ok "cannbot-skills cloned"
    fi
fi

step "[5/8] Linking skills..."
mkdir -p "$CONFIG_DIR/skills"
find "$CONFIG_DIR/skills" -xtype l -delete 2>/dev/null || true
local_skills="$AGENT_DIR/skills"
local_skill_count=0

if [ -d "$local_skills" ]; then
    for skill_dir in "$local_skills"/*; do
        [ -d "$skill_dir" ] || continue
        skill_name=$(basename "$skill_dir")
        if [ "$skill_name" = "cannbot_references.json" ]; then
            continue
        fi
        skill_dst="$CONFIG_DIR/skills/$skill_name"
        if [ -L "$skill_dst" ] || [ -e "$skill_dst" ]; then
            rm -rf "$skill_dst"
        fi
        ln -sf "$skill_dir" "$skill_dst"
        local_skill_count=$((local_skill_count + 1))
        ok "local skill: $skill_name"
    done
    [ "$local_skill_count" -eq 0 ] && warn "No local skills found in agent/skills/"
else
    warn "agent/skills/ directory not found, skipping"
fi

step "[6/8] Linking cannbot skills from cannbot_references.json..."
refs_json="$local_skills/cannbot_references.json"

if [ -f "$refs_json" ]; then
    cannbot_count=0
    cannbot_failed=0

    if command -v python3 &> /dev/null; then
        while IFS='|' read -r skill_name skill_path; do
            [ -z "$skill_name" ] && continue
            skill_src="$SKILLS_REPO/$skill_path"
            skill_dst="$CONFIG_DIR/skills/$skill_name"

            if [ -d "$skill_src" ]; then
                if [ -L "$skill_dst" ] || [ -e "$skill_dst" ]; then
                    rm -rf "$skill_dst"
                fi
                ln -sf "$skill_src" "$skill_dst"
                cannbot_count=$((cannbot_count + 1))
                ok "cannbot skill: $skill_name -> $skill_path"
            else
                warn "cannbot skill not found: $skill_name ($skill_path)"
                cannbot_failed=$((cannbot_failed + 1))
            fi
        done < <(python3 -c "
import json, sys
with open('$refs_json', 'r') as f:
    data = json.load(f)
for skill_name, paths in data.items():
    for p in paths:
        print(f'{skill_name}|{p}')
" 2>/dev/null) || warn "Failed to parse cannbot_references.json (empty or invalid JSON)"

        [ "$cannbot_count" -gt 0 ] && ok "Linked ${cannbot_count} cannbot skills"
        [ "$cannbot_failed" -gt 0 ] && warn "${cannbot_failed} cannbot skills not found"
    else
        warn "python3 not available, skipping cannbot skills linking"
    fi
else
    warn "cannbot_references.json not found, skipping cannbot skills linking"
fi

step "[7/8] Setting up external reference repos..."

AGENT_DIR_PATH="$OPS_SPARSE_DIR/.agent"

SAMPLES_TARGET="$AGENT_DIR_PATH/cann-samples"
if [ -n "$SAMPLES_PATH" ]; then
    if [ -L "$SAMPLES_TARGET" ] || [ -e "$SAMPLES_TARGET" ]; then
        rm -rf "$SAMPLES_TARGET"
    fi
    ln -sf "$SAMPLES_PATH" "$SAMPLES_TARGET"
    ok "cann-samples -> $SAMPLES_PATH (symlink)"
else
    if [ -d "$SAMPLES_TARGET/.git" ]; then
        info "cann-samples already exists, updating..."
        cd "$SAMPLES_TARGET"
        pull_err=$(git pull --quiet 2>&1) || warn "git pull failed: $pull_err"
        cd "$OPS_SPARSE_DIR"
        ok "cann-samples updated"
    elif [ -L "$SAMPLES_TARGET" ]; then
        ok "cann-samples symlink exists"
    else
        info "Cloning cann-samples from $CANN_SAMPLES_URL ..."
        clone_err=$(git clone --quiet "$CANN_SAMPLES_URL" "$SAMPLES_TARGET" 2>&1) || {
            warn "Failed to clone cann-samples from $CANN_SAMPLES_URL: $clone_err"
        }
        [ -d "$SAMPLES_TARGET" ] && ok "cann-samples cloned"
    fi
fi

ASC_TARGET="$AGENT_DIR_PATH/asc-devkit"
if [ -n "$ASC_PATH" ]; then
    if [ -L "$ASC_TARGET" ] || [ -e "$ASC_TARGET" ]; then
        rm -rf "$ASC_TARGET"
    fi
    ln -sf "$ASC_PATH" "$ASC_TARGET"
    ok "asc-devkit -> $ASC_PATH (symlink)"
else
    if [ -d "$ASC_TARGET/.git" ]; then
        info "asc-devkit already exists, updating..."
        cd "$ASC_TARGET"
        pull_err=$(git pull --quiet 2>&1) || warn "git pull failed: $pull_err"
        cd "$OPS_SPARSE_DIR"
        ok "asc-devkit updated"
    elif [ -L "$ASC_TARGET" ]; then
        ok "asc-devkit symlink exists"
    else
        info "Cloning asc-devkit from $ASC_DEVKIT_URL ..."
        clone_err=$(git clone --quiet "$ASC_DEVKIT_URL" "$ASC_TARGET" 2>&1) || {
            warn "Failed to clone asc-devkit from $ASC_DEVKIT_URL: $clone_err"
        }
        [ -d "$ASC_TARGET" ] && ok "asc-devkit cloned"
    fi
fi

step "[8/8] Generating opencode.json from model_config.json..."

if [ "$TARGET_ENV" = "opencode" ]; then
    CUSTOM_MODEL_JSON="$AGENT_DIR/agents/model_config.json"
    OPENCODE_JSON="$OPS_SPARSE_DIR/opencode.json"

    if [ -f "$CUSTOM_MODEL_JSON" ] && command -v python3 &> /dev/null; then
        AVAILABLE_MODELS=""
        if command -v opencode &> /dev/null; then
            AVAILABLE_MODELS=$(opencode models 2>/dev/null || true)
        fi
        python3 - "$CUSTOM_MODEL_JSON" "$OPENCODE_JSON" "$AVAILABLE_MODELS" << 'PYEOF'
import json, sys, os

custom_model_path = sys.argv[1]
opencode_json_path = sys.argv[2]
available_models_raw = sys.argv[3] if len(sys.argv) > 3 else ""

available_models = set()
for line in available_models_raw.strip().split("\n"):
    line = line.strip()
    if line:
        available_models.add(line)

with open(custom_model_path, 'r') as f:
    custom = json.load(f)

agent_config = {}
default_agents = []
for agent_name, cfg in custom.items():
    if agent_name.startswith("_") or agent_name == "comment":
        continue
    model = cfg.get("model", "default")
    if model and model != "default":
        if available_models and model not in available_models:
            print(f"  \033[0;33m⚠\033[0m\033[2m {agent_name}: model '{model}' not available, falling back to default\033[0m")
            default_agents.append(agent_name)
        else:
            agent_config[agent_name] = {"model": model}
    else:
        default_agents.append(agent_name)

existing = {}
if os.path.exists(opencode_json_path):
    with open(opencode_json_path, 'r') as f:
        existing = json.load(f)

existing_agents = existing.get("agent", {})
for name in default_agents:
    existing_agents.pop(name, None)

existing_agents.update(agent_config)

if existing_agents:
    existing["agent"] = existing_agents
else:
    existing.pop("agent", None)

if not existing:
    if os.path.exists(opencode_json_path):
        os.remove(opencode_json_path)
        print("  \033[0;32m✓\033[0m\033[2m opencode.json removed (all agents use default)\033[0m")
    else:
        print("  \033[2m→ All agents use default model, skipping opencode.json\033[0m")
    sys.exit(0)

with open(opencode_json_path, 'w') as f:
    json.dump(existing, f, indent=2, ensure_ascii=False)
    f.write("\n")

count = len(agent_config)
agents_list = ", ".join(agent_config.keys()) if agent_config else "none"
print(f"  \033[0;32m✓\033[0m\033[2m opencode.json updated ({count} custom agents: {agents_list})\033[0m")
PYEOF
    elif [ ! -f "$CUSTOM_MODEL_JSON" ]; then
        info "model_config.json not found, skipping"
    else
        warn "python3 not available, skipping opencode.json generation"
    fi
else
    info "Target is claude, skipping opencode.json generation"
fi

echo ""
echo -e "  ${GREEN}${BOLD}✓ Initialization completed!${NC}"
echo ""
echo -e "  ${BOLD}Usage:${NC}"
echo -e "  $QUICK_START_CMD"
echo ""
