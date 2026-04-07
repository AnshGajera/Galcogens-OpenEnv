#!/usr/bin/env bash
#
# validate-submission.sh — OpenEnv Submission Validator
#
# Usage:
#   ./validate-submission.sh https://your-space.hf.space
#

set -uo pipefail

PING_URL="${1:-}"

if [ -z "$PING_URL" ]; then
    printf "Usage: %s <ping_url>\n" "$0"
    exit 1
fi

PING_URL="${PING_URL%/}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

log()  { printf "[%s] %s\n" "$(date -u +%H:%M:%S)" "$*"; }
pass() { log "${GREEN}PASSED${NC} -- $1"; }
fail() { log "${RED}FAILED${NC} -- $1"; }

printf "\n${BOLD}========================================${NC}\n"
printf "${BOLD}  OpenEnv Submission Validator${NC}\n"
printf "${BOLD}========================================${NC}\n"
log "Ping URL: $PING_URL"
printf "\n"

# Step 1: Ping HF Space
log "${BOLD}Step 1/3: Pinging HF Space${NC} ($PING_URL/reset) ..."

HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST \
  -H "Content-Type: application/json" -d '{}' \
  "$PING_URL/reset" --max-time 30 2>/dev/null || echo "000")

if [ "$HTTP_CODE" = "200" ]; then
    pass "HF Space is live and responds to /reset"
else
    fail "HF Space /reset returned HTTP $HTTP_CODE (expected 200)"
    exit 1
fi

# Step 2: Check Dockerfile exists
log "${BOLD}Step 2/3: Checking Dockerfile${NC} ..."

if [ -f "Dockerfile" ] || [ -f "server/Dockerfile" ]; then
    pass "Dockerfile found"
else
    fail "No Dockerfile found"
    exit 1
fi

# Step 3: Check openenv.yaml
log "${BOLD}Step 3/3: Checking openenv.yaml${NC} ..."

if [ -f "openenv.yaml" ]; then
    pass "openenv.yaml found"
else
    fail "openenv.yaml not found"
    exit 1
fi

printf "\n${BOLD}========================================${NC}\n"
printf "${GREEN}${BOLD}  All 3/3 checks passed!${NC}\n"
printf "${GREEN}${BOLD}  Ready for submission!${NC}\n"
printf "${BOLD}========================================${NC}\n"
printf "\n"

exit 0