#!/bin/bash

# ==============================================================================
#  SASAIE Hummingbot Script Runner
# ==============================================================================
#
#  This script remotely starts a script inside the running Hummingbot
#  container by interacting with the Hummingbot REST API.
#
#  Usage:
#    ./run_hbot_script.sh <script_name.py>
#
#  Example:
#    ./run_hbot_script.sh sasaie_order_book_downloader.py
#
# ==============================================================================

# --- Configuration ---
API_BASE_URL="http://localhost:8000/api/v1"
API_USERNAME="${API_USERNAME:-admin}"
API_PASSWORD="${API_PASSWORD:-admin}"
# This should be the name of the API container, not the hummingbot client container
CONTAINER_NAME="sasaie-hummingbot-api-development"

# --- Functions ---

log_info() {
    echo -e "\e[34m[INFO]\e[0m $1"
}

log_success() {
    echo -e "\e[32m[SUCCESS]\e[0m $1"
}

log_error() {
    echo -e "\e[31m[ERROR]\e[0m $1"
    exit 1
}

check_command() {
    command -v "$1" >/dev/null 2>&1 || log_error "$1 is not installed. Please install it and try again."
}

# --- Main Script ---

# 1. Check for prerequisites
check_command curl
check_command jq

# 2. Check for script name argument
if [ -z "$1" ]; then
    log_error "Usage: $0 <script_name.py>"
fi
SCRIPT_NAME=$1

# 3. Check if API container is running
log_info "Checking if Hummingbot API container ('${CONTAINER_NAME}') is running..."
if ! docker ps --filter "name=${CONTAINER_NAME}" --format "{{.Names}}" | grep -q "${CONTAINER_NAME}"; then
    log_error "Container ${CONTAINER_NAME} is not running. Please start the system with 'python3 launch.py start'"
fi
log_info "API container is running."

# 4. Authenticate and get JWT token
log_info "Authenticating with the Hummingbot API..."
LOGIN_RESPONSE=$(curl -s -X POST "${API_BASE_URL}/login"
    -H "Content-Type: application/json"
    -d "{\"username\": \"${API_USERNAME}\", \"password\": \"${API_PASSWORD}\"}")

TOKEN=$(echo "${LOGIN_RESPONSE}" | jq -r '.token')

if [ -z "$TOKEN" ] || [ "$TOKEN" == "null" ]; then
    log_error "Failed to get authentication token. Response: ${LOGIN_RESPONSE}"
fi
log_info "Successfully authenticated."

# 5. Start the script
log_info "Sending request to start script: ${SCRIPT_NAME}..."
# Note: The script path inside the container is /home/hummingbot/scripts/
START_RESPONSE=$(curl -s -X POST "${API_BASE_URL}/scripts/start"
    -H "Content-Type: application/json"
    -H "Authorization: Bearer ${TOKEN}"
    -d "{\"script_name\": \"${SCRIPT_NAME}\"}")

# 6. Check response
if echo "${START_RESPONSE}" | jq -e '.status == "success"' > /dev/null; then
    MSG=$(echo "${START_RESPONSE}" | jq -r '.msg')
    log_success "Script started successfully: ${MSG}"
else
    ERROR_MSG=$(echo "${START_RESPONSE}" | jq -r '.msg')
    log_error "Failed to start script: ${ERROR_MSG}"
fi

exit 0
