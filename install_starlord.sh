#!/bin/bash

# --- Configuration ---
# IMPORTANT: Replace <YOUR_FUZZYMONK_REPO_URL> with the actual URL of your fuzzymonk monorepo
# If you have split the monorepo, this should be the URL of your *independent* starlord repository.
REPO_URL="git@github.com:ChrisCalvin/starlord.git"
INSTALL_DIR="/opt/starlord" # Or /opt/starlord if you've split the monorepo
STARLORD_DIR="/starlord" # Adjust if INSTALL_DIR is /opt/starlord
PYTHON_VENV_DIR="${STARLORD_DIR}/venv"

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

log_info "Starting SASAIE Starlord installation..."

# 1. Check for prerequisites
log_info "Checking for required tools: git, docker, docker compose, python3..."
check_command git
check_command docker
check_command docker compose
check_command python3

# 2. Install Docker if not present (basic check, assumes apt-get based system)
if ! command -v docker >/dev/null 2>&1; then
    log_info "Docker not found. Attempting to install Docker..."
    sudo apt-get update || log_error "Failed to update apt-get."
    sudo apt-get install -y ca-certificates curl gnupg || log_error "Failed to install ca-certificates, curl, gnupg."
    sudo install -m 0755 -d /etc/apt/keyrings || log_error "Failed to create /etc/apt/keyrings."
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg || log_error "Failed to add Docker GPG key."
    sudo chmod a+r /etc/apt/keyrings/docker.gpg || log_error "Failed to set permissions for Docker GPG key."
    echo \
      "deb [arch=\"$(dpkg --print-architecture)\" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
      \"$(. /etc/os-release && echo \"$VERSION_CODENAME\")\" stable" | \
      sudo tee /etc/apt/sources.list.d/docker.list > /dev/null || log_error "Failed to add Docker repository."
    sudo apt-get update || log_error "Failed to update apt-get after adding Docker repo."
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin || log_error "Failed to install Docker components."
    log_info "Adding current user to docker group..."
    sudo usermod -aG docker "$USER" || log_error "Failed to add user to docker group. Please log out and log back in."
    log_info "Docker installed. Please log out and log back in for Docker group changes to take effect, then re-run this script."
    exit 0
fi

# 3. Clone the fuzzymonk repository (or starlord independent repo)
if [ ! -d "${INSTALL_DIR}" ]; then
    log_info "Creating installation directory: ${INSTALL_DIR}"
    sudo mkdir -p "${INSTALL_DIR}" || log_error "Failed to create installation directory."
    sudo chown "$USER":"$USER" "${INSTALL_DIR}" || log_error "Failed to set ownership of installation directory."
fi

if [ ! -d "${INSTALL_DIR}/.git" ]; then
    log_info "Cloning repository into ${INSTALL_DIR}..."
    git clone "${REPO_URL}" "${INSTALL_DIR}" || log_error "Failed to clone repository. Check REPO_URL and permissions."
else
    log_info "Repository already exists. Pulling latest changes..."
    (cd "${INSTALL_DIR}" && git pull) || log_error "Failed to pull latest changes for repository."
fi

# 4. Navigate to starlord and set up Python environment
log_info "Setting up Python environment for Starlord..."
if [ ! -d "${STARLORD_DIR}" ]; then
    log_error "Starlord directory not found at ${STARLORD_DIR}. Ensure REPO_URL is correct and contains starlord."
fi

(
    cd "${STARLORD_DIR}" || log_error "Failed to change directory to ${STARLORD_DIR}".
    python3 -m venv "${PYTHON_VENV_DIR}" || log_error "Failed to create Python virtual environment."
    source "${PYTHON_VENV_DIR}/bin/activate" || log_error "Failed to activate virtual environment."
    pip install --upgrade pip || log_error "Failed to upgrade pip."
    pip install -r requirements.txt || log_error "Failed to install Python dependencies from requirements.txt."
    log_info "Python environment setup complete."
)

# 5. Build and start Docker Compose services
log_info "Building and starting Docker Compose services for Starlord..."
(
    cd "${STARLORD_DIR}" || log_error "Failed to change directory to ${STARLORD_DIR}".
    # Ensure .env file is present if needed by docker-compose (e.g., for passwords)
    # You might need to create a .env file here or pass environment variables
    # For now, assuming launch.py handles password prompting or it's not needed for initial 'up'

    # Use the launch.py script to start services
    log_info "Starting Docker services via launch.py..."
    python3 launch.py start --force-restart --profile development # Assuming 'development' profile
    # Note: launch.py start will prompt for password if not provided.
    # For full automation, you might need to pass --password or configure it differently.
)

log_success "SASAIE Starlord installation complete!"
log_info "To check status: cd ${STARLORD_DIR} && python3 launch.py status"
log_info "To stop services: cd ${STARLORD_DIR} && python3 launch.py stop"
log_info "IMPORTANT: If Docker was just installed, you might need to log out and log back in for user permissions to take effect."
log_info "IMPORTANT: Remember to replace <YOUR_FUZZYMONK_REPO_URL> in the script with your actual repository URL."
