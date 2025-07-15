#!/bin/bash

# Colors
RED='\033[1;31m'
GREEN='\033[1;32m'
BLUE='\033[1;34m'
YELLOW='\033[1;33m'
CYAN='\033[1;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}============================================================${NC}"
echo -e "${YELLOW}🔗 Installation Guide: ${NC} https://isaac-sim.github.io/IsaacLab/v2.0.0/source/setup/installation/pip_installation.html"
echo -e "${CYAN}============================================================${NC}\n"


##################################
# Activate the conda env to use. #
##################################
echo -e "${BLUE}🔹 Creating/activating conda environment...${NC}"

# Ensure Conda is initialized
if ! command -v conda &> /dev/null; then
    echo -e "${RED}❌ Error: Conda is not installed or not in PATH.${NC}"
    exit 1
fi

# Initialize Conda in case it's not initialized properly
eval "$(conda shell.bash hook 2> /dev/null || conda shell.zsh hook 2> /dev/null)"

# Prompt for conda environment name or empty to use active env
echo -e "${YELLOW}Enter name for new conda environment (or press Enter to use active env): ${NC}\c"
read env_name

if [ -z "$env_name" ]; then
    # Check if there's an active conda environment
    if [ -z "$CONDA_DEFAULT_ENV" ]; then
        echo -e "${RED}❌ Error: No conda environment name provided and no active conda environment.${NC}"
        exit 1
    else
        echo -e "${GREEN}✅ Using active conda environment: ${NC}${YELLOW}$CONDA_DEFAULT_ENV${NC}"
    fi
else
    # Check if environment already exists
    if conda env list | grep -q "^$env_name "; then
        echo -e "${GREEN}✅ Using existing conda environment: ${YELLOW}$env_name${NC}"
    else
        echo -e "${YELLOW}⚡ Creating conda environment: ${NC}$env_name"
        conda create -n "$env_name" python=3.10 -y
    fi
    echo -e "${CYAN}🔄 Activating conda environment: ${NC}$env_name"
    conda activate "$env_name"
fi


#########################
# Install torch. #
#########################
echo -e "${BLUE}🔹 Installing torch...${NC}"
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118


#########################
# Upgrade pip. #
#########################
echo -e "${BLUE}🔹 Upgrading pip...${NC}"
pip install --upgrade pip


########################################
# Install Isaac Sim and dependencies. #
########################################
echo -e "${BLUE}🔹 Installing Isaac Sim...${NC}"
pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com


#########################
# Install dependencies. #
#########################
echo -e "${BLUE}🔹 Installing dependencies...${NC}"
pip install -r requirements.txt


###################################
# Install ROS 2 and dependencies. #
###################################
echo -e "${BLUE}🔹 Install ROS 2 ${YELLOW}YOURSELF${NC} for ${CYAN}>>isaacsim${NC}"
echo -e "${YELLOW}📖 Docs at: https://docs.ros.org/en/dashing/Installation/Ubuntu-Install-Binary.html${NC}"
echo -e "${RED}⚠️  Note: ROS 2 bridge might require manual configuration.${NC}\n"


######################
# Install Isaac Lab. #
######################
echo -e "${BLUE}🔹 Installing Isaac Lab...${NC}"

# Check if current directory is IsaacLab
if [ -d "IsaacLab" ]; then
    echo -e "${GREEN}✅ Found IsaacLab in the current directory.${NC}"
else
    echo -e "${YELLOW}⚡ Cloning IsaacLab using gitman...${NC}"
    gitman update
fi
echo -e "${CYAN}🔄 Installing IsaacLab without RL frameworks...${NC}"
./IsaacLab/isaaclab.sh --install none
echo -e "${YELLOW}💡 Note: To create an environment manually, use:${NC} ./isaaclab.sh -i none -c my_env_name\n"


#############################
# Clone rsl_rl and install. #
#############################
echo -e "${BLUE}🔹 Installing rsl_rl...${NC}"
pip install -e drail_learning/rsl_rl


#############################
# Install drail_extensions. #
#############################
echo -e "${BLUE}🔹 Installing drail_extensions...${NC}"
pip install -e drail_extensions

########################
# Verify installation. #
########################
echo -e "${CYAN}==============================================================${NC}"
echo -e "${GREEN}🎉 Congratulations! You have set up the repo successfully. 🎉${NC}"
echo -e "${CYAN}==============================================================${NC}\n"

echo -e "${BLUE}🔹 Test installation (Option 1):${NC} ${YELLOW}./IsaacLab/isaaclab.sh -p IsaacLab/scripts/tutorials/00_sim/create_empty.py${NC}"
echo -e "${BLUE}🔹 Test installation (Option 2):${NC} ${YELLOW}python IsaacLab/scripts/tutorials/00_sim/create_empty.py${NC}\n"

echo -e "${GREEN}🚀 Happy coding! 🚀${NC}\n"