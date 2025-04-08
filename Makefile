# Makefile for Goo setup

# ANSI color codes
RED = \033[0;31m
GREEN = \033[0;32m
YELLOW = \033[0;33m
BLUE = \033[0;34m
MAGENTA = \033[0;35m
CYAN = \033[0;36m
WHITE = \033[0;37m
BOLD = \033[1m
RESET = \033[0m

# Load saved Blender path from .blender_path if it exists
-include .blender_path

# Initial setup
.PHONY: setup
setup:
	@if [ -f .blender_path ]; then \
		printf "$(YELLOW)Found existing Blender configuration. Use it? (y/n): $(RESET)" >&2; \
		read use_existing; \
		if [ "$$use_existing" = "y" ]; then \
			printf "$(GREEN)Using existing configuration...$(RESET)\n"; \
			$(MAKE) _setup; \
			exit 0; \
		fi; \
	fi; \
	while true; do \
		printf "$(YELLOW)Enter the path to your Blender executable: $(RESET)" >&2; \
		read blender_path; \
		if [ -f "$$blender_path" ]; then \
			printf "BLENDER_PATH := %s\n" "$$blender_path" > .blender_path; \
			break; \
		else \
			printf "$(RED)$(BOLD)Error:$(RESET) Blender executable not found at $$blender_path\n"; \
			printf "$(YELLOW)Please try again$(RESET)\n"; \
		fi; \
	done; \
	printf "$(YELLOW)Is this Blender 4.0? (y/n): $(RESET)" >&2; \
	read is_40; \
	if [ "$$is_40" = "y" ]; then \
		BLENDER_VERSION=4.0; \
		PYTHON_VERSION=3.10; \
		BLENDER_TAG=40; \
	else \
		BLENDER_VERSION=4.1; \
		PYTHON_VERSION=3.11; \
		BLENDER_TAG=45; \
	fi; \
	printf "BLENDER_VERSION := %s\n" "$$BLENDER_VERSION" >> .blender_path; \
	printf "PYTHON_VERSION := %s\n" "$$PYTHON_VERSION" >> .blender_path; \
	printf "BLENDER_TAG := %s\n" "$$BLENDER_TAG" >> .blender_path; \
	$(MAKE) _setup

# Internal setup target that requires BLENDER_PATH
.PHONY: _setup
_setup:
	@if [ -z "$(BLENDER_PATH)" ]; then \
		printf "$(RED)$(BOLD)Error:$(RESET) Please run 'make setup' first\n"; \
		exit 1; \
	fi
	@if [ ! -f "$(BLENDER_PATH)" ]; then \
		printf "$(RED)$(BOLD)Error:$(RESET) Blender executable not found at $(BLENDER_PATH)\n"; \
		printf "$(YELLOW)Please run 'make setup' again with the correct path$(RESET)\n"; \
		exit 1; \
	fi
	@printf "$(CYAN)$(BOLD)=== Goo Setup ===$(RESET)\n"
	@printf "$(GREEN)✓ Blender executable found$(RESET)\n"
	@printf "$(GREEN)Detected Blender version:$(RESET) $(BLENDER_VERSION)\n"
	@printf "$(GREEN)Using Python version:$(RESET) $(PYTHON_VERSION)\n\n"
	@printf "$(YELLOW)This setup will:$(RESET)\n"
	@printf "  1. Create a virtual environment\n"
	@printf "  2. Install dependencies\n"
	@printf "  3. Set up Blender hooks\n\n"
	@printf "$(YELLOW)Press Enter to continue or Ctrl+C to cancel...$(RESET)\n"
	@read
	$(MAKE) create_venv install_requirements create_hook

# Derive Python path from Blender path
BLENDER_DIR = $(dir $(BLENDER_PATH))
ifeq ($(shell uname),Darwin)
    BPY_PATH = $(BLENDER_DIR)../Resources/$(BLENDER_VERSION)/python/bin/python$(PYTHON_VERSION)
else
    BPY_PATH = $(BLENDER_DIR)$(BLENDER_VERSION)/python/bin/python$(PYTHON_VERSION)
endif

# Do not change these options!
VENV_DIR = .blender_venv$(BLENDER_VERSION)
HOOK_DIR = hook_blender$(BLENDER_VERSION)
VENV_PACKAGES = $(VENV_DIR)/lib/python$(PYTHON_VERSION)/site-packages
VENV_PYTHON = $(VENV_DIR)/bin/python
HOOK_PACKAGES = $(HOOK_DIR)/scripts/modules

TEST_DIR = unit_tests

.PHONY: update_modules clean create_venv install_requirements create_hook test testone install setup_hook all info

# --- Use these targets ---
info:
	@printf "$(CYAN)$(BOLD)=== Goo Setup Information ===$(RESET)\n"
	@printf "$(GREEN)Blender executable:$(RESET) $(BLENDER_PATH)\n"
	@printf "$(GREEN)Blender version:$(RESET) $(BLENDER_VERSION)\n"
	@printf "$(GREEN)Using Python version:$(RESET) $(PYTHON_VERSION)\n"
	@printf "$(GREEN)Virtual environment directory:$(RESET) $(VENV_DIR)\n"
	@printf "$(GREEN)Hook directory:$(RESET) $(HOOK_DIR)\n"
	@printf "$(CYAN)$(BOLD)===========================$(RESET)\n"

welcome:
	@printf "$(CYAN)$(BOLD)=== Goo Setup ===$(RESET)\n"
	@printf "$(YELLOW)Welcome to Goo setup!$(RESET)\n\n"
	@printf "$(GREEN)Detected Blender version:$(RESET) $(BLENDER_VERSION)\n"
	@printf "$(GREEN)Using Python version:$(RESET) $(PYTHON_VERSION)\n\n"
	@printf "$(YELLOW)This setup will:$(RESET)\n"
	@printf "  1. Create a virtual environment\n"
	@printf "  2. Install dependencies\n"
	@printf "  3. Set up Blender hooks\n\n"

check_blender:
	@printf "$(CYAN)$(BOLD)=== Checking Blender Installation ===$(RESET)\n"
	@printf "$(GREEN)✓ Blender executable found$(RESET)\n"
	@printf "$(GREEN)Detected Blender version:$(RESET) $(BLENDER_VERSION)\n"
	@printf "$(GREEN)Using Python version:$(RESET) $(PYTHON_VERSION)\n"

update_modules:
	@printf "$(YELLOW)$(BOLD)Updating modules...$(RESET)\n"
	@$(MAKE) create_hook

clean:
	@printf "$(YELLOW)$(BOLD)Cleaning up...$(RESET)\n"
	@rm -rf .blender_venv4.0 .blender_venv4.1
	@rm -rf hook_blender*
	@rm -rf temp_install
	@printf "$(GREEN)Cleanup complete!$(RESET)\n"

test:
	@if [ -z "$(BLENDER_PATH)" ]; then \
		printf "$(RED)$(BOLD)Error:$(RESET) Please run 'make setup' first\n"; \
		exit 1; \
	fi
	@printf "$(CYAN)$(BOLD)Running tests...$(RESET)\n"
ifeq ($(t),)
	$(BLENDER_PATH) --background --python-expr "import pytest; pytest.main(['-v', './tests'])"
else
	$(BLENDER_PATH) --background --python-expr "import pytest; pytest.main(['-v', '$(t)'])"
endif

# --- Don't use these targets ---
create_venv:
	@printf "$(CYAN)$(BOLD)=== Creating Virtual Environment ===$(RESET)\n"
	@if [ ! -d "$(VENV_DIR)" ]; then \
		printf "$(YELLOW)Creating virtual environment for Blender $(BLENDER_VERSION)...$(RESET)\n"; \
		$(BPY_PATH) -m venv $(VENV_DIR); \
		printf "$(GREEN)✓ Virtual environment created successfully!$(RESET)\n"; \
	else \
		printf "$(YELLOW)Virtual environment already exists.$(RESET)\n"; \
	fi

install_requirements:
	@printf "$(CYAN)$(BOLD)=== Installing Dependencies ===$(RESET)\n"
	@printf "$(YELLOW)This will install the following packages:$(RESET)\n"
	@printf "  - Core Goo dependencies\n"
	@printf "  - Blender $(BLENDER_VERSION) specific packages\n"
	@printf "  - Development tools\n"
	$(VENV_PYTHON) -m pip install -e ".[blender$(BLENDER_TAG),dev]"
	@printf "$(GREEN)✓ Dependencies installed successfully!$(RESET)\n"

create_hook:
	@printf "$(CYAN)$(BOLD)=== Setting Up Blender Hook ===$(RESET)\n"
	@printf "$(YELLOW)This will create a hook directory and copy necessary files.$(RESET)\n"
	@mkdir -p $(HOOK_PACKAGES)
	@cp -r src/goo $(HOOK_PACKAGES)/
	@cp -a $(VENV_PACKAGES)/. $(HOOK_PACKAGES)/
	@printf "$(GREEN)✓ Hook setup complete!$(RESET)\n"
	@printf "$(YELLOW)Registering script directory in Blender...$(RESET)\n"
	@$(BLENDER_PATH) --background --python-expr "import bpy; [bpy.ops.preferences.script_directory_remove(index=i) for i, dir in enumerate(bpy.context.preferences.filepaths.script_directories) if 'Goo-blender' in dir.name]; bpy.ops.preferences.script_directory_add(directory='$(abspath $(HOOK_DIR))/scripts'); bpy.context.preferences.filepaths.script_directories[-1].name = 'Goo-blender-$(BLENDER_VERSION)'; bpy.ops.wm.save_userpref()"
	@printf "$(GREEN)✓ Script directory registered in Blender preferences$(RESET)\n"

all: clean setup

help:
	@printf "$(CYAN)$(BOLD)=== Goo Makefile Help ===$(RESET)\n"
	@printf "$(GREEN)Available targets:$(RESET)\n"
	@printf "  $(YELLOW)info$(RESET)      - Show current setup information\n"
	@printf "  $(YELLOW)setup$(RESET)     - Interactive setup for Goo\n"
	@printf "  $(YELLOW)test$(RESET)      - Run tests (use t=path for specific test)\n"
	@printf "  $(YELLOW)clean$(RESET)     - Clean all generated files\n"
	@printf "  $(YELLOW)update_modules$(RESET) - Update hook modules\n"
	@printf "\n"
	@printf "$(GREEN)Usage:$(RESET)\n"
	@printf "  $(YELLOW)make setup$(RESET)\n"
	@printf "\n"
	@printf "$(GREEN)Example:$(RESET)\n"
	@printf "  $(YELLOW)make setup$(RESET)\n"
	@printf "$(CYAN)$(BOLD)========================$(RESET)\n"