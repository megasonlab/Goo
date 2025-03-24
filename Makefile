# Makefile for Goo setup

# Change the following two paths:
BLENDER_PATH = /Users/antoine/Harvard/MegasonLab/build_darwin/bin/Blender.app/Contents/MacOS/Blender
BPY_PATH = /Users/antoine/Harvard/MegasonLab/build_darwin/bin/Blender.app/Contents/Resources/4.5/python/bin/python3.11

# Do not change these options!
VENV_DIR = .blender_venv
HOOK_DIR = hook
VENV_PACKAGES = $(VENV_DIR)/lib/python3.11/site-packages
VENV_PYTHON = $(VENV_DIR)/bin/python
HOOK_PACKAGES = $(HOOK_DIR)/scripts/modules

TEST_DIR = unit_tests

.PHONY: setup update_modules clean create_venv install_requirements create_hook test testone

# --- Use these targets ---
setup: create_venv install_requirements create_hook

update_modules:
	@if [ -d "$(HOOK_PACKAGES)" ]; then \
		echo "Removing previous hook..."; \
		rm -rf $(HOOK_PACKAGES); \
	fi
	@echo "Updating modules..."
	@$(MAKE) create_hook

clean:
	rm -rf $(VENV_DIR)
	rm -rf $(HOOK_DIR)

test:
ifeq ($(t),)
	$(BLENDER_PATH) --background --python-expr "import pytest; pytest.main(['-v', './tests'])"
else
	$(BLENDER_PATH) --background --python-expr "import pytest; pytest.main(['-v', '$(t)'])"
endif


# --- Don't use these targets ---
create_venv:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "Creating virtual environment..."; \
		$(BPY_PATH) -m venv $(VENV_DIR); \
	else \
		echo "Virtual environment already created."; \
	fi

install_requirements:
	@echo "Installing dependencies to virtual environment..."
	$(VENV_PYTHON) -m pip install -r requirements.txt

create_hook:
	@echo "Creating hook directory..."
	mkdir -p $(HOOK_PACKAGES)
	cp -a $(VENV_PACKAGES)/. $(HOOK_PACKAGES)/