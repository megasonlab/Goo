Developer's Guide
==================

Development Environment
--------------------------

We recommend using either Visual Studio Code (VS Code) or Cursor as your primary IDE. The project uses a ``Makefile`` to manage the development environment and provides a robust setup for working with Blender's Python ecosystem.

Development Setup
~~~~~~~~~~~~~~~~~~

1. **Fork and Clone**

   Start by forking and cloning the `Goo repository <https://github.com/megasonlab/Goo>`__.

2. **Install Development Dependencies**

   Run the setup command:

   .. code-block:: bash

       make setup

   The setup process will:

   - Prompt for your Blender executable path
   - Create a virtual environment with the correct Python version
   - Install all required dependencies
   - Set up Blender hooks for development

   **Blender Executable Location:**
   
   - macOS: ``/Applications/Blender-x.x.app/Contents/MacOS/Blender``
   - Windows: ``C:\Program Files\Blender Foundation\Blender x.x\blender.exe``

   The path is saved in ``.blender_path`` for future use. Subsequent ``make setup`` runs will offer to reuse this configuration.

3. **Configure Your IDE**

   For VS Code users, we recommend installing:
   
   - Python extension
   - Ruff extension for linting and formatting
   
   The project includes pre-configured settings in ``pyproject.toml`` for:
   
   - Code formatting (88 character line length)
   - Import sorting
   - Linting rules optimized for Blender development

Development Workflow
---------------------

Testing
~~~~~~~~~

Run the test suite with:

.. code-block:: bash

    make test

To run a specific test file:

.. code-block:: bash

    make test t=tests/path/to/test.py

The test suite:

- Runs in Blender's Python environment
- Uses pytest for testing
- Includes unit tests for core functionality
- Excludes physics simulation accuracy tests

Code Quality
~~~~~~~~~~~~~

The project uses Ruff for code quality enforcement. The configuration in ``pyproject.toml`` follows:

- Google-style docstrings
- PEP 8 style guide with specific adaptations for Blender
- Automatic import sorting
- Type checking support

Documentation
~~~~~~~~~~~~~~~~

Documentation uses Sphinx and follows Google-style docstrings. To build the docs:

.. code-block:: bash

    make docs

This will:

1. Use Blender's Python interpreter to ensure compatibility
2. Build HTML documentation in ``docs/build/html``
3. Include all necessary dependencies from your setup

The documentation will be available at ``docs/build/html/index.html``.

Publishing documentation:

1. Build the documentation as shown above
2. Copy contents from ``docs/build/html`` to a temporary directory
3. Switch to the ``gh-pages`` branch
4. Copy the contents from the temporary directory to the root
5. Commit and create a pull request

Useful Commands
----------------

- ``make info``: Display current configuration
- ``make clean``: Clean build artifacts and hook directories
- ``make update_modules``: Update Blender hook modules
- ``make setup``: Initial setup or reconfiguration
- ``make test``: Run test suite
- ``make goo``: Update Goo library in both Blender's Python path and hook directory (useful during development)

For a complete list of available commands, run:

.. code-block:: bash

    make help