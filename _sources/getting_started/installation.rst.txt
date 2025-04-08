.. _installation:

Installation
=============

Prerequisites
-------------

Blender Installation
~~~~~~~~~~~~~~~~~~~~~

Goo runs in Blender. Currently supported versions:

- **Recommended:** Blender 4.0 LTS (`Download <https://www.blender.org/download/lts/4-0/>`__)
- **Supported:** Blender 3.3 through 4.0

.. warning::

   Blender 4.5 is currently not supported due to compatibility issues with the `RoadRunner Simulation Engine <https://libroadrunner.readthedocs.io/en/latest/index.html>`__. We are actively working on resolving this limitation.

System Requirements
~~~~~~~~~~~~~~~~~~~

- Python 3.10 or newer
- Operating System:
    - Windows 10/11
    - macOS 10.15 or newer
    - Linux (major distributions)

Installing Goo
---------------

1. **Clone Repository**

   Clone the Goo repository from GitHub:

   .. code-block:: bash

       git clone https://github.com/megasonlab/Goo.git
       cd Goo

2. **Run Setup**

   Execute the setup command:

   .. code-block:: bash

       make setup

   The setup will prompt for your Blender executable location:

   - **macOS:** ``/Applications/Blender-x.x.app/Contents/MacOS/Blender``
   - **Windows:** ``C:\\Program Files\\Blender Foundation\\Blender x.x\\blender.exe``
   - **Linux:** Typically ``/usr/bin/blender`` or the installation path

   Your Blender path is saved in ``.blender_path`` for future use.

How Setup Works
~~~~~~~~~~~~~~~~~

The setup process:

1. Creates a dedicated hook directory for Goo and its dependencies
2. Sets up an isolated Python environment
3. Installs all required packages
4. Configures Blender to recognize the Goo installation

This approach ensures:

- Clean separation from system Python
- Version-specific compatibility
- Easy updates and maintenance
- No conflicts with Blender's internal Python

Dependencies
------------

Core Dependencies
~~~~~~~~~~~~~~~~~

- **bpy** (Blender Python API)
    - Bundled with Blender
    - Provides 3D manipulation capabilities

Scientific Computing
~~~~~~~~~~~~~~~~~~~~~~

- **numpy** - Numerical computations
- **scipy** - Scientific algorithms
- **xarray** - N-D labeled arrays

Simulation Engines
~~~~~~~~~~~~~~~~~~~~

- **antimony** - Biological modeling language
- **libroadrunner** - SBML simulation engine

Data Handling
~~~~~~~~~~~~~~

- **h5py** - HDF5 file format support
- **tifffile** - TIFF file handling

All dependencies are automatically managed through our setup process. Manual installation is not recommended as it may lead to version conflicts or compatibility issues.

Verification
-------------

To verify your installation:

.. code-block:: bash

    make test

This will run the test suite to ensure all components are correctly installed and functioning.

For detailed information about your setup:

.. code-block:: bash

    make info

Next Steps
----------

After installation, we recommend:

1. Reviewing the :doc:`../user_guide/api` for API documentation
2. Exploring the Goo modules:
   - :doc:`../user_guide/goo.cell` for cell manipulation
   - :doc:`../user_guide/goo.force` for force calculations
   - :doc:`../user_guide/goo.growth` for growth models

.. _bpy: https://docs.blender.org/api/current/info_advanced_blender_as_bpy.html
.. _numpy: http://www.numpy.org/
.. _scipy: https://scipy.org/
.. _antimony: https://tellurium.readthedocs.io/en/latest/antimony.html
.. _libroadrunner: https://www.libroadrunner.org/
.. _h5py: https://www.h5py.org/
.. _xarray: https://xarray.dev/

.. note::

   It is possible to install the dependencies direclty in the Blender Python interpreter. However, we do not recommend this approach as it might lead to conflicts with the system Python interpreter and is harder to manage across different versions of Blender.

Documentation
~~~~~~~~~~~~~~~

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