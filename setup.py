from setuptools import setup, find_packages

setup(
    name='goo',
    version='1.1.0',    
    description='Goo is a library to simulate 3D biological cells, tissues and embryos in Blender.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/smegason/goo',
    author='Antoine A. Ruzette, Sean Megason',
    author_email='sean_megason@hms.harvard.edu',
    license='BSD 2-clause',
    packages=find_packages(where='scripts/modules'),
    package_dir={'': 'scripts/modules'},
    include_package_data=True,
    package_data={
        'goo': ['missile_launch_facility_01_4k.hdr'],
    },
    install_requires=[
        'numpy>=1.21.0',
        'scipy>=1.7.0',
        'sphinx>=4.0.0',
    ],
    python_requires='>=3.7',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)
