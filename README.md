[![License](https://img.shields.io/badge/License-BSD%203--Clause-orange.svg)](https://opensource.org/licenses/BSD-3-Clause)


# Goo is a Python-based Blender extension for modeling biological cells, tissues, and embryos

Goo is a modeling environment for creating physics based simulations of biological cells, tissues, and whole embryos. Goo contains built-in models for basic cell properties such as growth, division, adhesion, signaling and transduction. Goo creates realistic models of cells in 3D based on surface meshes enclosing compresible fluid. Goo is meant to fill a void in currently available models such as vertex and particle based models which are too simplified to capture essential featuers of cells, are often 2D, and can be difficult for new users to use.

Goo is built on top of Blender. Blender is a truly amazing open source computer graphics project for generating animations. Goo provides a library of helper functions in Python for creating cells with user defined properties that are used inside the scripting environment and GUI of Blender. Physics and rendering are handled by Blender. In combination Goo and Blender allow cell-based simulations to be interacted with via Python scripts and through a GUI.

Goo was started by the <a href="http://www.digitalfish.org">Megason lab at Harvard University</a> and is being developed through an open source collaboration. We would love for you to contribute! Please contact me if you are interested. megason AT hms.harvard.edu .

Our grand Driving Biological Problem is to simulate the first <a href= "https://www.youtube.com/watch?v=RQ6vkDr_Dec">24 hours of zebrafish development</a>. Our initial efforts are focussed on cleavage stage. We hope that Goo will be useful for simulating other biological tissues for understanding morphogenesis, embryonic development, growth of organoids, tissue engineering, and artificial life.

## Documentation
  Full documentation available on <a href="https://smegason.github.io/Goo/">Goo's website</a>. 

### User-interface

The library uses the Blender interface to launch simulations and visualize them in real-time.
![Goo UI](docs/source/_static/goo-GUI.png)

### Examples

Simulation of a growing and cleaving clump of cells, for 150 minutes at a time step of 1 minute.
![Division example](docs/source/_static/20240304_desynchronized_division_10001-0130.gif)


### Contributors (in order of appearance)
___
<li>Sean Megason, Harvard University
<li>Daniel Oo, Amherst University
<li>Kali Konstantinopoulos, Indiana University
<li>Michael Mitsch, Indiana University
<li>Drew Willis, Indiana University
<li>Ahmed Almaghasilah, University of Maine and Harvard University
<li>Antoine Ruzette, KU Leuven and Harvard University
<li>Vanshika Bidhan, KU Leuven
<li>Julie De Man, KU Leuven
<li>Nenghan Lin, KU Leuven
<li>Jiangli Gui, KU Leuven
<li>Rifa Gowani, Texas Academy of Mathematics and Science (TAMS) at the University of North Texas (UNT)
<li>Charles Dai, Harvard Medical School