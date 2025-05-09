import sys

from sphinx.cmd import build


first_sphinx_arg = sys.argv.index('-M')
build.make_main(sys.argv[first_sphinx_arg:])
