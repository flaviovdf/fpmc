# Simple makefile

PYTHON ?= python
NOSETESTS ?= nosetests

all:
	$(PYTHON) setup.py build_ext --inplace

clean:
	rm -rf build/
	find . -name "*.pyc" | xargs rm -f
	find . -name "*.c" | grep -v randomkit.c | xargs rm -f
	find . -name "*.so" | xargs rm -f

test:
	$(NOSETESTS)

trailing-spaces: 
	find -name "*.py" | xargs sed 's/^M$$//'
