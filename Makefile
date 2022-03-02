help:
	@echo "for now, you are on your own"

init:
	python3 -m venv --prompt cityscope .venv

install-mac:
	pip install -r requirements-mac.txt

install:
	pip install -r requirements-dev.txt

install-ci:
	pip install -r requirements-ci.txt

test: test/*.py
	python -m pytest -ra

run:
	python -m main
