
.PHONY: help install test train clean

help:
	@echo "Available targets: install, train, test, clean"

install:
	python -m pip install -r requirements.txt

train:
	python main.py --mode train --data_dir data/chest_xray

test:
	python -m pytest -q

clean:
	rm -rf build dist *.egg-info __pycache__
