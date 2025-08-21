SHELL := /bin/bash
.ONESHELL:

VENV := .venv

.PHONY: setup install pipeline app clean docker-build docker-run-pipeline docker-run-app

setup:
	python3 -m venv $(VENV)
	. $(VENV)/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

install:
	pip install --upgrade pip && pip install -r requirements.txt

pipeline:
	python -m src.pipeline

app:
	streamlit run app/app.py

docker-build:
	docker build -t credtech-xscore:latest .

docker-run-pipeline:
	docker run --rm -it -v $$(pwd):/workspace credtech-xscore:latest python -m src.pipeline

docker-run-app:
	docker run --rm -it -p 8501:8501 -v $$(pwd):/workspace credtech-xscore:latest streamlit run app/app.py

clean:
	rm -rf data/processed/* data/scores/* data/reports/* models/model.pkl
