.PHONY: install train serve predict refresh reset orchestrate drift demo test lint clean

install:
	pip install -r requirements.txt

train:
	python -m src.train

serve:
	uvicorn src.serve:app --host 0.0.0.0 --port 8000

predict:
	python scripts/predict_once.py

reset:
	python -m src.data reset

refresh:
	python -m src.data refresh

# refresh + inject a distribution shift
drift:
	python -m src.data refresh --inject-drift

# one orchestration cycle (append a drifted batch, detect drift, retrain, promote)
orchestrate:
	python -m src.orchestrate --inject-drift

# one-command local demo (trains, serves, predicts, then runs the closed loop)
demo:
	bash scripts/run_demo.sh

test:
	pytest -q

lint:
	ruff check src tests

clean:
	python -m src.data reset
	rm -rf models/* runs/*
