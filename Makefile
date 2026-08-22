.PHONY: install test demo web docker lint ci

install:
	python -m pip install -e ".[web,plot,dev]"

test:
	python -m unittest discover -s tests -v

demo:
	bash scripts/run_demo.sh

web:
	streamlit run streamlit_app.py

docker:
	docker compose up --build

lint:
	ruff check src tests streamlit_app.py

ci: lint test demo
