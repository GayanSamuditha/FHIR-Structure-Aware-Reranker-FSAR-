.PHONY: install ingest embed eval test run docker-build docker-run

install:
	pip install -r requirements.txt
	pip install streamlit plotly

ingest:
	python -m src.ingest
	python tests/test_m1.py

embed:
	python -m src.embed
	python tests/test_m2.py

eval-gen:
	python -m src.eval_gen
	python tests/test_m3.py

test:
	python tests/test_m4.py
	python tests/test_m5.py

test-all:
	python tests/test_m1.py
	python tests/test_m2.py
	python tests/test_m3.py
	python tests/test_m4.py
	python tests/test_m5.py

run:
	streamlit run app/streamlit_app.py

docker-build:
	docker build -t fsar:latest .

docker-run:
	docker-compose up

k8s-deploy:
	kubectl apply -f k8s/
