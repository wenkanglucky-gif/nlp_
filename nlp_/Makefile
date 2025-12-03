.PHONY: install test run clean lint format

install:
	pip install -r requirements.txt
	pip install pytest black flake8

test:
	python -m pytest test_app.py -v

run:
	python app.py

clean:
	rm -rf __pycache__
	rm -rf monitor_logs/*.csv monitor_logs/*.log
	find . -type f -name "*.pyc" -delete

lint:
	flake8 app.py --max-line-length=100 --ignore=E501,W503

format:
	black app.py test_app.py

check:
	@echo "Running code quality checks..."
	black --check app.py
	flake8 app.py --max-line-length=100
	@echo "All checks passed!"
