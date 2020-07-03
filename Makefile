test:
	poetry run pytest tests $(ARGS)

jupyter:
	poetry run jupyter notebook

py:
	poetry run python $(ARGS)

mypy:
	poetry run mypy incubator

lint:
	poetry run flake8 incubator
	poetry run pyright incubator

run:
	poetry run python -m incubator.run $(ARGS)

train:
	poetry run python -m incubator.run train $(ARGS)
