test:
	@poetry run pytest tests $(ARGS)

jupyter:
	@poetry run jupyter notebook

repl:
	@poetry run python

mypy:
	@poetry run mypy incubator

run:
	@poetry run python -m incubator.run