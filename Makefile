# Mirror of tasks.py targets (D9: `make` is absent on the primary build machine;
# this exists so the reproducibility contract in project.md §11 reads naturally on Linux).
# No logic is duplicated -- every target just delegates to tasks.py.

.PHONY: test lint smoke repro-quick repro-all run sweep figures paper slides clean

test:
	uv run python tasks.py test

lint:
	uv run python tasks.py lint

smoke:
	uv run python tasks.py smoke

repro-quick:
	uv run python tasks.py repro-quick

repro-all:
	uv run python tasks.py repro-all

run:
	uv run python tasks.py run $(ARGS)

sweep:
	uv run python tasks.py sweep $(ARGS)

figures:
	uv run python tasks.py figures

paper:
	uv run python tasks.py paper

slides:
	uv run python tasks.py slides

clean:
	uv run python tasks.py clean
