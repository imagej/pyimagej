help:
	@echo "Available targets:\n\
		clean - remove build files and directories\n\
		lint  - run code formatters and linters\n\
		test  - run automated test suite\n\
		docs  - generate documentation site\n\
		dist  - generate release archives\n\
	"

clean:
	bin/clean.sh

check:
	@bin/check.sh

lint: check
	bin/lint.sh

test: check
	bin/test.sh

docs:
	cd doc && $(MAKE) html

dist: check clean
	uv run python -m build

.PHONY: tests
