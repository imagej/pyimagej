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

lint:
	bin/lint.sh

fmt:
	bin/fmt.sh

test:
	bin/test.sh

docs:
	cd doc && $(MAKE) html

dist: clean
	uv run python -m build

.PHONY: tests
