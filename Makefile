help:
	@echo "Available targets:\n\
		clean - remove build files and directories\n\
		setup - create mamba developer environment\n\
		lint  - run code formatters and linters\n\
		test  - run automated test suite\n\
		docs  - generate documentation site\n\
		dist  - generate release archives\n\
	\n\
	Remember to 'mamba activate pyimagej-dev' first!"

clean:
	bin/clean.sh

setup:
	bin/setup.sh

check:
	@bin/check.sh

lint: check
	black src tests

test: check
	bin/test.sh

docs: check
	cd doc && $(MAKE) html

dist: check
	python -m build

.PHONY: tests
