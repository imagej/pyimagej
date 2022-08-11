help:
	@echo "Available targets:\n\
		clean - remove build files and directories\n\
		lint  - run code formatters and linters\n\
		test  - run automated test suite\n\
		docs  - generate documentation site\n\
		dist  - generate release archives\n\
	\n\
	Remember to 'mamba activate pyimagej-dev' first!"

clean:
	bin/clean.sh

lint:
	black src test

test:
	bin/test.sh

docs:
	cd doc && $(MAKE) html

dist:
	python -m build

.PHONY: test
