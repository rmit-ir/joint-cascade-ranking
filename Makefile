
all:
	make -C eval

.PHONY: check
check:
	python -m pytest -lv test

.PHONY: lint
lint:
	find bin cascade test -name "*.py" | xargs yapf -vv -i

.PHONY: tags
tags:
	find bin cascade test -name "*.py" | xargs ctags
