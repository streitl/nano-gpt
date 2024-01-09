SHELL = /bin/bash

## help: Display list of commands
.PHONY: help
help: Makefile
	@sed -n 's|^##||p' $< | column -t -s ':' | sed -e 's|^| |'

## style: Check lint, code styling rules.
.PHONY: style
style:
	bash style.sh --style $(FILES_TO_STYLE)

## format: Check lint, code styling rules.
.PHONY: format
format:
	bash style.sh --format $(FILES_TO_STYLE)