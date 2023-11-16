# These have been configured to only really run short tasks. Longer form tasks
# are usually completed in github actions.

NAME := Owner avatar
PACKAGE_NAME := OneShotForecastingNAS 

DIR := "${CURDIR}"
SOURCE_DIR := ${PACKAGE_NAME}
.PHONY: help install-dev check format clean help:
	@echo "Makefile ${NAME}"
	@echo "* install-dev      to install all dev requirements and install pre-commit"
	@echo "* clean            to clean any doc or build files"
	@echo "* check            to check the source code for issues"
	@echo "* format           to format the code with black and isort"
	PYTHON ?= python
PIP ?= python -m pip
MAKE ?= make
PYDOCSTYLE ?= pydocstyle
install-dev:
	$(PIP) install -e ".[dev]"
	check-pydocstyle:
	$(PYDOCSTYLE) ${SOURCE_DIR} || :

check: check-pydocstyle

format:

# Clean up any builds in ./dist as well as doc, if present
clean: 