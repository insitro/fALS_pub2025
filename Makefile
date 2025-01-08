.PHONY: all
all: setup


.PHONY: setup
setup:
	@echo "Installing package..."
	@pixi install


# Run jupyterhub within the activated environment
.PHONY: notebook
notebook:
	@echo "Running JupyterLab..."
	@pixi run jupyter lab
