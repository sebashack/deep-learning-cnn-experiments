SHELL := bash

py_bin := ${PYTHON_VENV_DIR}/bin/python
pip_bin := ${PYTHON_VENV_DIR}/bin/pip
py_test := ${PYTHON_VENV_DIR}/bin/py.test

.PHONY: venv
venv:
	python3 -m venv ${PYTHON_VENV_DIR}
	${pip_bin} install -r ${ROOT}/requirements.txt
	${pip_bin} install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
	${pip_bin} install torch_geometric pyg_lib torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.6.0+cu126.html

.PHONY: clean
clean:
	rm -rf ${ROOT}/venv
	rm -rf ${ROOT}/_runs
	rm -rf ${ROOT}/_data
	rm -rf ${ROOT}/_download
