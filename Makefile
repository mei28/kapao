jupyter:
	rye run python -m jupyterlab --no-browser --port=9096 --ip=0.0.0.0 --allow-root --NotebookApp.token=''

build_cpu:
	cd ./environments/cpu && docker compose build

build_gpu:
	cd ./environments/gpu && docker compose build

build_nc_cpu:
	cd ./environments/cpu && docker compose build --no-cache

build_nc_gpu:
	cd ./environments/gpu && docker compose build --no-cache

up_cpu:
	cd ./environments/cpu && docker compose up -d && docker exec -it gbm_cpu /bin/bash

up_gpu:
	cd ./environments/gpu && docker compose up -d && docker exec -it gbm_gpu /bin/bash

run_cpu:
	cd ./environments/cpu && docker compose run --rm --service-ports core; true

run_gpu:
	cd ./environments/gpu && docker compose run --rm --service-ports core; true


down_cpu:
	cd ./environments/cpu && docker compose down

down_gpu:
	cd ./environments/gpu && docker compose down

nvimei_cpu:
	cd ./environments/cpu && docker compose run --rm --name nvimei nvimei /bin/bash

nvimei_gpu:
	cd ./environments/gpu && docker compose run --rm --name nvimei nvimei /bin/bash
