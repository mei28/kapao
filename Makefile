build:
	sudo docker build -t kapao --build-arg UID=`id -u` -f docker/Dockerfile .

run:
	sudo docker run -it --rm --gpus all -v `pwd`:/work  --name kapao-container kapao

jupyter:
	sudo docker run -d -it --rm --gpus all -v `pwd`:/work -p 8888:8888 --name kapao-container kapao jupyter-lab --no-browser --port=8888 --ip=0.0.0.0 --allow-root --NotebookApp.toke=''

