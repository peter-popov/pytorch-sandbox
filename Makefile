dockerimage ?= peterpopov/pytorch-app
dockerfile ?= Dockerfile.cpu
srcdir ?= $(shell pwd)
datadir ?= $(shell pwd)

install:
	@docker build -t $(dockerimage) -f $(dockerfile) .

i: install


update:
	@docker build -t $(dockerimage) -f $(dockerfile) . --pull --no-cache

u: update

requirements.txt: requirements.in
	docker run --volume $(CURDIR):/usr/src/app --rm $(dockerimage) pip-compile --generate-hashes
	touch requirements.txt

run:
	@docker run -it --rm --ipc=host --runtime=nvidia \
	  -v $(srcdir)/:/usr/src/app/ \
	  -v $(datadir):/data \
	  --entrypoint=/bin/bash \
	  $(dockerimage)

r: run

run_cpu:
	@docker run -it --rm --ipc=host \
	  -v $(srcdir)/:/usr/src/app/ \
	  -v $(datadir):/data \
	  --entrypoint=/bin/bash \
	  $(dockerimage)

rc: run_cpu

notebook: install
	@docker run                              \
	  --ipc=host                             \
	  --rm                                   \
	  -v $(srcdir):/usr/src/app/             \
	  -v $(datadir):/data                    \
	  -p 8888:8888							 \
	  -it $(dockerimage)                     \
	  jupyter notebook --ip=0.0.0.0 --no-browser --allow-root

n: notebook


.PHONY: install run update notebook
