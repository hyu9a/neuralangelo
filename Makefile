SHARED_REGISTRY := $(shell ovhai registry list -o json | jq -r '.[] | select(.kind == "Shared") | .url')
REGISTRY ?= $(SHARED_REGISTRY)
REPOSITORY ?= ai-core/nvidia
VERSION_colmap ?= 3.8
VERSION_neuralangelo ?= 23.04-py3

DATASTORE ?= S3
BUCKET ?= neuralangelo-lego
GPU ?= 3
GROUP ?= experiments
MODEL ?= lego

OUTPUT_DIR := logs/$(GROUP)/$(MODEL)

# preprocess config
DOWNSAMPLE_RATE ?= 2
SCENE_TYPE ?= object

.IMAGES=colmap neuralangelo
.BUILDS=$(addprefix docker-build-,$(.IMAGES))
.PUSHES=$(addprefix docker-push-,$(.IMAGES))

dockertag = $(REGISTRY)/$(REPOSITORY)/$(1):$(VERSION_$(1))-ovhcloud

.PHONY: init
init:
	git submodule update --init --recursive

.PHONY: $(.BUILDS)
$(.BUILDS): docker-build-%:
	docker build . -t $(call dockertag,$*) --build-arg FROM_IMAGE=chenhsuanlin/$*:$(VERSION_$*) -f docker/Dockerfile-ovhcloud

.PHONY: $(.PUSHES)
$(.PUSHES): docker-push-%:
	docker push $(call dockertag,$*)

.PHONY: prepare
prepare:
	ovhai job run \
		-o json \
		-g $(GPU) \
		-v $(BUCKET)@$(DATASTORE):/workspace:rw:cache \
		$(call dockertag,colmap) -- \
		bash projects/neuralangelo/scripts/preprocess.sh $(MODEL) input/$(MODEL).mp4 $(DOWNSAMPLE_RATE) $(SCENE_TYPE) \
		| jq -r .id | xargs -r ovhai job logs -f

.PHONY: process
process:
	ovhai job run \
		-o json \
		-g $(GPU) \
		-v $(BUCKET)@$(DATASTORE):/workspace:rw:cache \
		$(call dockertag,neuralangelo) -- \
			torchrun --nproc_per_node=$(GPU) train.py \
				--logdir=$(OUTPUT_DIR) \
				--show_pbar \
				--config=projects/neuralangelo/configs/custom/$(MODEL).yaml \
				--data.readjust.scale=0.5 \
				--max_iter=20000 \
				--validation_iter=99999999 \
				--model.object.sdf.encoding.coarse2fine.step=200 \
				--model.object.sdf.encoding.hashgrid.dict_size=19 \
				--optim.sched.warm_up_end=200 \
				--optim.sched.two_steps=[12000,16000] \
		| jq -r .id | xargs -r ovhai job logs -f

.PHONY: $(OUTPUT_DIR)/latest_checkpoint.txt
$(OUTPUT_DIR)/latest_checkpoint.txt:
	ovhai bucket object download $(BUCKET)@$(DATASTORE) $@

.PHONY: extract
extract: $(OUTPUT_DIR)/latest_checkpoint.txt
	ovhai job run \
		-o json \
		-g $(GPU) \
		-v $(BUCKET)@$(DATASTORE):/workspace:rw:cache \
		$(call dockertag,neuralangelo) -- \
			torchrun --nproc_per_node=$(GPU) projects/neuralangelo/scripts/extract_mesh.py \
				--config=$(OUTPUT_DIR)/config.yaml \
				--checkpoint=$(OUTPUT_DIR)/$(shell cat $<) \
				--output_file=$(OUTPUT_DIR)/mesh.ply \
				--resolution=300 --block_res=128 \
				--textured \
		| jq -r .id | xargs -r ovhai job logs -f

.PHONY: clear-data
clear-data:
	ovhai bucket object rm $(BUCKET)@$(DATASTORE) --all -y

.PHONY: push-data
push-data:
	ovhai bucket object upload $(BUCKET)@$(DATASTORE) .

.PHONY: pull-data
pull-data:
	ovhai bucket object download $(BUCKET)@$(DATASTORE)

.PHONY: clean
clean:
	rm -Rf datasets/*/
	rm -Rf logs/
