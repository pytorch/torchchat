MODEL_NAME=stories15M
MODEL_PATH=checkpoints/${MODEL_NAME}/${MODEL_NAME}.pt
MODEL_DSO=./${MODEL_NAME}.so

aoti:
	AOT_INDUCTOR_DEBUG_COMPILE=1 TORCHINDUCTOR_ABI_COMPATIBLE=1 python export_aoti.py --checkpoint_path ${MODEL_PATH} --prompt "Once upon a time" --device cpu --temperature 0 --output_path ${MODEL_DSO}
	python generate.py --checkpoint_path ${MODEL_PATH} --prompt "Once upon a time" --device cpu --temperature 0 --dso_path ${MODEL_DSO}

clean:
	rm -rf *.so
