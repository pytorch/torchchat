aoti:
	AOT_INDUCTOR_DEBUG_COMPILE=1 TORCHINDUCTOR_ABI_COMPATIBLE=1 python aoti_export.py --checkpoint_path checkpoints/stories15M/stories15M.pt --prompt "Once upon a time" --device cpu --temperature 0 --output_path stories15M.so
	python generate.py --checkpoint_path checkpoints/stories15M/stories15M.pt --prompt "Once upon a time" --device cpu --temperature 0 --dso_path stories15M.so

clean:
	rm -rf *.so
