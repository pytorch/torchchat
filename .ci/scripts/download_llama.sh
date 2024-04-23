#!/usr/bin/env bash

set -xeou pipefail

install_huggingface_cli() {
	pip install -U "huggingface_hub[cli]"
}

download_checkpoint() {
	# This funciton is "technically re-usable but ymmv"
	# includes org name, like <org>/<repo>
	local repo_name=$1
	local include=$2
	# basically just removes the org in <org>/<repo>
	local local_dir=${repo_name##/*}

	mkdir -p "${local_dir}"
	huggingface-cli download \
		"${repo_name}" \
		--include "${include}" \
		--local-dir "${local_dir}"
}

normalize_llama_checpoint() {
	# normalizes the checkpoint file into something that the rest of
	# the testing scripts understand
	local repo_name=$1
	local local_dir=${repo_name##/*}
	mkdir -p "${local_dir}"
	mv "${local_dir}/original/*" "${local_dir}"
	mv "${local_dir}/consolidated.00.pth" "${local_dir}/model.pth"
	rmdir "${local_dir/original/}"
}

# install huggingface-cli if not already installed
if ! command -v huggingface-cli; then
	install_huggingface_cli
fi

# TODO: Eventually you could extend this to download different models
# taking in some arguments similar to .ci/scripts/wget_checkpoint.sh
download_checkpoint "meta-llama/Meta-Llama-3-8B" "original/*"
normalize_llama_checpoint "meta-llama/Meta-Llama-3-8B"
