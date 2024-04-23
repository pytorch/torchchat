#!/usr/bin/env bash

set -xeou pipefail

shopt -s globstar

install_huggingface_cli() {
	pip install -U "huggingface_hub[cli]"
}

download_checkpoint() {
	# This funciton is "technically re-usable but ymmv"
	# includes org name, like <org>/<repo>
	local repo_name=$1
	local include=$2
	# basically just removes the org in <org>/<repo>
	local local_dir="checkpoints/${repo_name}"

	mkdir -p "${local_dir}"
	huggingface-cli download \
		"${repo_name}" \
		--quiet \
		--include "${include}" \
		--local-dir "${local_dir}"
}

# install huggingface-cli if not already installed
if ! command -v huggingface-cli; then
	install_huggingface_cli
fi

# TODO: Eventually you could extend this to download different models
# taking in some arguments similar to .ci/scripts/wget_checkpoint.sh
download_checkpoint "meta-llama/Meta-Llama-3-8B" "original/*"
