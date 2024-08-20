

# Copyright (c) Meta Platforms, Inc. and affiliates.
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
