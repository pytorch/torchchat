import json

transformer_configs = {
    "CodeLlama-7b-Python-hf": {
        "block_size": 16384,
        "vocab_size": 32000,
        "n_layers": 32,
        "dim": 4096,
        "rope_base": 1000000,
    },
    "7B": {"n_layers": 32, "n_heads": 32, "dim": 4096},
    "13B": {"n_layers": 40, "n_heads": 40, "dim": 5120},
    "30B": {"n_layers": 60, "n_heads": 52, "dim": 6656},
    "34B": {
        "n_layers": 48,
        "n_heads": 64,
        "dim": 8192,
        "vocab_size": 32000,
        "n_local_heads": 8,
        "hidden_dim": 22016,
        "rope_base": 1000000,
    },  # CodeLlama-34B-Python-hf
    "70B": {
        "n_layers": 80,
        "n_heads": 64,
        "dim": 8192,
        "n_local_heads": 8,
        "hidden_dim": 28672,
    },
    "Meta-Llama-3-8B": {
        "dim": 4096,
        "ffn_dim_multiplier": 1.3,
        "multiple_of": 1024,
        "n_heads": 32,
        "n_local_heads": 8,  # n_kv_heads
        "n_layers": 32,
        "rope_base": 500000.0,  # rope_theta
        "vocab_size": 128256,
        "use_tiktoken": True,
    },
    "Mistral-7B": {
        "n_layers": 32,
        "n_heads": 32,
        "n_local_heads": 8,
        "dim": 4096,
        "hidden_dim": 14336,
        "vocab_size": 32000,
    },
    "stories15M": {"n_layers": 6, "n_heads": 6, "dim": 288},
    "stories110M": {"n_layers": 12, "n_heads": 12, "dim": 768},
}

for key, config in transformer_configs.items():
    with open(f"build/known_model_params/{key}.json", "w") as c:
        json.dump(config, c)

print("completed successfully")
