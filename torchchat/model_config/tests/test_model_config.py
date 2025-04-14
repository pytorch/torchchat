import pytest
from torchchat.model_config.model_config import load_model_configs, resolve_model_config


TEST_CONFIG = "meta-llama/llama-3.2-11b-vision"
TEST_CONFIG_NAME = "meta-llama/Llama-3.2-11B-Vision"


@pytest.mark.model_config
def test_load_model_configs():
    configs = load_model_configs()
    assert TEST_CONFIG in configs
    assert configs[TEST_CONFIG].name == TEST_CONFIG_NAME


@pytest.mark.model_config
def test_resolve_model_config():
    config = resolve_model_config(TEST_CONFIG)
    print(config)
    assert config.name == TEST_CONFIG_NAME
    assert config.checkpoint_file == "model.pth"

    with pytest.raises(ValueError):
        resolve_model_config("UnknownModel")
