import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TEST_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # same tokenizer and chat template as Qwen2.5-7B


@pytest.fixture(scope="session")
def tok():
    return AutoTokenizer.from_pretrained(TEST_MODEL)


@pytest.fixture(scope="session")
def model():
    m = AutoModelForCausalLM.from_pretrained(TEST_MODEL, dtype=torch.float32, device_map={"": "cpu"})
    return m.eval()
