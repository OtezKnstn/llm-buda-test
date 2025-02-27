# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

#  GPT Neo Demo - CausalLM

import os

import pybuda
import torch
from pybuda._C.backend_api import BackendDevice
from pybuda.transformers.pipeline import pipeline as pybuda_pipeline
from transformers import AutoTokenizer, GPTNeoConfig, GPTNeoForCausalLM


def run_gptneo_causal_lm(variant="EleutherAI/gpt-neo-125M", batch_size=1):
    available_devices = pybuda.detect_available_devices()
    # Set random seed for repeatability
    torch.manual_seed(42)

    # Variants: # EleutherAI/gpt-neo-125M, EleutherAI/gpt-neo-1.3B, EleutherAI/gpt-neo-2.7B
    model_ckpt = variant

    # Set PyBuda configurations
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b
    if variant == "EleutherAI/gpt-neo-1.3B":
        compiler_cfg.amp_level = 1
        if available_devices[0] == BackendDevice.Grayskull:
            compiler_cfg.balancer_policy = "Ribbon"

    if variant == "EleutherAI/gpt-neo-2.7B":
        compiler_cfg.amp_level = 1
        compiler_cfg.default_dram_parameters = True

        if available_devices:
            if available_devices[0] == BackendDevice.Grayskull:
                compiler_cfg.enable_auto_fusing = False
                compiler_cfg.balancer_policy = "Ribbon"

    # Modify Config
    config = GPTNeoConfig.from_pretrained(model_ckpt)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    config_dict["use_cache"] = False
    config = GPTNeoConfig(**config_dict)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPTNeoForCausalLM.from_pretrained(model_ckpt, config=config)

    # Sample input text
    prompt = ["My name is Bert, and I am"] * batch_size

    # Instantiate PyBuda pipeline
    text_generator = pybuda_pipeline("text-generation", model=model, tokenizer=tokenizer, batch_size=batch_size)

    # Run inference
    answer = text_generator(
        prompt,
        max_length=20,
        num_beams=1,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id,
        no_repeat_ngram_size=2,
    )

    # Report output
    for sample_id in range(batch_size):
        print(f"Sample ID: {sample_id}")
        print(f"Prefix text: {prompt[sample_id]}")
        print(f"Generated text: {answer[sample_id]}")


if __name__ == "__main__":
    import time
    start_time = time.time()

    run_gptneo_causal_lm()
    print('\n\n\nAnswer time: ', time.time() - start_time)
