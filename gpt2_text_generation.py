# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# GPT2 demo script - Text generation
import pybuda
from pybuda.transformers.pipeline import pipeline as pybuda_pipeline
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer


def gpt_test():

    batch_size = 1

    # Set model configurations
    config = GPT2Config.from_pretrained("gpt2")
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    config_dict["use_cache"] = False
    config = GPT2Config(**config_dict)

    compiler_cfg = pybuda.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b

    # Load tokenizer and model from HuggingFace
    model = GPT2LMHeadModel.from_pretrained("gpt2", config=config)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token, tokenizer.pad_token_id = (
        tokenizer.eos_token,
        tokenizer.eos_token_id,
    )
    prefix_text = ["My name is Thomas and my main"] * batch_size

    # Initialize pipeline
    text_generator = pybuda_pipeline("text-generation", model=model, tokenizer=tokenizer, batch_size=batch_size)

    # Run inference on Tenstorrent device
    answer = text_generator(
        prefix_text,
        max_length=30,
        num_beams=1,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id,
        no_repeat_ngram_size=2,
    )

    # Report output
    for sample_id in range(batch_size):
        print("\n\n")
        print(f"Sample ID: {sample_id}")
        print(f"Prefix text: {prefix_text[sample_id]}")
        print(f"Generated text: {answer[sample_id]}")
        print("\n\n")


if __name__ == "__main__":
    import time
    start_time = time.time()
    gpt_test()
    print('\n\n\nAnswer time: ', time.time() - start_time)
