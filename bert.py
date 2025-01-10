import os
import pybuda
from transformers import BertForQuestionAnswering, BertTokenizer

# Load BERT tokenizer and model from HuggingFace for Q&A task
model_ckpt = "bert-large-cased-whole-word-masking-finetuned-squad"
tokenizer = BertTokenizer.from_pretrained(model_ckpt)
model = BertForQuestionAnswering.from_pretrained(model_ckpt)

# Load data sample from SQuADv1.1
context = """Super Bowl 50 was an American football game to determine the champion of the National Football League
(NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the
National Football Conference (NFC) champion Carolina Panthers 24\u201310 to earn their third Super Bowl title.
The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California.
As this was the 50th Super Bowl, the league emphasized the \"golden anniversary\" with various gold-themed
initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals
(under which the game would have been known as \"Super Bowl L\"), so that the logo could prominently
feature the Arabic numerals 50."""

question = "Which NFL team represented the AFC at Super Bowl 50?"

# Data preprocessing
input_tokens = tokenizer(
    question,  # pass question
    context,  # pass context
    max_length=384,  # set the maximum input context length
    padding="max_length",  # pad to max length for fixed input size
    truncation=True,  # truncate to max length
    return_tensors="pt",  # return PyTorch tensor
)

# Set PyBuda configurations
compiler_cfg = pybuda.config._get_global_compiler_config()
compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b
compiler_cfg.default_dram_parameters = False
compiler_cfg.balancer_policy = "Ribbon"
os.environ["PYBUDA_RIBBON2"] = "1"
os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = f"{81*1024}"

tt0 = pybuda.TTDevice(
    name="tt_device_0",  # here we can give our device any name we wish, for tracking purposes
)

# Create a TTDevice instance with a specified architecture
tt0 = pybuda.TTDevice(
    name="tt_device_0",  # You can give your device any name for tracking purposes
    arch=pybuda.BackendDevice.Grayskull  # Optionally set the target device architecture
)

# Create module
pybuda_module = pybuda.PyTorchModule(
    name = "pt_bert_question_answering",  # give the module a name, this will be used for tracking purposes
    module=model  # specify the model that is being targeted for compilation
)

# Place module on device
tt0.place_module(module=pybuda_module)

# Run inference on Tenstorrent device
output_q = pybuda.run_inference()  # executes compilation (if first time) + runtime
output = output_q.get()  # get last value from output queue

# Data postprocessing
answer_start = output[0].value().argmax().item()
answer_end = output[1].value().argmax().item()
answer = tokenizer.decode(input_tokens["input_ids"][0, answer_start : answer_end + 1])

# Print outputs
print(f"Input context:\n{context}")
print(f"\nInput question:\n{question}")
print(f"\nOutput from model running on TTDevice:\n{answer}")

pybuda.shutdown()