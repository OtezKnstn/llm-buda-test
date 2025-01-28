import queue
import torch
import pybuda
import pytest
from pybuda.config import _get_global_compiler_config

from pybuda.schedulers import LearningRateScheduler
from pybuda.pybudaglobal import pybuda_reset
from pybuda._C.backend_api import BackendDevice, BackendType, DeviceMode 
from test.utils import download_model
def _safe_read(q):
    """
    Read a queue, but return None if an error was raised in the meantime, preventing a hang on error.
    """
    while True:
        try:
            data = q.get(timeout = 0.5)
            return data
        except queue.Empty as _:
            if pybuda.error_raised():
                raise RuntimeError("Error raised in pybuda")
        except KeyboardInterrupt:
            return None

#
# Run inference pipeline on a Transformers model
#
def test_transformers_pipeline_inference():

    from transformers import BertModel, BertTokenizer

    tokenizer = download_model(BertTokenizer.from_pretrained, "prajjwal1/bert-tiny")
    input_sentence = "BERT is a transformers model pretrained on a large corpus of English data in a self-supervised fashion. This means it was pretrained on the raw texts only, with no humans labelling them in any way (which is why it can use lots of publicly available data) with an automatic process to generate inputs and labels from those texts. More precisely, it was pretrained with two objectives: Masked language modeling (MLM): taking a sentence, the model randomly masks 15% of the words in the input then run the entire masked sentence through the model and has to predict the masked words. This is different from traditional recurrent neural networks (RNNs) that usually see the words one after the other, or from autoregressive models like GPT which internally mask the future tokens. It allows the model to learn a bidirectional representation of the sentence."
    input_tokens = tokenizer.encode(input_sentence, max_length=128, pad_to_max_length=True)

    model = download_model(BertModel.from_pretrained, "prajjwal1/bert-tiny", torchscript=False, add_pooling_layer=False)
    cpu0 = pybuda.CPUDevice("cpu0", module=pybuda.PyTorchModule("bert_embeddings", model.embeddings))
    tt0 = pybuda.TTDevice("tt1", module=pybuda.PyTorchModule("bert_encoder", model.encoder))

    cpu0.push_to_inputs(torch.Tensor(input_tokens).int().unsqueeze(0))
    output_q = pybuda.run_inference()

    print(_safe_read(output_q))

#
# Run inference pipeline on a Transformers model, enabling cpu fallback on unsupported ops
#
def test_transformers_pipeline_fallback_inference():

    from transformers import BertModel, BertTokenizer

    compiler_cfg = pybuda.config._get_global_compiler_config() 

    tokenizer = download_model(BertTokenizer.from_pretrained, "prajjwal1/bert-tiny")
    input_sentence = "BERT is a transformers model pretrained on a large corpus of English data in a self-supervised fashion. This means it was pretrained on the raw texts only, with no humans labelling them in any way (which is why it can use lots of publicly available data) with an automatic process to generate inputs and labels from those texts. More precisely, it was pretrained with two objectives: Masked language modeling (MLM): taking a sentence, the model randomly masks 15% of the words in the input then run the entire masked sentence through the model and has to predict the masked words. This is different from traditional recurrent neural networks (RNNs) that usually see the words one after the other, or from autoregressive models like GPT which internally mask the future tokens. It allows the model to learn a bidirectional representation of the sentence."
    input_tokens = tokenizer.encode(input_sentence, max_length=128, pad_to_max_length=True)

    model = download_model(BertModel.from_pretrained, "prajjwal1/bert-tiny", torchscript=False, add_pooling_layer=False)
    tt0 = pybuda.TTDevice("tt0", module=pybuda.PyTorchModule("bert", model))

    for i in range(5):
        tt0.push_to_inputs(torch.Tensor(input_tokens).int().unsqueeze(0))
        output_q = pybuda.run_inference()
        print(_safe_read(output_q))

#
# Run training through setup + manual loop of fwd/bwd/opt
#

if __name__ == "__main__":
    test_transformers_pipeline_inference()
    test_transformers_pipeline_fallback_inference()