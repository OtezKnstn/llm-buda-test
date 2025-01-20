# import the pybuda library and additional libraries required for this tutorial
import os
from typing import Dict, Tuple, Union

import nest_asyncio
import pybuda
import torch
from transformers import BertForSequenceClassification, BertTokenizer

class BERTHandler:
    """
    A class to represent a BERT model RESTful API handler.

    ...

    Attributes
    ----------
    initialized : bool
        Flag to mark if model as been compiled or not
    device0 : pybuda.TTDevice
        Tenstorrent device object which represents the hardware target to deploy model on
    seqlen : int
        Input sequence length

    Methods
    -------
    initialize():
        Initializes the model by downloading the weights, selecting the hardware target, and compiling the model
    preprocess(input_text):
        Preprocess the input (apply tokenization)
    inference(processed_inputs):
        Run inference on device
    postprocess(logits):
        Run post-processing on logits from model
    handle(input_text):
        Run all of the steps on user inputs
    """

    def __init__(self, seqlen: int = 128):
        """
        Constructs all the necessary attributes for the BERTHandler object.

        Parameters
        ----------
        seqlen : int, optional
            Input sequence length, by default 128
        batch_size : int, optional
            Input batch size, by default 1
        """
        self.initialized = False
        self.device0 = None
        self.seqlen = seqlen

    def initialize(self):
        """
        Initialize and compile model pipeline.
        """

        # Set logging levels
        os.environ["LOGURU_LEVEL"] = "ERROR"
        os.environ["LOGGER_LEVEL"] = "ERROR"

        # Load BERT tokenizer and model from HuggingFace for text classification task
        model_ckpt = "assemblyai/bert-large-uncased-sst2"
        model = BertForSequenceClassification.from_pretrained(model_ckpt)
        self.tokenizer = BertTokenizer.from_pretrained(model_ckpt)

        # Initialize TTDevice object
        tt0 = pybuda.TTDevice(
            name="tt_device_0",  # here we can give our device any name we wish, for tracking purposes
        )

        # Create PyBUDA module
        pybuda_module = pybuda.PyTorchModule(
            name = "pt_bert_text_classification",  # give the module a name, this will be used for tracking purposes
            module=model  # specify the model that is being targeted for compilation
        )

        # Place module on device
        tt0.place_module(module=pybuda_module)
        self.device0 = tt0

        # Load data sample to compile model
        sample_input = self.preprocess("sample input text")

        # Push input to model
        self.device0.push_to_inputs(*sample_input)

        # Compile & initialize the pipeline for inference, with given shapes
        output_q = pybuda.run_inference()
        _ = output_q.get()

        # Configure initialization flag
        self.initialized = True
        print("BERTHandler initialized.")

    def preprocess(self, input_text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocess the user inputs.

        Parameters
        ----------
        input_text : str
            User input

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Processed outputs: `input_ids` and `attention_mask`
        """

        input_tokens = self.tokenizer(
            input_text,
            max_length=self.seqlen,  # set the maximum input context length
            padding="max_length",  # pad to max length for fixed input size
            truncation=True,  # truncate to max length
            return_tensors="pt",  # return PyTorch tensor
        )

        return (input_tokens["input_ids"], input_tokens["attention_mask"])

    def inference(self, processed_inputs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Run inference on Tenstorrent hardware.

        Parameters
        ----------
        processed_inputs : Tuple[torch.Tensor, torch.Tensor]
            Processed inputs: `input_ids` and `attention_mask`

        Returns
        -------
        torch.Tensor
            Output logits from model
        """

        self.device0.push_to_inputs(*processed_inputs)
        output_q = pybuda.run_inference()
        output = output_q.get()
        logits = output[0].value().detach()
        return logits

    def postprocessing(self, logits: torch.Tensor) -> Dict[str, Union[str, float]]:
        """
        Post-process logits and return dictionary with prediction and confidence score.

        Parameters
        ----------
        logits : torch.Tensor
            Predicted logits from model

        Returns
        -------
        Dict[str, Union[str, float]]
            Output dictionary with predicted class and confidence score
        """

        probabilities = torch.softmax(logits, dim=1)
        confidences, predicted_classes = torch.max(probabilities, dim=1)
        confidences = confidences.cpu().tolist()[0]
        predicted_classes = predicted_classes.cpu()
        output = {
            "predicted sentiment": "positive" if predicted_classes else "negative",
            "confidence": confidences
        }

        return output

    def handle(self, text_input: str) -> Dict[str, Union[str, float]]:
        """
        Handler function which runs end-to-end model pipeline

        Parameters
        ----------
        text_input : str
            User input

        Returns
        -------
        Dict[str, Union[str, float]]
            Output dictionary with predicted class and confidence score
        """

        # Data preprocessing
        processed_text = self.preprocess(text_input)

        # Run inference
        logits = self.inference(processed_text)

        # Data postprocessing
        output = self.postprocessing(logits)

        return output
if __name__ == "__main__":
    import time
    start_time = time.time()
    model = BERTHandler()
    model.initialize()
    print('\nCompile time: ', time.time() - start_time)
    for i in range(5):
        start_time = time.time()
        print(model.handle(input("Input:")))
        print('\nAnswer time: ', time.time() - start_time)
