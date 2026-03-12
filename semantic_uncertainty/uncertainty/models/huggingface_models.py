"""Implement HuggingfaceModel models."""
import copy
import logging
from collections import Counter
import torch

import accelerate

from transformers import AutoTokenizer
from transformers import AutoConfig
from transformers import AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from transformers import StoppingCriteria
from transformers import StoppingCriteriaList
from huggingface_hub import snapshot_download


from uncertainty.models.base_model import BaseModel
from uncertainty.models.base_model import STOP_SEQUENCES


class StoppingCriteriaSub(StoppingCriteria):
    """Stop generations when they match a particular text or token."""
    def __init__(self, stops, tokenizer, match_on='text', initial_length=None):
        super().__init__()
        self.stops = stops
        self.initial_length = initial_length
        self.tokenizer = tokenizer
        self.match_on = match_on
        if self.match_on == 'tokens':
            self.stops = [torch.tensor(self.tokenizer.encode(i)).to('cuda') for i in self.stops]
            print(self.stops)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        del scores  # `scores` arg is required by StoppingCriteria but unused by us.
        for stop in self.stops:
            if self.match_on == 'text':
                generation = self.tokenizer.decode(input_ids[0][self.initial_length:], skip_special_tokens=False)
                match = stop in generation
            elif self.match_on == 'tokens':
                match = stop in input_ids[0][-len(stop):]
            else:
                raise ValueError("Invalid match_on type")
            if match:
                return True
        return False


def remove_split_layer(device_map_in):
    """Modify device maps s.t. individual layers are not spread across devices."""
    device_map = copy.deepcopy(device_map_in)
    destinations = list(device_map.keys())
    counts = Counter(['.'.join(i.split('.')[:2]) for i in destinations])

    found_split = False
    for layer, count in counts.items():
        if count == 1:
            continue
        if found_split:
            raise ValueError('More than one split layer found.')
        
        logging.info(f'Split layer is {layer}.')
        for name in list(device_map.keys()):
            if name.startswith(layer):
                device = device_map.pop(name)
        device_map[layer] = device
        found_split = True
    return device_map


class HuggingfaceModel(BaseModel):
    """Hugging Face Model."""

    def __init__(self, model_name, stop_sequences=None, max_new_tokens=None):
        if max_new_tokens is None:
            raise ValueError("max_new_tokens must be specified")
        self.max_new_tokens = max_new_tokens

        if stop_sequences == 'default':
            stop_sequences = STOP_SEQUENCES

        # --- MODEL CONFIGURATION ---
        eightbit = False
        kwargs = {"device_map": "auto"}

        if model_name.endswith('-8bit'):
            eightbit = True
            model_name = model_name[:-len('-8bit')]
            kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True
            )

        # Determine Hub Base
        if 'llama-3' in model_name.lower():
            base = 'meta-llama'
        elif 'llama-2' in model_name.lower():
            base = 'meta-llama'
            if not model_name.endswith('-hf'):
                model_name += '-hf'
        else:
            base = 'huggyllama' # Legacy fallback

        full_model_id = f"{base}/{model_name}"
        print(f"Loading model: {full_model_id}")

        self.tokenizer = AutoTokenizer.from_pretrained(full_model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load Model
        self.model = AutoModelForCausalLM.from_pretrained(
            full_model_id,
            **kwargs
        )

        self.model_name = model_name
        self.stop_sequences = stop_sequences + [self.tokenizer.eos_token]
        # Adjust context limits for Llama 3
        self.token_limit = 8192 if 'llama-3' in model_name.lower() else 4096

    def predict(self, input_data, temperature, return_full=False):
        # Ensure input_data is a string (library sometimes passes unexpected types)
        inputs = self.tokenizer(str(input_data), return_tensors="pt").to(self.model.device)
        
        if 'token_type_ids' in inputs:
            del inputs['token_type_ids']

        if self.stop_sequences is not None:
            stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(
                stops=self.stop_sequences,
                initial_length=inputs['input_ids'].shape[1],
                tokenizer=self.tokenizer)])
        else:
            stopping_criteria = None

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                output_hidden_states=True,
                temperature=temperature,
                do_sample=True if temperature > 0 else False,
                stopping_criteria=stopping_criteria,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode using input_ids specifically to avoid Encoding object error
        full_answer = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        prompt_text = self.tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)

        if return_full:
            return full_answer

        # Slice answer to remove prompt
        if full_answer.startswith(prompt_text):
            answer = full_answer[len(prompt_text):]
        else:
            # Fallback for weird tokenization artifacts
            answer = full_answer

        # Clean up stop sequences
        sliced_answer = answer
        if self.stop_sequences is not None:
            for stop in self.stop_sequences:
                if stop in sliced_answer:
                    sliced_answer = sliced_answer.split(stop)[0]
        
        sliced_answer = sliced_answer.strip()

        # Metadata for Uncertainty Calculations
        token_stop_index = outputs.sequences.shape[1]
        n_input_token = inputs['input_ids'].shape[1]
        n_generated = token_stop_index - n_input_token

        if n_generated <= 0:
            n_generated = 1

        # Handle Hidden States (Embeddings)
        hidden = outputs.hidden_states
        # Access the last generated token's hidden state
        # generate() returns tuple of tuples: (gen_tokens, layers, batch, seq, hidden)
        last_input = hidden[-1] 
        last_layer = last_input[-1]
        last_token_embedding = last_layer[:, -1, :].cpu()

        # Handle Likelihoods
        transition_scores = self.model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True)
        
        log_likelihoods = transition_scores[0].cpu().numpy().tolist()

        return sliced_answer, log_likelihoods, last_token_embedding

    def get_p_true(self, input_data):
        input_data += ' A'
        tokenized = self.tokenizer(input_data, return_tensors='pt').to(self.model.device)
        target_ids = tokenized.input_ids.clone()
        target_ids[0, :-1] = -100

        with torch.no_grad():
            outputs = self.model(tokenized.input_ids, labels=target_ids)
        
        return -outputs.loss.item()
    