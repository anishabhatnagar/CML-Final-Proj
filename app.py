import gradio as gr
import random
import numpy as np

import torch
import torch.backends.cudnn as cudnn

from transformers import AutoTokenizer
from transformers import MT5ForConditionalGeneration

def setup_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def encode_str(text, tokenizer, seq_len):  
    """ Tokenize, pad to max length and encode to ids 
        Returns tensor with token ids """
    input_ids = tokenizer.encode(
        text=text,
        return_tensors = 'pt',
        padding = 'max_length',
        truncation = True,
        max_length = seq_len)

    return input_ids[0]

def infer(input_text):
    print(f'\n\n-----------------------{input_text}----------------------\n\n')
    input_ids = encode_str(text = input_text, tokenizer = tokenizer, seq_len = 40)
    input_ids = input_ids.unsqueeze(0).to(device)
    output_tokens = model.generate(input_ids, num_beams=10, num_return_sequences=1, length_penalty = 1, no_repeat_ngram_size=2)
    for token_set in output_tokens:
        prediction = tokenizer.decode(token_set,skip_special_tokens=True)
    return prediction


print('Initialization')
setup_seeds(0)

model_repo = 'google/mt5-small'

print("torch version")
print(torch.__version__)
print("cuda version")
print(torch.version.cuda)

if torch.cuda.is_available():
    device = 'cuda'
else:
    device='cpu'

print(device)
tokenizer = AutoTokenizer.from_pretrained(model_repo)

LANG_TOKEN_MAPPING = {
    'identify language': '<idf.lang>'
}
special_tokens_dict = {'additional_special_tokens': list(LANG_TOKEN_MAPPING.values())}
print(special_tokens_dict)

tokenizer.add_special_tokens(special_tokens_dict)


model = MT5ForConditionalGeneration.from_pretrained(model_repo, device_map="auto")
model.resize_token_embeddings(len(tokenizer))
model.load_state_dict(torch.load("Step-5623_checkpoint_lang_pred.pt",map_location=torch.device(device)))
model = model.to(device)

print("initialization complete")

print("creating gradio interface")
description = """
<p>This interactive application employs the powerful <strong>mT5 model</strong> for accurate and swift language detection. The mT5, a multilingual variant of the T5 model, is adept at understanding and processing a wide range of languages, making it an ideal choice for global language detection tasks.</p>

<h2>How It Works:</h2>
<ul>
  <li>Simply input your text into the provided field.</li>
  <li>Our mT5 model will analyze the text and identify the language it's written in.</li>
  <li>You'll receive instant results, showcasing the model's language detection capabilities.</li>
</ul>
"""

article = "Authors : Anisha Bhatnagar, Divyanshi Parashar"

demo = gr.Interface(fn=infer, inputs="text", outputs="text", title='Language detection using mT5', description=description, article=article) #, examples=examples)

demo.launch()
# demo.launch(share=True,show_api=False)   


