# Fine-tune mBERT

This was a university project. It was designed to fine-tune pre-trained models with the HuggingFace API. We fine-tuned the mBERT model on PoS tagging for different languages. 
<br>
<br>
The corpus we used is a Universal Dependencies corpus. As this formal is already pre-tokenized, the pre_treatment.py file contains functions to read and "normalize" the PoS taggs as well as to align the UD tokens with the tokens form the mBERT tokenizer. 
<br>
<br>
That step done, the fine_tuning.py file contains functions to fine-tune and evaluate the model. Once training completed, the model is evaluated on 5 different languages. I fine-tuned the model on five different languages and tested it on the same languages. Here are my test results: 


![screenshot](results.jpeg)
