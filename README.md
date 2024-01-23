# CLEX: Continuous Length Extrapolation for Large Language Models
This repo provides the official implementation of our paper "CLEX: Continuous Length Extrapolation for Large Language Models"

<div style='display:flex; gap: 0.25rem; '>
<!-- <a href='https://huggingface.co/DAMO-NLP-SG'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Checkpoint-blue'></a>  -->
<a href='https://huggingface.co/spaces/DAMO-NLP-SG/CLEX-Chat'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue'></a>
<a href='https://huggingface.co/papers/2310.16450'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Paper-blue'></a>
</div>

## News
- [2024.1.19] 🔥 Release the CLEX-Mixtral-8x7B-32K, CLEX-LLaMA-2-7B-64K, and CLEX-Phi-2-7B-32K (and refactor the codes to support different models), which all support more than 100k context length! 
- [2024.1.16] 🌟 CLEX has been accepted to ICLR 2024!
- [2023.10.25] 🚀 Release the code of **CLEX** and the long-context base & chat models trained with CLEX. 

## Features and Highlights of CLEX
![CLEX_diagram](https://github.com/DAMO-NLP-SG/CLEX/assets/18526640/063ffe34-0116-4759-92bf-e22fc7264cdf)

- **Simple and Clear**: _MINIMAL_ code and architecture changes. Only one up-and-down projection layer introduced, _NO_ recurrent memory caching or sparse attention required.
- **Train Short, Test Long**: _NO_ performance drop on the sequences _4x~8x longer_ than the training ones (see [here](https://github.com/DAMO-NLP-SG/CLEX#language-modelling)). 
- **Continuous Length Extrapolation**: Explicitly modeling the continuous dynamics of context window size during length extrapolation.

If you have any questions, feel free to contact us. (Emails: guanzzh.chen@gmail.com, lixin4ever@gmail.com)

## Model Zoo
<div align="center">

| Model Name | Model Type | Starting Point | Train Data |Train Length | MAX Test Length | HF Repo |
|:-----|:-----|:-----------|:-----------|:-----------|:-----------|:------:|
| CLEX-LLaMA-2-7B-16K | base | LLaMA-2-7B | [Redpajama-Book](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T) | 16K | 64K | [link](https://huggingface.co/DAMO-NLP-SG/CLEX-7B-16K) |
| CLEX-LLaMA-2-7B-Chat-16K | chat | CLEX-7B-16K | [UltraChat](https://github.com/thunlp/UltraChat) | 16K | 64K | [link](https://huggingface.co/DAMO-NLP-SG/CLEX-7B-Chat-16K) |
| CLEX-LLaMA-2-7B-64K | base | LLaMA-2-7B | [Redpajama-Book](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T) | 64k | 256K | [link](https://huggingface.co/DAMO-NLP-SG/CLEX-LLaMA-2-7B-64K) |
| CLEX-Phi-2-32K | base | Phi-2-2.7B | [LongCorpus-2.5B](https://huggingface.co/datasets/DAMO-NLP-SG/LongCorpus-2.5B) | 32k | 128K | [link](https://huggingface.co/DAMO-NLP-SG/CLEX-Phi-2-32K) |
| CLEX-Mixtral-8x7B-32K | base | Mixtral-8x7B-v0.1 | [LongCorpus-2.5B](https://huggingface.co/datasets/DAMO-NLP-SG/LongCorpus-2.5B) | 32k | >128K | [link](https://huggingface.co/DAMO-NLP-SG/CLEX-Mixtral-8x7B-32K) |
| CLEX-Mixtral-8x7B-Chat-32k | chat | CLEX-Mixtral-8x7B-32K | [Ultrachat 200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) | 32k | >128K | [link](https://huggingface.co/DAMO-NLP-SG/CLEX-Mixtral-8x7B-Chat-32K) |
</div>

## Supported LLMs
- [x] LLaMA-2
- [x] Phi-2
- [x] Mixtral-8x7B
- [ ] Mistral
- [ ] Falcon
- [ ] GPT-NeoX
- [ ] QWen



## Usage

### Environment Setup
```bash
conda create -yn clex python=3.9
conda activate clex

git clone https://github.com/DAMO-NLP-SG/CLEX.git
cd CLEX
pip install -r requirements.txt
# install flash-attn separately
pip install flash-attn --no-build-isolation
```

### Code Snippet for Minimal Usage

```bash
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("DAMO-NLP-SG/CLEX-7B-16K", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
  "DAMO-NLP-SG/CLEX-7B-16K",
  torch_dtype=torch.bfloat16,
  trust_remote_code=True,
  use_flash_attention_2=True
)
inputs = tokenizer("What is CLEX?", return_tensors="pt")
sample = model.generate(**inputs, max_length=128)
print(tokenizer.decode(sample[0]))
```



### Inference with Command Line Interface
We replicate the command line interface of [FastChat](https://github.com/lm-sys/FastChat) here.
You can use the command below to enable the streaming chatting upon CLEX. The CLEX-7B-Chat-4K supports the input sequence lengths up to 16k. 
```bash
python3 serve/cli.py --model-path DAMO-NLP-SG/CLEX-7B-Chat-4K --num-gpu 1
```
You can also try our web GUI demo [here](https://huggingface.co/spaces/DAMO-NLP-SG/CLEX-Chat).


## LongCorpus-2.5B
We collect a 2.5B training dataset from various domains for long-context continual pre-training. The composition of this dataset is as follows (partially inspired by [Long-Data-Collection](https://huggingface.co/datasets/togethercomputer/Long-Data-Collections)):

| Domain        | Proportion | Source |
| ------------- | ---------- | ------ |
| Book          | 40%        | [Redpajama-Book](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T)   |
| Arxiv         | 20%        | [Redpajama-Arxiv](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T)    |
| General       | 20%        | [Redpajama](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T)    |
| Code          | 10%        | [LCC-Python](https://huggingface.co/datasets/microsoft/LCC_python)    |
| QA            | 5%         | [Natural Questions](https://ai.google.com/research/NaturalQuestions/)   |
| Summarization | 5%         | [BookSum](https://github.com/salesforce/booksum)   |

We have also curated a test dataset comprising 250 million tokens, mirroring the same composition. The selection criteria ensured that the average n-gram similarity (for n=2, 3, 4) with the training set is below 10%. This threshold effectively excludes all QA and Summarization data, resulting in a test corpus where the distribution of tokens across Book, Arxiv, General, and Code categories follows a ratio of 4:2:2:1, respectively.


## Training

To train the long-context LLM with CLEX, run the script `scripts/train_lm.sh` as follows:
```bash
./scripts/train_lm.sh
```
For training the chat model, run the script `scripts/train_chat.sh` instead.

Note that we use an on-the-fly tokenization, which supports any desired training length without pre-tokenizing. So if you use a learning rate scheduler (e.g., cosine), you may need to specify the arg `max_steps` in the training arguments (You can estimate it depending on training data size).

## Customization
We now support LLaMA-2, Phi-2, and Mixtral-8x7B. If you want to customize your LLM equipped with RoPE, please follow three steps:
1. [Init](https://github.com/DAMO-NLP-SG/CLEX/blob/f9adf565a90459644c1cd61a55e23cc631ac940e/CLEX/llama/modeling_llama_clex.py#L1027) the CLEX layer and acquire the packed [cos and sin embeddings](https://github.com/DAMO-NLP-SG/CLEX/blob/f9adf565a90459644c1cd61a55e23cc631ac940e/CLEX/llama/modeling_llama_clex.py#L1118) of CLEX-scaled RoPE.
2. Pass the cos and sin embeddings to the attention layer.
3. [Move](https://github.com/DAMO-NLP-SG/CLEX/blob/f9adf565a90459644c1cd61a55e23cc631ac940e/CLEX/llama/modeling_llama_clex.py#L426) the update of `past_key_value` **before** applying the RoPE. This ensures all keys would be rotated by the same cos and sin embeddings.


## Evaluation
### Language Modelling
Here are the evaluation PPLs of the base models trained with CLEX. We apply training and evaluation on a subset of 2B tokens from the [RedPajama-Book](https://github.com/togethercomputer/RedPajama-Data) corpus, where the training and test sets are split by 99:1.

| Models | Train Length | Eval.(4k) | Eval.(8k) | Eval.(16k) | Eval.(32k) | Eval.(64k) |
| --- | --- | --- | --- | --- | --- | --- |
| LLaMA-2-7B | 4k  | 6.04 | 20.54 | >100 | >1000 | >1000 |
| CodeLLaMA-7B | 16k | 7.6 | 7.4 | 7.33 | 15.12 | 52.02 |
| Naive FT | 16k | 5.98 | 5.93 | 5.91 | 18.31 | > 100 |
| PI  | 16k | 5.9 | 5.71 | 5.72 | 6.05 | 8.75 |
| Yarn (s=16) | 16k | 6.5 | 5.71 | 5.73 | 5.99 | 8.51 |
| Yarn (s=32) | 16k | 6.61 | 5.94 | 5.96 | 6.08 | 6.22 |
| CL-Scaling | 16k | 24.99 | 5.86 | 5.87 | 10.56 | 41.09 |
| ALIBI | 4k  | 6.34 | 6.39 | 6.41 | 6.5 | 6.51 |
| RandomPos | 4k  | 5.88 | >100 | >1000 | >1000 | >1000 |
| CLEX-LLaMA-2-7B-4K | 4k  | 5.86 | 5.7 | 5.87 | 14.53 | 30.51 |
| CLEX-LLaMA-2-7B-16K | 16k | 5.88 | 5.68 | 5.52 | 5.55 | 5.64 |
| CLEX-LLaMA-2-13B-4k | 4k  | 5.43 | 5.31 | 5.34 | 6.40 | 12.15 |


|                 | Train Length | Eval.(32k) | Eval.(64k) | Eval.(128k) | Eval.(256k) |
| --------------- | ------------ | ---------- | ---------- | ----------- | ----------- |
| CLEX-LLaMA-2-7B | 64k          | 5.99       | 5.89       | 6.04        | 5.98        |



The CLEX-Phi-2-2.7B and CLEX-Mixtral-8x7B are trained on [LongCorpus-2.5B](https://huggingface.co/datasets/DAMO-NLP-SG/LongCorpus-2.5B), where the eval results on test set are listed below.

|                   | Train Length | Eval.(32k) | Eval.(64k) | Eval.(128k) | Eval.(256k) |
| ----------------- | ------------ | ---------- | ---------- | ----------- | ----------- |
| Phi-2-2.7B        | 2k           | >100       | >100       | >100        | >100        |
| CLEX-Phi-2-2.7B   | 32k          | 5.11       | 5.17       | 6.55        | -           |
| Mixtral-8x7B      | 32k          | 2.78       | 3.44       | 5.88        | 14.20       |
| CLEX-Mixtral-8x7B | 32k          | 2.56       | 2.53       | 2.57        | 3.78        |



### LongBench

We evaluate the chat models trained with CLEX on the [LongBench](https://github.com/THUDM/LongBench), where the average length of most tasks ranges from 5k to 16k. Except for those marked with † are evaluated by ourselves, the baseline results are retrieved from the leaderboard of LongBench. ** denotes the method that needs to truncate the input sequence to the train length.

| Model              | Train Length | Avg.  | Single-Document QA | Multi-Document QA | Summarization | Few-shot Learning | Sythetic Task | Code Completion |
| ------------------ | ------------ | ----- | ------------------ | ----------------- | ------------- | ----------------- | ------------- | --------------- |
| GPT-3.5-Turbo-16K      | -         | 44.66 | 45.1               | 36.23             | 23.9          | 57.58             | 51            | 54.15           |
| CodeLLaMA-7B<sup>†</sup>       | 16k          | 33.42 | 32.19              | 21.49             | 20.06         | 57.73             | 8.92          | 60.11           |
| Vicuna-v1.5-7B     | 16k          | 30.54 | 31.75              | 18.8              | 23.25         | 56.83             | 5.33          | 47.25           |
| LongChat-v1.5-7B   | 32k          | 31.59 | 28.78              | 20.33             | 22.45         | 50.8              | 13.03         | 54.15           |
| XGen-7B<sup>**</sup>        | 8k           | 24.96 | 22.15              | 18.02             | 19.05         | 47.23             | 4.7           | 38.6            |
| InternLM-7B<sup>**</sup>    | 8k           | 22.64 | 21.45              | 17.9              | 15.2          | 41.55             | 3.3           | 36.45           |
| Llama2-7B-chat<sup>**</sup> | 4k           | 26.76 | 21.65              | 18.2              | 18.53         | 49.95             | 4.13          | 48.1            |
| Baichuan-13B<sup>†</sup> (ALiBi)       | 4k           | 13.49 | 18.36              | 6.79              | 9.93          | 11.72             | 1.85          | 32.28           |
| ALiBi-7B-4K<sup>†</sup>        | 4k           | 9.93  | 7.23               | 5.98              | 7.4           | 5.69              | 0.67          | 32.61           |
| CLEX-7B-Chat-4K         | 4k           | 32.72 | 29.38              | 20.08             | 23.25         | 56.02             | 9.67          | 57.94           |


## InfiniteBench
We also evaluate CLEX-Mixtral-8x7B-Chat-32k on [InfiniteBench](https://github.com/OpenBMB/InfiniteBench), which is a 128k-length benchmark covering various tasks. We compare our CLEX-Mixtral-8x7B-Chat-32k with GPT-4, Claude, KimiChat, and vanilla Mixtral-8x7B.

| Task Name           | GPT-4  | YaRN-Mistral-7B | Kimi-Chat | Claude 2 | CLEX-Mixtral-8x7B-Chat-32k | Mixtral-8x7B-Instruct-v0.1 |
| ------------------- | ------ | --------------- | --------- | -------- | -------------------------- | -------------------------- |
| Retrieve.PassKey    | 100%   | 92.71%          | 98.14%    | 97.80%   | 99.72%                     | 96.78%                     |
| **Retrieve.Number** | 100%   | 56.61%          | 95.42%    | 98.14%   | 76.10%                     | 76.61%                     |
| **Retrieve.KV**     | 89.00% | < 5%            | 53.60%    | 65.40%   | <5%                        | <%5                        |
| En.Sum              | 14.73% | 9.09%           | 17.93%    | 14.45%   | 15.48%                     | 14.3%                      |
| En.QA               | 22.22% | 9.55%           | 16.52%    | 11.97%   | 15.52%                     | 16.81%                     |
| En.MC               | 67.25% | 27.95%          | 72.49%    | 62.88%   | 58.96%                     | 56.77%                     |
| En.Dia              | 8.50%  | 7.50%           | 11.50%    | 46.50%   | 9%                         | <5%                        |
| Code.Debug          | 39.59% | < 5%            | 18.02%    | < 5%     | 21.32%                     | <5%                        |
| Code.Run            | 23.25% | < 5%            | < 5%      | < 5%     | < 5%                       | <5%                        |
| Math.Calc           | < 5%   | < 5%            | < 5%      | < 5%     | < 5%                       | <5%                        |
| Math.Find           | 60.00% | 17.14%          | 12.57%    | 32.29%   | 28%                        | 26.57%                     |

Key points:
- We found Mixtral-8x7B-Instruct-v0.1 has some extrapolation ability, by setting the `rope_theta` as 1e6 following CodeLLaMA.
- Our CLEX-Mixtral-8x7B-Chat-32k is also trained on 32k but perform better than vanilla Mixtral-8x7B-Instruct-v0.1 on most tasks. 
- Note that we only apply a "toy" SFT on Ultrachat 200K for one epoch, so the many bad cases of our model may be caused by the unsolid instuction-following ability (verbose or incomplete answers). The performance may hold great potential to be improved.


## Acknowledgement
We would like to express our gratitude to the following open-sourcing efforts our CLEX benefits from:
- [LLaMA-2](https://github.com/facebookresearch/llama): Open Foundation and Fine-Tuned Chat Models
- [FastChat](https://github.com/lm-sys/FastChat): An Open Platform for Training, Serving, and Evaluating Large Language Models.
- [RedPajama-Data](https://github.com/togethercomputer/RedPajama-Data): An Open Source Recipe to Reproduce LLaMA training dataset
- [Pile](https://pile.eleuther.ai/): An 800GB Dataset of Diverse Text for Language Modeling
- [PG-19](https://openreview.net/pdf?id=SylKikSYDH): Language Modeling Language Modeling Benchmark
- [UltraChat](https://github.com/thunlp/UltraChat): Large-scale, Informative, and Diverse Multi-round Dialogue Data, and Models
- [InfiniteBench](https://github.com/OpenBMB/InfiniteBench): 100k+ Long-Context Benchmark for Large Language Models

## Citation
If you find our project useful, hope you can star our repo and cite our paper as follows:
```
@article{damonlpsg2023clex,
  author = {Chen, Guanzheng and Li, Xin and Meng, Zaiqiao and Liang, Shangsong and Bing, Lidong},
  title = {CLEX: Continuous Length Extrapolation for Large Language Models},
  year = 2023,
  journal = {arXiv preprint arXiv:2310.16450},
  url = {https://arxiv.org/abs/2310.16450}
}
```

