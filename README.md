# Infini-Megrez: Open Source LLM Family Developed By Infinigence-AI

<p align="center">
    <img src="assets/megrez_logo.png" width="400"/>
<p>

<p align="center">
        🤗 <a href="https://huggingface.co/Infinigence/Megrez-3B-Instruct">HuggingFace</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://www.modelscope.cn/models/InfiniAI/Megrez-3b-Instruct">ModelScope</a>&nbsp&nbsp | &nbsp&nbsp🧙 <a href="https://modelers.cn/models/INFINIGENCE-AI/Megrez-3B-Instruct">Modelers</a>&nbsp&nbsp <br> &nbsp&nbsp🏠 <a href="https://cloud.infini-ai.com/genstudio/model/mo-c73owqiotql7lozr">Infini-AI mass</a>&nbsp&nbsp | &nbsp&nbsp🖥️ <a href="https://huggingface.co/spaces/Infinigence/Infinigence-AI-Chat-Demo">HF Demo</a>&nbsp&nbsp |  &nbsp&nbsp📖 <a href="https://cloud.infini-ai.com/assets/png/wechat_official_account.1f7e61401727063822266.png">WeChat Official</a>&nbsp&nbsp | &nbsp&nbsp💬 <a href="https://cloud.infini-ai.com/assets/png/wechat_community.7dbbc0b51727063822266.png">WeChat Groups</a>&nbsp&nbsp   
</p>

## Introduction
Megrez-3B-Instruct is a large language model independently trained by Infinigence AI. Designed through an integrated approach to software and hardware optimization, Megrez-3B aims to provide ultra-fast inference, compact yet powerful performance, and highly accessible edge-side intelligent solutions. The model offers several key advantages:

- **High Accuracy**: Despite its relatively small size of 3 billion parameters, Megrez-3B significantly narrows the performance gap through substantial improvements in data quality. It effectively compresses the capabilities of a previous 14-billion parameter model into a 3-billion parameter framework, achieving exceptional performance on mainstream benchmarks.
- **High Speed**: A smaller model does not inherently guarantee faster speeds. Megrez-3B leverages software and hardware co-optimization to ensure high compatibility with mainstream hardware, delivering a 300% improvement in inference speed compared to models of equivalent accuracy.
- **Ease of Use**: During the model's development, we deliberated on whether to prioritize structural designs that would allow for enhanced software and hardware collaboration (e.g., through ReLU activation, sparsity, and more streamlined architectures) or to maintain a classical structure for ease of use. We opted for the latter, implementing the traditional LLaMA2 architecture. This decision allows developers to deploy the model across various platforms without modification, thereby reducing the complexity of further development.
- **Versatile Applications**: Based on Megrez-3B-Instruct, we provide a comprehensive WebSearch solution. In comparison to search_with_lepton, we have conducted targeted training on the model, enabling it to automatically determine when to call the search function and provide superior summarization results. Users can build their own Kimi or Perplexity based on this feature, overcoming common issues of hallucinations and knowledge-reserve limitations of smaller models.


<a name="news-and-updates"></a>
## News and Updates
- 2024.09.24: We released the Megrez-3B-Instruct.


<a name="quick-start"></a>
## Quick Start
### Requirements
```
torch==2.1.2
numpy==1.26.4
transformers==4.44.2
accelerate==0.34.2
vllm==0.6.1.post2
```

### Inference
#### 🤗 HuggingFace Transformers
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "Infinigence/Megrez-3B-Instruct"
device = "cuda"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_romote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True)

messages = [{"role": "user", "content": "How to make braised chicken in brown sauce?"}]
model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(device)
model_outputs = model.generate(
    model_inputs,
    do_sample = True,
    max_new_tokens=2048,
    top_p=0.9,
    temperature=0.7
)

output_token_ids = [
    model_outputs[i][len(model_inputs[i]):] for i in range(len(model_inputs))
]
responses = tokenizer.batch_decode(output_token_ids, skip_special_tokens=True)[0]
print(responses)
```

#### 🤖 ModelScope
```python
import torch
from modelscope import AutoTokenizer, AutoModelForCausalLM

model_path = "Infinigence/Megrez-3B-Instruct"
device = "cuda"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_romote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True)

messages = [{"role": "user", "content": "How to make braised chicken in brown sauce?"}]
model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(device)
model_outputs = model.generate(
    model_inputs,
    do_sample = True,
    max_new_tokens=2048,
    top_p=0.9,
    temperature=0.7
)

output_token_ids = [
    model_outputs[i][len(model_inputs[i]):] for i in range(len(model_inputs))
]
responses = tokenizer.batch_decode(output_token_ids, skip_special_tokens=True)[0]
print(responses)
```

#### 💻 vLLM
```python
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

model_name = "Infinigence/Megrez-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
llm = LLM(
    model=model_name,
    trust_remote_code=True,
    tensor_parallel_size=1
)

messages = [{"role": "user", "content": "How to make braised chicken in brown sauce?"}]
input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
sampling_params = SamplingParams(top_p=0.9, temperature=0.7, max_tokens=2048, repetition_penalty=1.02)
outputs = llm.generate(prompts=input_text, sampling_params=sampling_params)

print(outputs[0].outputs[0].text)
```

### Deployment
#### vLLM API Server
```bash
python -m vllm.entrypoints.openai.api_server --served-model-name Megrez-3B-Instruct --model /local/path/to/Megrez-3B-Instruct --port 8000 --tensor-parallel-size 1
```

### Tool Use
Megrez-3B-Instruct supports function-calling, especially optimized for web-search agents. Please refer to our release [InfiniWebSearch](https://github.com/infinigence/InfiniWebSearch) framework for a more detailed information.


### Throughput Benchmarking
```bash
python benchmark_throughput.py --model /local/path/to/Qwen-7B-Chat/ --input-len 128 --output-len 128 --max-num-seqs 8 --max-model-len 256 --trust-remote-code
```


<a name="performance"></a>
## Performance
We have evaluated Megrez-3B-Instruct using the open-source evaluation tool [OpenCompass](https://github.com/open-compass/opencompass) on several important benchmarks. Some of the evaluation results are shown in the table below. For more evaluation results, please visit the [OpenCompass leaderboard](https://rank.opencompass.org.cn/).

The inference speeds reported here were all obtained using [vllm](https://github.com/vllm-project/vllm). The experimental configuration is `batch_size=8`, `prefill_tokens=128` and `decode_tokens=128`.


### Model Card
|   Architecture   | Context length | # Params (Total) | # Non-Emb Params | Vocab Size | Training data | Supported languages |
| :--------------: | :------------: | :------------: | :----------------------------------------: | :--------: | :-----------: | :-----------------: |
| Llama-2 with GQA |   4K tokens    |     2.92B      |                   2.29B                    |   122880   |   2T tokens   |  Chinese & English  |

### General Benchmark
|        Models         | Instruction-tuned | # Non-Emb Params (B) | Inference Speed (tokens/s) | C-EVAL | CMMLU | MMLU  | MMLU-Pro | HumanEval | MBPP  | GSM8K | MATH  |
| :-------------------: | :---------------: | :------------: | :------------------------: | :----: | :---: | :---: | :------: | :-------: | :---: | :---: | :---: |
|  Megrez-3B-Instruct   |         Y         |      2.3       |          2329.38           |  81.4  | 74.5  | 70.6  |   48.2   |   62.2    | 77.4  | 64.8  | 26.5  |
|      Qwen2-1.5B       |                   |      1.3       |          3299.53           |  70.6  | 70.3  | 56.5  |   21.8   |   31.1    | 37.4  | 58.5  | 21.7  |
|     Qwen2.5-1.5B      |                   |      1.3       |          3318.81           |   -    |   -   | 60.9  |   28.5   |   37.2    | 60.2  | 68.5  | 35.0  |
|      MiniCPM-2B       |                   |      2.4       |          1930.79           |  51.1  | 51.1  | 53.5  |    -     |   50.0    | 47.3  | 53.8  | 10.2  |
|      Qwen2.5-3B       |                   |      2.8       |          2248.33           |   -    |   -   | 65.6  |   34.6   |   42.1    | 57.1  | 79.1  | 42.6  |
|  Qwen2.5-3B-Instruct  |         Y         |      2.8       |          2248.33           |   -    |   -   |   -   |   43.7   |   74.4    | 72.7  | 86.7  | 65.9  |
|      Qwen1.5-4B       |                   |      3.2       |          1837.91           |  67.6  | 66.7  | 56.1  |    -     |   25.6    | 29.2  | 57.0  | 10.0  |
| Phi-3.5-mini-instruct |         Y         |      3.6       |          1559.09           |  46.1  | 46.9  | 69.0  |    -     |   62.8    | 69.6  | 86.2  | 48.5  |
|      MiniCPM3-4B      |         Y         |      3.9       |           901.05           |  73.6  | 73.3  | 67.2  |    -     |   74.4    | 72.5  | 81.1  | 46.6  |
|       Yi-1.5-6B       |                   |      5.5       |          1542.66           |   -    | 70.8  | 63.5  |    -     |   36.5    | 56.8  | 62.2  | 28.4  |
|      Qwen1.5-7B       |                   |      6.5       |          1282.27           |  74.1  | 73.1  | 61.0  |   29.9   |   36.0    | 51.6  | 62.5  | 20.3  |
|       Qwen2-7B        |                   |      6.5       |          1279.37           |  83.2  | 83.9  | 70.3  |   40.0   |   51.2    | 65.9  | 79.9  | 44.2  |
|      Qwen2.5-7B       |                   |      6.5       |          1283.37           |   -    |   -   | 74.2  |   45.0   |   57.9    | 74.9  | 85.4  | 49.8  |
|   Meta-Llama-3.1-8B   |                   |      7.0       |          1255.91           |   -    |   -   | 66.7  |   37.1   |     -     |   -   |   -   |   -   |
|     GLM-4-9B-chat     |         Y         |      8.2       |          1076.13           |  75.6  | 71.5  | 72.4  |    -     |   71.8    |   -   | 79.6  | 50.6  |
|  Baichuan2-13B-Base   |                   |      12.6      |           756.71           |  58.1  | 62.0  | 59.2  |    -     |   17.1    | 30.2  | 52.8  | 10.1  |
|      Qwen1.5-14B      |                   |      12.6      |           735.61           |  78.7  | 77.6  | 67.6  |    -     |   37.8    | 44.0  | 70.1  | 29.2  |

### Chat Benchmark 
|       Models        | # Non-Emb Params (B) | Inference Speed (tokens/s) | MT-Bench | AlignBench (ZH) |
| :-----------------: | :------------: | :------------------------: | :------: | :-------------: |
| Megrez-3B-Instruct  |      2.3       |          2329.38           |   8.76   |      6.91       |
| MiniCPM-2B-sft-bf16 |      2.4       |          1930.79           |    -     |      4.64       |
| MiniCPM-2B-dpo-bf16 |      2.4       |          1930.79           |   7.25   |        -        |
| Qwen2.5-3B-Instruct |      2.8       |          2248.33           |    -     |        -        |
|     MiniCPM3-4B     |      3.9       |           901.05           |   8.41   |      6.74       |
|   Yi-1.5-6B-Chat    |      5.5       |          1542.66           |   7.5    |       6.2       |
|   Qwen1.5-7B-Chat   |      6.5       |          1282.27           |   7.6    |       6.2       |
|    Qwen2-7B-Chat    |      6.5       |          1279.37           |   8.41   |      7.21       |
| Qwen2.5-7B-Instruct |      6.5       |          1283.37           |   8.75   |        -        |
|    GLM4-9B-Chat     |      8.2       |          1076.13           |   8.35   |      7.01       |
| Baichuan2-13B-Chat  |      12.6      |           756.71           |    -     |      5.25       |

### LLM Leaderboard
|         Models         | # Non-Emb Params (B) | Inference Speed (tokens/s) | IFEval |  BBH  | ARC_C | HellaSwag | WinoGrande | TriviaQA |
| :--------------------: | :------------: | :------------------------: | :----: | :---: | :---: | :-------: | :--------: | :------: |
|   Megrez-3B-Instruct   |      2.3       |          2329.38           |  78.4  | 61.0  | 90.9  |   83.6    |    72.7    |   82.5   |
|       MiniCPM-2B       |      2.4       |          1930.79           |   -    | 36.9  | 68.0  |   68.3    |     -      |   32.5   |
|       Qwen2.5-3B       |      2.8       |          2248.33           |   -    | 56.3  | 56.5  |   74.6    |    71.1    |    -     |
|  Qwen2.5-3B-instruct   |      2.8       |          2248.33           |  58.2  |   -   |   -   |     -     |     -      |    -     |
| Phi-3.5-mini-instruct  |      3.6       |          1559.09           |   -    | 69.0  | 84.6  |   69.4    |    68.5    |    -     |
|      MiniCPM3-4B       |      3.9       |           901.05           |  68.4  | 70.2  |   -   |     -     |     -      |    -     |
|     Qwen2-7B-Chat      |      6.5       |          1279.37           |   -    | 62.6  | 60.6  |   80.7    |    77.0    |    -     |
| Meta-Llama-3.1-8B-inst |      7.0       |          1255.91           |  71.5  | 28.9  | 83.4  |     -     |     -      |    -     |


## Application
<details><summary><b>On-device Inference</b></summary>
MLCChat is a on-device chat app, coming soon.
<div align="center">
    <img src="assets/deployment_android.gif" width="50%" alt="Example GIF">
</div>

</details>

<details><summary><b>Web-Search Agent</b></summary>
InfiniWebSearch is our open-sourced web-search agent. Please refer to https://github.com/infinigence/InfiniWebSearch.
<div align="center">
    <img src="assets/websearch_demo.gif" width="100%" alt="Example GIF">
</div>

</details>


<a name="limitations"></a>
## Limitations
- **Hallucination**: LLMs inherently suffer from hallucination issues. Users are advised not to fully trust the content generated by the model. If more factually accurate outputs are desired, we recommend utilizing our WebSearch framework, as detailed in [InfiniWebSearch](https://github.com/infinigence/InfiniWebSearch).
- **Mathematics & Reasoning**: SLMs tend to produce incorrect calculations or flawed reasoning chains in tasks involving mathematics and reasoning, leading to erroneous outputs. Notably, the softmax distribution of SLMs is less sharp compared to LLMs, making them more prone to inconsistent reasoning results, particularly under higher temperature settings. This is especially evident in deterministic tasks such as mathematics and logical reasoning. We recommend lowering the temperature or verifying through multiple inference attempts in such cases.
- **System Prompt**: As with most LLMs, we recommend using the default system prompt from the `chat_template` in the configuration file for a stable and balanced performance. This model release has de-emphasized capabilities related to domain-specific applications such as role-playing. For users with specific domain needs, we suggest fine-tuning the model accordingly.
- **Values & Safety**: Every effort has been made to ensure the compliance of the data used during the training of this model. However, given the large scale and complexity of the data, unforeseen issues may still arise. We do not assume any responsibility for any issues that may result from the use of this open-source model, including but not limited to data security concerns, risks related to public opinion, or any risks and problems arising from the misguidance, misuse, or dissemination of the model.


<a name="license"></a>
## License
All our open-source models are licensed under Apache 2.0. You can find the license files in this repository and the respective Hugging Face repositories. 


<a name="contact"></a>
## Contact
If you are interested to leave a message to either our research team or product team, join our [WeChat groups](https://cloud.infini-ai.com/assets/png/wechat_community.7dbbc0b51727063822266.png).
