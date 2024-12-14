# Megrez-3B: The integration of software and hardware unleashes the potential of edge intelligence

<p align="center">
    <img src="assets/megrez_logo.png" width="400"/>
<p>

<p align="center">
        🤗 <a href="https://huggingface.co/Infinigence/Megrez-3B-Instruct">HuggingFace</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://www.modelscope.cn/models/InfiniAI/Megrez-3b-Instruct">ModelScope</a>&nbsp&nbsp | &nbsp&nbsp🧙 <a href="https://modelers.cn/models/INFINIGENCE-AI/Megrez-3B-Instruct">Modelers</a>&nbsp&nbsp <br> &nbsp&nbsp🏠 <a href="https://cloud.infini-ai.com/genstudio/model/mo-c73owqiotql7lozr">Infini-AI mass</a>&nbsp&nbsp | &nbsp&nbsp📖 <a href="https://cloud.infini-ai.com/assets/png/wechat_official_account.1f7e61401727063822266.png">WeChat Official</a>&nbsp&nbsp | &nbsp&nbsp💬 <a href="https://cloud.infini-ai.com/assets/png/wechat_community.7dbbc0b51727063822266.png">WeChat Groups</a>&nbsp&nbsp   
</p>

## Introduction
Megrez-3B is a large language model trained by [Infinigence AI](https://cloud.infini-ai.com/platform/ai). Megrez-3B aims to provide a fast inference, compact, and powerful edge-side intelligent solution through software-hardware co-design. Megrez-3B has the following advantages:

- High Accuracy: Megrez-3B successfully compresses the capabilities of the previous 14 billion model into a 3 billion size, and achieves excellent performance on mainstream benchmarks.
- High Speed: A smaller model does not necessarily bring faster speed. Megrez-3B ensures a high degree of compatibility with mainstream hardware through software-hardware co-design, leading an inference speedup up to 300% compared to previous models of the same accuracy.
- Easy to Use: In the beginning, we had a debate about model design: should we design a unique but efficient model structure, or use a classic structure for ease of use? We chose the latter and adopt the most primitive LLaMA structure, which allows developers to deploy the model on various platforms without any modifications and minimize the complexity of future development.
- Rich Applications: We have provided a full stack WebSearch solution. Our model is functionally trained on web search tasks, enabling it to automatically determine the timing of search invocations and provide better summarization results. The complete deployment code is released on [github](https://github.com/infinigence/InfiniWebSearch). 


<a name="news-and-updates"></a>
## News and Updates
- 2024.12.16: We released the Megrez-3B-Instruct.


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
    temperature=0.2
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
    temperature=0.2
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
sampling_params = SamplingParams(top_p=0.9, temperature=0.2, max_tokens=2048)
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
| Model Name |    Architecture   | Context length | # Total Params | # Non-Emb Params | Vocab Size | Training data | Supported languages |
|:------:|:--------------: | :------------: | :------------: | :----------------------------------------: | :--------: | :-----------: | :-----------------: |
| Megrez-3B-Instruct | Llama-2 with GQA |   32K tokens    |     2.92B      |                   2.29B                    |   122880   |   3T tokens   |  Chinese & English  |

### General Benchmark
|        Model Name         | Chat Model | # Non-Emb Params (B) | Inference Speed (tokens/s) | C-EVAL | CMMLU | MMLU  | MMLU-Pro | HumanEval | MBPP  | GSM8K | MATH  |
| :-------------------: | :---------------: | :------------: | :------------------------: | :----: | :---: | :---: | :------: | :-------: | :---: | :---: | :---: |
|  Megrez-3B-Instruct   |         ✔         |      2.3       |          2329.38           |  81.4  | 74.5  | 70.6  |   48.2   |   62.2    | 77.4  | 64.8  | 26.5  |
|      Qwen2-1.5B       |                   |      1.3       |          3299.53           |  70.6  | 70.3  | 56.5  |   21.8   |   31.1    | 37.4  | 58.5  | 21.7  |
|     Qwen2.5-1.5B      |                   |      1.3       |          3318.81           |   -    |   -   | 60.9  |   28.5   |   37.2    | 60.2  | 68.5  | 35.0  |
|      MiniCPM-2B       |                   |      2.4       |          1930.79           |  51.1  | 51.1  | 53.5  |    -     |   50.0    | 47.3  | 53.8  | 10.2  |
|      Qwen2.5-3B       |                   |      2.8       |          2248.33           |   -    |   -   | 65.6  |   34.6   |   42.1    | 57.1  | 79.1  | 42.6  |
|  Qwen2.5-3B-Instruct  |         ✔         |      2.8       |          2248.33           |   -    |   -   |   -   |   43.7   |   74.4    | 72.7  | 86.7  | 65.9  |
|      Qwen1.5-4B       |                   |      3.2       |          1837.91           |  67.6  | 66.7  | 56.1  |    -     |   25.6    | 29.2  | 57.0  | 10.0  |
| Phi-3.5-mini-instruct |         ✔         |      3.6       |          1559.09           |  46.1  | 46.9  | 69.0  |    -     |   62.8    | 69.6  | 86.2  | 48.5  |
|      MiniCPM3-4B      |         ✔         |      3.9       |           901.05           |  73.6  | 73.3  | 67.2  |    -     |   74.4    | 72.5  | 81.1  | 46.6  |
|       Yi-1.5-6B       |                   |      5.5       |          1542.66           |   -    | 70.8  | 63.5  |    -     |   36.5    | 56.8  | 62.2  | 28.4  |
|      Qwen1.5-7B       |                   |      6.5       |          1282.27           |  74.1  | 73.1  | 61.0  |   29.9   |   36.0    | 51.6  | 62.5  | 20.3  |
|       Qwen2-7B        |                   |      6.5       |          1279.37           |  83.2  | 83.9  | 70.3  |   40.0   |   51.2    | 65.9  | 79.9  | 44.2  |
|      Qwen2.5-7B       |                   |      6.5       |          1283.37           |   -    |   -   | 74.2  |   45.0   |   57.9    | 74.9  | 85.4  | 49.8  |
|   Meta-Llama-3.1-8B   |                   |      7.0       |          1255.91           |   -    |   -   | 66.7  |   37.1   |     -     |   -   |   -   |   -   |
|     GLM-4-9B-chat     |         ✔         |      8.2       |          1076.13           |  75.6  | 71.5  | 72.4  |    -     |   71.8    |   -   | 79.6  | 50.6  |
|  Baichuan2-13B-Base   |                   |      12.6      |           756.71           |  58.1  | 62.0  | 59.2  |    -     |   17.1    | 30.2  | 52.8  | 10.1  |
|      Qwen1.5-14B      |                   |      12.6      |           735.61           |  78.7  | 77.6  | 67.6  |    -     |   37.8    | 44.0  | 70.1  | 29.2  |

### Chat Benchmark 
|       Model Name        | # Non-Emb Params (B) | Inference Speed (tokens/s) | MT-Bench | AlignBench (ZH) |
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
|         Model Name         | # Non-Emb Params (B) | Inference Speed (tokens/s) | IFEval |  BBH  | ARC_C | HellaSwag | WinoGrande | TriviaQA |
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
