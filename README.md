# Infini-Megrez: Open Source LLM Family Developed By Infinigence-AI

<p align="center">
        🤗 <a href="https://huggingface.co/Infinigence/Megrez-3B-Instruct">HuggingFace</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/organization/Infinigence">ModelScope</a>&nbsp&nbsp
<br>
🖥️ <a href="http://39.107.190.207:8888">Demo</a>&nbsp&nbsp | &nbsp&nbsp💬 <a href="https://cloud.infini-ai.com/assets/png/wechat_community.7dbbc0b51726745659605.png">WeChat Groups</a>&nbsp&nbsp | &nbsp&nbsp🫨 <a href="https://cloud.infini-ai.com/assets/png/wechat_official_account.1f7e61401726745659605.png">WeChat Official</a>&nbsp&nbsp
</p>


## Table of Contents
- [Introduction](#leaderboard)
- [News and Updates](#news-and-updates)
- [Quick Start](#quick-start)
- [Performance](#performance)
- [Limitations](#limitations)
- [License](#license)
- [Citation](#citation)
- [Contact](#contact)


<a name="news-and-updates"></a>
## News and Updates
- 2024.09.27: We released the Megrez-3B.


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

### Download Models
You can download our models through HuggingFace or ModelScope.
| Model Name             | HF Link                                               | MS Link |
| ---------------------- | ----------------------------------------------------- | ------- |
| Megrez-3B-Instruct     | https://huggingface.co/Infinigence/Megrez-3B-Instruct | xxxxx   |

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


<a name="performance"></a>
## Performance
We have evaluated Megrez-3B using the open-source evaluation tool [OpenCompass](https://github.com/open-compass/opencompass) on several important benchmarks. Some of the evaluation results are shown in the table below. For more evaluation results, please visit the [OpenCompass leaderboard](https://rank.opencompass.org.cn/).

### Model Card

### General Ability

### Chat Ability 

### Other Ability

<details><summary><b>On-device Inference</b></summary>
    
<div align="center">
    <img src="assets/deployment_android.gif" width="50%" alt="Example GIF">
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


<a name="citation"></a>
## Citation
```
@misc{yuan2024lveval,
      title={LV-Eval: A Balanced Long-Context Benchmark with 5 Length Levels Up to 256K}, 
      author={Tao Yuan and Xuefei Ning and Dong Zhou and Zhijie Yang and Shiyao Li and Minghui Zhuang and Zheyue Tan and Zhuyu Yao and Dahua Lin and Boxun Li and Guohao Dai and Shengen Yan and Yu Wang},
      year={2024},
      eprint={2402.05136},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```


<a name="contact"></a>
## Contact
If you are interested to leave a message to either our research team or product team, join our [WeChat groups](https://infinigence.feishu.cn/3aa43f45-6d47-47cd-b544-38cef330db84)
