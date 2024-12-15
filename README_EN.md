# Megrez-3B: The integration of software and hardware unleashes the potential of edge intelligence

<p align="center">
    <img src="assets/megrez_logo.png" width="400"/>
<p>

<p align="center">
        🤗 <a href="https://huggingface.co/Infinigence/Megrez-3B-Instruct">Megrez-3B-Instruct</a>&nbsp&nbsp| &nbsp&nbsp🤗 <a href="https://huggingface.co/Infinigence/Megrez-3B-Omni"> Megrez-3B-Omni</a>&nbsp&nbsp  &nbsp | &nbsp&nbsp📖 <a href="https://cloud.infini-ai.com/assets/png/wechat_official_account.1f7e61401727063822266.png">WeChat Official</a>&nbsp&nbsp  |  &nbsp&nbsp💬 <a href="https://cloud.infini-ai.com/assets/png/wechat_community.7dbbc0b51727063822266.png">WeChat Groups</a>&nbsp&nbsp   
</p>
<h4 align="center">
    <p>
        <a  href="https://github.com/infinigence/Infini-Megrez/blob/main/README.md">中文</a> | <b>English</b>
    <p>
</h4>

# 目录

- [Model Downloads](#model-downloads)
- [Megrez-3B-Omni](#megrez-3b-omni)
  - [Evaluation Results](#evaluation-results)
    - [Image Understanding](#image-understanding)
    - [Text Understanding](#text-understanding)
    - [Audio Understanding](#audio-understanding)
    - [Inference Speed](#inference-speed)
  - [Qucikstart](#qucikstart)
    - [Online Experience](#online-experience)
    - [Local Deployment](#local-deployment)
    - [Notes](#notes)
- [Megrez-3B](#megrez-3b)
- [Open Source License and Usage Statement](#open-source-license-and-usage-statement)

# Model Downloads
| HuggingFace                                                  | ModelScope                  |Modelers
| :-----------------------------------------------------------:|:---------------------------:|:--------:|
| [Megrez-3B-Instruct-Omni](https://huggingface.co/Infinigence/Megrez-3B-Omni) | [Megrez-3B-Instruct-Omni](https://www.modelscope.cn/models/InfiniAI/Megrez-3B-Omni) |[Megrez-3B-Instruct-Omni](https://modelers.cn/models/NFINIGENCE-AI/Megrez-3B-Omni)  |
| [Megrez-3B-Instruct](https://huggingface.co/Infinigence/Megrez-3B-Instruct) | [Megrez-3B-Instruct](https://www.modelscope.cn/models/InfiniAI/Megrez-3b-Instruct)|[Megrez-3B-Instruct](https://modelers.cn/models/INFINIGENCE-AI/Megrez-3B-Instruct)|




# Megrez-3B-Omni
**Megrez-3B-Omni** is an edge-side multimodal understanding model developed by **Infinigence AI** ([Infinigence AI](https://cloud.infini-ai.com/platform/ai)). It is an extension of the Megrez-3B-Instruct model and supports comprehensive understanding and analysis of image, text, and audio modalities. The model achieves state-of-the-art accuracy in all three domains:

- Image Understanding: By utilizing SigLip-400M for constructing image tokens, Megrez-3B-Omni outperforms models with more parameters such as LLaVA-NeXT-Yi-34B, in overall performance. It is one of the highest-accuracy image understanding models across multiple mainstream benchmarks, including MME, MMVet, OCRBench, and MMMU. It demonstrates excellent performance in tasks such as scene understanding and OCR.

- Language Understanding: Megrez-3B-Omni retains exceptional text understanding capabilities without significant trade-offs. Compared to its single-modal counterpart (Megrez-3B-Instruct), its overall accuracy variation is less than 2%, maintaining state-of-the-art performance on benchmarks like C-EVAL, MMLU (Pro), and AlignBench. It continues to outperform previous-generation models with 14B parameters.

- Speech Understanding: Equipped with the encoder head of Whisper-large-v3 (~600M parameters), the model supports both Chinese and English speech input, multi-turn conversations, and voice-based questions about input images. It can directly respond to voice commands with text and has achieved leading results across multiple benchmark tasks.

## Evaluation Results
### Image Understanding


- The above image compares the performance of Megrez-3B-Omni with other open-source models on mainstream image multimodal tasks.
- The below image shows the performance of Megrez-3B-Omni on the OpenCompass test set. Image reference: [InternVL 2.5 Blog Post](https://internvl.github.io/blog/2024-12-05-InternVL-2.5/)

![Multitask](assets/multitask.jpg)

![OpencompassBmk](assets/opencompass.jpg)

| model                 | basemodel             | release time       | OpenCompass | MME      | MMMU val  | OCRBench | MathVista | RealWorldQA | MMVet  | hallusionBench | MMB TEST (en) | MMB TEST (zh) | TextVQA val | AI2D_TEST | MMstar    | DocVQA_TEST |
|:-----------------------:|:-----------------------:|:----------------:|:--------------------:|:----------:|:-----------:|:----------:|:-----------------:|:-------------:|:--------:|:----------------:|:--------------:|:--------------:|:-------------:|:-----------:|:-----------:|:-------------:|
| **Megrez-3B-Omni**    | **Megrez-3B**         | **2024.12.16** | **66.2**           | **2315** | **51.89** | **82.8** | **62**          | **71.89**   | **60** | **50.12**      | **80.8**     | **82.3**     | **80.3**    | **82.05** | **60.46** | **91.62**   |
| Qwen2-VL-2B-Instruct  | Qwen2-1.5B            | 2024.08.28     | 57.2               | 1872     | 41.1      | 79.4     | 43              | 62.9        | 49.5   | 41.7           | 74.9         | 73.5         | 79.7        | 74.7      | 48        | 90.1        |
| InternVL2.5-2B        | Internlm2.5-1.8B-chat | 2024.12.06     | 59.9               | 2138     | 43.6      | 80.4     | 51.3            | 60.1        | 60.8   | 42.6           | 74.7         | 71.9         | 74.3        | 74.9      | 53.7      | 88.7        |
| BlueLM-V-3B           | -                     | 2024.11.29     | 66.1               | -        | 45.1      | 82.9     | 60.8            | 66.7        | 61.8   | 48             | 83           | 80.5         | 78.4        | 85.3      | 62.3      | 87.8        |
| InternVL2.5-4B        | Qwen2.5-3B-Instruct   | 2024.12.06     | 65.1               | 2337     | 52.3      | 82.8     | 60.5            | 64.3        | 60.6   | 46.3           | 81.1         | 79.3         | 76.8        | 81.4      | 58.3      | 91.6        |
| Baichuan-Omni         | Unknown-7B            | 2024.10.11     | -                  | 2186     | 47.3      | 70.0     | 51.9            | 62.6        | 65.4   | 47.8           | 76.2         | 74.9         | 74.3        | -         | -         | -           |
| MiniCPM-V-2.6         | Qwen2-7B              | 2024.08.06     | 65.2               | 2348     | 49.8      | 85.2     | 60.6            | 69.7        | 60     | 48.1           | 81.2         | 79           | 80.1        | 82.1      | 57.26     | 90.8        |
| Qwen2-VL-7B-Instruct  | Qwen2-7B              | 2024.08.28     | 67                 | 2326     | 54.1      | 84.5     | 58.2            | 70.1        | 62     | 50.6           | 83           | 80.5         | 84.3        | 83        | 60.7      | 94.5        |
| MiniCPM-Llama3-V-2.5  | Llama3-Instruct 8B    | 2024.05.20     | 58.8               | 2024     | 45.8      | 72.5     | 54.3            | 63.5        | 52.8   | 42.4           | 77.2         | 74.2         | 76.6        | 78.4      | -         | 84.8        |
| VITA                  | Mixtral 8x7B          | 2024.08.12     | -                  | 2097     | 47.3      | 67.8     | 44.9            | 59          | 41.6   | 39.7           | 74.7         | 71.4         | 71.8        | -         | -         | -           |
| GLM-4V-9B             | GLM-4-9B              | 2024.06.04     | 59.1               | 2018     | 46.9      | 77.6     | 51.1            | -           | 58     | 46.6           | 81.1         | 79.4         | -           | 81.1      | 58.7      | -           |
| LLaVA-NeXT-Yi-34B     | Yi-34B                | 2024.01.18     | 55                 | 2006     | 48.8      | 57.4     | 40.4            | 66          | 50.7   | 34.8           | 81.1         | 79           | 69.3        | 78.9      | 51.6      | -           |
| Qwen2-VL-72B-Instruct | Qwen2-72B             | 2024.08.28     | 74.8               | 2482     | 64.5      | 87.7     | 70.5            | 77.8        | 74     | 58.1           | 86.5         | 86.6         | 85.5        | 88.1      | 68.3      | 96.5        |


### Text Understanding

|                       |          |             |                                       | Chat&Instruction |                 |        | Zh&En Tasks |            |       |          |  Code |       | Math |       |
|:---------------------:|:--------:|:-----------:|:-------------------------------------:|:---------:|:---------------:|:------:|:-------------:|:----------:|:-----:|:--------:|:---------:|:-----:|:--------:|:-----:|
|         models        | Instruction |   Release Time  | Transformer #Params （w/o emb&softmax） |  MT-Bench | AlignBench (ZH) | IFEval |  C-EVAL (ZH)  | CMMLU (ZH) | MMLU  | MMLU-Pro | HumanEval |  MBPP |   GSM8K  |  MATH |
| Megrez-3B-Omni        |     Y    |  2024.12.16 |                  2.3                  |    8.4    |       6.94       |  66.5  |     84.0      |    75.3    | 73.3  |   45.2   |   72.6    | 60.6  |   63.8   | 27.3  |
| Megrez-3B-Instruct    |     Y    |  2024.12.16 |                  2.3                  |   8.64    |      7.06       |  68.6  |     84.8      |    74.7    | 72.8  |   46.1   |   78.7    | 71.0  |   65.5   | 28.3  |
| Baichuan-Omni         |     Y    |  2024.10.11 |                  7.0                  |     -     |        -        |    -   |     68.9      |    72.2    |  65.3 |     -    |     -     |   -   |     -    |   -   |
| VITA                  |     Y    |  2024.08.12 |                 12.9                  |     -     |        -        |    -   |     56.7      |    46.6    | 71.0  |     -    |     -     |   -   |   75.7   |   -   |
| Qwen1.5-7B            |          |  2024.02.04 |                  6.5                  |     -     |        -        |    -   |     74.1      |    73.1    | 61.0  |   29.9   |   36.0    | 51.6  |   62.5   | 20.3  |
| Qwen1.5-7B-Chat       |     Y    |  2024.02.04 |                  6.5                  |   7.60    |      6.20       |    -   |     67.3      |      -     | 59.5  |   29.1   |   46.3    | 48.9  |   60.3   | 23.2  |
| Qwen1.5-14B           |          |  2024.02.04 |                 12.6                  |     -     |        -        |    -   |     78.7      |    77.6    | 67.6  |     -    |   37.8    | 44.0  |   70.1   | 29.2  |
| Qwen1.5-14B-Chat      |     Y    |  2024.02.04 |                 12.6                  |    7.9    |        -        |    -   |       -       |      -     |   -   |     -    |     -     |   -   |     -    |   -   |
| Qwen2-7B              |          |  2024.06.07 |                  6.5                  |     -     |        -        |    -   |     83.2      |    83.9    | 70.3  |   40.0   |   51.2    | 65.9  |   79.9   | 44.2  |
| Qwen2-7b-Instruct     |     Y    |  2024.06.07 |                  6.5                  |   8.41    |      7.21       |  51.4  |     80.9      |    77.2    | 70.5  |   44.1   |   79.9    | 67.2  |   85.7   | 52.9  |
| Qwen2.5-3B-Instruct   |     Y    |  2024.9.19  |                  2.8                  |     -     |        -        |    -   |       -       |      -     |   -   |   43.7   |   74.4    | 72.7  |   86.7   | 65.9  |
| Qwen2.5-7B            |          |  2024.9.19  |                  6.5                  |     -     |        -        |    -   |       -       |      -     | 74.2  |   45.0   |   57.9    | 74.9  |   85.4   | 49.8  |
| Qwen2.5-7B-Instruct   |     Y    |  2024.09.19 |                  6.5                  |   8.75    |        -        |  74.9  |       -       |      -     |   -   |   56.3   |   84.8    | 79.2  |   91.6   | 75.5  |
| Llama-3.1-8B          |          |  2024.07.23 |                  7.0                  |    8.3    |       5.7       |  71.5  |     55.2      |    55.8    | 66.7  |   37.1   |     -     |   -   |   84.5   | 51.9  |
| Llama-3.2-3B          |          |  2024.09.25 |                  2.8                  |     -     |        -        |  77.4  |       -       |      -     | 63.4  |     -    |     -     |   -   |   77.7   | 48.0  |
| Phi-3.5-mini-instruct |     Y    |  2024.08.23 |                  3.6                  |    8.6    |       5.7       |  49.4  |     46.1      |    46.9    | 69.0  |   47.4   |   62.8    | 69.6  |   86.2   | 48.5  |
| MiniCPM3-4B           |     Y    |  2024.09.05 |                  3.9                  |   8.41    |      6.74       |  68.4  |     73.6      |    73.3    | 67.2  |     -    |   74.4    | 72.5  |   81.1   | 46.6  |
| Yi-1.5-6B-Chat        |     Y    |  2024.05.11 |                  5.5                  |   7.50    |      6.20       |    -   |     74.2      |    74.7    | 61.0  |     -    |   64.0    | 70.9  |   78.9   | 40.5  |
| GLM-4-9B-chat         |     Y    |  2024.06.04 |                  8.2                  |   8.35    |      7.01       |  64.5  |     75.6      |    71.5    | 72.4  |     -    |   71.8    |   -   |   79.6   | 50.6  |
| Baichuan2-13B-Base    |          |  2023.09.06 |                 12.6                  |     -     |      5.25       |    -   |     58.1      |    62.0    | 59.2  |     -    |   17.1    | 30.2  |   52.8   | 10.1  |

- The metrics for the Qwen2-1.5B model differ between the original paper and the Qwen2.5 report. Currently, the accuracy figures from the original paper are being used.

### Audio Understanding

|       Model      |     Base model     | Release Time | Fleurs test-zh | WenetSpeech test_net | WenetSpeech test_meeting |
|:----------------:|:------------------:|:-------------:|:--------------:|:--------------------:|:------------------------:|
|  Megrez-3B-Omni  | Megrez-3B-Instruct |   2024.12.16  |      10.8      |           -          |           16.4           |
| Whisper-large-v3 |          -         |   2023.11.06  |      12.4      |         17.5         |           30.8           |
|  Qwen2-Audio-7B  |      Qwen2-7B      |   2024.08.09  |        9       |          11          |           10.7           |
|  Baichuan2-omni  |     Unknown-7B     |   2024.10.11  |        7       |          6.9         |            8.4           |
|       VITA       |    Mixtral 8x7B    |   2024.08.12  |        -       |      -/12.2(CER)     |        -/16.5(CER)       |

### Inference Speed

|                | image_tokens | prefill (tokens/s) | decode (tokens/s) |
|:----------------:|:------------:|:------------------:|:-----------------:|
| Megrez-3B-Omni |      448     |       6312.66      |       1294.9      |
| Qwen2-VL-2B    |     1378     |       7349.39      |       685.66      |
| MiniCPM-V-2_6  |      448     |       2167.09      |       452.51      |

Setup:  

- The testing environment utilizes an NVIDIA H100 GPU with VLLM. Each test includes 128 text tokens and a 720×1480 image as input, producing 128 output tokens, with `num_seqs` fixed at 8.  
- Under this setup, the decoding speed of **Qwen2-VL-2B** is slower than **Megrez-3B-Omni**, despite having a smaller base LLM. This is due to the larger number of image tokens generated when encoding images of the specified size, which impacts actual inference speed.  



## Qucikstart

### Online Experience

[HF Chat Demo](https://huggingface.co/spaces/Infinigence/Megrez-3B-Omni)(recommend) 

### Local Deployment

For environment installation and Vllm inference code deployment, refer to [Infini-Megrez-Omni](https://github.com/infinigence/Infini-Megrez-Omni)

Below is an example of using transformers for inference. By passing text, image, and audio in the content field, you can interact with various modalities and models.
```python
import torch
from transformers import AutoModelForCausalLM

path = "{{PATH_TO_PRETRAINED_MODEL}}"  # Change this to the path of the model.

model = (
    AutoModelForCausalLM.from_pretrained(
        path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    .eval()
    .cuda()
)

# Chat with text and image
messages = [
    {
        "role": "user",
        "content": {
            "text": "Please describe the content of the image.",
            "image": "./data/sample_image.jpg",
        },
    },
]

# Chat with audio and image
messages = [
    {
        "role": "user",
        "content": {
            "image": "./data/sample_image.jpg",
            "audio": "./data/sample_audio.m4a",
        },
    },
]

MAX_NEW_TOKENS = 100
response = model.chat(
    messages,
    sampling=False,
    max_new_tokens=MAX_NEW_TOKENS,
    temperature=0,
)
print(response)

```

## Notes
1. Please input the images in the first round to ensure inference effectiveness. There are no such restrictions for audio and text, which can be switched freely.
2. In the Automatic Speech Recognition (ASR) scenario, simply change content['text'] to "Convert speech to text."
3. In the OCR scenario, enabling sampling may introduce language model hallucinations that cause text changes. Consider disabling sampling for inference (sampling=False). However, disabling sampling may introduce model repetition.


# Megrez-3B

Megrez-3B is a large language model trained by Infinigence AI. Megrez-3B aims to provide a fast inference, compact, and powerful edge-side intelligent solution through software-hardware co-design. Megrez-3B has the following advantages:

- High Accuracy: Megrez-3B successfully compresses the capabilities of the previous 14 billion model into a 3 billion size, and achieves excellent performance on mainstream benchmarks.
- High Speed: A smaller model does not necessarily bring faster speed. Megrez-3B ensures a high degree of compatibility with mainstream hardware through software-hardware co-design, leading an inference speedup up to 300% compared to previous models of the same accuracy.
- Easy to Use: In the beginning, we had a debate about model design: should we design a unique but efficient model structure, or use a classic structure for ease of use? We chose the latter and adopt the most primitive LLaMA structure, which allows developers to deploy the model on various platforms without any modifications and minimize the complexity of future development.
- Rich Applications: We have provided a full stack WebSearch solution. Our model is functionally trained on web search tasks, enabling it to automatically determine the timing of search invocations and provide better summarization results. The complete deployment code is released on [github](https://github.com/infinigence/InfiniWebSearch).

The specific model capabilities and deployment code can be referenced. [Infini-Megrez](https://github.com/infinigence/Infini-Megrez/megrez/README_EN.md)


## Open Source License and Usage Statement

- **License**: The code in this repository is open-sourced under the [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) license.  
- **Hallucination**: Large models inherently have hallucination issues. Users should not completely trust the content generated by the model. 
- **Values and Safety**: While we have made every effort to ensure compliance of the data used during training, the large volume and complexity of the data may still lead to unforeseen issues. We disclaim any liability for problems arising from the use of this open-source model, including but not limited to data security issues, public opinion risks, or risks and problems caused by misleading, misuse, propagation, or improper utilization of the model.  

