# Benchmarking Open Source LLMs in Challenging Radiology Cases

![Percentage correct across all cases.](results_overview.png)

**Files contained in this repo**

- eurorad_cases.csv: Full List of Case IDs selected to be included in our study. To review an individual case by its CaseID, go to https://eurorad.org/case/CaseID
- os_llm_benchmark.py: Script that performs the entire evaluation pipeline


**LLMs used in our study**

Please note that all models were quantized (gguf format) and used with [llama_cpp_python v. 0.2.89](https://github.com/abetlen/llama-cpp-python/releases/tag/v0.2.89)
|Model|# Params|Base Model|HF Link|Reference|
---|---|---|---|---
BioMistral-7B-DARE-GGUF-Q5_K_M.gguf | 7B | Mistral-7B | https://huggingface.co/BioMistral/BioMistral-7B-DARE-GGUF | https://arxiv.org/abs/2402.10373
Gemma-2-27b-it-Q5_K_M.gguf | 27B | Gemma-2 |https://huggingface.co/bartowski/gemma-2-27b-it-GGUF | https://arxiv.org/abs/2408.00118
Llama-2-70b-chat.Q4_K_M.gguf | 70B | Llama-2 | https://huggingface.co/TheBloke/Llama-2-70B-Chat-GGUF | https://arxiv.org/abs/2307.09288
Medalpaca-13b.Q5_K_M.gguf | 13B | LLaMA | https://huggingface.co/mradermacher/medalpaca-13b-GGUF | https://arxiv.org/abs/2304.08247
Meditron-7b-chat.Q5_K_M.gguf | 7B | Llama-2 | https://huggingface.co/TheBloke/meditron-7B-chat-GGUF | https://arxiv.org/abs/2311.16079
Meta-Llama-3-8B-Instruct-Q5_K_M.gguf | 8B | Llama-3 | https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF | https://arxiv.org/abs/2407.21783
Meta-Llama-3-70B-Instruct-Q4_K_M.gguf | 70B | Llama-3 | https://huggingface.co/bartowski/Meta-Llama-3-70B-Instruct-GGUF | https://arxiv.org/abs/2407.21783
Mistral-Nemo-Instruct-2407-Q5_K_M.gguf | 12B | Mistral-Nemo | https://huggingface.co/bartowski/Mistral-Nemo-Instruct-2407-GGUF | https://mistral.ai/news/mistral-nemo/
Mistral-Small-Instruct-2409-Q5_K_M.gguf | 22B | Mistral-Small |https://huggingface.co/bartowski/Mistral-Small-Instruct-2409-GGUF | https://mistral.ai/news/september-24-release/
Mixtral-8x7b-instruct-v0.1.Q5_K_M.gguf | 8x7B MoE | Mistral-7B | https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF | https://arxiv.org/abs/2401.04088
OpenBioLLM-Llama3-8B-Q5_K_M.gguf | 8B | Llama-3 | https://huggingface.co/bartowski/OpenBioLLM-Llama3-8B-GGUF | https://huggingface.co/blog/aaditya/openbiollm
OpenBioLLM-Llama3-70B.Q4_K_M.gguf | 70B | Llama-3 | https://huggingface.co/mradermacher/OpenBioLLM-Llama3-70B-GGUF | https://huggingface.co/blog/aaditya/openbiollm
Phi-3-medium-128k-instruct-Q5_K_M.gguf | 14B | Phi-3 | https://huggingface.co/bartowski/Phi-3-medium-128k-instruct-GGUF | https://arxiv.org/abs/2404.14219
Qwen2.5-32b-instruct-q5_k_m.gguf | 32B | Qwen-2.5 | https://huggingface.co/Qwen/Qwen2.5-32B-Instruct-GGUF | https://qwenlm.github.io/blog/qwen2.5/ & https://arxiv.org/abs/2407.10671
Vicuna-13b-v1.5.Q5_K_M.gguf | 13B | Llama-2 | https://huggingface.co/TheBloke/vicuna-13B-v1.5-GGUF | https://lmsys.org/blog/2023-03-30-vicuna/
