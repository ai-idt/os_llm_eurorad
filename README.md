# Benchmarking Open Source LLMs in Challenging Radiology Cases

![Percentage correct across all cases.](results_overview.png)

**Files contained in this repo**

- eurorad_cases.csv: Full List of Case IDs selected to be included in our study. To review an individual case by its CaseID, go to https://eurorad.org/case/CaseID
- os_llm_benchmark.py: Script that performs the entire evaluation pipeline


**LLMs used in our study**

Please note that all models were quantized (gguf format) and used with [llama_cpp_python v. 0.2.79](https://github.com/abetlen/llama-cpp-python/releases/tag/v0.2.79)
- [BioMistral-7B-DARE-GGUF-Q5_K_M.gguf](https://huggingface.co/BioMistral/BioMistral-7B-DARE-GGUF)
- [Gemma-2-27b-it-Q5_K_M.gguf](https://huggingface.co/bartowski/gemma-2-27b-it-GGUF)
- [Llama-2-70b-chat.Q4_K_M.gguf](https://huggingface.co/TheBloke/Llama-2-70B-Chat-GGUF)
- [Medalpaca-13b.Q5_K_M.gguf](https://huggingface.co/mradermacher/medalpaca-13b-GGUF)
- [Meditron-7b-chat.Q5_K_M.gguf](https://huggingface.co/TheBloke/meditron-7B-chat-GGUF)
- [Meta-Llama-3-70B-Instruct-Q4_K_M.gguf](https://huggingface.co/bartowski/Meta-Llama-3-70B-Instruct-GGUF)
- [Meta-Llama-3-8B-Instruct-Q5_K_M.gguf](https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF)
- [Mixtral-8x7b-instruct-v0.1.Q5_K_M.gguf](https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF)
- [OpenBioLLM-Llama3-8B-Q5_K_M.gguf](https://huggingface.co/bartowski/OpenBioLLM-Llama3-8B-GGUF)
- [Phi-3-medium-128k-instruct-Q5_K_M.gguf](https://huggingface.co/bartowski/Phi-3-medium-128k-instruct-GGUF)
- [Vicuna-13b-v1.5.Q5_K_M.gguf](https://huggingface.co/TheBloke/vicuna-7B-v1.5-GGUF)
