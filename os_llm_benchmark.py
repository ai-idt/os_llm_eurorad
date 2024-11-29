from llama_cpp import Llama
import pandas as pd

"""
HYPER_PARAMETERS
NOTE:   In the call to load the model, we defined to fully offload the model to the GPU (n_gpu_layers=-1).
        If a model does not fit into your GPU's VRAM (in particular, the 70B models), you must either (i) omit this model (outcomment it in "MODEL_ZOO" below) or (ii) change the offloading
"""
max_tokens = 1024
n_ctx = 4096
temperature = 0.0
verbose = False

base_prompt = 'You are a senior radiologist. Below, you will find information about a patient: First, the clinical presentation, followed by imaging findings. Based on this information, name the three most likely differential diagnoses, with a short rationale for each.'
correctness_prompt = 'You are a senior radiologist. Below, you will find the correct diagnosis (given after "Correct Diagnosis:"), followed by the differential diagnoses given by a Radiology Assistant during an exam. Please grade if the radiology assistant provided the correct diagnosis in his differential diagnosis. Only reply with either "correct" (when the correct diagnosis is contained in the answer by the radiology assistant) or "wrong", if it is not.'

model_folder = "/mnt/8tb_slot8/benedikt/" # Adjust this to where you store your .gguf files
output_file = "/home/benedikt/output.csv" # This is where the final .csv (with model response and LLM-Judge evaluation) is stored

judge_llm = {"Model": 'Meta-Llama-3-70B-Instruct-Q4_K_M.gguf',
            "PromptStyle": "llama-3"} #This is the LLM to judge the responses

"""
MODEL_ZOO
NOTE: This is a dict, where keys are model file names and values are the prompt construction template to be used
"""
models = {'Phi-3-medium-128k-instruct-Q5_K_M.gguf': "phi-3", #https://huggingface.co/bartowski/Phi-3-medium-128k-instruct-GGUF
 'gemma-2-27b-it-Q5_K_M.gguf': "gemma", #https://huggingface.co/bartowski/gemma-2-27b-it-GGUF
 'BioMistral-7B-DARE-GGUF-Q5_K_M.gguf': "mistral", #https://huggingface.co/BioMistral/BioMistral-7B-DARE-GGUF
 'Meta-Llama-3-70B-Instruct-Q4_K_M.gguf': "llama-3", #https://huggingface.co/bartowski/Meta-Llama-3-70B-Instruct-GGUF
 'Meta-Llama-3-8B-Instruct-Q5_K_M.gguf': "llama-3", #https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF
 'OpenBioLLM-Llama3-8B-Q5_K_M.gguf': "mistral", #https://huggingface.co/bartowski/OpenBioLLM-Llama3-8B-GGUF
 'llama-2-70b-chat.Q4_K_M.gguf': "llama-2", #https://huggingface.co/TheBloke/Llama-2-70B-Chat-GGUF
 'medalpaca-13b.Q5_K_M.gguf': "medalpaca", #https://huggingface.co/mradermacher/medalpaca-13b-GGUF
 'meditron-7b-chat.Q5_K_M.gguf': "meditron", #https://huggingface.co/TheBloke/meditron-7B-chat-GGUF
 'mixtral-8x7b-instruct-v0.1.Q5_K_M.gguf': "mixtral", #https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF
 'vicuna-13b-v1.5.Q5_K_M.gguf': "vicuna", #https://huggingface.co/TheBloke/vicuna-13B-v1.5-GGUF
 'Mistral-Nemo-Instruct-2407-Q5_K_M.gguf': "mistral", # https://huggingface.co/bartowski/Mistral-Nemo-Instruct-2407-GGUF
 'Mistral-Small-Instruct-2409-Q5_K_M.gguf': "mistral", # https://huggingface.co/bartowski/Mistral-Small-Instruct-2409-GGUF
 'qwen2.5-32b-instruct-q5_k_m.gguf': "qwen", # https://huggingface.co/Qwen/Qwen2.5-32B-Instruct-GGUF
 'OpenBioLLM-Llama3-70B.Q4_K_M.gguf': "mistral"} #https://huggingface.co/mradermacher/OpenBioLLM-Llama3-70B-GGUF

"""
PROMPT_HELPER
"""
def construct_prompt(system_prompt_, user_prompt_, prompt_format_):
    """
    Constructs a prompt for use with the create_completion() function of llama_cpp_python according to prompt_format_
    """
    if prompt_format_ == "llama-2":
        formatted_prompt_ = "[INST] <<SYS>>" + system_prompt_ + "<</SYS>>\n{"+user_prompt_+"}[/INST]"

    elif prompt_format_ == "llama-3":
        formatted_prompt_ = "<|start_header_id|>system<|end_header_id|>\n{"+system_prompt_+"}<|eot_id|><|start_header_id|>user<|end_header_id|>\n{"+user_prompt_+"}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

    elif prompt_format_ == "mistral":
        formatted_prompt_ = "<|im_start|>system{"+system_prompt_+"}<|im_end|>\n<|im_start|>user{"+user_prompt_+"}<|im_end|>\n<|im_start|>assistant"

    elif prompt_format_ == "mixtral":
        formatted_prompt_ = "[INST] " + system_prompt_ + "\n" + user_prompt_ + " [/INST]"

    elif prompt_format_ == "vicuna":
        formatted_prompt_ = system_prompt_+"\nUSER: {" + user_prompt_ + "}\nASSISTANT:"

    elif prompt_format_ == "medalpaca":
        formatted_prompt_ = "Context: {"+system_prompt_+"}\nQuestion:{" + user_prompt_ + "}\nResponse:"

    elif prompt_format_ == "meditron":
        formatted_prompt_ = "Instruction:\n{" + system_prompt_ + "\n" + user_prompt_ + "}\nResponse:"

    elif prompt_format_ == "phi-3":
        formatted_prompt_ = "<|user|>\n" + system_prompt_ + "\n"+user_prompt_+"}\n<|end|><|assistant|><|end|>"

    elif prompt_format_ == "gemma":
        formatted_prompt_ = "<start_of_turn>user\n{" + system_prompt_ + "\n"+user_prompt_+"}\n<end_of_turn><start_of_turn>model<end_of_turn><start_of_turn>model"

    elif prompt_format_ == "qwen":
        formatted_prompt_ = "<|im_start|>system\n"+system_prompt_+"<|im_end|>\n<|im_start|>user\n"+user_prompt_+"}<|im_end|>\n<|im_start|>assistant\n"

    else:
        print("---------------------------------------\n")
        print("!UNKNOWN PROMPT FORMAT!\n")
        print("---------------------------------------")
        formatted_prompt_ = "Instruction:\n" + system_prompt_ + "\nQuestion:\n" + user_prompt_ + "\nResponse:"

    return formatted_prompt_

"""
MAIN
"""

res_df = pd.DataFrame()
case_list = [
            {"CaseDescription": "The patient is a 20 year old female with a cystic, contrast-enhancing mass in the left cerebellar hemisphere. The solid nodule with this cystic mass has strongly elevated perfusion signal.", "CorrectDiagnosis": "Medulloblastoma"}
            ] #Replace this with your own data; "case" should be a dict with a "CaseDescription" and "CorrectDiagnosis"
for case in case_list:
    for llm_model in models.keys():
        llm = Llama(model_path=model_folder+llm_model, n_gpu_layers=-1, n_ctx=n_ctx, verbose=verbose)
        prompt = construct_prompt(base_prompt,case["CaseDescription"],prompt_format_=models[llm_model])
        response = llm.create_completion(prompt,max_tokens=max_tokens,temperature=temperature)

        res_dict = {"Model": llm_model,
                    "Temperature": temperature,
                    "Prompt": prompt,
                    "CaseDescription": case["CaseDescription"],
                    "Reply": response["choices"][0]["text"],
                    "True_Diagnosis": case["CorrectDiagnosis"]}

        del(llm) #Free up memory

        #This judge evaluates, if the correct diagnosis is given by the model
        prompt_to_judge = "Correct Diagnosis:\n" + res_dict["True_Diagnosis"] + "\n" + "Radiology Assistant:\n" + res_dict["Reply"]
        prompt = construct_prompt(correctness_prompt,prompt_to_judge,prompt_format_=judge_llm["PromptStyle"])
        llm = Llama(model_path=model_folder+judge_llm["Model"], n_gpu_layers=-1, n_ctx=n_ctx, verbose=verbose)
        response = llm.create_completion(prompt,max_tokens=max_tokens,temperature=temperature)
        res_dict["Judge"] = response["choices"][0]["text"]
        del(llm)

        res_df = pd.concat([res_df,pd.DataFrame(res_dict,index=[0])],ignore_index=True)
        res_df.to_csv(output_file,index=False)