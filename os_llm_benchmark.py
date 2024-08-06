import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" #Or whatever your large GPU is
from llama_cpp import Llama
from glob import glob
import json
import pandas as pd

"""
HYPER_PARAMETERS
"""
max_tokens = 1024
temperature = 0.0
verbose = False
base_prompt = 'You are a senior radiologist. Below, you will find information about a patient: First, the clinical presentation, followed by imaging findings. Based on this information, name the three most likely differential diagnoses, with a short rationale for each.'
correctness_prompt = 'You are a senior radiologist. Below, you will find the correct diagnosis (given after "Correct Diagnosis:"), followed by the differential diagnoses given by a Radiology Assistant during an exam. Please grade if the radiology assistant provided the correct diagnosis in his differential diagnosis. Only reply with either "correct" (when the correct diagnosis is contained in the answer by the radiology assistant) or "wrong", if it is not.'

"""
MODEL_ZOO
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
 'vicuna-13b-v1.5.Q5_K_M.gguf': "vicuna"} #https://huggingface.co/TheBloke/vicuna-7B-v1.5-GGUF

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

    else:
        print("---------------------------------------\n")
        print("!UNKNOWN PROMPT FORMAT!\n")
        print("---------------------------------------")
        formatted_prompt_ = "Instruction:\n" + system_prompt_ + "\nQuestion:\n" + user_prompt_ + "\nResponse:"

    return formatted_prompt_

"""
MAIN
"""

ordner = "/home/benedikt/eurorad_cases/"

case_files = glob(ordner + "/**/*_description.json",recursive=True)
case_files.sort()

res_df = pd.DataFrame()

for case_file in case_files:
    with open(case_file) as user_file:
        file_contents = user_file.read()
        cur_case = json.loads(file_contents)
    #We want to filter cases with short imaging descriptions, and those that have the diagnosis in the vignette     
    if len(cur_case["Imaging Findings"]) > 500:

        #This judge (LLama3-70B) decides if the diagnosis is mentioned in the case description, and returns either "mentioned" or "not mentioned"
        judge_prompt = "You are a senior radiologist. Below, you will find a case description for a patient. This patient was diagnosed with " + cur_case["Diagnosis"] + ". We want to use this case description for an exam. Please check if the diagnosis or part of it is mentioned, discussed or suggested in the case description. Please reply by saying either 'mentioned' (if the diagnosis is mentioned, discussed or suggested) or 'not mentioned', and nothing else."
        user_prompt = "Case Description:\n" + cur_case["Clinical Description"] + "\n" + cur_case["Imaging Findings"]
        llm = Llama(model_path="/mnt/8tb_slot8/benedikt/Meta-Llama-3-70B-Instruct-Q4_K_M.gguf", n_gpu_layers=-1, n_ctx=0, verbose=verbose)
        prompt_to_judge = "<|start_header_id|>system<|end_header_id|>\n{"+judge_prompt+"}<|eot_id|><|start_header_id|>user<|end_header_id|>\n{"+user_prompt+"}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        response = llm.create_completion(prompt_to_judge,max_tokens=1024,temperature=0.0)
        print(response["choices"][0]["text"])
        del(llm)

        if "not" in response["choices"][0]["text"]: #Cases w/o mentioning the diagnosis
            for repetition in [1]:
                prompt_to_summarize = user_prompt + "\n"
                for llm_model in models.keys():
                    llm = Llama(model_path="/mnt/8tb_slot8/benedikt/"+llm_model, n_gpu_layers=-1, n_ctx=0, verbose=verbose)
                    prompt = construct_prompt(base_prompt,user_prompt,prompt_format_=models[llm_model])
                    response = llm.create_completion(prompt,max_tokens=max_tokens,temperature=temperature)

                    res_dict = {"CaseID": os.path.basename(case_file)[:-22],
                                    "Repetition": repetition,
                                    "Model": llm_model,
                                    "Prompt": prompt,
                                    "Reply": response["choices"][0]["text"],
                                    "True_Diagnosis": cur_case["Diagnosis"]}

                    del(llm) #Free up memory

                    #This judge evaluates, if the correct diagnosis is given by the model
                    prompt_to_judge = "Correct Diagnosis:\n" + res_dict["True_Diagnosis"] + "\n" + "Radiology Assistant:\n" + res_dict["Reply"]
                    llm = Llama(model_path="/mnt/8tb_slot8/benedikt/Meta-Llama-3-70B-Instruct-Q4_K_M.gguf", n_gpu_layers=-1, n_ctx=0, verbose=verbose)
                    prompt_to_judge = "<|start_header_id|>system<|end_header_id|>\n{"+correctness_prompt+"}<|eot_id|><|start_header_id|>user<|end_header_id|>\n{"+prompt_to_judge+"}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
                    response = llm.create_completion(prompt_to_judge,max_tokens=1024,temperature=0.0)
                    res_dict["Judge"] = response["choices"][0]["text"]
                    del(llm)

                    res_df = pd.concat([res_df,pd.DataFrame(res_dict,index=[0])],ignore_index=True)
                    res_df.to_csv(ordner+"os-llm-benchmark-eurorad.csv",index=False)