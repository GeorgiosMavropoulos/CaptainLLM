"""This file contains a method used to change prompt template to ALPACA-STYLE"""
from datasets.download_dataset_with_instructions import data
def format_input(entry):
    """Provide instruction text"""
    instruction_test = (
        f"Below is an instruction that describes a task.\n"
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = (
        f"\n\n### Input:{entry['input']}" if entry ['input'] else ""
    )
    return instruction_test + input_text


#try to find out if the method worked
model_input = format_input(data[999])
desired_response = f"\n\n### Response:\n {data[999]['output']}"
#print(model_input + desired_response)

#method to format on Phi-3 style
def format_ph3(entry):
     user_prompt_template = "|<user>|"

     input_text = f"{user_prompt_template}\n"

     if entry.get("instruction"):
        input_text += f"{entry['instruction']}\n"

     if entry.get("input"):
        input_text += f"\nFollowing word: {entry['input']}\n"

     return input_text

input = format_ph3(data[50])
response = f"\n<|assistant|>\n{data[50]['output']}"
#print(input + response)

