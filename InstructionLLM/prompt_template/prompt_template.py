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
print(model_input + desired_response)