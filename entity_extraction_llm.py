import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import bitsandbytes as bnb
import json

def load_model(model_name, quantize=False):
    """
    Loads the model with optional quantization and moves it to the available device (CPU/GPU).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if quantize:
        # Load the model with 8-bit quantization using bitsandbytes
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            load_in_8bit=True, 
            device_map='auto', 
            quantization_config=bnb.QuantizationConfig(
                load_in_8bit=True
            )
        )
    else:
        # Load the model in mixed precision (fp16)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16,
            device_map='auto'
        )

    return model, tokenizer

def extract_json_from_output(output_text):
    """
    Extracts JSON content from the model output using regex and stops at <END>.
    """
    try:
        # Remove everything after <END> (if present)
        output_text = output_text.split("<END>")[0]
        print(f"Processed Output:\n{output_text}")
        
        # Extract the JSON part from the text
        json_match = re.search(r"\{.*?\}", output_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            json_obj = json.loads(json_str)
            return json_obj
        else:
            raise ValueError("No JSON found in model output.")
    except Exception as e:
        print(f"Error extracting JSON: {e}")
        return None

def extract_entities_llm(user_input, model_name="microsoft/Phi-3.5-mini-instruct", quantize=False):
    """
    Extracts entities from the user input using a language model.
    """
    # Load the model and tokenizer with optional quantization
    model, tokenizer = load_model(model_name, quantize)
    
    # Set pad_token to eos_token if no padding token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Add few-shot examples to guide the model
    few_shot_examples = """
    Input: Rotate the car by azimuth +45 degrees and polar +30 degrees.
    Output: {"object": "car", "azimuth": 45.0, "polar": 30.0}
    <END>

    Input: Adjust the table by azimuth -90 degrees and polar +15 degrees.
    Output: {"object": "table", "azimuth": -90.0, "polar": 15.0}
    <END>
    """

    # Construct the prompt with few-shot examples
    prompt = f"""Extract the object class, azimuth angle, and polar angle from the text as entity retrieval task.
Provide the output in JSON format with keys "object", "azimuth", and "polar", where angles are in degrees as floats. The output must end with <END>.
Here are few examples to guide you:
{few_shot_examples}

Now perform the task and output only single JSON object, ending the response after:
Input: {user_input}
Output:
"""
    
    # Tokenize input and pass attention mask
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)

    # Generate output
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,  # Explicitly pass attention_mask
            max_length=inputs.input_ids.shape[1] + 50,
            temperature=0.5,
            do_sample=True,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Get the length of the input tokens
    input_length = inputs.input_ids.shape[1]
    
    # Decode only the generated text (ignore the input prompt)
    generated_tokens = outputs[0, input_length:]  # Slice only the generated tokens
    output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    print("Model Output:\n" + output_text)
    
    # Extract the JSON part
    json_obj = extract_json_from_output(output_text)
    
    if json_obj:
        object_name = json_obj.get("object")
        azimuth = float(json_obj.get("azimuth", 0))
        polar = float(json_obj.get("polar", 0))
        print(f"Extracted JSON: {json_obj}")
        return object_name, azimuth, polar
    else:
        print("Failed to extract entities from user input.")
        return None, None, None

# Example usage
if __name__ == "__main__":
    user_input = "Please rotate the chair by azimuth 72 degrees and polar -5 degrees."
    object_name, azimuth, polar = extract_entities_llm(user_input, model_name="microsoft/Phi-3.5-mini-instruct", quantize=True)
    print(f"Object: {object_name}, Azimuth: {azimuth}, Polar: {polar}")
