import torch
import fire
from peft import (
    PeftModel,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer,AutoTokenizer
from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
# Assuming device and model setup remains as described in your initial script


def main(
    load_8bit: bool = False,
    base_model: str = "",
    lora_weights_path: str = "",
    lora_config_path: str= "", # provide only the file path, excluding the file name 'adapter_config.json'
    prompt_template: str = ""):

    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    if not lora_weights_path.endswith(".bin"):
        if device == "cuda":
            model = LlamaForCausalLM.from_pretrained(
                base_model,
                load_in_8bit=load_8bit,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            model = PeftModel.from_pretrained(
                model,
                lora_weights_path,
                torch_dtype=torch.float16,
            )
        else:
            model = LlamaForCausalLM.from_pretrained(
                base_model, device_map={"": device}, low_cpu_mem_usage=True
            )
            model = PeftModel.from_pretrained(
                model,
                lora_weights_path,
                device_map={"": device},
            )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = prepare_model_for_int8_training(model)
        config = LoraConfig.from_pretrained(lora_config_path)
        lora_weights = torch.load(lora_weights_path)
        model = PeftModel(model, config)
        set_peft_model_state_dict(model,lora_weights,"default")
        del lora_weights

    model.eval()


    # Evaluation
    correct_predictions = 0
    test_cases_path = './data/testing/shakespeare_instruction_response_pairs_all.json'
    with open(test_cases_path, 'r') as f:
        test_cases = json.load(f)
    for case in test_cases:
        # Generate the prompt using your logic; adjust as necessary
        prompt = prompter.generate_prompt(case["instruction"], case["context"])
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Generate output
        with torch.no_grad():
            outputs = model.generate(inputs["input_ids"], max_length=1, num_return_sequences=1)
        
        # Decode generated ID to text
        predicted_char = tokenizer.decode(outputs[:, -1][0], skip_special_tokens=True)
        expected_char = case["response"].strip()

        # Evaluate prediction
        if predicted_char.lower() == expected_char.lower():
            correct_predictions += 1
        else:
            print(f"Wrong prediction for: {prompt}\nExpected: {expected_char}, Got: {predicted_char}")

    # Calculate and print accuracy
    accuracy = correct_predictions / len(test_cases)
    print(f"Accuracy: {accuracy:.2f} ({correct_predictions}/{len(test_cases)})")


if __name__ == "__main__":
    fire.Fire(main)