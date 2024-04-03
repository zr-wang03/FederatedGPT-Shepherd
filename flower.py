# pip install -U flwr["simulation"]

from collections import OrderedDict
from typing import List, Tuple
import matplotlib.pyplot as plt
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import flwr as fl
from flwr.common import Metrics
from src.model import pyramidnet, TwoCNN
from src.model import resnet18
from dataset import cifar10
from sampling import get_splits, get_splits_fig
from datasets_utils import Subset
from config import args_parser


class GeneralClient(fl.client.NumPyClient):
    def __init__(self, client_id, model, data_path, output_dir):
        self.client_id = client_id
        self.model = model
        self.local_data_path = os.path.join(data_path, "local_training_{}.json".format(self.client_id))
        self.local_data = load_dataset("json", data_files=self.local_data_path)
        self.output_dir = output_dir
        self.local_output_dir = os.path.join(self.output_dir, "trainer_saved", "local_output_{}".format(self.client_id))

    def preprare_local_dataset(self, generate_and_tokenize_prompt, local_val_set_size):
        if local_val_set_size > 0:
            local_train_val = self.local_data["train"].train_test_split(
                test_size=local_val_set_size, shuffle=True, seed=42
            )
            self.local_train_dataset = (
                local_train_val["train"].shuffle().map(generate_and_tokenize_prompt)
            )
            self.local_eval_dataset = (
                local_train_val["test"].shuffle().map(generate_and_tokenize_prompt)
            )
        else:
            self.local_train_dataset = self.local_data["train"].shuffle().map(generate_and_tokenize_prompt)
            self.local_eval_dataset = None
        self.local_val_set_size = local_val_set_size

    def build_local_trainer(self,
                            tokenizer,
                            local_micro_batch_size,
                            gradient_accumulation_steps,
                            local_num_epochs,
                            local_learning_rate,
                            group_by_length,
                            ddp):
        self.train_args = transformers.TrainingArguments(
            per_device_train_batch_size=local_micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=0,
            num_train_epochs=local_num_epochs,
            learning_rate=local_learning_rate,
            fp16=False,
            logging_steps=1,
            optim="adamw_torch",
            evaluation_strategy="steps" if self.local_val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=200 if self.local_val_set_size > 0 else None,
            save_steps=200,
            output_dir=self.local_output_dir,
            save_total_limit=1,
            load_best_model_at_end=True if self.local_val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            dataloader_drop_last=False
        )
        self.local_trainer = transformers.Trainer(model=self.model,
                                                  train_dataset=self.local_train_dataset,
                                                  eval_dataset=self.local_eval_dataset,
                                                  args=self.train_args,
                                                  data_collator=transformers.DataCollatorForSeq2Seq(
                                                      tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
                                                  ),
                                                  )

    def initiate_local_training(self):
        self.model.config.use_cache = False
        self.params_dict_old = copy.deepcopy(
            OrderedDict((name, param.detach()) for name, param in self.model.named_parameters() if
                        "default" in name))
        self.params_dict_new = OrderedDict((name, param.detach()) for name, param in self.model.named_parameters() if
                                           "default" in name)
        self.model.state_dict = (
            lambda instance, *_, **__: get_peft_model_state_dict(
                instance, self.params_dict_new, "default"
            )
        ).__get__(self.model, type(self.model))

    def fit(self, config):
        self.build_local_trainer()

        self.initiate_local_training()
        self.local_trainer.train()


    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        set_peft_model_state_dict(self.model, parameters, "default")
        

    def get_parameters(self,config):
        return self.terminate_local_training()


    def terminate_local_training(self):

        new_adapter_weight = self.model.state_dict()
        # single_output_dir = os.path.join(self.output_dir, str(epoch), "local_output_{}".format(self.client_id))
        # os.makedirs(single_output_dir, exist_ok=True)
        # torch.save(new_adapter_weight, single_output_dir + "/pytorch_model.bin")

        older_adapter_weight = get_peft_model_state_dict(self.model, self.params_dict_old, "default")
        set_peft_model_state_dict(self.model, older_adapter_weight, "default")

        return new_adapter_weight



def evaluate_global(
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






def client_fn(cid: str) -> FlowerClient:
    """Create a Flower client representing a single organization."""

    # Load model
    if args.model == "pyramidnet":
        net = pyramidnet().to(DEVICE)
    elif args.model == "resnet":
        net = resnet18().to(DEVICE)
    else:
        net = pyramidnet().to(DEVICE)

    # Load data (CIFAR-10)
    client_idxs = {dataset_type: splits[dataset_type].idxs[int(cid)] if splits[dataset_type] is not None else None for dataset_type in splits}
    trainloader = DataLoader(Subset(datasets['train'], client_idxs['train']), batch_size=args.train_bs, shuffle=True) if len(client_idxs['train']) > 0 else None
    valloader = trainloader
    
    # Create a  single Flower client representing a single organization
    return FlowerClient(net, trainloader, valloader)





client_resources = {"num_gpus": 1, "num_cpus":35}
strategy = fl.server.strategy.FedAvg(
        fraction_fit=args.frac_clients,  
        fraction_evaluate=1,  
        min_fit_clients=1,  
        min_available_clients=1,  
        evaluate_fn=evaluate,
        evaluate_metrics_aggregation_fn=weighted_average,
        on_evaluate_config_fn=evaluate_config,
)

# Start simulation
fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=args.num_clients,
    config=fl.server.ServerConfig(num_rounds=args.rounds),
    strategy=strategy,
    client_resources=client_resources,
)