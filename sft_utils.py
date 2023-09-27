
from datasets import load_dataset
from tqdm import tqdm
from trl.trainer import ConstantLengthDataset

# some useful function related to the Stack Exchange dataset
def chars_token_ratio(dataset, tokenizer, nb_examples=400):
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        text = prepare_sample_text(example)
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))
    return total_characters / total_tokens

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def prepare_sample_text(example):
    text = f"Question: {example['question']}\n\nAnswer: {example['response_j']}"
    return text

def create_datasets(tokenizer, streaming=True):
    dataset = load_dataset(
        "stack-exchange-paired", # local training dataset
        data_dir="data/finetune",
        split="train",
        num_proc=4 if not True else None,
        streaming=True,
    )
    if streaming:
        print("Loading the dataset in streaming mode")
        valid_data = dataset.take(4000)
        train_data = dataset.skip(4000)
        train_data = train_data.shuffle(buffer_size=5000, seed=None)
    else:
        dataset = dataset.train_test_split(test_size=0.005, seed=None)
        train_data = dataset["train"]
        valid_data = dataset["test"]
        print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")

    chars_per_token = chars_token_ratio(train_data, tokenizer)
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        formatting_func=prepare_sample_text,
        infinite=True,
        seq_length=1024,
        chars_per_token=chars_per_token,
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer,
        valid_data,
        formatting_func=prepare_sample_text,
        infinite=False,
        seq_length=1024,
        chars_per_token=chars_per_token,
    )
    return train_dataset, valid_dataset
