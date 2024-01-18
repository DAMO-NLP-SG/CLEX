from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk, Dataset
from datasets.builder import DatasetGenerationError
from torch.utils.data import IterableDataset
import random
import torch
from tqdm import tqdm

class ConstantLengthDataset(IterableDataset):
    """
    Iterable dataset that returns constant length chunks of tokens from stream of text files.
        Args:
            tokenizer (Tokenizer): The processor used for proccessing the data.
            dataset (dataset.Dataset): Dataset with text files.
            infinite (bool): If True the iterator is reset after dataset reaches end else stops.
            seq_length (int): Length of token sequences to return.
            num_of_sequences (int): Number of token sequences to keep in buffer.
            chars_per_token (int): Number of characters per token used to estimate number of tokens in text buffer.
            shuffle (bool): If true, the samples in each buffer are suffled. Default is `True`.
            add_eos_token (bool): If true, each buffer is delimited with eos token. Default is `True`.
    """

    def __init__(
        self,
        tokenizer,
        dataset,
        infinite=False,
        seq_length=1024,
        num_of_sequences=64,
        chars_per_token=3.4,
        content_field="content",
        shuffle=True,
        add_eos_token=True,
    ):
        self.tokenizer = tokenizer
        self.concat_token_id = tokenizer.eos_token_id
        self.dataset = dataset
        self.seq_length = seq_length
        self.infinite = infinite
        self.current_size = 0
        self.max_buffer_size = seq_length * chars_per_token * num_of_sequences
        self.content_field = content_field
        self.shuffle = shuffle
        self.add_eos_token = add_eos_token
        # print(f"Max Buffer: {self.max_buffer_size}")

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.max_buffer_size:
                    break
                try:
                    buffer.append(next(iterator)[self.content_field])
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                    else:
                        more_examples = False
                        break
            tokenized_inputs = self.tokenizer(buffer, truncation=False)["input_ids"]
            all_token_ids = []
            for tokenized_input in tokenized_inputs:
                if self.add_eos_token:
                    tokenized_input = tokenized_input + [self.concat_token_id]
                all_token_ids.extend(tokenized_input)
            examples = []
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i : i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    examples.append(input_ids)
            if self.shuffle:
                random.shuffle(examples)
            for example in examples:
                self.current_size += 1
                yield {
                    "input_ids": torch.LongTensor(example),
                    "labels": torch.LongTensor(example),
                }


def chars_token_ratio(dataset, tokenizer, data_column, nb_examples=40):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        total_characters += len(example[data_column])
        total_tokens += len(tokenizer(example[data_column]).tokens())

    return total_characters / total_tokens


def create_datasets(tokenizer, training_args, data_path):
    raw_datasets = load_dataset("json", data_files=data_path)
    train_data, valid_data = raw_datasets["train"], raw_datasets["test"]
    train_dataset, valid_dataset = None, None
    if training_args.do_train:
        print(f"Size of the train set: {len(train_data)}")
        column_names = train_data.column_names
        dataset_text_field = "text" if "text" in column_names else column_names[0]
        # chars_per_token = chars_token_ratio(train_data, tokenizer, dataset_text_field)
        # print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")
        train_dataset = ConstantLengthDataset(
            tokenizer,
            train_data,
            infinite=True,
            seq_length=training_args.model_max_length,
            chars_per_token=3.4,
            num_of_sequences=256,
            content_field=dataset_text_field,
            shuffle=True,
            add_eos_token=False,
        )
    if training_args.do_eval or training_args.do_predict:
        # valid_data = test_dataset
        column_names = valid_data.column_names
        dataset_text_field = "text" if "text" in column_names else column_names[0]
        # chars_per_token = chars_token_ratio(valid_data, tokenizer, dataset_text_field)
        print(f"Size of the validation set: {len(valid_data)}")
        valid_dataset = ConstantLengthDataset(
            tokenizer,
            valid_data,
            infinite=False,
            seq_length=training_args.model_max_length,
            chars_per_token=3.4,
            num_of_sequences=2,
            content_field=dataset_text_field,
            shuffle=False,
            add_eos_token=False,
        )

    return train_dataset, valid_dataset