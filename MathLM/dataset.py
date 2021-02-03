import pandas as pd
from nlp import load_dataset
from transformers import T5Tokenizer
from torch.utils.data import Dataset

class math_problems(Dataset):
    def __init__(self, type_path: str, 
                       input_length: int, 
                       output_length: int,
                       num_samples: int = None,
                       tokenizer = T5Tokenizer.from_pretrained('t5-small')) -> None:      

        self.dataset =  pd.read_csv(f"{type_path}.csv")
        self.dataset.answer = self.dataset.answer.astype(str)
        if num_samples:
            self.dataset = self.dataset.iloc[:num_samples]
        self.input_length = input_length
        self.tokenizer = tokenizer
        self.output_length = output_length
  
    def __len__(self) -> int:
        return self.dataset.shape[0]
    
    def convert_to_features(self, example_batch):   
        input_ = example_batch['question']
        target_ = example_batch['answer']
        
        source = self.tokenizer.batch_encode_plus([input_], max_length=self.input_length, 
                                                     padding='max_length', truncation=True, return_tensors="pt")
        
        targets = self.tokenizer.batch_encode_plus([target_], max_length=self.output_length, 
                                                     padding='max_length', truncation=True, return_tensors="pt")   
        return source, targets
  
    def __getitem__(self, index: int) -> dict:
        source, targets = self.convert_to_features(self.dataset.iloc[index])
        
        source_ids = source["input_ids"].squeeze()
        target_ids = targets["input_ids"].squeeze()

        src_mask    = source["attention_mask"].squeeze()
        target_mask = targets["attention_mask"].squeeze()

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}


def get_dataset(tokenizer, type_path: str, num_samples: int, args):
      return math_problems(type_path=type_path, 
                    num_samples=num_samples,  
                    input_length=args.max_input_length, 
                    output_length=args.max_output_length)