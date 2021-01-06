from nlp import load_dataset
from torch.utils.data import Dataset

class wikisql(Dataset):
    def __init__(self, tokenizer, 
                       type_path: str, 
                       num_samples: int,
                       input_length: int, 
                       output_length: int, 
                       sql2txt: bool = True) -> None:      

        self.dataset =  load_dataset('wikisql', 'all', data_dir='data/', split=type_path)
        if num_samples:
            self.dataset = self.dataset.select(list(range(0, num_samples)))
        self.input_length = input_length
        self.tokenizer = tokenizer
        self.output_length = output_length
        self.sql2txt = sql2txt
  
    def __len__(self) -> int:
        return self.dataset.shape[0]
    
    def clean_text(self, text: str) -> str:
        return text.replace('\n','').replace('``', '').replace('"', '')

    
    def convert_to_features(self, example_batch):                
        # text to sql
        if self.sql2txt:
            input_ = self.clean_text(example_batch['question'])
            target_ = self.clean_text(example_batch['sql']['human_readable'])
        else: # sql to text 
            input_ = self.clean_text(example_batch['sql']['human_readable'])
            target_ = self.clean_text(example_batch['question'])
        
        source = self.tokenizer.batch_encode_plus([input_], max_length=self.input_length, 
                                                     padding='max_length', truncation=True, return_tensors="pt")
        
        targets = self.tokenizer.batch_encode_plus([target_], max_length=self.output_length, 
                                                     padding='max_length', truncation=True, return_tensors="pt")
    
       
        return source, targets
  
    def __getitem__(self, index: int) -> dict:
        source, targets = self.convert_to_features(self.dataset[index])
        
        source_ids = source["input_ids"].squeeze()
        target_ids = targets["input_ids"].squeeze()

        src_mask    = source["attention_mask"].squeeze()
        target_mask = targets["attention_mask"].squeeze()

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}


def get_dataset(tokenizer, type_path: str, num_samples: int, args) -> wikisql:
      return wikisql(tokenizer=tokenizer, type_path=type_path, num_samples=num_samples,  input_length=args.max_input_length, 
                        output_length=args.max_output_length)