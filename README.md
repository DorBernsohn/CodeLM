# SQLM
An SQL Language Model {SQL -> en, en -> SQL}

## Example {SQL -> en}:
+ **SQL**:  `translate English to SQL: SELECT people FROM peoples where age > 10`
+ **Description prediction**:  What people are older than 10?

+ **SQL**:  `translate English to SQL: SELECT COUNT Params from model where location=HF-Hub`
+ **Description prediction**:  How many params from model location is hf-hub?

## Example {en -> SQL}:
+ **Text**:  
+ **SQL prediction**:  

## Requirments:
+ pytorch_lightning==0.9
+ transformers
+ rouge_score