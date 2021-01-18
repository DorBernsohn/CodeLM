# SQLM
An SQL Language Model {SQL -> en, en -> SQL}
Based on wikisql dataset

## Example {SQL -> en}:
+ **SQL**:  translate SQL to English: `SELECT people FROM peoples where age > 10`
+ **Description prediction**:  What people are older than 10?

+ **SQL**:  translate SQL to English: `SELECT COUNT Params from model where location=HF-Hub`
+ **Description prediction**:  How many params from model location is hf-hub?

## Example {en -> SQL}:
+ **Text**: what are the names of all the people in the USA? 
+ **SQL prediction**: `SELECT Name FROM table WHERE Country = USA`

## Requirments:
+ pytorch_lightning==0.9
+ transformers
+ rouge_score

## HuggingFace models URLs:
+ https://huggingface.co/dbernsohn/t5_wikisql_SQL2en
+ https://huggingface.co/dbernsohn/t5_wikisql_en2SQL