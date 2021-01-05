# SQLM
An SQL Language Model {SQL -> en, en -> SQL}

## Example {SQL -> en}:
+ **SQL**:  `SELECT Teams FROM table WHERE League = bundesliga AND Away = 3-2`
+ **Description prediction**:  Which teams have a League of bundesliga, and an Away of 3-2?

## Example {en -> SQL}:
+ **Text**:  
+ **SQL prediction**:  

## Requirments:
+ pytorch_lightning==0.9
+ transformers
+ rouge_score