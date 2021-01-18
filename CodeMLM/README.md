# CodeMLM
A Roberta Language Model {MLM - Masked Language Model}
Based on CodeSearchNet dataset

## Example {Python}:
+ **Code**:  
```python
new_dict = {}
for k, v in my_dict.<mask>():
    new_dict[k] = v**2
```
+ **Prediction**:  
```
[('items', 0.7376779913902283),
 ('keys', 0.16238391399383545),
 ('values', 0.03965481370687485),
 ('iteritems', 0.03346433863043785),
 ('splitlines', 0.0032723243348300457)]
```
![roberta python loss](roberta-python-loss.pdf "roberta python loss")

## Example {Java}:
+ **Code**:
```java
```
+ **Prediction**:  

## Example {go}:
+ **Code**:
```go
```
+ **Prediction**:  

## Example {php}:
+ **Code**:
```php
```
+ **Prediction**:  

## Example {javascript}:
+ **Code**:
```javascript
```
+ **Prediction**: 

## Example {ruby}:
+ **Code**:
```ruby
```
+ **Prediction**: 

## Requirments:
+ transformers

## HuggingFace models URLs:
+ https://huggingface.co/dbernsohn/roberta-python
+ ..