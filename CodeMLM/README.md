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
    String[] cars = {"Volvo", "BMW", "Ford", "Mazda"};
    for (String i : cars) {
    System.out.<mask>(i);
    }
    ```
+ **Prediction**: 
    ```
    [('println', 0.32571351528167725),
    ('get', 0.2897663116455078),
    ('remove', 0.0637081190943718),
    ('exit', 0.058875661343336105),
    ('print', 0.034190207719802856)]
    ```
![roberta java loss](roberta-java-loss.pdf "roberta java loss")

## Example {go}:
+ **Code**:
    ```go
    package main

    import (
        "fmt"
        "runtime"
    )

    func main() {
        fmt.Print("Go runs on ")
        switch os := runtime.<mask>; os {
        case "darwin":
            fmt.Println("OS X.")
        case "linux":
            fmt.Println("Linux.")
        default:
            // freebsd, openbsd,
            // plan9, windows...
            fmt.Printf("%s.\n", os)
        }
    }
    ```
![roberta go loss](roberta-go-loss.pdf "roberta go loss")

+ **Prediction**:  
    ```
    [('GOOS', 0.11810332536697388),
    ('FileInfo', 0.04276798665523529),
    ('Stdout', 0.03572738170623779),
    ('Getenv', 0.025064032524824142),
    ('FileMode', 0.01462600938975811)]
    ```
## Example {php}:
+ **Code**:
    ```php
    $people = array(
        array('name' => 'Kalle', 'salt' => 856412),
        array('name' => 'Pierre', 'salt' => 215863)
    );

    for($i = 0; $i < count($<mask>); ++$i) {
        $people[$i]['salt'] = mt_rand(000000, 999999);
    }
    ```
![roberta php loss](roberta-php-loss.pdf "roberta php loss")

+ **Prediction**:
    ```
    [('people', 0.785636842250824),
    ('parts', 0.006270722020417452),
    ('id', 0.0035842324141412973),
    ('data', 0.0025512021966278553),
    ('config', 0.002258970635011792)]
    ```
## Example {javascript}:
+ **Code**:
    ```javascript
    var i;
    for (i = 0; i < cars.<mask>; i++) {
    text += cars[i] + "<br>";
    }
    ```
+ **Prediction**:
    ```
    [('length', 0.9959614872932434),
    ('i', 0.00027875584783032537),
    ('len', 0.0002283261710545048),
    ('nodeType', 0.00013731322542298585),
    ('index', 7.5289819505997e-05)]
    ```
![roberta javascript loss](roberta-javascript-loss.pdf "roberta javascript loss")

## Requirments:
+ transformers

## HuggingFace models URLs:
+ https://huggingface.co/dbernsohn/roberta-python
+ https://huggingface.co/dbernsohn/roberta-java
+ https://huggingface.co/dbernsohn/roberta-go
+ https://huggingface.co/dbernsohn/roberta-php
+ https://huggingface.co/dbernsohn/roberta-javascript