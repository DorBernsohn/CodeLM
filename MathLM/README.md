# MathLM
A T5 Language Model {math question -> answer}
Based on [mathematics_dataset](https://github.com/deepmind/mathematics_dataset)

To run the experimnt go to the notebook MathLM\run_Text_to_Text_on_math_data.ipynb

## Example:
+ `Suppose -d = 5 - 16. Let b = -579 + 584. Solve -b*c + 36 = d for c.`
+ **Answer**: 5 
+ **Pred**: 5
----
+ `Suppose 3v - l + 9 = 4v, 0 = -5v + 5l - 5. Let f(s) = 3s**2 + 1. Let g be f(-1). Suppose 63 = gx - x. Solve -5*i + v + x = 0 for i.`
+ **Answer**: 5 
+ **Pred**: 5
----
+ `Let w be 2 - (0 - 0)/(-2). Let f = -110 - -110. Suppose fm - 4m + 3m = 0. Solve mv = -w*v for v.`
+ **Answer**: 0 
+ **Pred**: 0
----
+ `Let a(h) = -34h**3 - 15 + 3h + 36h**3 + 8h*2 + 5h*2. Let r be a(-6). Solve 2z = r*z for z.`
+ **Answer**: 0 
+ **Pred**: 0
----
+ `Suppose -3p + 24 = -3c, 0c + 6 = -2c. Suppose -67 = 4i + 289. Let t = i + 94. Solve t = 2y - p for y.`
+ **Answer**: 5 
+ **Pred**: 5
----
+ `Let b = -36 + 53. Suppose -7u - b = -73. Solve j + 3j = -u for j.`
+ **Answer**: -2 
+ **Pred**: -2
----
+ `Let h be 8*((-2)/2 + 14)1. Let y = -101 + h. Solve yp = -p for p.`
+ **Answer**: 0 
+ **Pred**: 0
----
+ `Let b = 178 - 79. Let s be 9/(-1 - 2 - b/(-22)). Solve s = -k - k for k.`
+ **Answer**: -3 
+ **Pred**: -3
----
+ `Suppose 31 = -4z + 11, -3k - 5z - 22 = 0. Solve 23 = -11p + k for p.`
+ **Answer**: -2 
+ **Pred**: -2
----
+ `Calculate the greatest common factor of 3470 and 97090.`
+ **Answer**: 10 
+ **Pred**: 10
----
+ `Calculate the highest common factor of 3480 and 775431.`
+ **Answer**: 87 
+ **Pred**: 87
----
+ `How many minutes are there between 2:09 PM and 2:27 PM?`
+ **Answer**:  18 
+ **Pred**:  18
----
+ `What is 116 minutes after 10:06 AM?`
+ **Answer**:  12:02 PM 
+ **Pred**:  12:02 PM


## Requirments:
+ transformers

## HuggingFace models URLs:
+ https://huggingface.co/dbernsohn/algebra_linear_1d
+ https://huggingface.co/dbernsohn/algebra_linear_1d_composed
+ https://huggingface.co/dbernsohn/t5_numbers_gcd
+ https://huggingface.co/dbernsohn/t5_measurement_time