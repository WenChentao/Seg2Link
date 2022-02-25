# Questions and answers
**Q1: Why dose the soft crashes if I press a hot-key more than once?**

A1: Requesting the execution of a new instruction 
(e.g. Merge, delete, division, etc.) before the last instruction 
was completed can cause the program to crash. Please wait until
the last instruction was completed.

**Q2: How to prepare a cell/non-cell image**

A2: You can train U-Net or other deep neural networks to predict 
cell/non-cell regions. We have written a program 2D U-Net programs
for training and prediction. Find it [here]().

**Q3: Do I need a computer with GPU?**

A3: To use Seg2Link, the answer is No. The core functions of 
Seg2Link were realized using Numpy, which only need a CPU. 

On the other hand, you will need a GPU PC to train a deep neural 
network to prediction cell/non-cell regions, or you may try 
free cloud resources such as Google Colab instead of buying GPU 
by yourself.


