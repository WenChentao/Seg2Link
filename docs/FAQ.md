# Frequently asked questions
**Q1: The software crashes when I press a hotkey more than once. Why?**

A1: Requesting the execution of a new operation before the previous operation has finished can cause the program to crash. Please wait until the previous instruction has been finished.

**Q2: How to prepare a cell/non-cell image**

A2: You can use either commercial software or a free program to train a deep neural network to predict cell/non-cell regions. We wrote a program for training and prediction with 2D U-Net.
Find it [here](https://github.com/WenChentao/seg2link_unet2d).

**Q3: Do I need a GPU to use Seg2Link?**

A3: No. Seg2Link's core functions are realized using numpy, which only requires a CPU.

To train a deep neural network to predict cell/non-cell regions, however, you may need a GPU PC or free GPU resources like Google Colab.

