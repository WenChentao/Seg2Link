# Frequently asked questions
**Q1: Is Seg2Link compatible with Windows/macOS/Linux?**

A1: Yes. Seg2Link can be used in any of these three operating systems, though their appearance in each differs slightly.

**Q2: The software crashes when I press a hotkey more than once. Why?**

A2: Requesting a new operation before the previous one has finished can cause the program to crash. Please wait until the previous instruction is completed.

**Q3: The layout of the main window is incorrect. Why?**

A3: One possible reason is that your monitor's resolution is insufficient. We recommend that users use a monitor with at least 1920 x 1080 resolution.

The scale function in Windows could also cause problems. Try to modify or stop it to get an acceptable appearance.


**Q4: Do I need a GPU to use Seg2Link?**

A4: No, the core functions of Seg2Link are realized using numpy and numpy-based libraries, which rely solely on the CPU.

However, keep in mind that in order to train a deep neural network to predict cell/non-cell regions, you may require a GPU PC or free GPU resources from sites like Google Colab.

**Q5: When I try to save the segmentation result, the software crashes with an OSError message. Why?**

A5: The most likely cause is a lack of hard disk space to store the result. Please check the available space and increase it if necessary.

**Q6: (In Round 1) I divided one cell into two cells with hotkey K/R, but they merged back into one after performing *Next slice*. Why?**

A6: This is due to the fact that the two cells in the following slice are not well separated by a correct boundary. To solve it, correct the boundary in the new slice, then select the cell and press R. The merged cell will be separated.