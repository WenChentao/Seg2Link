### Division+Relink

When a label has been corrected segmented in slice i but not in slice i+1, you can use [**Division+Relink**]() to automatically link cells after Division, instead of apply [**Division**](./divide.md) and manually linking the results afterwards.

1. Before operation:

    Slice 1:
    ![select](./pictures/division_relink_preslice.png)

    Slice 2:
    ![select](./pictures/division_1_annotation.png)

2. Correct the cell boundary in slice i+1 (See procedures 2 and 3 in [**Division**](./divide.md)). 

    ![edit](./pictures/division_3_annotation.png)

3. Switch to the **Picker Mode** by pressing **L**. Click the cell to be divided and then press **R**.

    ![divide](./pictures/division_relink_annotation.png)

    ***Now the cell in slice 2 has been divided and correctly linked to the cells in slice 1!***
