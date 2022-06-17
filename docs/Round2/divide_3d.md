### Division (3D)

Divide a cell based on 3D connectivity. It should be used in following two cases:

#### Divide incorrect merged cells that are spatially separated
1. The cell 2 should be divided into two cells due to the careless incorrect merge operation by the user.

    ![slice1](./pictures/division_3d_1_annotation.png)

2. Choose **3D mode**. Click the cell 2 in **Picker Mode**. Then press **K** to divide it. 
    ![slice1](./pictures/division_3d_2_annotation.png)

    - *The cell was correctly divided into two cells.*

#### Divide two cells connected along z-axis
1. Here are three slices of cell 36. Suppose we need to divide it into two cells, with slice 10 as the boundary.

    Slice 9
    ![slice1](./pictures/division_3d_s9_annotation.png)
    Slice 10
    ![slice1](./pictures/division_3d_s10_annotation.png)
    Slice 11
    ![slice1](./pictures/division_3d_s11_annotation.png)

2. Move to slice 10. Use bucket tool to fill the cell region with label 0 (i.e. as boundary)
    
    ![slice1](./pictures/division_3d_s10_clean_annotation.png)

4. Go back to slice 9. Click cell 36. Press K in 3D mode.

    Slice 9
    ![slice1](./pictures/division_3d_s9_2.png)
    Slice 10
    ![slice1](./pictures/division_3d_s10_clean.png)
    Slice 11
    ![slice1](./pictures/division_3d_s11_2.png)

    *The cell 36 has been divided into two cells along z-axis!*