### Division (2D)

#### Division
1. The cell 76 in slice 1 should be divided into 3 cells.

    ![slice1](./pictures/division_2d_1_annotation.png)

2. Choose **2D mode** and max division = **Inf**. 

    Click the cell 76 in **Picker Mode**. Then press **K** to divide it. 
    ![slice1](./pictures/division_2d_2_annotation.png)

    - *The cell was over-segmented with 2D watershed algorithm.*

    - *If necessary, correct the cell boundary before pressing K*

4. Correct the over-segmentation by [Merge](./merge.md). 

    ![slice1](./pictures/division_2d_3.png)


#### Division+Relink

1. The cell 76 in slice 2 should be divided similarly as in slice 1.

    ![slice2](./pictures/division_2d_s2_1.png)

2. Click the cell 76 in **Picker Mode**. Then press **R** to divide it and relink the results to slice 1. 

    ![slice2](./pictures/division_2d_s2_2.png)

    *The over-segmentation was solved automatically by relinking.*

4. The cell 76 in slice 3.

    ![slice2](./pictures/division_2d_s3_1.png)

    Similarly, press **R** solves everything. 

    ![slice2](./pictures/division_2d_s3_2.png)
