## Workflow of round 2

1. Specify the images and the parameters ([Detailed instructions](./parameter_setting.md))
     
    ![set_para](./pictures/round2_set_para.png)

2. Launch the main window of Seg2D+Link ([Panel descriptions](./panel_description.md))
   
    ![open_round2](./pictures/round2.png)

3. Perform inspections and corrections with following operations:

      1. [*Localize*](./localize.md) to a cell.
      2. Correct the segmentation in slice **i** with following operations:
          - [*Merge*](./merge.md) / [*Delete*](./delete.md) / [*Division*](./divide.md#division) / [*Division-Relink*](./divide.md#divisionrelink) 
         / [*3D Division*](./divide_3d.md) / [*Insert*](./insert.md)
      3. [*Sort*](./sort_remove.md#sort-cells) cells by sizes and [*Remove tiny cells*](./sort_remove.md#remove-tiny-cells).
      4. [*Save and load*](./save_load_export.md#saveload) intermediate segmentation results.
      5. [*Export*](./save_load_export.md#export) the segmentation as 2D Tiff images.