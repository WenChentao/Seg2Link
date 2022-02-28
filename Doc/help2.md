# Help information for the second round

**Check a cell**

1. Select a label by pressing **Select label**
2. Press Q to show the selected cell. Press Q again to show all cells.
3. If the cell is now seen, press **Locate** to jump to the slice with the largest area of the cell. The coordinates [row, column] will also been shown.

**Sort the cells by size and removing the tiny cells**

- By default, the exported segmentation file (.npy) from the first round has been sorted by cell size from largest to smallest. 
- After manual corrections, the users may want to re-sort the cells. This can be done by following procedures:
  - Modify the **Max cell No.**, so that only this number of largest cells will be kept. 
  - Press **Estimate**, then the soft will calculate how many cells will be removed. If the **Max cell No.** value is larger than the cell number, cells will still be re-sorted but without removing small cells.
  - Press **Remove tiny cells, Sorting and Save (.npy)** button. The sorting and removing will be performed, and the segmentation before/and the operation will be saved as 'seg-modified_before_removing_tiny_cells.npy' and 'seg-modified_after_removing_tiny_cells.npy', respectively

**Divide one cell:**

1. Select a cell by clicking it in Pick Mode
2. Modify the cell region by eraser or other tools
3. Select a mode from the three modes
   - **_2D**: The selected cell region in current slice will be divided into multiple new cells
   - **_2D_Link**: The selected cell region in current slice will be divided and then linked with the previous slice
   - **_3D**: The selected cell region in all slices will be divided based on the 3D connectivity
4. In case **_2D** / **_2D_Link** mode was selected, modify **min_area (2D)** (percent value), so that the small sub-regions will be inhibited during the division process. In **_3D** mode, this option will be ignored.
5. Press K to divide the selected cell.
6. Press Q to show one of the divided cells. Press Q again to go back showing all cells.
7. Press **Check it** to select another divided cell. The viewer will jump to the slice with the largest area of this cell. The coordinates will also be shown after the **Locate** button

**Merge:**
1. Select a cell in Pick Mode
2. Add the selected cell into the merge list by pressing A
3. Repeat 1 and 2 to add all the cells to be merged
4. In case incorrect cells were added, press C to clean the merge list
5. Press M to merge the cells in the merge list

**Delete:**

1. Select a cell in Pick Mode
2. Press D to remove the selected cell

**Undo:**
- Press U to undo one action in the **Cached action** list

**Redo:**

- Press F to redo one action in the **Cached action** list'

**Save/load the segmentation**

- Press '**Save (.npy)** button. The segmentation will be saved as 'seg-modified.npy'.
- To save the segmentation in a different folder, or with a different name, press **Save As (.npy)** button.
- To load a saved segmentation, Press **Load segmentation (.npy)** button and choose a saved file.

**Export the segmentation as tiff files**
- Press **Export segmentation as .tiff files**, and choose a folder to save the segmentation as tiff files.
- If the users need to add cell boundary lines between the neiboughing cells, choose **Boundary - Add** option and then press the **Export** button.
- If the users need to remove cell boundary lines between the neiboughing cells, choose **Boundary - Remove** option and then press the **Export** button. By default, the cells will dilate one pixel in x-y plane to remove the cell boundaries.





