## Panels description

### Panels
![para_panels](./pictures/round2_annotation.png)

### Panels 1, 2, 3, 5, 7, and 9
See [descriptions of in round 1](../Round1/panel_description.md).

### 4. Current layer [cursor position] selected label
Together with the [Localize](./localize.md) function, the **cursor position** can assist users in locating a cell.

### 6. States
Four types of information/functions are supplied here:

1. The largest cell number in current segmentation result.
2. The cached states that can be retrieved with undo/redo.
3. Select a cell ID and locate its position. See [Localize](./localize.md)
4. The contents of the label list, used in [Merge](./merge.md) and [Delete](./delete.md)

### 8. Divide a single cell
- **Mode**
    - 2D: [Divide](./divide.md#division) / [Divide-Relink](./divide.md#divisionrelink) a cell in a specific slice.
    - 3D: [Divide a cell in 3D space](./divide_3d.md).
- **Max division**
    - Only used in **2D mode**.
    - The max number of cells allowed in the division result. 
    - If more cells are obtained, smaller cells will be merged with nearby cells. 
    - **Inf** means no limitation of the cell number.
- **Divide cell**
    - Show information of division result.
- **Check it**
    - Select a cell in the division result and jump to the slice of its center.
    
### 10. Save/Export 
- **Save segmentation**
    - Save: Save the current result as seg-modified.npy. 
    - Save as: Save the current result with a custom filename.
- **Load segmentation**
    - Load the segmentation result saved as npy format.
- [**Sort labels and remove tiny cells**](./sort_remove.md)
- [**Export segmentation as .tiff files**](./save_load_export.md)


