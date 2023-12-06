import os
import numpy as np
import rasterio
from rasterio.merge import merge
def merge_tiff_files(input_directory, output_filename):
  """
    Merges the given input tif images. 
    Args:
        input_directory(str): Path to the tif images.
        output_filename (str): Path to save the output merged file.
    """
    # Get list of all tif files in the directory
    tiff_files = [os.path.join(input_directory, f) for f in os.listdir(input_directory) if
                  f.endswith('.tif') and 'udm' not in f]
    # Ensure there are tif files in the directory
    if not tiff_files:
        print("No tif files found in the directory!")
        return
    # Open the tiff files using rasterio
    src_files_to_mosaic = [rasterio.open(tiff_file) for tiff_file in tiff_files]
    # Merge function returns a single mosaic array and the transformation info
    mosaic, out_trans = merge(src_files_to_mosaic)
    # Convert the mosaic array to float32
    mosaic = mosaic.astype(np.float32)
    # Copy the metadata from the first TIFF file
    out_meta = src_files_to_mosaic[0].meta.copy()
    # Update the metadata to use the mosaic's data
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans,
        "crs": src_files_to_mosaic[0].crs,
        "compress": "none",
        "dtype": 'float32'
    })
    with rasterio.open(output_filename, "w", **out_meta) as dest:
        dest.write(mosaic)
    for src in src_files_to_mosaic:
        src.close()
if __name__ == "__main__":
    input_dir = "OP_Figures"
    output_file_path = os.path.join(input_dir, "REFERENCED_MERGE.tif")
    merge_tiff_files(input_dir, output_file_path)
