import os
import pyheif
from PIL import Image

def convert_heic_to_jpg(folder_path, target_size=(720, 1280)):
    # Ensure the folder exists
    if not os.path.isdir(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return
    
    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".heic"):
            heic_path = os.path.join(folder_path, filename)
            jpg_path = os.path.splitext(heic_path)[0] + ".jpg"
            
            # Convert HEIC to JPG
            try:
                heif_file = pyheif.read(heic_path)
                image = Image.frombytes(
                    heif_file.mode, 
                    heif_file.size, 
                    heif_file.data,
                    "raw",
                    heif_file.mode,
                    heif_file.stride,
                )
                
                # Resize image if target_size is specified
                if target_size is not None:
                    image.thumbnail(target_size, Image.ANTIALIAS)  # Maintain aspect ratio
                    print(f"Resized {filename} to max {target_size}")

                # Save as JPG
                image.save(jpg_path, "JPEG")
                print(f"Converted: {filename} -> {os.path.basename(jpg_path)}")
                
                # Remove the original HEIC file
                os.remove(heic_path)
                print(f"Deleted: {filename}")

            except Exception as e:
                print(f"Failed to convert {filename}: {e}")


# Example usage:
folder_path = "database/body_img/"  # Replace with your folder path
convert_heic_to_jpg(folder_path)