import os
import random
import shutil

import OpenEXR
import Imath
import numpy as np
import cv2
from matplotlib import pyplot as plt

EXT_COLOR_IMG = '-transparent-rgb-img.jpg'
EXT_DEPTH_IMG = '-transparent-depth-img.exr'
EXT_MASK = '-mask.png'


def show_image(image: np.ndarray, title: str = "Image") -> None:
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Image has 3 channels (RGB)
        plt.imshow(image)
    elif len(image.shape) == 3 and image.shape[2] == 4:
        # Image has 4 channels (RGBA)
        plt.imshow(image)
    elif len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
        # Image has 1 channel (grayscale)
        if len(image.shape) == 3:
            image = image[:, :, 0]
        plt.imshow(image, cmap='gray')
    else:
        raise ValueError("Unsupported image format. Image must have 1, 3, or 4 channels.")

    plt.title(title)
    plt.axis('off')
    plt.show()


def rename_files_in_directory(directory: str, postfix: str) -> None:
    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        # Split the file name and extension
        name, ext = os.path.splitext(filename)

        # Split the name by '-' and take the first part (assuming it's numeric)
        parts = name.split('-')
        if parts:
            base_name = parts[0]
            # Generate new file name using the extracted numeric part and postfix
            new_name = base_name + postfix
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_name)

            # Rename the file
            os.rename(old_path, new_path)
            print(f"Renamed {old_path} to {new_path}")
        else:
            print(f"No '-' found in {filename}, skipping...")


def rename_color_images(directory: str) -> None:
    rename_files_in_directory(directory, EXT_COLOR_IMG)


def rename_depth_images(directory: str) -> None:
    rename_files_in_directory(directory, EXT_DEPTH_IMG)


def rename_mask_images(directory: str) -> None:
    rename_files_in_directory(directory, EXT_MASK)


def save_image_to_exr(image_path: str, exr_output_path: str, image_type: str) -> None:
    # Read the image (assuming it's a 4-channel image for depth or 3-channel for normals)
    image: np.ndarray = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if image is None:
        raise ValueError(f"Cannot read the image from {image_path}")

    if image_type == 'depth':
        # Extract the depth information from the appropriate channel (assuming the first channel)
        channel: np.ndarray = image[:, :, 0]

        # Ensure the depth image is in float32 format
        if channel.dtype != np.float32:
            channel = channel.astype(np.float32)

        # Get the dimensions of the image
        height: int
        width: int
        height, width = channel.shape

        # Create an EXR file with the same dimensions
        header: OpenEXR.Header = OpenEXR.Header(width, height)
        header['channels'] = {'R': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))}
        exr_file: OpenEXR.OutputFile = OpenEXR.OutputFile(exr_output_path, header)

        # Convert the depth image to the EXR format
        exr_data: bytes = channel.tobytes()

        # Save the data to the EXR file
        exr_file.writePixels({'R': exr_data})
        exr_file.close()

    elif image_type == 'normal':
        # Ensure the normal image is in float32 format and normalize
        if image.dtype != np.float32:
            image = image.astype(np.float32) / 255.0  # Normalize if necessary

        # Get the dimensions of the image
        height: int
        width: int
        height, width, channels = image.shape

        # Create an EXR file with the same dimensions
        header: OpenEXR.Header = OpenEXR.Header(width, height)
        header['channels'] = {
            'R': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
            'G': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
            'B': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
        }
        exr_file: OpenEXR.OutputFile = OpenEXR.OutputFile(exr_output_path, header)

        # Convert the normal image to the EXR format
        r_channel: bytes = image[:, :, 0].tobytes()
        g_channel: bytes = image[:, :, 1].tobytes()
        b_channel: bytes = image[:, :, 2].tobytes()

        # Save the data to the EXR file
        exr_file.writePixels({'R': r_channel, 'G': g_channel, 'B': b_channel})
        exr_file.close()

    else:
        raise ValueError("Invalid image type specified. Choose 'depth' or 'normal'.")


def exr_to_image(exr_input_path: str, image_type: str) -> np.ndarray:
    # Open the EXR file
    exr_file: OpenEXR.InputFile = OpenEXR.InputFile(exr_input_path)

    # Get the image size
    header: dict = exr_file.header()
    dw: Imath.Box2i = header['dataWindow']
    width: int = dw.max.x - dw.min.x + 1
    height: int = dw.max.y - dw.min.y + 1

    if image_type == 'depth':
        # Read the depth channel (assuming it's stored in 'R' channel)
        pt: Imath.PixelType = Imath.PixelType(Imath.PixelType.FLOAT)
        depth_data: bytes = exr_file.channel('R', pt)

        # Convert the raw data to a numpy array
        depth_array: np.ndarray = np.frombuffer(depth_data, dtype=np.float32).reshape((height, width))

        # Map the depth image values to 8-bit for visualization
        depth_min: float = depth_array.min()
        depth_max: float = depth_array.max()
        depth_mapped: np.ndarray = ((depth_array - depth_min) / (depth_max - depth_min) * 255.0).astype(np.uint8)

        # Create a 4-channel image by replicating the depth information across all channels
        image: np.ndarray = cv2.merge([depth_mapped] * 3 + [np.full_like(depth_mapped, 255)])

    elif image_type == 'normal':
        # Read the RGB channels
        pt: Imath.PixelType = Imath.PixelType(Imath.PixelType.FLOAT)
        r_channel: bytes = exr_file.channel('R', pt)
        g_channel: bytes = exr_file.channel('G', pt)
        b_channel: bytes = exr_file.channel('B', pt)

        # Convert the raw data to numpy arrays
        r_array: np.ndarray = np.frombuffer(r_channel, dtype=np.float32).reshape((height, width))
        g_array: np.ndarray = np.frombuffer(g_channel, dtype=np.float32).reshape((height, width))
        b_array: np.ndarray = np.frombuffer(b_channel, dtype=np.float32).reshape((height, width))

        # Combine the channels and normalize to 8-bit for visualization
        image: np.ndarray = np.stack([r_array, g_array, b_array], axis=-1)
        image = (image * 255.0).astype(np.uint8)

    else:
        raise ValueError("Invalid image type specified. Choose 'depth' or 'normal'.")

    return image


def convert_directory_images_to_exr(png_directory: str, exr_directory: str, image_type: str) -> None:
    # Ensure the output directory exists
    if not os.path.exists(exr_directory):
        os.makedirs(exr_directory)

    # Iterate over all files in the input directory
    for idx, filename in enumerate(os.listdir(png_directory)):
        if filename.endswith('.png'):
            png_path: str = os.path.join(png_directory, filename)
            exr_filename: str = os.path.splitext(filename)[0] + '.exr'
            exr_path: str = os.path.join(exr_directory, exr_filename)

            try:
                save_image_to_exr(png_path, exr_path, image_type)
                print(f"Converted {png_path} to {exr_path}")

                if idx % 100 == 0:
                    # Convert EXR back to image for comparison
                    converted_image: np.ndarray = exr_to_image(exr_path, image_type)
                    original_image: np.ndarray = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)

                    # Display the original and converted images side by side
                    plt.figure(figsize=(10, 5))
                    plt.subplot(1, 2, 1)
                    plt.title("Original Image")
                    plt.imshow(
                        cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB) if image_type == 'normal' else cv2.cvtColor(
                            original_image, cv2.COLOR_BGRA2RGBA))
                    plt.axis('off')

                    plt.subplot(1, 2, 2)
                    plt.title("Converted Image")
                    plt.imshow(
                        cv2.cvtColor(converted_image, cv2.COLOR_BGR2RGB) if image_type == 'normal' else cv2.cvtColor(
                            converted_image, cv2.COLOR_BGRA2RGBA))
                    plt.axis('off')

                    plt.show()

            except Exception as e:
                print(f"Failed to convert {png_path}: {e}")


def pick_files_for_validation(imgs_directory: str, masks_directory: str, depth_exr_directory: str,
                              output_directory: str, num_files: int = 7) -> None:
    # Create output directory if it does not exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Collect all base names from the image directory
    base_names = set(os.path.splitext(f)[0].split('-')[0] for f in os.listdir(imgs_directory) if f.endswith('.jpg'))

    # Randomly select the specified number of base names
    selected_base_names = random.sample(list(base_names), num_files)

    # Copy the selected files to the output directory
    for base_name in selected_base_names:
        for postfix in [EXT_COLOR_IMG, EXT_DEPTH_IMG, EXT_MASK]:
            file_name = base_name + postfix
            for directory in [imgs_directory, masks_directory, depth_exr_directory]:
                file_path = os.path.join(directory, file_name)
                if os.path.exists(file_path):
                    shutil.copy(file_path, output_directory)
                    print(f"Copied {file_path} to {output_directory}")


def view_exr(_sample):
    image = exr_to_image(_sample, "normal")
    plt.imshow(image, interpolation='nearest')
    plt.show()


if __name__ == "__main__":
    # Paths to the input and output files
    generated_depth_images_directory = '/home/noot_noot/Dev/cleargrasp/pytorch_networks/caustics/synthetic_data/Normal'

    imgs_directory = '/home/noot_noot/Dev/cleargrasp/pytorch_networks/caustics/synthetic_data/Picture_Caustic'
    masks_directory = '/home/noot_noot/Dev/cleargrasp/pytorch_networks/caustics/synthetic_data/Seg_Processed'

    depth_directory = '/home/noot_noot/Dev/cleargrasp/pytorch_networks/caustics/synthetic_data/Depth'
    depth_exr_directory = '/home/noot_noot/Dev/cleargrasp/pytorch_networks/caustics/synthetic_data/Depth_exr'

    output_directory = '/home/noot_noot/Dev/cleargrasp/pytorch_networks/caustics/synthetic_data/cleargrasp_val'
    sample = '/home/noot_noot/Dev/cleargrasp/pytorch_networks/caustics/synthetic_data/cleargrasp_val/200-depth-rectified.exr'

    legacy_depth_exr = '/home/noot_noot/Dev/cleargrasp/data/sample_dataset/real-val/d435/000000080-opaque-depth-img.exr'
    # image = exr_to_image(legacy_depth_exr, image_type="normal")


    # Pick 7 files from each directory and copy them to the output directory
    pick_files_for_validation(imgs_directory, masks_directory, depth_exr_directory, output_directory, num_files=7)
