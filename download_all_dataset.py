import os
import requests
import zipfile
import argparse
from tqdm import tqdm # For progress bar

# Dataset names and their Hugging Face download URLs
# (Extracted from your DATASETS.md)
DATASET_URLS = {
    "BTMRI": "https://huggingface.co/datasets/TahaKoleilat/BiomedCoOp/resolve/main/BTMRI.zip",
    "BUSI": "https://huggingface.co/datasets/TahaKoleilat/BiomedCoOp/resolve/main/BUSI.zip",
    "CHMNIST": "https://huggingface.co/datasets/TahaKoleilat/BiomedCoOp/resolve/main/CHMNIST.zip",
    "COVID_19": "https://huggingface.co/datasets/TahaKoleilat/BiomedCoOp/resolve/main/COVID_19.zip",
    "CTKidney": "https://huggingface.co/datasets/TahaKoleilat/BiomedCoOp/resolve/main/CTKidney.zip",
    "DermaMNIST": "https://huggingface.co/datasets/TahaKoleilat/BiomedCoOp/resolve/main/DermaMNIST.zip",
    "KneeXray": "https://huggingface.co/datasets/TahaKoleilat/BiomedCoOp/resolve/main/KneeXray.zip",
    "Kvasir": "https://huggingface.co/datasets/TahaKoleilat/BiomedCoOp/resolve/main/Kvasir.zip",
    "LungColon": "https://huggingface.co/datasets/TahaKoleilat/BiomedCoOp/resolve/main/LungColon.zip", # Assuming LC25000 is LungColon
    "OCTMNIST": "https://huggingface.co/datasets/TahaKoleilat/BiomedCoOp/resolve/main/OCTMNIST.zip",
    "RETINA": "https://huggingface.co/datasets/TahaKoleilat/BiomedCoOp/resolve/main/RETINA.zip",
}

def download_file_with_progress(url, destination_path):
    """Downloads a file from a URL to a destination path with a progress bar."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes
        total_size = int(response.headers.get('content-length', 0))
        
        with open(destination_path, 'wb') as f, tqdm(
            desc=os.path.basename(destination_path),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                size = f.write(chunk)
                bar.update(size)
        print(f"Successfully downloaded {os.path.basename(destination_path)}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return False
    except IOError as e:
        print(f"Error writing file {destination_path}: {e}")
        return False

def unzip_file(zip_path, extract_to_dir):
    """Unzips a file to a specified directory."""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            print(f"Unzipping {os.path.basename(zip_path)} to {extract_to_dir}...")
            zip_ref.extractall(extract_to_dir)
        print(f"Successfully unzipped {os.path.basename(zip_path)}.")
        return True
    except zipfile.BadZipFile:
        print(f"Error: {zip_path} is not a valid zip file or is corrupted.")
        return False
    except Exception as e:
        print(f"Error unzipping {zip_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download and setup biomedical datasets.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Directory to download and extract datasets into (default: ./data)."
    )
    parser.add_argument(
        "--datasets",
        nargs="+", # Allows specifying multiple datasets
        default=list(DATASET_URLS.keys()), # Default to all datasets
        choices=list(DATASET_URLS.keys()) + ["all"],
        help=(
            "Specify which datasets to download. Use 'all' for all datasets. "
            "Example: --datasets BTMRI OCTMNIST. Default: all."
        )
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing dataset directories if they exist."
    )

    args = parser.parse_args()

    # Ensure base data directory exists
    os.makedirs(args.data_dir, exist_ok=True)

    datasets_to_download = []
    if "all" in args.datasets:
        datasets_to_download = list(DATASET_URLS.keys())
    else:
        datasets_to_download = args.datasets

    for dataset_name in datasets_to_download:
        if dataset_name not in DATASET_URLS:
            print(f"Warning: Dataset '{dataset_name}' not found in predefined URLs. Skipping.")
            continue

        print(f"\nProcessing dataset: {dataset_name}")
        dataset_url = DATASET_URLS[dataset_name]
        
        # Create dataset-specific directory (e.g., ./data/BTMRI/)
        # This will be the directory where the ZIP contents are extracted.
        # The MD file implies that the zip might contain a folder with the same name,
        # e.g., BTMRI.zip unzips to a BTMRI/ folder.
        # The target structure is data/BTMRI/ (containing BTMRI/ and split_BTMRI.json)
        dataset_base_path = os.path.join(args.data_dir, dataset_name)

        if os.path.exists(dataset_base_path) and not args.overwrite:
            print(f"Directory {dataset_base_path} already exists. Skipping (use --overwrite to force).")
            continue
        elif os.path.exists(dataset_base_path) and args.overwrite:
            print(f"Directory {dataset_base_path} exists. Overwriting.")
            # Potentially add shutil.rmtree(dataset_base_path) here if you want a clean overwrite
            # For now, it will just overwrite files during unzip.
        
        os.makedirs(dataset_base_path, exist_ok=True)

        # Path for the downloaded zip file
        zip_file_name = f"{dataset_name}.zip"
        zip_download_path = os.path.join(dataset_base_path, zip_file_name) # Download inside dataset folder

        # 1. Download the zip file
        print(f"Downloading {dataset_name}.zip from {dataset_url}...")
        if not download_file_with_progress(dataset_url, zip_download_path):
            print(f"Skipping {dataset_name} due to download error.")
            continue

        # 2. Unzip the file
        # The extraction path is dataset_base_path.
        # If BTMRI.zip contains a "BTMRI" folder, it will be extracted to ./data/BTMRI/BTMRI/
        # This matches the example structure: data/BTMRI/ (parent) -> BTMRI/ (unzipped folder)
        if not unzip_file(zip_download_path, dataset_base_path):
            print(f"Skipping further processing for {dataset_name} due to unzip error.")
            continue
            
        # 3. Optional: Clean up the downloaded zip file
        try:
            os.remove(zip_download_path)
            print(f"Removed temporary zip file: {zip_download_path}")
        except OSError as e:
            print(f"Warning: Could not remove zip file {zip_download_path}: {e}")

        print(f"Successfully processed dataset: {dataset_name}")

    print("\nAll selected datasets processed.")

if __name__ == "__main__":
    main()