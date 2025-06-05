from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError # For more specific error catching
from argparse import ArgumentParser
import os

# --- Configuration for downloading multiple checkpoints ---

# List all datasets you want to download checkpoints for.
# These names should match the dataset names used in the Hugging Face Hub repository path.
ALL_DATASETS = [
    "BTMRI", "BUSI", "CHMNIST", "COVID_19", "CTKidney",
    "DermaMNIST", "KneeXray", "Kvasir", "LungColon",
    "OCTMNIST", "RETINA"
]

# Define shots configurations for the "few_shot" task
FEW_SHOT_CONFIG_SHOTS = [1, 2, 4, 8, 16]
# For "base2new" task, shots are fixed at 16 as per the original script logic

# Define trainers. Add more if you have checkpoints for other trainers.
ALL_TRAINERS = ["BiomedCoOp_BiomedCLIP"] # Example, can be ["TrainerA", "TrainerB"]

# Define seeds
ALL_SEEDS = [1, 2, 3]

# Constants from your original script for path construction
NCTX_CSC_CTP_SUFFIX = "nctx4_cscFalse_ctpend"
REPO_ID = "TahaKoleilat/BiomedCoOp"

def download_checkpoint(task, dataset, shots, trainer, seed):
    """
    Constructs the path and downloads a single checkpoint.
    """
    print(f"\nAttempting download for: Task={task}, Dataset={dataset}, Shots={shots}, Trainer={trainer}, Seed={seed}")

    if task == "few_shot":
        base_path = f"few_shot/{dataset.lower()}/shots_{shots}/{trainer}/{NCTX_CSC_CTP_SUFFIX}"
        model_name = "model.pth.tar-100"
    elif task == "base2new":
        # Original script logic: base2new task uses shots_16 and model.pth.tar-50
        if shots != 16:
            print(f"Warning: 'base2new' task expects 16 shots. Using 16 shots instead of {shots}.")
        current_shots_for_path = 16 # Path for base2new always uses shots_16
        base_path = f"base2new/train_base/{dataset}/shots_{current_shots_for_path}/{trainer}/{NCTX_CSC_CTP_SUFFIX}"
        model_name = "model.pth.tar-50"
    else:
        print(f"Unknown task: {task}. Skipping.")
        return

    file_path_on_hub = os.path.join(base_path, f"seed{seed}", "prompt_learner", model_name)
    
    # Ensure the local directory structure will be created by hf_hub_download
    # local_dir="." means it will create folders like ./few_shot/BTMRI/...
    # The target local file will be local_dir / file_path_on_hub
    expected_local_file_path = os.path.join(".", file_path_on_hub)

    if os.path.exists(expected_local_file_path):
        print(f"Checkpoint already exists locally: {expected_local_file_path}. Skipping download.")
        return

    print(f"Downloading from Hub: {REPO_ID}/{file_path_on_hub}")
    try:
        downloaded_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=file_path_on_hub,
            local_dir=".",  # Downloads to current_directory/filename_on_hub
            repo_type="model",
            resume_download=True # Resume if download was interrupted
        )
        print(f"Successfully downloaded to: {downloaded_path}")
    except HfHubHTTPError as e:
        # This exception is typically raised for 404 Not Found or other HTTP errors
        print(f"Failed to download {file_path_on_hub}. File might not exist on Hub or other HTTP error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while downloading {file_path_on_hub}: {e}")


def main():
    parser = ArgumentParser(description="Download multiple checkpoints from Hugging Face Hub.")
    parser.add_argument(
        "--target_tasks",
        nargs="+",
        choices=["few_shot", "base2new", "all"],
        default=["all"],
        help="Specify which main tasks to download checkpoints for ('few_shot', 'base2new', or 'all')."
    )
    parser.add_argument(
        "--target_datasets",
        nargs="+",
        default=["all"],
        help="Specify dataset names, or 'all' for all configured datasets."
    )
    parser.add_argument(
        "--target_trainers",
        nargs="+",
        default=["all"],
        help="Specify trainer names, or 'all' for all configured trainers."
    )

    args = parser.parse_args()

    tasks_to_run = []
    if "all" in args.target_tasks:
        tasks_to_run = ["few_shot", "base2new"]
    else:
        tasks_to_run = args.target_tasks

    datasets_to_process = []
    if "all" in args.target_datasets:
        datasets_to_process = ALL_DATASETS
    else:
        datasets_to_process = [ds for ds in args.target_datasets if ds in ALL_DATASETS]
        if not datasets_to_process and "all" not in args.target_datasets:
             print(f"Warning: None of the specified datasets ({args.target_datasets}) are in the configured ALL_DATASETS list.")
             return


    trainers_to_process = []
    if "all" in args.target_trainers:
        trainers_to_process = ALL_TRAINERS
    else:
        trainers_to_process = [tr for tr in args.target_trainers if tr in ALL_TRAINERS]
        if not trainers_to_process and "all" not in args.target_trainers:
            print(f"Warning: None of the specified trainers ({args.target_trainers}) are in the configured ALL_TRAINERS list.")
            return


    for task_type in tasks_to_run:
        for dataset_name in datasets_to_process:
            for trainer_name in trainers_to_process:
                current_shots_list = []
                if task_type == "few_shot":
                    current_shots_list = FEW_SHOT_CONFIG_SHOTS
                elif task_type == "base2new":
                    current_shots_list = [16] # Fixed for base2new as per original logic

                for shots_val in current_shots_list:
                    for seed_val in ALL_SEEDS:
                        download_checkpoint(
                            task=task_type,
                            dataset=dataset_name,
                            shots=shots_val,
                            trainer=trainer_name,
                            seed=seed_val
                        )
    print("\nAll specified checkpoint download attempts finished.")

if __name__ == "__main__":
    main()