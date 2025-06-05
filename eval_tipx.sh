#!/bin/bash

# --- Configuration ---
# Ensure these dataset names EXACTLY match the keys in your datasets/__init__.py
DATASETS=("BTMRI" "OCTMNIST" "CHMNIST" "DermaMNIST" "BUSI") # Customize this list
SHOTS_CONFIG=(1 2 4 8 16)
SEEDS=(1 2 3)

# Fixed parameters for the prompt learner (BiomedCoOp)
PROMPT_MODEL_TYPE="BiomedCLIP"
PROMPT_NCTX=4
PROMPT_CSC="False" # Bash treats False as a string
PROMPT_CTP="end"
PROMPT_LOADEP=100
PROMPT_METHOD_BASE="BiomedCoOp"
PROMPT_TRAINER_NAME="${PROMPT_METHOD_BASE}_${PROMPT_MODEL_TYPE}"

# Parameters for Tip-X run
TIPX_SEARCH_BETA_MIN=1.0
TIPX_SEARCH_BETA_MAX=10.0
TIPX_SEARCH_BETA_STEPS=10

TIPX_SEARCH_ALPHA_MIN=1.0
TIPX_SEARCH_ALPHA_MAX=10.0
TIPX_SEARCH_ALPHA_STEPS=10

TIPX_SEARCH_GAMMA_MIN=0.0
TIPX_SEARCH_GAMMA_MAX=1.0
TIPX_SEARCH_GAMMA_STEPS=10

TIPX_LR=0.001
TIPX_TRAIN_EPOCH=10 # Dummy, as we are not fine-tuning Tip-X here
TIPX_FINETUNE_ARG="" # Set to "--finetune" if needed, otherwise empty

# --- Command-line Arguments ---
DATA_ROOT="$1"
OUTPUT_ROOT="${2:-output_eval_tipx}" # Default to output_eval_tipx if not provided
PYTHON_EXECUTABLE="${3:-python}"    # Default to python if not provided

if [ -z "$DATA_ROOT" ]; then
    echo "Error: Data root directory not provided."
    echo "Usage: $0 /path/to/data_root [optional_output_root_path] [optional_python_executable]"
    exit 1
fi

if [ ! -d "$DATA_ROOT" ]; then
    echo "Error: Data root directory '$DATA_ROOT' does not exist."
    exit 1
fi

mkdir -p "$OUTPUT_ROOT"

# --- Main Loop ---
for DATASET_NAME in "${DATASETS[@]}"; do
    for SHOTS in "${SHOTS_CONFIG[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            echo ""
            echo "--- Processing: Dataset=${DATASET_NAME}, Shots=${SHOTS}, Seed=${SEED} ---"

            # 1. Construct path for the pre-trained prompt learner model
            PROMPT_MODEL_DIR="few_shot/${DATASET_NAME}/shots_${SHOTS}/${PROMPT_TRAINER_NAME}/nctx${PROMPT_NCTX}_csc${PROMPT_CSC}_ctp${PROMPT_CTP}/seed${SEED}"
            PROMPT_MODEL_CHECKPOINT_PATH="${PROMPT_MODEL_DIR}/prompt_learner/model.pth.tar-${PROMPT_LOADEP}"

            # 2. Check if prompt learner checkpoint exists, try to download if not
            if [ ! -f "$PROMPT_MODEL_CHECKPOINT_PATH" ]; then
                echo "Prompt learner checkpoint not found at: ${PROMPT_MODEL_CHECKPOINT_PATH}"
                echo "Attempting to download checkpoint..."
                # Adjust download_ckpts.py arguments if needed (NCTX, CSC, CTP etc.)
                "$PYTHON_EXECUTABLE" download_ckpts.py \
                    --task "few_shot" \
                    --dataset "${DATASET_NAME}" \
                    --shots "${SHOTS}" \
                    --trainer "${PROMPT_TRAINER_NAME}"

                if [ ! -f "$PROMPT_MODEL_CHECKPOINT_PATH" ]; then
                    echo "Download script ran, but checkpoint still not found at ${PROMPT_MODEL_CHECKPOINT_PATH}. Skipping this run."
                    continue
                fi
                echo "Checkpoint downloaded to (or verified at): ${PROMPT_MODEL_DIR}"
            else
                echo "Prompt learner checkpoint found at: ${PROMPT_MODEL_DIR}"
            fi

            # 3. Construct output directory for Tip-X results
            OUTPUT_DIR_TIPX="${OUTPUT_ROOT}/${DATASET_NAME}/shots_${SHOTS}/${PROMPT_TRAINER_NAME}/nctx${PROMPT_NCTX}_csc${PROMPT_CSC}_ctp${PROMPT_CTP}/seed${SEED}"

            # 4. Check if Tip-X results already exist
            # Checking for a log file as an indicator
            LOG_FILE_PATH="${OUTPUT_DIR_TIPX}/tipx_evaluation_log.txt"
            if [ -f "$LOG_FILE_PATH" ]; then
                echo "Tip-X results (log file) already exist in ${LOG_FILE_PATH}. Skipping."
                continue
            fi

            mkdir -p "$OUTPUT_DIR_TIPX"

            # 5. Build and execute the command for run_tipx_with_imports.py
            echo "Running Tip-X evaluation for ${DATASET_NAME}, ${SHOTS}-shots, seed ${SEED}..."
            echo "Log will be saved to: ${LOG_FILE_PATH}"

            # Prepare arguments for run_tipx_with_imports.py
            TIPX_ARGS=(
                "--root_path" "$DATA_ROOT"
                "--dataset" "$DATASET_NAME"
                "--shots" "$SHOTS"
                "--prompt_method" "$PROMPT_TRAINER_NAME"
                "--prompt_model_dir" "$PROMPT_MODEL_DIR"
                "--prompt_model_epoch" "$PROMPT_LOADEP"
                "--search_beta" "$TIPX_SEARCH_BETA_MIN" "$TIPX_SEARCH_BETA_MAX" "$TIPX_SEARCH_BETA_STEPS"
                "--search_alpha" "$TIPX_SEARCH_ALPHA_MIN" "$TIPX_SEARCH_ALPHA_MAX" "$TIPX_SEARCH_ALPHA_STEPS"
                "--search_gamma" "$TIPX_SEARCH_GAMMA_MIN" "$TIPX_SEARCH_GAMMA_MAX" "$TIPX_SEARCH_GAMMA_STEPS"
                "--lr" "$TIPX_LR"
                "--train_epoch" "$TIPX_TRAIN_EPOCH"
            )
            if [ -n "$TIPX_FINETUNE_ARG" ]; then
                TIPX_ARGS+=("$TIPX_FINETUNE_ARG")
            fi

            # Execute and redirect output
            "$PYTHON_EXECUTABLE" run_tipx_with_imports.py "${TIPX_ARGS[@]}" > "$LOG_FILE_PATH" 2>&1

            if [ $? -eq 0 ]; then
                echo "Tip-X evaluation completed successfully for ${OUTPUT_DIR_TIPX}."
            else
                echo "Error: Tip-X evaluation failed for ${OUTPUT_DIR_TIPX}. See log: ${LOG_FILE_PATH}"
            fi
        done
    done
done

echo ""
echo "--- All evaluations finished. ---"
