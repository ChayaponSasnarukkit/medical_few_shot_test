# run_tipx_with_imports.py
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# --- Clean Imports from your project structure ---
from datasets import build_dataset
from datasets.utils import build_data_loader
from trainers import __dict__ as all_methods
from utils import get_arguments as get_original_arguments, pre_load_features
from open_clip.src.open_clip import create_model_from_pretrained

# === ASSUMPTION: Your TIPX_BiomedCLIP class is saved in trainers/tipx.py ===
# If it's in another file, please adjust the import path.
from trainers.TipX import TIPX_BiomedCLIP

# --- PROMPT LEARNER CODE (This part remains the same) ---
class TextEncoder(nn.Module):
    def __init__(self, biomedclip_model):
        super().__init__()
        self.model = biomedclip_model
        self.dtype = biomedclip_model.visual.conv1.weight.dtype
    def forward(self, prompts, tokenized_prompts):
        return self.model.encode_text(prompts, use_embedded_prompts=True, tokenized_prompts=tokenized_prompts)

def build_classifier_from_learned_prompt(prompt_learner, clip_model):
    with torch.no_grad():
        prompt_embeddings = prompt_learner()
        tokenized_prompts = prompt_learner.tokenized_prompts.cuda()
        text_encoder = TextEncoder(clip_model)
        text_features = text_encoder(prompt_embeddings, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features


def main():
    # 1. --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run Tip-X with a pre-trained learned prompt using imports.")
    parser.add_argument('--root_path', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--shots', type=int, default=16)
    parser.add_argument('--prompt_method', type=str, required=True)
    parser.add_argument('--prompt_model_dir', type=str, required=True)
    parser.add_argument('--prompt_model_epoch', type=int, required=True)
    # Hyperparameter search ranges, matching the names in your TIPX class
    parser.add_argument('--search_beta', type=float, nargs=3, default=[1.0, 10.0, 10])
    parser.add_argument('--search_alpha', type=float, nargs=3, default=[1.0, 10.0, 10])
    parser.add_argument('--search_gamma', type=float, nargs=3, default=[0.0, 1.0, 10])
    # Dummy args that might be expected by the TIPX __init__ method
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--train_epoch', type=int, default=10)
    parser.add_argument('--finetune', action='store_true', default=False)
    args = parser.parse_args()

    # 2. --- Load Model and Dataset ---
    print("Loading base model and dataset...")
    clip_model, _, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    original_encode_text = clip_model.encode_text
    def patched_encode_text(text_or_embeds, use_embedded_prompts=False, tokenized_prompts=None):
        if use_embedded_prompts:
            return original_encode_text(text_or_embeds, use_embedded_prompts=True, tokenized_prompts=tokenized_prompts)
        else:
            return original_encode_text(text_or_embeds)
    clip_model.encode_text = patched_encode_text
    clip_model.cuda().eval()

    # Create a config dictionary from args for compatibility
    cfg = vars(args)

    # Add other necessary fields to cfg
    original_cfg = get_original_arguments()
    cfg.update(original_cfg)
    cfg['DATASET'] = {'NAME': args.dataset, 'ROOT': args.root_path, 'NUM_SHOTS': args.shots}
    cfg['method'] = args.prompt_method
    cfg['TRAINER'] = {'BIOMEDCOOP': {'N_CTX': 4, 'CTX_INIT': '', 'CSC': False, 'CLASS_TOKEN_POSITION': 'end'}}
    # Add initial HP values for TIP-X
    cfg['init_beta'], cfg['init_alpha'], cfg['init_gamma'] = 1.0, 1.0, 0.5

    dataset = build_dataset(cfg)
    train_loader = build_data_loader(data_source=dataset.train_x, batch_size=cfg['shots'] * len(dataset.classnames), is_train=False, tfm=preprocess, shuffle=False)
    val_loader = build_data_loader(data_source=dataset.val, batch_size=64, is_train=False, tfm=preprocess, shuffle=False)
    test_loader = build_data_loader(data_source=dataset.test, batch_size=64, is_train=False, tfm=preprocess, shuffle=False)
    
    # 3. --- Load Learned Prompt and Generate Classifier ---
    print(f"Loading learned prompt from: {args.prompt_model_dir}")
    trainer = all_methods[args.prompt_method](cfg)
    trainer.load_model(args.prompt_model_dir, epoch=args.prompt_model_epoch)
    prompt_learner = trainer.model.prompt_learner.cuda()
    print("Building text classifier from learned prompt...")
    learned_text_features = build_classifier_from_learned_prompt(prompt_learner, clip_model).cuda()

    # 4. --- Pre-load Test Features ---
    print("Pre-loading test features...")
    test_features, test_labels = pre_load_features(cfg, "test", clip_model, test_loader)

    # 5. --- Instantiate and Run the TIPX Trainer ---
    print("\nInstantiating TIPX_BiomedCLIP trainer...")
    # The cfg dictionary now holds all necessary parameters
    tipx_trainer = TIPX_BiomedCLIP(cfg)

    print("Executing TIP-X forward pass for HP search and evaluation...")
    # The forward method handles the entire workflow
    _, test_acc = tipx_trainer.forward(
        train_loader=train_loader,
        val_loader=val_loader,
        test_features=test_features.cuda(),
        test_labels=test_labels.cuda(),
        text_weights=learned_text_features,
        model=clip_model,
        classnames=dataset.classnames
    )

    print("-" * 50)
    print(f"Final Test Accuracy from TIP-X run: {test_acc:.2f}%")
    print("-" * 50)


if __name__ == '__main__':
    main()