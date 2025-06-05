# run_tip_adapter_learned_prompt.py
import os
import random
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from datasets import build_dataset
from datasets.utils import build_data_loader
import clip
from trainers import __dict__ as all_methods
from utils import load_cfg_from_cfg_file, merge_cfg_from_list, get_arguments as get_original_arguments, pre_load_features
from open_clip.src.open_clip import create_model_from_pretrained
from clip.pmcclip import ModifiedResNet, image_transform # Assuming you have these locally
from main import PMCCLIP, download_file, pubmedclip_files, pmcclip_files, directory # Re-use model loading from original main

# --- Utility function to generate classifier from a learned prompt ---

def build_classifier_from_learned_prompt(classnames, prompt_learner, clip_model):
    """
    Generates text features (classifier weights) using a trained prompt_learner.
    """
    with torch.no_grad():
        # Get the learned prompt embeddings
        prompt_embeds = prompt_learner.token_prefix
        # Get the class name token embeddings
        tokenized_classnames = torch.cat([clip.tokenize(c) for c in classnames]).cuda()
        classname_embeds = clip_model.token_embedding(tokenized_classnames).type(clip_model.dtype)
        
        # Combine learned prompt with class names
        # This logic should match the forward pass of your prompt_learner
        # Example for a simple prefix prompt:
        n_cls = len(classnames)
        n_ctx = prompt_learner.n_ctx
        
        # Create full embeddings: [n_cls, 77, 512]
        full_embeds = torch.zeros(n_cls, 77, classname_embeds.shape[-1]).type(clip_model.dtype).cuda()
        
        for i in range(n_cls):
            # Assumes the prompt is a prefix
            prefix = prompt_embeds[i] if prompt_embeds.dim() == 3 else prompt_embeds
            suffix = classname_embeds[i]
            
            # Simple concatenation logic, adjust if your prompt learner is different
            # e.g., for "a photo of a [CLASS]" -> "[V1]...[V_N_CTX] [CLASS] [EOT]..."
            eot_idx = tokenized_classnames[i].argmax()
            
            full_embeds[i, :n_ctx, :] = prefix
            full_embeds[i, n_ctx:eot_idx, :] = suffix[1:eot_idx-n_ctx+1] # Avoid [SOS] token

        # Get text features by passing through the transformer
        text_features = clip_model.transformer(full_embeds)
        text_features = clip_model.ln_final(text_features)
        
        # Take the feature from the EOT token
        text_features = text_features[torch.arange(n_cls), tokenized_classnames.argmax(dim=-1)] @ clip_model.text_projection
        
        # Normalize features
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
    return text_features


# --- Main execution ---

def main():
    # 1. --- Simplified Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run Tip-Adapter with a pre-trained learned prompt.")
    parser.add_argument('--root_path', type=str, required=True, help='Root directory of the dataset.')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset (e.g., btmri).')
    parser.add_argument('--shots', type=int, default=16, help='Number of few-shots.')
    parser.add_argument('--clip_model', type=str, default='BiomedCLIP', help='Which CLIP backbone to use.')
    parser.add_argument('--backbone', type=str, default='ViT-B-16', help='Backbone architecture for standard CLIP.')
    parser.add_argument('--prompt_method', type=str, required=True, help='The prompt learning method used (e.g., BiomedCoOp_BiomedCLIP).')
    parser.add_argument('--prompt_model_dir', type=str, required=True, help='Directory of the saved prompt learner model.')
    parser.add_argument('--prompt_model_epoch', type=int, required=True, help='Epoch of the prompt model to load.')
    parser.add_argument('--beta', type=float, default=1.0, help='Tip-Adapter beta parameter.')
    parser.add_argument('--alpha', type=float, default=1.0, help='Tip-Adapter alpha parameter.')
    
    args = parser.parse_args()
    
    # 2. --- Load the Base CLIP Model ---
    print(f"Loading base model: {args.clip_model}")
    if args.clip_model == 'CLIP':
        clip_model, preprocess = clip.load(args.backbone)
    elif args.clip_model == 'BiomedCLIP':
        clip_model, _, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    elif args.clip_model == 'PubMedCLIP':
        # Your PubMedCLIP loading logic here
        for filename, url in pubmedclip_files.items():
            filepath = os.path.join(directory, filename)
            if not os.path.exists(filepath): download_file(url, filepath)
        clip_model, preprocess = clip.load('ViT-B/32')
        checkpoint = torch.load(os.path.join(directory,"PubMedCLIP_ViT32.pth"), map_location="cpu")
        clip_model.load_state_dict(checkpoint['state_dict'])
    else:
        raise ValueError(f"Unsupported clip_model: {args.clip_model}")
    clip_model.cuda().eval()

    # 3. --- Load Dataset ---
    print(f"Loading dataset: {args.dataset} with {args.shots} shots.")
    
    # Create a mock cfg object for compatibility with existing functions
    cfg = get_original_arguments() # Load base configs
    cfg.DATASET.NAME = args.dataset
    cfg.DATASET.ROOT = args.root_path
    cfg.DATASET.NUM_SHOTS = args.shots
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"
    cfg.method = args.prompt_method # Important for loading the correct prompt learner class
    
    dataset = build_dataset(cfg)
    test_loader = build_data_loader(data_source=dataset.test, batch_size=64, is_train=False, tfm=preprocess, shuffle=False)
    
    # 4. --- Load the Learned Prompt and Generate Classifier ---
    print(f"Loading learned prompt from: {args.prompt_model_dir}")
    prompt_learner_trainer = all_methods[args.prompt_method](cfg)
    prompt_learner_trainer.load_model(args.prompt_model_dir, epoch=args.prompt_model_epoch)
    prompt_learner = prompt_learner_trainer.prompt_learner.cuda()

    print("Building text classifier from learned prompt...")
    # NOTE: This function assumes a specific prompt learner structure. You may need to adapt it.
    learned_text_features = build_classifier_from_learned_prompt(dataset.classnames, prompt_learner, clip_model)
    
    # 5. --- Run Tip-Adapter Logic ---
    print("Executing Tip-Adapter with the learned classifier.")
    
    # Create few-shot training data loader
    few_shot_dataset = dataset.generate_fewshot_dataset(shot_num=args.shots, split='train')
    train_loader = build_data_loader(data_source=few_shot_dataset, batch_size=len(few_shot_dataset), is_train=False, tfm=preprocess, shuffle=False)
    
    # --- Create the cache for Tip-Adapter ---
    with torch.no_grad():
        # Get features and labels for the few-shot training images
        train_images, train_labels = next(iter(train_loader))
        train_features = clip_model.encode_image(train_images.cuda())
        train_features /= train_features.norm(dim=-1, keepdim=True)
        
        # Pre-load test features
        print("Pre-loading test features...")
        test_features, test_labels = pre_load_features(cfg, "test", clip_model, test_loader)

    # --- Construct the Tip-Adapter cache ---
    # Keys are the normalized training features
    cache_keys = train_features
    # Values are the one-hot encoded labels
    cache_values = F.one_hot(train_labels, num_classes=len(dataset.classnames)).float().cuda()
    
    # --- Evaluation with Tip-Adapter ---
    # Zero-shot prediction using the learned prompt classifier
    zeroshot_logits = test_features @ learned_text_features.T
    
    # Tip-Adapter affinity calculation
    tip_logits = test_features @ cache_keys.T
    
    # Final prediction combines zero-shot and cache predictions
    final_logits = zeroshot_logits + args.beta * (tip_logits ** args.alpha) @ cache_values
    
    # Calculate accuracy
    acc = (final_logits.argmax(dim=1) == test_labels.cuda()).float().mean().item()
    
    print("-" * 30)
    print(f"Dataset: {args.dataset}, Shots: {args.shots}")
    print(f"Tip-Adapter (alpha={args.alpha}, beta={args.beta}) with learned prompt from {args.prompt_method}")
    print(f"Final Accuracy = {acc * 100:.2f}%")
    print("-" * 30)


if __name__ == '__main__':
    main()