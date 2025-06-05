# run_tip_adapter_learned_prompt_CORRECTED.py
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from datasets import build_dataset
from datasets.utils import build_data_loader
from trainers import __dict__ as all_methods
from utils import get_arguments as get_original_arguments, pre_load_features
from open_clip.src.open_clip import create_model_from_pretrained
from main import PMCCLIP, download_file, pubmedclip_files, pmcclip_files, directory # Re-use model loading

# --- CODE FROM YOUR PROMPT LEARNER ---
# This TextEncoder is copied directly from the code you provided.
class TextEncoder(nn.Module):
    def __init__(self, biomedclip_model):
        super().__init__()
        self.model = biomedclip_model
        # The original code had a typo here, it should likely be this to get the dtype
        self.dtype = biomedclip_model.visual.conv1.weight.dtype 

    def forward(self, prompts, tokenized_prompts):
        # We need to replicate the behavior of your custom TextEncoder
        # The 'prompts' are the already-embedded vectors
        # The second argument to encode_text in your BiomedCLIP seems to be a flag to use pre-computed embeddings
        x = self.model.encode_text(prompts, use_embedded_prompts=True, tokenized_prompts=tokenized_prompts)
        return x

# --- CORRECTED UTILITY FUNCTION ---
def build_classifier_from_learned_prompt(prompt_learner, clip_model):
    """
    CORRECTED: Generates text features by directly using the PromptLearner's
    forward pass and the TextEncoder, perfectly matching the training logic.
    """
    with torch.no_grad():
        # 1. Get the assembled prompt embeddings directly from the learner's forward pass
        # This calls `construct_prompts` internally and handles all logic correctly.
        prompt_embeddings = prompt_learner()

        # 2. Get the corresponding tokenized prompts (needed by the TextEncoder)
        tokenized_prompts = prompt_learner.tokenized_prompts.cuda()

        # 3. Use the TextEncoder wrapper to get the final features
        text_encoder = TextEncoder(clip_model)
        text_features = text_encoder(prompt_embeddings, tokenized_prompts)

        # 4. Normalize the features to use as a classifier
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
    return text_features


# --- Main execution (mostly unchanged) ---
def main():
    # 1. --- Argument Parsing ---
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
    if args.clip_model == 'BiomedCLIP':
        # IMPORTANT: We need to modify the encode_text method to accept pre-computed embeddings
        clip_model, _, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        
        # Monkey-patch the encode_text method to match your training code's expectation
        original_encode_text = clip_model.encode_text
        def patched_encode_text(text_or_embeds, use_embedded_prompts=False, tokenized_prompts=None):
            if use_embedded_prompts:
                # The input is already embedded, pass it to the transformer
                return original_encode_text(text_or_embeds, use_embedded_prompts=True, tokenized_prompts=tokenized_prompts)
            else:
                return original_encode_text(text_or_embeds)
        clip_model.encode_text = patched_encode_text
    else:
        raise ValueError(f"This corrected script is tailored for your BiomedCLIP-based PromptLearner.")
    clip_model.cuda().eval()

    # 3. --- Load Dataset ---
    print(f"Loading dataset: {args.dataset} with {args.shots} shots.")
    cfg = get_original_arguments()
    cfg.DATASET.NAME = args.dataset
    cfg.DATASET.ROOT = args.root_path
    cfg.DATASET.NUM_SHOTS = args.shots
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"
    cfg.method = args.prompt_method
    # Add dummy values for cfg fields your PromptLearner needs
    cfg.TRAINER.BIOMEDCOOP = {'N_CTX': 4, 'CTX_INIT': '', 'CSC': False, 'CLASS_TOKEN_POSITION': 'end'}

    dataset = build_dataset(cfg)
    test_loader = build_data_loader(data_source=dataset.test, batch_size=64, is_train=False, tfm=preprocess, shuffle=False)
    
    # 4. --- Load the Learned Prompt and Generate Classifier ---
    print(f"Loading learned prompt from: {args.prompt_model_dir}")
    # We instantiate the trainer which contains the model and prompt_learner
    trainer = all_methods[args.prompt_method](cfg)
    # The load_model function in your code loads weights into `trainer.model`, which is the CustomCLIP instance
    trainer.load_model(args.prompt_model_dir, epoch=args.prompt_model_epoch)
    prompt_learner = trainer.model.prompt_learner.cuda()

    print("Building text classifier from learned prompt...")
    learned_text_features = build_classifier_from_learned_prompt(prompt_learner, clip_model)
    
    # 5. --- Run Tip-Adapter Logic ---
    print("Executing Tip-Adapter with the learned classifier.")
    few_shot_dataset = dataset.generate_fewshot_dataset(shot_num=args.shots, split='train')
    train_loader = build_data_loader(data_source=few_shot_dataset, batch_size=len(few_shot_dataset), is_train=False, tfm=preprocess, shuffle=False)
    
    with torch.no_grad():
        train_images, train_labels = next(iter(train_loader))
        train_features = clip_model.encode_image(train_images.cuda())
        train_features /= train_features.norm(dim=-1, keepdim=True)
        print("Pre-loading test features...")
        test_features, test_labels = pre_load_features(cfg, "test", clip_model, test_loader)

    cache_keys = train_features
    cache_values = F.one_hot(train_labels, num_classes=len(dataset.classnames)).float().cuda()
    
    zeroshot_logits = test_features @ learned_text_features.T
    tip_logits = test_features @ cache_keys.T
    final_logits = zeroshot_logits + args.beta * (tip_logits ** args.alpha) @ cache_values
    
    acc = (final_logits.argmax(dim=1) == test_labels.cuda()).float().mean().item()
    
    print("-" * 30)
    print(f"Dataset: {args.dataset}, Shots: {args.shots}")
    print(f"Tip-Adapter (alpha={args.alpha}, beta={args.beta}) with learned prompt from {args.prompt_method}")
    print(f"Final Accuracy = {acc * 100:.2f}%")
    print("-" * 30)

if __name__ == '__main__':
    main()