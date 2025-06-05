import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import argparse
import numpy as np

from tqdm import tqdm
from trainers.method import FSCLIPmethod
from trainers.utils import build_cache_model, search_hp_tip_biomedclip, cls_acc

def _compute_distributions(features, text_weights, temp=0.5):
    """Computes softmax class distributions for given features."""
    # The temperature `temp` can be tuned; 0.5 is a common default.
    dists = 85.2323 * features @ text_weights
    dists = F.softmax(dists / temp, dim=1)
    return dists

# +++ Helper function for Scaling - remains the same +++
def _scale(x, target_range_tensor):
    """Scales tensor x to the min/max range of target_range_tensor."""
    x_min, x_max = x.min(), x.max()
    target_min, target_max = target_range_tensor.min(), target_range_tensor.max()
    
    if (x_max - x_min) == 0:
        return torch.full_like(x, (target_min + target_max) / 2) # Return mid-point of target
        
    y = (x - x_min) / (x_max - x_min)
    y = y * (target_max - target_min) + target_min
    return y

# +++ REFACTORED KL-Divergence Function +++
def _compute_scaled_kl_affinity(eval_dists, support_dists, affinity_for_scaling):
    """
    Computes KL-divergence, negates it, and scales it to the range of the
    provided affinity matrix, fully implementing the logic from the Tip-X paper.
    """
    kl_sim = torch.zeros((eval_dists.shape[0], support_dists.shape[0]), device=eval_dists.device)
    for i in tqdm(range(eval_dists.shape[0]), desc="Computing KL-Div Sim", leave=False):
        p = eval_dists[i].unsqueeze(0)
        q = support_dists
        kl_values = (p * (p.log() - q.log())).sum(dim=1)
        # Negate the KL divergence values to turn distance into similarity
        kl_sim[i] = -kl_values

    # Scale the KL similarities to match the range of cosine similarities (affinity_for_scaling)
    scaled_kl_affinity = _scale(kl_sim, affinity_for_scaling)
    
    return scaled_kl_affinity

def search_hp_tipx_biomedclip(cfg, cache_keys, cache_values, val_features, val_labels, text_weights):
    """
    Performs a 3D grid search for the best alpha, beta, and gamma.
    """
    print("\n**** Searching for best hyperparameters (alpha, beta, gamma) for Tip-X ****")
    
    # Pre-compute constant logits for efficiency
    clip_logits = 85.2323 * val_features @ text_weights
    affinity = val_features @ cache_keys
    
    # +++ TIP-X: Pre-compute KL logits +++
    val_dists = _compute_distributions(val_features, text_weights)
    support_dists = _compute_distributions(cache_keys.t(), text_weights)
    kl_sim = _compute_scaled_kl_affinity(val_dists, support_dists, affinity)
    # Note: Unlike the original Tip-X which scales KL, we'll let gamma handle the magnitude.
    kl_logits = kl_sim @ cache_values.float()

    # Define search ranges from the config
    beta_search_range = np.linspace(cfg['search_beta'][0], cfg['search_beta'][1], cfg['search_beta'][2])
    alpha_search_range = np.linspace(cfg['search_alpha'][0], cfg['search_alpha'][1], cfg['search_alpha'][2])
    gamma_search_range = np.linspace(cfg['search_gamma'][0], cfg['search_gamma'][1], cfg['search_gamma'][2]) # New search range for gamma

    best_acc = 0
    best_beta, best_alpha, best_gamma = 0, 0, 0

    for beta in tqdm(beta_search_range, desc="Searching Beta"):
        for alpha in alpha_search_range:
            for gamma in gamma_search_range:
                # Calculate cache_logits with the current beta
                cache_logits = ((-1) * (beta - beta * affinity.float())).exp() @ cache_values.float()
                
                # Combine all three logit components
                tipx_logits = clip_logits + cache_logits * alpha + kl_logits * gamma
                acc = cls_acc(tipx_logits, val_labels)

                if acc > best_acc:
                    best_acc = acc
                    best_beta = beta
                    best_alpha = alpha
                    best_gamma = gamma
    
    print(f"**** Best val accuracy: {best_acc:.2f} | Best Beta: {best_beta:.2f}, Best Alpha: {best_alpha:.2f}, Best Gamma: {best_gamma:.2f} ****")
    return best_beta, best_alpha, best_gamma


class TIPX_BiomedCLIP(FSCLIPmethod):
    '''
    TIP-X methods
    '''

    def __init__(self, args: argparse.Namespace):
        # self.normalize = args.normalize
        super().__init__(args)
        self.cfg = args
        self.lr = args['lr']
        self.epoch = args['train_epoch']
        self.shot = args['shots']
        self.init_beta = args['init_beta']
        self.init_alpha = args['init_alpha']
        self.finetune = args['finetune']

    def forward(self,
                train_loader: torch.utils.data.DataLoader,
                val_loader: torch.utils.data.DataLoader,
                test_features: torch.tensor,
                test_labels: torch.tensor,
                text_weights: torch.tensor,
                model: nn.Module,
                classnames):
        """
        inputs:
            train_loader : torch.utils.data.DataLoader
            test_features : torch.Tensor of shape [test_data_size, 1024]
            test_labels : torch.Tensor of shape [test_data_size]
            text_weights : torch.Tensor of shape [num_shot*num_classes, 1024]
                            text embeddings of prompts for each class
                           NOTE?: doesn't this suppose to be [num_classes, 1024] ?
        """

        # cache_keys: torch.Tensor of shape [num_shot*num_classes, 1024]
        #              Image embeddings of support set
        # cache_values: torch.Tensor of shape [num_shot*num_classes, num_classes]
        cache_keys, cache_values = build_cache_model(self.cfg, model, train_loader)
        # hyperparameters weight reliable of each method
        beta, alpha, gamma = self.cfg['init_beta'], self.cfg['init_alpha'], self.cfg['init_gamma']
        
        # Feature Extraction for Validation
        print("\nExtracting visual features and labels from val set.")
        val_features, val_labels = [], []
        with torch.no_grad():
            for i, (images, target) in enumerate(tqdm(val_loader)):
                images, target = images.cuda(), target.cuda()
                image_features = model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                val_features.append(image_features)
                val_labels.append(target)
        # val_features: Image embeddings of valiadarion/query set
        # val_labels: Ground truth class of that image in valiadation set
        val_features, val_labels = torch.cat(val_features), torch.cat(val_labels)

        start_time = time.time()
        if not self.finetune:
            # Zero-shot BiomedCLIP
            clip_logits = 85.2323 * val_features @ text_weights # [num_valid, num_classes]
            # cal acc
            acc = cls_acc(clip_logits, val_labels)
            print("\n**** Zero-shot BiomedCLIP's val accuracy: {:.2f}. ****\n".format(acc))

            # Tip-Adapter
            
            # affinity score, the dot product similarity between each val_feature and cache_keys
            affinity = val_features @ cache_keys # [num_valid, num_shot*num_classes]
            # cache_logits: affinity score that become distance via activation function @ cache_values
            cache_logits = ((-1) * (beta - beta * affinity.float())).exp() @ cache_values.float()
            # cache_logits = beta * affinity @ cache_values
            
            tip_logits = clip_logits + cache_logits * alpha
            acc = cls_acc(tip_logits, val_labels)
            print("**** Tip-Adapter's val accuracy: {:.2f}. ****\n".format(acc))


            # +++ TIP-X Extension: Calculate KL logits for validation set +++
            print("Computing KL-Divergence based logits...")
            # calculate cosine similarity of val and support set
            val_dists = _compute_distributions(val_features, text_weights)
            support_dists = _compute_distributions(cache_keys.t(), text_weights) # Note the transpose for cache_keys
            # based on those similarity distributions, calculate kl divergence between each validations and all supports
            kl_sim = _compute_scaled_kl_affinity(val_dists, support_dists, affinity) # [num_valid, num_shot*num_classes]
            kl_logits = kl_sim @ cache_values.float() # [num_valid, num_classes]
            
            # ensembles all logits
            tipx_logits = clip_logits + cache_logits * alpha + kl_logits * gamma
            acc = cls_acc(tipx_logits, val_labels)
            print("**** Tip-X's val accuracy (with initial HPs): {:.2f}. ****\n".format(acc))
            

            # Search Hyperparameters
            best_beta, best_alpha, best_gamma = search_hp_tipx_biomedclip(self.cfg, cache_keys, cache_values, val_features, val_labels, text_weights)            
            
            # Zero-shot BiomedCLIP
            clip_logits = 85.2323 * test_features @ text_weights
            acc = cls_acc(clip_logits, test_labels)
            print("\n**** Zero-shot BiomedCLIP's test accuracy: {:.2f}. ****\n".format(acc))

            # Tip-Adapter    
            affinity = test_features @ cache_keys
            cache_logits = ((-1) * (best_beta - best_beta * affinity.float())).exp() @ cache_values.float()
            
            # tip_logits = clip_logits + cache_logits * best_alpha
            # acc = cls_acc(tip_logits, test_labels)
            # print("**** Tip-Adapter's test accuracy: {:.2f}. ****\n".format(acc))

            print("Computing KL-Divergence based logits...")
            # calculate cosine similarity of val and support set
            test_dists = _compute_distributions(test_features, text_weights)
            support_dists = _compute_distributions(cache_keys.t(), text_weights) # Note the transpose for cache_keys
            # based on those similarity distributions, calculate kl divergence between each validations and all supports
            kl_sim = _compute_scaled_kl_affinity(test_dists, support_dists, affinity) # [num_valid, num_shot*num_classes]
            kl_logits = kl_sim @ cache_values.float() # [num_valid, num_classes]
            
            # ensembles all logits
            tipx_logits = clip_logits + cache_logits * best_alpha + kl_logits * best_gamma
            acc = cls_acc(tipx_logits, val_labels)
            
            return None, acc
        
        """ ######################################################################################### """
        # NOTE: Following code is not modified yet, it's still the tip-F one
        # Enable the cached keys to be learnable
        adapter = nn.Linear(cache_keys.shape[0], cache_keys.shape[1], bias=False).to(model.text.transformer.dtype).cuda()
        adapter.weight = nn.Parameter(cache_keys.t())
        
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=self.cfg['lr'], eps=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.cfg['train_epoch'] * len(train_loader))

        
        # alpha initialization
        if self.cfg["grid_search"]:
            best_acc = 0.0
            print("**** Searching for best initialization of alpha **** \n")
            for init_alpha in range(self.cfg['init_alpha_scale']):
                init_adapter = self.search_init_hp(init_alpha, beta, train_loader, model, cache_keys, cache_values, text_weights)
                affinity = init_adapter(val_features)
                cache_logits = ((-1) * (beta - beta * affinity.float())).exp() @ cache_values.float()
                clip_logits = 85.2323 * val_features @ text_weights
                tip_logits = clip_logits + cache_logits * init_alpha
                acc = cls_acc(tip_logits, val_labels)
                if acc > best_acc:
                    best_acc = acc
                    alpha = init_alpha
                    adapter = init_adapter
            print(alpha)
            print(beta)
        
        # Training Prodecure
        print("**** Start Training **** \n")
        best_acc, best_epoch = 0.0, 0
        for train_idx in range(self.cfg['train_epoch']):
            # Train
            adapter.train()
            correct_samples, all_samples = 0, 0
            loss_list = []
            print('Train Epoch: {:} / {:}'.format(train_idx, self.epoch))

            for i, (images, target) in enumerate(tqdm(train_loader)):
                
                images, target = images.cuda(), target.cuda()
                print("Extraction")
                with torch.no_grad():
                    image_features = model.encode_image(images)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                affinity = adapter(image_features)
                cache_logits = ((-1) * (beta - beta * affinity.float())).exp() @ cache_values.float()
                clip_logits = 85.2323 * image_features @ text_weights
                tip_logits = clip_logits + cache_logits * alpha

                loss = F.cross_entropy(tip_logits, target)

                acc = cls_acc(tip_logits, target)
                correct_samples += acc / 100 * len(tip_logits)
                all_samples += len(tip_logits)
                loss_list.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                

            current_lr = scheduler.get_last_lr()[0]
            print('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples, correct_samples, all_samples, sum(loss_list)/len(loss_list)))

            # Eval
            adapter.eval()

            affinity = adapter(val_features)
            cache_logits = ((-1) * (beta - beta * affinity.float())).exp() @ cache_values.float()
            clip_logits = 85.2323 * val_features @ text_weights
            tip_logits = clip_logits + cache_logits * alpha
            acc = cls_acc(tip_logits, val_labels)

            print("**** Tip-Adapter-F's val accuracy: {:.2f}. ****\n".format(acc))
            if acc > best_acc:
                best_acc = acc
                best_epoch = train_idx
                torch.save(adapter.weight, self.cfg['cache_dir'] + "/best_F_" + str(self.cfg['shots']) + "shots.pt")
        
        adapter.weight = torch.load(self.cfg['cache_dir'] + "/best_F_" + str(self.cfg['shots']) + "shots.pt", weights_only=True)
        print(f"**** After fine-tuning, Tip-Adapter-F's best test accuracy: {best_acc:.2f}, at epoch: {best_epoch}. ****\n")

        """
        affinity = adapter(test_features)
        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
        clip_logits = 85.2323 * test_features @ text_weights
        tip_logits = clip_logits + cache_logits * alpha
        acc = cls_acc(tip_logits, test_labels)
        print("**** Tip-Adapter-F's test accuracy before search : {:.2f}. ****\n".format(acc))
        """
        print("Total time = {:.4f}".format(time.time()-start_time))
        # Search Hyperparameters
        best_beta, best_alpha = search_hp_tip_biomedclip(self.cfg, affinity, cache_values, val_features, val_labels, text_weights, adapter=adapter)
        print("\n-------- Evaluating on the test set. --------")
        
        affinity = adapter(test_features)
        cache_logits = ((-1) * (best_beta - best_beta * affinity.float())).exp() @ cache_values.float()
        clip_logits = 85.2323 * test_features @ text_weights
        tip_logits = clip_logits + cache_logits * best_alpha
        acc = cls_acc(tip_logits, test_labels)
        print("**** Tip-Adapter-F's test accuracy after search: {:.2f}. ****\n".format(acc))
        
        return loss, acc

    def search_init_hp(self, alpha, beta, val_loader, model, cache_keys, cache_values, text_weights):
        adapter = nn.Linear(cache_keys.shape[0], cache_keys.shape[1], bias=False).to(model.text.transformer.dtype).cuda()
        adapter.weight = nn.Parameter(cache_keys.t())
        
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=self.cfg['lr'], eps=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.cfg['train_epoch'] * len(val_loader))

        for val_idx in range(self.cfg['train_epoch']):
            # finetune on validation
            adapter.train()
            correct_samples, all_samples = 0, 0
            loss_list = []
            print('Val Epoch: {:} / {:}'.format(val_idx, self.epoch))

            for i, (images, target) in enumerate(tqdm(val_loader)):
                images, target = images.cuda(), target.cuda()
                with torch.no_grad():
                    image_features = model.encode_image(images)
                    image_features /= image_features.norm(dim=-1, keepdim=True)

                affinity = adapter(image_features)
                cache_logits = ((-1) * (beta - beta * affinity.float())).exp() @ cache_values.float()
                clip_logits = 85.2323 * image_features @ text_weights
                tip_logits = clip_logits + cache_logits * alpha
                acc = cls_acc(tip_logits, target)
                loss = F.cross_entropy(tip_logits, target)

                correct_samples += acc / 100 * len(tip_logits)
                all_samples += len(tip_logits)
                loss_list.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                
            current_lr = scheduler.get_last_lr()[0]
            print('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples, correct_samples, all_samples, sum(loss_list)/len(loss_list)))

            # Eval
            adapter.eval()

        return adapter