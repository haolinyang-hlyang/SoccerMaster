import os
import torch
from torch import nn
import torch.nn.functional as F
import math

from models.soccer_master import SoccerMaster
from models.deformable_detr.deformable_detr import build_deformable_detr_head
from models.lines_detection import build_lines_detection_head
from models.keypoints_detection import build_keypoints_detection_head
from models.video_caption import build_video_caption_head
from models.caption_classification import build_caption_classification_head
from safetensors import safe_open

def interpolate_position_embedding(checkpoint_pos_embed, model_pos_embed, patch_size=16):
    """
    Interpolate position embeddings to allow loading weights across models with different resolutions.
    
    Args:
        checkpoint_pos_embed: Position embedding weights from checkpoint [num_positions_old, embed_dim]
        model_pos_embed: Position embedding weights of the current model [num_positions_new, embed_dim]
        patch_size: Patch size, default 16
    
    Returns:
        interpolated_pos_embed: Interpolated position embedding [num_positions_new, embed_dim]
    """
    num_positions_checkpoint = checkpoint_pos_embed.shape[0]
    num_positions_model = model_pos_embed.shape[0]
    embed_dim = checkpoint_pos_embed.shape[1]
    
    if num_positions_checkpoint == num_positions_model:
        return checkpoint_pos_embed
    
    sqrt_num_positions_checkpoint = int(num_positions_checkpoint ** 0.5)
    sqrt_num_positions_model = int(num_positions_model ** 0.5)
    
    # Reshape position embedding to 2D grid
    # [num_positions, embed_dim] -> [1, sqrt_num, sqrt_num, embed_dim]
    checkpoint_pos_embed_2d = checkpoint_pos_embed.reshape(1, sqrt_num_positions_checkpoint, sqrt_num_positions_checkpoint, embed_dim)
    # [1, sqrt_num, sqrt_num, embed_dim] -> [1, embed_dim, sqrt_num, sqrt_num]
    checkpoint_pos_embed_2d = checkpoint_pos_embed_2d.permute(0, 3, 1, 2)
    
    # Bicubic interpolation for upsampling or downsampling
    interpolated_pos_embed = F.interpolate(
        checkpoint_pos_embed_2d,
        size=(sqrt_num_positions_model, sqrt_num_positions_model),
        mode='bicubic',
        align_corners=False
    )
    
    # [1, embed_dim, sqrt_num_new, sqrt_num_new] -> [1, sqrt_num_new, sqrt_num_new, embed_dim]
    interpolated_pos_embed = interpolated_pos_embed.permute(0, 2, 3, 1)
    # [1, sqrt_num_new, sqrt_num_new, embed_dim] -> [num_positions_new, embed_dim]
    interpolated_pos_embed = interpolated_pos_embed.view(num_positions_model, embed_dim)
    
    return interpolated_pos_embed

class MultiTaskingModel(nn.Module):
    def __init__(self, config, logger=None):
        super().__init__()
        self.config = config
        
        backbone_type = config['SIGLIP_BACKBONE_TYPE'].lower()
        
        if backbone_type == 'soccer_master':
            BackboneType = SoccerMaster
        else:
            raise ValueError(f"Unsupported SIGLIP_BACKBONE_TYPE: {backbone_type}. Supported types: 'soccer_master'")
        
        # Prepare backbone initialization arguments
        backbone_args = {
            'num_frames': config['NUM_FRAMES'],
            'ckpt_path': config['CKPT_PATH'],
            'text_encoder_ckpt_path': config['TEXT_ENCODER_CKPT_PATH'],
            'use_lora': False,
            'use_temporal_gate': config['BACKBONE_USE_TEMPORAL_GATE'],
            'freeze_vision_encoder': config['FREEZE_VISION_ENCODER'],
            'freeze_text_encoder': config['FREEZE_TEXT_ENCODER'],
            'temporal_start_layer': config['TEMPORAL_START_LAYER']
        }
        
        self.backbone = BackboneType(**backbone_args)
        if logger is not None:
            logger.info(f"Using Backbone type: {backbone_type}")
        else:
            print(f"Using Backbone type: {backbone_type}")
        
        self.multi_task_head = nn.ModuleDict()
        self.datasets_to_heads = config["DATASETS_TO_HEADS"]
        all_heads = []
        for dataset, heads in self.datasets_to_heads.items():
            all_heads.extend(heads)
        all_heads = list(set(all_heads))
        all_heads.sort()
        for head in all_heads:
            if head == "SoccerNetGSR_Detection":
                self.multi_task_head[head] = build_deformable_detr_head(config)
            elif head == "LinesDetection":
                self.multi_task_head[head] = build_lines_detection_head(config)
            elif head == "KeypointsDetection":
                self.multi_task_head[head] = build_keypoints_detection_head(config)
            elif head == "VideoCaption":
                self.multi_task_head[head] = build_video_caption_head(config)
            elif head == "CaptionClassification":
                self.multi_task_head[head] = build_caption_classification_head(config)
            else:
                raise ValueError(f"Head {head} is not supported.")
            
    def forward(self, images, dataset_name, metas=None, text=None):
        backbone_outputs = self.backbone(images, text=text)
        
        outputs = {}
        for head in self.datasets_to_heads[dataset_name]:
            outputs[head] = self.multi_task_head[head](backbone_outputs, metas)
        
        return outputs
    
    def save_checkpoint(self, checkpoint_dir: str, logger=None):
        """
        Save model checkpoint including backbone, text encoder, and task heads
        
        Args:
            checkpoint_dir: Directory to save checkpoint
            logger: Logger instance for logging messages
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        backbone_path = os.path.join(checkpoint_dir, 'backbone.pt')
        torch.save(self.backbone.vision_model.state_dict(), backbone_path)
        if logger is not None:
            logger.info(f"Saved custom backbone weights to: {backbone_path}")
        else:
            print(f"Saved custom backbone weights to: {backbone_path}")
        
        for head_name, head in self.multi_task_head.items():
            head_path = os.path.join(checkpoint_dir, f'{head_name}.pt')
            torch.save(head.state_dict(), head_path)
            if logger is not None:
                logger.info(f"Saved {head_name} head to: {head_path}")
            else:
                print(f"Saved {head_name} head to: {head_path}")
    
    def _interpolate_pos_embed_if_needed(self, checkpoint_state_dict: dict, logger=None):
        """
        Check and interpolate position embeddings to allow loading weights across models with different resolutions.
        
        Args:
            checkpoint_state_dict: State dict loaded from checkpoint
            logger: Logger instance for logging messages
            
        Returns:
            Modified state dict
        """
        pos_embed_keys = [
            'vision_model.embeddings.position_embedding.weight',  # SiglipVisionModel
            'vision_model_embedding.position_embedding.weight',  # SoccerMaster
        ]
        
        checkpoint_pos_embed_key = None
        for key in pos_embed_keys:
            if key in checkpoint_state_dict:
                checkpoint_pos_embed_key = key
                break
        
        if checkpoint_pos_embed_key is None:
            if logger is not None:
                logger.info("No position embedding found, skipping interpolation")
            else:
                print("No position embedding found, skipping interpolation")
            return checkpoint_state_dict
        
        model_pos_embed = None
        try:
            if hasattr(self.backbone.vision_model, 'vision_model'):
                # Standard SiglipVisionModel
                model_pos_embed = self.backbone.vision_model.vision_model.embeddings.position_embedding.weight
            elif hasattr(self.backbone.vision_model, 'vision_model_embedding'):
                # UniSoccerBackbone
                model_pos_embed = self.backbone.vision_model.vision_model_embedding.position_embedding.weight
        except AttributeError:
            if logger is not None:
                logger.warning("Could not access model's position embedding, skipping interpolation")
            else:
                print("Warning: Could not access model's position embedding, skipping interpolation")
            return checkpoint_state_dict
        
        if model_pos_embed is None:
            return checkpoint_state_dict
        
        checkpoint_pos_embed = checkpoint_state_dict[checkpoint_pos_embed_key]
        
        if checkpoint_pos_embed.shape[0] != model_pos_embed.shape[0]:
            if logger is not None:
                logger.info(f"Position embedding size mismatch: checkpoint={checkpoint_pos_embed.shape[0]}, model={model_pos_embed.shape[0]}")
                logger.info(f"Interpolating position embedding from {int(checkpoint_pos_embed.shape[0]**0.5)}x{int(checkpoint_pos_embed.shape[0]**0.5)} to {int(model_pos_embed.shape[0]**0.5)}x{int(model_pos_embed.shape[0]**0.5)}")
            else:
                print(f"Position embedding size mismatch: checkpoint={checkpoint_pos_embed.shape[0]}, model={model_pos_embed.shape[0]}")
                print(f"Interpolating position embedding from {int(checkpoint_pos_embed.shape[0]**0.5)}x{int(checkpoint_pos_embed.shape[0]**0.5)} to {int(model_pos_embed.shape[0]**0.5)}x{int(model_pos_embed.shape[0]**0.5)}")
            
            interpolated_pos_embed = interpolate_position_embedding(
                checkpoint_pos_embed, 
                model_pos_embed
            )
            checkpoint_state_dict[checkpoint_pos_embed_key] = interpolated_pos_embed
            
            if logger is not None:
                logger.info("Position embedding interpolation completed successfully")
            else:
                print("Position embedding interpolation completed successfully")
        
        return checkpoint_state_dict
    
    def load_checkpoint(self, checkpoint_dir: str, logger=None, load_heads: bool = True):
        """
        Load model checkpoint including backbone, text encoder, and task heads
        
        Args:
            checkpoint_dir: Directory to load checkpoint from
            logger: Logger instance for logging messages
        """
        backbone_ckpt_path = os.path.join(checkpoint_dir, "backbone.pt")
        if os.path.exists(backbone_ckpt_path):
            backbone_state_dict = torch.load(backbone_ckpt_path, map_location="cpu", weights_only=True)
            
            backbone_state_dict = self._interpolate_pos_embed_if_needed(backbone_state_dict, logger)
            
            res = self.backbone.vision_model.load_state_dict(backbone_state_dict, strict=False)
            if logger is not None:
                logger.info(f"Loaded backbone weights from: {backbone_ckpt_path}")
                if res.missing_keys:
                    logger.warning(f"Missing keys: {res.missing_keys}")
                if res.unexpected_keys:
                    logger.warning(f"Unexpected keys: {res.unexpected_keys}")
            else:
                print(f"Loaded backbone weights from: {backbone_ckpt_path}")
                if res.missing_keys:
                    print(f"Missing keys: {res.missing_keys}")
                if res.unexpected_keys:
                    print(f"Unexpected keys: {res.unexpected_keys}")
        else:
            if logger is not None:
                logger.warning(f"Warning: backbone checkpoint not found at {backbone_ckpt_path}")
            else:
                print(f"Warning: backbone checkpoint not found at {backbone_ckpt_path}")
        
        # Load task heads
        if load_heads:
            for head in self.multi_task_head:
                head_ckpt_path = os.path.join(checkpoint_dir, f"{head}.pt")
                if os.path.exists(head_ckpt_path):
                    if logger is not None:
                        logger.info(f"Loading {head} head from: {head_ckpt_path}")
                    else:
                        print(f"Loading {head} head from: {head_ckpt_path}")
                    head_state_dict = torch.load(head_ckpt_path, map_location="cpu", weights_only=True)
                    self.multi_task_head[head].load_state_dict(head_state_dict)
                else:
                    if logger is not None:
                        logger.warning(f"Warning: {head} head checkpoint not found at {head_ckpt_path}")
                    else:
                        print(f"Warning: {head} head checkpoint not found at {head_ckpt_path}")
        else:
            if logger is not None:
                logger.info(f"Skipping loading task heads from: {checkpoint_dir}")
            else:
                print(f"Skipping loading task heads from: {checkpoint_dir}")
