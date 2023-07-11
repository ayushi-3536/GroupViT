# -------------------------------------------------------------------------
# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/GroupViT/blob/main/LICENSE
#
# Written by Jiarui Xu
# -------------------------------------------------------------------------

import os.path as osp

import matplotlib.pyplot as plt
import mmcv
import numpy as np
import os
import torch
import torch.nn.functional as F
from einops import rearrange
from mmseg.models import EncoderDecoder
from PIL import Image
from utils import get_logger
import cv2
import random
import colorsys
from io import BytesIO
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import torch
import torch.nn as nn
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image

GROUP_PALETTE = np.loadtxt(osp.join(osp.dirname(osp.abspath(__file__)), 'group_palette.txt'), dtype=np.uint8)[:, ::-1]

def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
    return image


def random_colors(N, bright=True):
    """
    Generate random colors.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def display_instances(image, mask, fname="test", figsize=(5, 5), blur=False, contour=False, alpha=0.5):
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()

    N = 1
    mask = mask[None, :, :]
    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    margin = 0
    ax.set_ylim(height + margin, -margin)
    ax.set_xlim(-margin, width + margin)
    ax.axis('off')
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]
        _mask = mask[i]
        if blur:
            _mask = cv2.blur(_mask,(10,10))
        # Mask
        masked_image = apply_mask(masked_image, _mask, color, alpha)
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        if contour:
            padded_mask = np.zeros((_mask.shape[0] + 2, _mask.shape[1] + 2))
            padded_mask[1:-1, 1:-1] = _mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8), aspect='auto')
    fig.savefig(fname)
    #print(f"{fname} saved.")
    return

def resize_attn_map(attentions, h, w, align_corners=False):
    """

    Args:
        attentions: shape [B, num_head, H*W, groups]
        h:
        w:

    Returns:

        attentions: shape [B, num_head, h, w, groups]


    """
    scale = (h * w // attentions.shape[2])**0.5
    if h > w:
        w_featmap = w // int(np.round(scale))
        h_featmap = attentions.shape[2] // w_featmap
    else:
        h_featmap = h // int(np.round(scale))
        w_featmap = attentions.shape[2] // h_featmap

    print(f'attention map size: {attentions.shape}')
    
    assert attentions.shape[
        2] == h_featmap * w_featmap, f'{attentions.shape[2]} = {h_featmap} x {w_featmap}, h={h}, w={w}'

    bs = attentions.shape[0]
    nh = attentions.shape[1]  # number of head
    groups = attentions.shape[3]  # number of group token
    # [bs, nh, h*w, groups] -> [bs*nh, groups, h, w]
    attentions = rearrange(
        attentions, 'bs nh (h w) c -> (bs nh) c h w', bs=bs, nh=nh, h=h_featmap, w=w_featmap, c=groups)
    attentions = F.interpolate(attentions, size=(h, w), mode='bilinear', align_corners=align_corners)
    #  [bs*nh, groups, h, w] -> [bs, nh, h*w, groups]
    attentions = rearrange(attentions, '(bs nh) c h w -> bs nh h w c', bs=bs, nh=nh, h=h, w=w, c=groups)

    return attentions


def top_groups(attn_map, k):
    """
    Args:
        attn_map: (B, H, W, G)
        k: int

    Return:
        (B, H, W, k)
    """

    attn_map = attn_map.clone()

    for i in range(attn_map.size(0)):
        # [H*W, G]
        flatten_map = rearrange(attn_map[i], 'h w g -> (h w) g')
        kept_mat = torch.zeros(flatten_map.shape[0], device=flatten_map.device, dtype=torch.bool)
        area_per_group = flatten_map.sum(dim=0)
        top_group_idx = area_per_group.topk(k=k).indices.cpu().numpy().tolist()
        for group_idx in top_group_idx:
            kept_mat[flatten_map.argmax(dim=-1) == group_idx] = True
        # [H, W, 2]
        coords = torch.stack(
            torch.meshgrid(
                torch.arange(attn_map[i].shape[0], device=attn_map[i].device, dtype=attn_map[i].dtype),
                torch.arange(attn_map[i].shape[1], device=attn_map[i].device, dtype=attn_map[i].dtype)),
            dim=-1)
        coords = rearrange(coords, 'h w c -> (h w) c')

        # calculate distance between each pair of points
        # [non_kept, kept]
        dist_mat = torch.sum((coords[~kept_mat].unsqueeze(1) - coords[kept_mat].unsqueeze(0))**2, dim=-1)

        flatten_map[~kept_mat] = flatten_map[kept_mat.nonzero(as_tuple=True)[0][dist_mat.argmin(dim=-1)]]

        attn_map[i] = flatten_map.reshape_as(attn_map[i])

    return attn_map


def seg2coord(seg_map):
    """
    Args:
        seg_map (np.ndarray): (H, W)

    Return:
        dict(group_id -> (x, y))
    """
    h, w = seg_map.shape
    # [h ,w, 2]
    coords = np.stack(np.meshgrid(np.arange(h), np.arange(w), indexing='ij'), axis=-1)
    labels = np.unique(seg_map)
    coord_map = {}
    for label in labels:
        coord_map[label] = coords[seg_map == label].mean(axis=0)
    return coord_map


class GroupViTSegInference(EncoderDecoder):

    def __init__(self, model, text_embedding, with_bg, test_cfg=dict(mode='whole', bg_thresh=.9)):
        super(EncoderDecoder, self).__init__()
        if not isinstance(test_cfg, mmcv.Config):
            test_cfg = mmcv.Config(test_cfg)
        self.test_cfg = test_cfg
        
        self.model = model
        # [N, C]
        self.register_buffer('text_embedding', text_embedding)
        self.with_bg = with_bg
        self.bg_thresh = test_cfg['bg_thresh']
        #print('bg_thresh', self.bg_thresh)
        if self.with_bg:
            self.num_classes = len(text_embedding) + 1
        else:
            self.num_classes = len(text_embedding)
        self.align_corners = False
        self.output_dir=''
        ##logger = get_logger()
        #logger.info(
        #    f'Building GroupViTSegInference with {self.num_classes} classes, test_cfg={test_cfg}, with_bg={with_bg}')

    def forward_train(self, img, img_metas, gt_semantic_seg):
        raise NotImplementedError

    def get_attn_maps(self, img, return_onehot=False, rescale=False):
        """
        Args:
            img: [B, C, H, W]

        Returns:
            attn_maps: list[Tensor], attention map of shape [B, H, W, groups]
        """
        #logger = get_logger()
        results = self.model.img_encoder(img, return_attn=True, as_dict=True)

        attn_maps = []
        with torch.no_grad():
            prev_attn_masks = None
            for idx, attn_dict in enumerate(results['attn_dicts']):
                if attn_dict is None:
                    assert idx == len(results['attn_dicts']) - 1, 'only last layer can be None'
                    continue
                # [B, G, HxW]
                # B: batch size (1), nH: number of heads, G: number of group token
                attn_masks = attn_dict['soft']
                # [B, nH, G, HxW] -> [B, nH, HxW, G]
                #print('attn_masks before', attn_masks.shape)
                attn_masks = rearrange(attn_masks, 'b h g n -> b h n g')
                #print('attn_masks after', attn_masks.shape)
                if prev_attn_masks is None:
                    prev_attn_masks = attn_masks
                else:
                    #print('attn_masks', attn_masks.shape)
                    #print('prev_attn_masks', prev_attn_masks.shape)
                    prev_attn_masks = prev_attn_masks @ attn_masks
                #[B, nH, HxW, G] -> [B, nH, H, W, G]
                attn_maps.append(resize_attn_map(prev_attn_masks, *img.shape[-2:]))
                #attn_maps.append(resize_attn_map(attn_masks, *img.shape[-2:]))


        for i in range(len(attn_maps)):
            attn_map = attn_maps[i]
            # [B, nh, H, W, G]
            assert attn_map.shape[1] == 1
            # [B, H, W, G]
            attn_map = attn_map.squeeze(1)

            if rescale:
                attn_map = rearrange(attn_map, 'b h w g -> b g h w')
                attn_map = F.interpolate(
                    attn_map, size=img.shape[2:], mode='bilinear', align_corners=self.align_corners)
                attn_map = rearrange(attn_map, 'b g h w -> b h w g')

            if return_onehot:
                # [B, H, W, G]
                attn_map = F.one_hot(attn_map.argmax(dim=-1), num_classes=attn_map.shape[-1]).to(dtype=attn_map.dtype)

            attn_maps[i] = attn_map

        return attn_maps

    def save_heat_maps(self, data, file_path, title):
        fig, ax = plt.subplots()
        data = data.cpu().numpy()
        im = ax.imshow(data, cmap='viridis')
        cbar = ax.figure.colorbar(im, ax=ax)
        ax.set_title(title)
        plt.savefig(file_path)
        plt.close()
    
    def save_class_affinity_maps(self, data, file_path, title):
        fig, ax = plt.subplots()
        im = ax.imshow(data, cmap='viridis')
        cbar = ax.figure.colorbar(im, ax=ax)
        ax.set_title(title)
        ax.set_xticks(range(len(self.CLASSES[1:])))
        ax.set_xticklabels(self.CLASSES[1:], rotation=90, ha='right')
        plt.savefig(file_path)
        plt.close()
    
    def plot_attn_map(self,attn_map, index, file_path, title='Group {}', format=True, format_value='0'):
            """
            Plots an attention map for a specific index in a list.
            """
            fig, ax = plt.subplots()
            if np.any(attn_map[:, :, index] != 0):
                print("index", index)
                print('attn_map[:, :, index]', attn_map[:, :, index].shape)
                im = ax.imshow(attn_map[:, :, index], cmap='viridis')
                cbar = ax.figure.colorbar(im, ax=ax)
                # This code is used to plot the data for each group/class.
                # The variable 'i' is the group/class number.
                if format:
                    ax.set_title(title.format(format_value))
                    plt.savefig(file_path.format(format_value))
                else:
                    ax.set_title(title.format(index))
                    plt.savefig(file_path.format(index))
                plt.close()

    def save_entropy_maps(self, attn_map, img):
        """save attention map"""
        attn_map_results_path = osp.join(self.output_dir, 'attn_map')
        mmcv.mkdir_or_exist(attn_map_results_path)
        attn_map_results_path = osp.join(attn_map_results_path,'attn_map{}.png')

        for i in range(attn_map.shape[-1]):
            print("index for attn map", i)
            self.plot_attn_map(attn_map.detach().cpu().numpy(), i, attn_map_results_path, 'Group {}', format==False)

        """calculate shannon entropy for each group"""
        dist_attn_map = F.softmax(attn_map, dim=-1)
        shannon_entropy = -dist_attn_map * torch.log(dist_attn_map)
        sum_shannon_entropy = shannon_entropy[:-1].sum(dim=-1)

        """save entropy map"""
        entropy_map_path = osp.join(self.output_dir, 'entropy_map')
        mmcv.mkdir_or_exist(entropy_map_path)
        sum_entropy_map_path = osp.join(entropy_map_path, 'sum_entropy_map.png')
        self.save_heat_maps(sum_shannon_entropy, sum_entropy_map_path, 'Sum Entropy Of Attention Map Distrubtion Across Groups')

        img_outs = self.model.encode_image(img, return_feat=True, as_dict=True)
        #print("img_outs", img_outs.keys())
        # [B, L, C] -> [L, C]
        grouped_img_tokens = img_outs['image_feat'].squeeze(0)
        #print("grouped_img_tokens", grouped_img_tokens.shape)

        img_avg_feat = img_outs['image_x']
        #print("img_avg_feat", img_avg_feat.shape)
        
        # [G, C]
        grouped_img_tokens = F.normalize(grouped_img_tokens, dim=-1)
        img_avg_feat = F.normalize(img_avg_feat, dim=-1)

        num_fg_classes = self.text_embedding.shape[0]
        class_offset = 1 if self.with_bg else 0
        text_tokens = self.text_embedding
        num_classes = num_fg_classes + class_offset

        logit_scale = torch.clamp(self.model.logit_scale.exp(), max=100)
        # [G, N]
        """calculate affinity matrix"""
        group_affinity_mat = (grouped_img_tokens @ text_tokens.T) * logit_scale
        pre_group_affinity_mat = F.softmax(group_affinity_mat, dim=-1)

        """calculate average affinity matrix"""
        avg_affinity_mat = (img_avg_feat @ text_tokens.T) * logit_scale
        avg_affinity_mat = F.softmax(avg_affinity_mat, dim=-1)

        """calculate group affinity for top-5 classes"""
        affinity_mask = torch.zeros_like(avg_affinity_mat)
        avg_affinity_topk = avg_affinity_mat.topk(dim=-1, k=min(5, num_fg_classes))
        affinity_mask.scatter_add_(
            dim=-1, index=avg_affinity_topk.indices, src=torch.ones_like(avg_affinity_topk.values))
        group_affinity_mat.masked_fill_(~affinity_mask.bool(), float('-inf'))

        group_affinity_mat = F.softmax(group_affinity_mat, dim=-1)

        # TODO: check if necessary
        group_affinity_mat *= pre_group_affinity_mat

        
        """calculate entropy of attention map distribution across classes"""
        prob_affinity_value = (attn_map @ group_affinity_mat)
        #print("affinity value", prob_affinity_value)
        dist_prob_affinity_value = F.softmax(prob_affinity_value, dim=-1)
        class_path = osp.join(self.output_dir, 'entropy_affinity')        
        mmcv.mkdir_or_exist(class_path)
        class_path = osp.join(class_path, 'class_entropy.png')
        entropy = -torch.sum(dist_prob_affinity_value * torch.log(dist_prob_affinity_value), dim=-1)
        self.save_heat_maps(entropy, class_path, 'Entropy Of Attention Map Distrubtion Across Classes')



    def save_all_visualization(self, results):
        attn_map = results['attention_map']
        onehot_attn_map = results['onehot_attn_map']
        group_affinity_mat = results['group_affinity_mat']
        pre_group_affinity_mat = results['pre_group_affinity_mat']
        avg_affinity_mat = results['avg_affinity_mat']
        affinity_value = results['affinity_value']
        
        """save attention map"""
        attn_map_results_path = osp.join(self.output_dir, 'attn_map')
        mmcv.mkdir_or_exist(attn_map_results_path)
        attn_map_results_path = osp.join(attn_map_results_path,'attn_map{}.png')
        for i in range(attn_map.shape[-1]):
            self.plot_attn_map(attn_map.detach().cpu().numpy(), i, attn_map_results_path, 'Group {}')

        """calculate shannon entropy for each group"""
        dist_attn_map = F.softmax(attn_map, dim=-1)
        shannon_entropy = -dist_attn_map * torch.log(dist_attn_map)
        sum_shannon_entropy = shannon_entropy[:-1].sum(dim=-1)

        """save entropy map"""
        entropy_map_path = osp.join(self.output_dir, 'entropy_map')
        mmcv.mkdir_or_exist(entropy_map_path)
        sum_entropy_map_path = osp.join(entropy_map_path, 'sum_entropy_map.png')
        self.save_heat_maps(sum_shannon_entropy, sum_entropy_map_path, 'Sum Entropy Of Attention Map Distrubtion Across Groups')

        """save onehot attention map"""
        onehot_attn_map_path = osp.join(self.output_dir, 'one_hot_attn')
        mmcv.mkdir_or_exist(onehot_attn_map_path)
        onehot_attn_map_path = osp.join(onehot_attn_map_path, 'onehot_attn_map{}.png')
        for i in range(onehot_attn_map.shape[-1]):
            self.plot_attn_map(onehot_attn_map.detach().cpu().numpy(), i, onehot_attn_map_path, 'Group {}')

        """save affinity matrix"""
        group_text_affinity_metric_path = osp.join(self.output_dir, 'group_text_affinity_metric.png')
        pre_group_text_affinity_metric_path = osp.join(self.output_dir, 'pre_group_text_affinity_metric.png')        
        avg_affinity_metric_path = osp.join(self.output_dir, 'avg_affinity_metric.png')
        
        self.save_class_affinity_maps(group_affinity_mat.detach().cpu().numpy(),
                                    group_text_affinity_metric_path,
                                    'Visual_Text Token Affinity')
        self.save_class_affinity_maps(pre_group_affinity_mat.detach().cpu().numpy(),
                                       pre_group_text_affinity_metric_path, 
                                       'Softmax Visual_Text Token Affinity')
        self.save_class_affinity_maps(avg_affinity_mat.detach().cpu().numpy(),
                                       avg_affinity_metric_path, 
                                       'Avg Visual_Text Token Affinity')
        
        """save class-threshold attention map(masks) affinity heat map"""
        class_path = osp.join(self.output_dir, 'label_affinity_onehot')        
        mmcv.mkdir_or_exist(class_path)
        class_path = osp.join(class_path, 'class_{}.png')
        for i in range(affinity_value.shape[-1]):
             self.plot_attn_map(affinity_value.detach().cpu().numpy(), i,class_path, 'Class {}', format=True, format_value=self.CLASSES[i+1])
        

        """calculate entropy of attention map distribution across classes"""
        prob_affinity_value = (attn_map @ group_affinity_mat)
        #print("affinity value", prob_affinity_value)
        dist_prob_affinity_value = F.softmax(prob_affinity_value, dim=-1)
        class_path = osp.join(self.output_dir, 'entropy_affinity')        
        mmcv.mkdir_or_exist(class_path)
        class_path = osp.join(class_path, 'class_entropy.png')
        entropy = -torch.sum(dist_prob_affinity_value * torch.log(dist_prob_affinity_value), dim=-1)
        self.save_heat_maps(entropy, class_path, 'Entropy Of Attention Map Distrubtion Across Classes')


    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""

        assert img.shape[0] == 1, 'batch size must be 1'
        print("img shape", img.shape)
        print("img_metas", img_metas)

        # [B, C, H, W], get the last one only
        attn_map = self.get_attn_maps(img, rescale=True)[-1]
        # [H, W, G], select batch idx 0
        attn_map = attn_map[0]


        """Thresholding the attention map"""
        onehot_attn_map = F.one_hot(attn_map.argmax(dim=-1), num_classes=attn_map.shape[-1]).to(dtype=attn_map.dtype)
        
        num_fg_classes = self.text_embedding.shape[0]
        class_offset = 1 if self.with_bg else 0
        text_tokens = self.text_embedding
        num_classes = num_fg_classes + class_offset

        logit_scale = torch.clamp(self.model.logit_scale.exp(), max=100)

        img_outs = self.model.encode_image(img, return_feat=True, as_dict=True)
        #print("img_outs", img_outs.keys())
        # [B, L, C] -> [L, C]
        grouped_img_tokens = img_outs['image_feat'].squeeze(0)
        #print("grouped_img_tokens", grouped_img_tokens.shape)

        img_avg_feat = img_outs['image_x']
        #print("img_avg_feat", img_avg_feat.shape)
        
        # [G, C]
        grouped_img_tokens = F.normalize(grouped_img_tokens, dim=-1)
        img_avg_feat = F.normalize(img_avg_feat, dim=-1)

        # [G, N]
        """calculate affinity matrix"""
        group_affinity_mat = (grouped_img_tokens @ text_tokens.T) * logit_scale
        pre_group_affinity_mat = F.softmax(group_affinity_mat, dim=-1)

        """calculate average affinity matrix"""
        avg_affinity_mat = (img_avg_feat @ text_tokens.T) * logit_scale
        avg_affinity_mat = F.softmax(avg_affinity_mat, dim=-1)

        """calculate group affinity for top-5 classes"""
        affinity_mask = torch.zeros_like(avg_affinity_mat)
        avg_affinity_topk = avg_affinity_mat.topk(dim=-1, k=min(5, num_fg_classes))
        affinity_mask.scatter_add_(
            dim=-1, index=avg_affinity_topk.indices, src=torch.ones_like(avg_affinity_topk.values))
        group_affinity_mat.masked_fill_(~affinity_mask.bool(), float('-inf'))

        group_affinity_mat = F.softmax(group_affinity_mat, dim=-1)

        # TODO: check if necessary
        group_affinity_mat *= pre_group_affinity_mat

        """get similarity of patch and text"""
        pred_logits = torch.zeros(num_classes, *attn_map.shape[:2], device=img.device, dtype=img.dtype)
        # print("pred_logits", pred_logits.shape)
        # print("onehot_attn_map", onehot_attn_map.shape)
        pred_logits[class_offset:] = rearrange(onehot_attn_map @ group_affinity_mat, 'h w c -> c h w')
        affinity_value = (onehot_attn_map @ group_affinity_mat)
        max_affinity_value = affinity_value.max(dim=-1).values
        if self.with_bg:
            bg_thresh = min(self.bg_thresh, group_affinity_mat.max().item())
            pred_logits[0, max_affinity_value < bg_thresh] = 1

        # dict_result = dict(attention_map=attn_map,
        #                    onehot_attn_map=onehot_attn_map,
        #                    group_affinity_mat=group_affinity_mat,
        #                    pre_group_affinity_mat=pre_group_affinity_mat,
        #                    avg_affinity_mat=avg_affinity_mat,
        #                    affinity_value=affinity_value,
        #                    pred_logits=pred_logits)
        # self.save_all_visualization(results=dict_result)

        return pred_logits.unsqueeze(0)

    def blend_result(self, img, result,  only_label=None, palette=None, out_file=None, opacity=0.5, with_bg=False):
        img = mmcv.imread(img)
        img = img.copy()
        seg = result[0]
        if palette is None:
            palette = self.PALETTE
        palette = np.array(palette)
        assert palette.shape[1] == 3, palette.shape
        assert len(palette.shape) == 2
        assert 0 < opacity <= 1.0
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            if only_label is not None and label != only_label:
                ##print(f"label:{label} is not equal to only_label:{only_label}")
                continue
            color_seg[seg == label, :] = color
        # convert to BGR
        color_seg = color_seg[..., ::-1]

        if with_bg:
            fg_mask = seg != 0
            img[fg_mask] = img[fg_mask] * (1 - opacity) + color_seg[fg_mask] * opacity
        else:
            img = img * (1 - opacity) + color_seg * opacity
        img = img.astype(np.uint8)

        if out_file is not None:
            mmcv.imwrite(img, out_file)

        return img
    #Hacky way to get set output dir
    #Since this class is a sub-class of EncoderDecoder class there are several checks on return object and 
    # argument which are hard to parse. So to set up an assessment pipeline this method helps in setting up several variables
    def set_output_dir(self, output_file_path):
        self.output_dir = output_file_path

    
    def show_result(self, img_show, img_tensor, result, out_file, vis_mode='input'):

        assert vis_mode in [
            'input', 'heatmap','pred', 'input_pred', 'all_groups', 'second_group', 'first_group',
             'final_group', 'input_pred_label', 'input_pred_distinct_labels', 'final_group_pred', 'entropy_map'
        ], vis_mode
        #imgtensor: [B,C,H,W]
        #
        if vis_mode == 'input':
            mmcv.imwrite(img_show, out_file)
        elif vis_mode == 'pred':
            output = Image.fromarray(result[0].astype(np.uint8)).convert('P')
            output.putpalette(np.array(self.PALETTE).astype(np.uint8))
            mmcv.mkdir_or_exist(osp.dirname(out_file))
            output.save(out_file.replace('.jpg', '.png'))
        elif vis_mode == 'heatmap':
            output = Image.fromarray(result[0].astype(np.uint8)).convert('P')
            output.putpalette(np.array(self.PALETTE).astype(np.uint8))
            mmcv.mkdir_or_exist(osp.dirname(out_file))
            display_instances(img_show, result[0], fname= out_file, blur=False)
            
        elif vis_mode == 'input_pred':
            self.blend_result(img=img_show, result=result, out_file=out_file, opacity=0.5, with_bg=self.with_bg)
        elif vis_mode == 'input_pred_label':
            labels = np.unique(result[0])
            coord_map = seg2coord(result[0])
            # reference: https://github.com/open-mmlab/mmdetection/blob/ff9bc39913cb3ff5dde79d3933add7dc2561bab7/mmdet/models/detectors/base.py#L271 # noqa
            blended_img = self.blend_result(
                img=img_show, result=result, out_file=None, opacity=0.5, with_bg=self.with_bg)
            blended_img = mmcv.bgr2rgb(blended_img)
            width, height = img_show.shape[1], img_show.shape[0]
            EPS = 1e-2
            fig = plt.figure(frameon=False)
            canvas = fig.canvas
            dpi = fig.get_dpi()
            fig.set_size_inches((width + EPS) / dpi, (height + EPS) / dpi)

            # remove white edges by set subplot margin
            plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
            ax = plt.gca()
            ax.axis('off')
            for i, label in enumerate(labels):
                if self.with_bg and label == 0:
                    continue
                center = coord_map[label].astype(np.int32)
                label_text = self.CLASSES[label]
                ax.text(
                    center[1],
                    center[0],
                    f'{label_text}',
                    bbox={
                        'facecolor': 'black',
                        'alpha': 0.5,
                        'pad': 0.7,
                        'edgecolor': 'none'
                    },
                    color='orangered',
                    fontsize=16,
                    verticalalignment='top',
                    horizontalalignment='left')
            plt.imshow(blended_img)

            stream, _ = canvas.print_to_buffer()
            buffer = np.frombuffer(stream, dtype='uint8')
            img_rgba = buffer.reshape(height, width, 4)
            rgb, alpha = np.split(img_rgba, [3], axis=2)
            img = rgb.astype('uint8')
            img = mmcv.rgb2bgr(img)
            mmcv.imwrite(img, out_file)
            plt.close()
        elif vis_mode == 'input_pred_distinct_labels':
            labels = np.unique(result[0])
            coord_map = seg2coord(result[0])
            from pathlib import Path
            path = Path(out_file)
            parent_dir = path.parent
            # reference: https://github.com/open-mmlab/mmdetection/blob/ff9bc39913cb3ff5dde79d3933add7dc2561bab7/mmdet/models/detectors/base.py#L271 # noqa
            for i, label in enumerate(labels):
                blended_img = self.blend_result(img=img_show, result=result, only_label=label,
                                                 out_file=None, opacity=0.75, with_bg=self.with_bg)
                blended_img = mmcv.bgr2rgb(blended_img)
                width, height = img_show.shape[1], img_show.shape[0]
                EPS = 1e-2
                fig = plt.figure(frameon=False)
                canvas = fig.canvas
                dpi = fig.get_dpi()
                fig.set_size_inches((width + EPS) / dpi, (height + EPS) / dpi)

                # remove white edges by set subplot margin
                plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
                ax = plt.gca()
                ax.axis('off')
                if self.with_bg and label == 0:
                    continue
                center = coord_map[label].astype(np.int32)
                label_text = self.CLASSES[label]
                ax.text(
                    center[1],
                    center[0],
                    f'{label_text}',
                    bbox={
                        'facecolor': 'black',
                        'alpha': 0.5,
                        'pad': 0.7,
                        'edgecolor': 'none'
                    },
                    color='orangered',
                    fontsize=16,
                    verticalalignment='top',
                    horizontalalignment='left')
                #plt.imshow(blended_img)
                ax.imshow(blended_img)
                stream, _ = canvas.print_to_buffer()
                buffer = np.frombuffer(stream, dtype='uint8')
                img_rgba = buffer.reshape(height, width, 4)
                rgb, alpha = np.split(img_rgba, [3], axis=2)
                img = rgb.astype('uint8')
                img = mmcv.rgb2bgr(img)
                mmcv.imwrite(img, str(parent_dir)+'/'+str(label)+'.jpg')
                plt.close()
        elif vis_mode == 'final_group_pred':
            from pathlib import Path
            path = Path(out_file)
            parent_dir = path.parent
            os.makedirs(parent_dir) if not os.path.exists(parent_dir) else None
            meta_info = self.CLASSES
            np.savetxt(str(parent_dir) +'/labels.txt', np.array(meta_info), delimiter=',', fmt='%s')
            attn_map_list = self.get_attn_maps(img_tensor)
            text_gt_affinity = self.get_text_gt_affinity(img_tensor)
            assert len(attn_map_list) in [1, 2, 3]
            num_groups = [attn_map_list[layer_idx].shape[-1] for layer_idx in range(len(attn_map_list))]
            attn_map = attn_map_list[-1]
            attn_map = rearrange(attn_map, 'b h w g -> b g h w')
            attn_map = F.interpolate(
                    attn_map, size=img_show.shape[:2], mode='bilinear', align_corners=self.align_corners)
            
            pre_attention_map = attn_map.squeeze(0)
            for i in range(pre_attention_map.shape[0]):
                group_atten_map = pre_attention_map[i, :, :]
                labels = np.unique(group_atten_map)
            group_result = attn_map.argmax(dim=1).cpu().numpy()
            patches = np.unique(group_result)
            layer_out_file = out_file
            self.blend_result(img=img_show, result=group_result, 
                    palette=GROUP_PALETTE[sum(num_groups[:1]):sum(num_groups[:2])],
                    out_file=layer_out_file,
                    opacity=0.75)
            coord_map = seg2coord(group_result.squeeze(0))
            # reference: https://github.com/open-mmlab/mmdetection/blob/ff9bc39913cb3ff5dde79d3933add7dc2561bab7/mmdet/models/detectors/base.py#L271 # noqa
            for i, patch in enumerate(patches):
                patch_group_result = group_result.copy()
                patch_group_result[patch_group_result != patch] = 0
                blended_img = self.blend_result(img=img_show, result=patch_group_result, 
                                                palette=GROUP_PALETTE[sum(num_groups[:1]):sum(num_groups[:2])],
                                                out_file=str(parent_dir)+'/'+str(patch)+'.jpg',
                                                opacity=0.75)
                blended_img = mmcv.bgr2rgb(blended_img)
                label_indexes = np.where(text_gt_affinity == patch)[0]
                center = coord_map[patch].astype(np.int32)
                if label_indexes.size != 0:
                    for label in label_indexes:
                        if self.with_bg and label == 0:
                            continue
                        width, height = img_show.shape[1], img_show.shape[0]
                        EPS = 1e-2
                        fig = plt.figure(frameon=False)
                        canvas = fig.canvas
                        dpi = fig.get_dpi()
                        fig.set_size_inches((width + EPS) / dpi, (height + EPS) / dpi)

                        # remove white edges by set subplot margin
                        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
                        ax = plt.gca()
                        ax.axis('off')
                        label_blended_img = blended_img.copy()
                        label_text = self.CLASSES[label]
                        ax.text(
                            center[1],
                            center[0],
                            f'{label_text}',
                            bbox={
                                'facecolor': 'black',
                                'alpha': 0.5,
                                'pad': 0.7,
                                'edgecolor': 'none'
                            },
                            color='orangered',
                            fontsize=16,
                            verticalalignment='top',
                            horizontalalignment='left',
                            wrap=True, )
                        plt.imshow(label_blended_img)
                        stream, _ = canvas.print_to_buffer()
                        buffer = np.frombuffer(stream, dtype='uint8')
                        img_rgba = buffer.reshape(height, width, 4)
                        rgb, alpha = np.split(img_rgba, [3], axis=2)
                        img = rgb.astype('uint8')
                        img = mmcv.rgb2bgr(img)
                        mmcv.imwrite(img, str(parent_dir)+'/'+str(patch)+'_'+str(label)+'.jpg')
                        plt.close()
        elif vis_mode == 'all_groups' or vis_mode == 'final_group' or vis_mode == 'first_group' or vis_mode == 'second_group':
            attn_map_list = self.get_attn_maps(img_tensor)
            assert len(attn_map_list) in [1, 2, 3]
            # only show 16 groups for the first stage
            # if len(attn_map_list) == 1:
            #     attn_map_list[0] = top_groups(attn_map_list[0], k=5)

            num_groups = [attn_map_list[layer_idx].shape[-1] for layer_idx in range(len(attn_map_list))]
            for layer_idx, attn_map in enumerate(attn_map_list):
                if vis_mode == 'first_group' and layer_idx != 0:
                    continue
                if vis_mode == 'second_group' and layer_idx != 1:
                    continue
                if vis_mode == 'final_group' and layer_idx != len(attn_map_list) - 1:
                    continue
                attn_map = rearrange(attn_map, 'b h w g -> b g h w')
                attn_map = F.interpolate(
                    attn_map, size=img_show.shape[:2], mode='bilinear', align_corners=self.align_corners)
                group_result = attn_map.argmax(dim=1).cpu().numpy()
                if vis_mode == 'all_groups':
                    layer_out_file = out_file.replace(
                        osp.splitext(out_file)[-1], f'_layer{layer_idx}{osp.splitext(out_file)[-1]}')
                else:
                    layer_out_file = out_file
                self.blend_result(
                    img=img_show,
                    result=group_result,
                    palette=GROUP_PALETTE[sum(num_groups[:layer_idx]):sum(num_groups[:layer_idx + 1])],
                    out_file=layer_out_file,
                    opacity=0.5)
        elif vis_mode == 'entropy_map':
            attn_map = self.get_attn_maps(img_tensor)[-1]
            print("attention maps", attn_map.shape)
            attn_map = rearrange(attn_map, 'b h w g -> b g h w')
            attn_map = F.interpolate(
                    attn_map, size=img_show.shape[:2], mode='bilinear', align_corners=self.align_corners)
            print("attention maps after interpolation", attn_map.shape)
             
            attn_map = rearrange(attn_map, 'b g h w -> b h w g')
            attn_map = attn_map.squeeze(0)
            print("attention maps after squeeze", attn_map.shape)
            self.save_entropy_maps(attn_map, img_tensor)
        else:
            raise ValueError(f'Unknown vis_type: {vis_mode}')
