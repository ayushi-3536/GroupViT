import torch
from einops import rearrange
import PIL.Image
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA

from typing import List, Tuple

# def pca(image_paths, load_size: int = 224, layer: int = 11, facet: str = 'key', bin: bool = False, stride: int = 4,
#         model_type: str = 'dino_vits8', n_components: int = 4,
#         all_together: bool = True) -> List[Tuple[Image.Image, np.ndarray]]:
#     """
#     finding pca of a set of images.
#     :param image_paths: a list of paths of all the images.
#     :param load_size: size of the smaller edge of loaded images. If None, does not resize.
#     :param layer: layer to extract descriptors from.
#     :param facet: facet to extract descriptors from.
#     :param bin: if True use a log-binning descriptor.
#     :param model_type: type of model to extract descriptors from.
#     :param stride: stride of the model.
#     :param n_components: number of pca components to produce.
#     :param all_together: if true apply pca on all images together.
#     :return: a list of lists containing an image and its principal components.
#     """
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     descriptors_list = []
#     image_pil_list = []
#     num_patches_list = []
#     load_size_list = []

#     # extract descriptors and saliency maps for each image
#     for image_path in image_paths:
#         image_batch, image_pil = extractor.preprocess(image_path, load_size)
#         image_pil_list.append(image_pil)
#         descs = extractor.extract_descriptors(image_batch.to(device), layer, facet, bin, include_cls=False).cpu().numpy()
#         curr_num_patches, curr_load_size = extractor.num_patches, extractor.load_size
#         num_patches_list.append(curr_num_patches)
#         load_size_list.append(curr_load_size)
#         descriptors_list.append(descs)
#     if all_together:
#         descriptors = np.concatenate(descriptors_list, axis=2)[0, 0]
#         pca = PCA(n_components=n_components).fit(descriptors)
#         pca_descriptors = pca.transform(descriptors)
#         split_idxs = np.array([num_patches[0] * num_patches[1] for num_patches in num_patches_list])
#         split_idxs = np.cumsum(split_idxs)
#         pca_per_image = np.split(pca_descriptors, split_idxs[:-1], axis=0)
#     else:
#         pca_per_image = []
#         for descriptors in descriptors_list:
#             pca = PCA(n_components=n_components).fit(descriptors[0, 0])
#             pca_descriptors = pca.transform(descriptors[0, 0])
#             pca_per_image.append(pca_descriptors)
#     results = [(pil_image, img_pca.reshape((num_patches[0], num_patches[1], n_components))) for
#                (pil_image, img_pca, num_patches) in zip(image_pil_list, pca_per_image, num_patches_list)]
#     return results

# def plot_pca(pil_image: Image.Image, pca_image: np.ndarray, save_dir: str, last_components_rgb: bool = True,
#              save_resized=True, save_prefix: str = ''):
#     """
#     finding pca of a set of images.
#     :param pil_image: The original PIL image.
#     :param pca_image: A numpy tensor containing pca components of the image. HxWxn_components
#     :param save_dir: if None than show results.
#     :param last_components_rgb: If true save last 3 components as RGB image in addition to each component separately.
#     :param save_resized: If true save PCA components resized to original resolution.
#     :param save_prefix: optional. prefix to saving
#     :return: a list of lists containing an image and its principal components.
#     """
#     save_dir = Path(save_dir)
#     save_dir.mkdir(exist_ok=True, parents=True)
#     pil_image_path = save_dir / f'{save_prefix}_orig_img.png'
#     pil_image.save(pil_image_path)

#     n_components = pca_image.shape[2]
#     for comp_idx in range(n_components):
#         comp = pca_image[:, :, comp_idx]
#         comp_min = comp.min(axis=(0, 1))
#         comp_max = comp.max(axis=(0, 1))
#         comp_img = (comp - comp_min) / (comp_max - comp_min)
#         comp_file_path = save_dir / f'{save_prefix}_{comp_idx}.png'
#         pca_pil = Image.fromarray((comp_img * 255).astype(np.uint8))
#         if save_resized:
#             pca_pil = pca_pil.resize(pil_image.size, resample=PIL.Image.NEAREST)
#         pca_pil.save(comp_file_path)

#     if last_components_rgb:
#         comp_idxs = f"{n_components-3}_{n_components-2}_{n_components-1}"
#         comp = pca_image[:, :, -3:]
#         comp_min = comp.min(axis=(0, 1))
#         comp_max = comp.max(axis=(0, 1))
#         comp_img = (comp - comp_min) / (comp_max - comp_min)
#         comp_file_path = save_dir / f'{save_prefix}_{comp_idxs}_rgb.png'
#         pca_pil = Image.fromarray((comp_img * 255).astype(np.uint8))
#         if save_resized:
#             pca_pil = pca_pil.resize(pil_image.size, resample=PIL.Image.NEAREST)
#         pca_pil.save(comp_file_path)


def plot_heatmap(matrix):
    plt.imshow(matrix, cmap='jet', interpolation='nearest')
    plt.colorbar()
    plt.title("Cosine Similarity Matrix")
    return plt

def chunk_cosine_sim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """ Computes cosine similarity between all possible pairs in two sets of vectors.
    Operates on chunks so no large amount of GPU RAM is required.
    :param x: an tensor of descriptors of shape Bx1x(t_x)xd' where d' is the dimensionality of the descriptors and t_x
    is the number of tokens in x.
    :param y: a tensor of descriptors of shape Bx1x(t_y)xd' where d' is the dimensionality of the descriptors and t_y
    is the number of tokens in y.
    :return: cosine similarity between all descriptors in x and all descriptors in y. Has shape of Bx1x(t_x)x(t_y) """
    result_list = []
    num_token_x = x.shape[1]
    for token_idx in range(num_token_x):
        token = x[:,token_idx, :].unsqueeze(dim=1)  # Bx1x1xd'
        print("token.shape", token.shape)
        print("y.shape", y.shape)
        result_list.append(torch.nn.CosineSimilarity(dim=2)(token, y))  # Bx1xt
    return torch.stack(result_list, dim=2)  # Bx1x(t_x)x(t_y)

gvit_feat = torch.rand(1, 196, 384)
dino_feat = torch.rand(1, 196, 384)

cosine_simmat  = chunk_cosine_sim(gvit_feat, dino_feat)
print("cosine_simmat.shape", cosine_simmat.shape)

#visualize the cosine similarity matrix as heatmap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
cosine_simmat = cosine_simmat.squeeze(0)
print("cosine_simmat.shape", cosine_simmat.shape)
cosine_simmat = cosine_simmat.cpu().numpy()
plt = plot_heatmap(cosine_simmat)
#save the figure
plt.savefig("cosine_similarity_matrix.png")
cosine_simmat = cosine_simmat[0,:]
cosine_simmat = cosine_simmat.reshape((14,14))
#print("after reshaping", curr_similarities.shape)
plt.imshow(cosine_simmat, cmap='jet')
plt.savefig('similarity1.png')

print("gvit features.dtype", gvit_feat.dtype)
print("dino features.dtype", dino_feat.dtype)

