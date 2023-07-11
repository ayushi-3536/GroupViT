import torch
from einops import rearrange
import PIL.Image
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from typing import List, Tuple
import torch.nn.functional as F
from sklearn.cluster import KMeans

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


def perform_clustering(features, n_clusters=10):
    # Normalize features
    features = F.normalize(features, p=2, dim=1)
    # Convert the features to float32
    features = features.cpu().detach().numpy().astype('float32')
    # Initialize a k-means clustering index with the desired number of clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    # Train the k-means index with the features
    kmeans.fit(features)
    # Assign the features to their nearest cluster
    labels = kmeans.predict(features)
    return labels

# Load the saved tensor
prob_tensor = torch.load('/misc/student/sharmaa/groupvit/GroupViT/saved_soft_attn.pt', map_location=torch.device('cpu'))
print("prob_tensor.shape", prob_tensor.shape)

#save the tensor in a csv file
np.savetxt("/misc/student/sharmaa/groupvit/GroupViT/saved_soft_attn.csv", prob_tensor, delimiter=",")
print("prob_tensor", prob_tensor)
# Get the indices of the maximum values along the last dimension
probability_across_patch = F.softmax(prob_tensor, dim=-1)
max_indices = torch.argmax(probability_across_patch, dim=-1)
print("max_indices.shape", max_indices.shape)
print("max_indices", max_indices)

# Convert the indices to one-hot vectors
onehot_prob_tensor = F.one_hot(max_indices, num_classes=prob_tensor.shape[0]).to(dtype=prob_tensor.dtype)

# Print the one-hot encoded matrix
print(onehot_prob_tensor)

updated_prob = prob_tensor * onehot_prob_tensor.T
print("updated_prob.shape", updated_prob.shape)
print("updated_prob", updated_prob)
np.savetxt("/misc/student/sharmaa/groupvit/GroupViT/updated_prob.csv", updated_prob, delimiter=",", fmt='%.1f')

#onehot_prob_tensor = max_values, _ = torch.max(prob_tensor, dim=-1) #F.one_hot(prob_tensor.argmax(dim=-1)).to(dtype=prob_tensor.dtype)
print("onehot_prob_tensor.shape", onehot_prob_tensor.shape)
print("onehot_prob_tensor", onehot_prob_tensor)
np.savetxt("/misc/student/sharmaa/groupvit/GroupViT/onehot.csv", onehot_prob_tensor.T, delimiter=",", fmt='%.1f')


#dino distance matrix
dino_dist_mat = torch.load('/misc/student/sharmaa/groupvit/GroupViT/saved_gvit_cosine.pt', map_location=torch.device('cpu'))
print("dino distance matrix", dino_dist_mat)
print("max indices", max_indices.shape)
temp_mi = max_indices[:,None]
print("temp_mi", temp_mi.shape)

temp_mi2 = max_indices[None,:]
print("temp_mi2", temp_mi2.shape)
print("comp",(max_indices[:, None] != max_indices[None, :]).shape)
dino_dist_mat[max_indices[:, None] != max_indices[None, :]] = 0
# Print the loaded tensor
print("dino distance matrix after", dino_dist_mat)
np.savetxt("/misc/student/sharmaa/groupvit/GroupViT/dinodist.csv", dino_dist_mat, delimiter=",", fmt='%.1f')
 
print("dino distance matrix", dino_dist_mat.shape)
#b,g,p 
print("updated_prob", updated_prob.shape)
print(np.add(updated_prob.numpy(),dim=-1))
#loss = F.softmax(updated_prob,dim=-1) @ dino_dist_mat
print("updated_prob", updated_prob)
loss = updated_prob @ dino_dist_mat
np.savetxt("/misc/student/sharmaa/groupvit/GroupViT/loss.csv", loss, delimiter=",", fmt='%.1f')

print("loss.shape", loss.shape)
#sum the loss
loss = torch.mean(loss) 
print("loss", loss)
batch_loss = loss*256
print("loss", batch_loss)




