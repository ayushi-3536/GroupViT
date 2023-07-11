import torch
from einops import rearrange
def co_pca_dino_gvit(features, target_dim=384):
    
    batch_size = features.shape[0]
    spatial = features.shape[1]
    #reshape the tensor to [256*196, 768]
    features = features.reshape(-1, features.shape[-1])
    features = features.unsqueeze(0)
    print("reshaped_features.shape", features.shape)

    mean = torch.mean(features[0], dim=0, keepdim=True)
    print("mean.shape", mean.shape)
    centered_features = features[0] - mean
    print("centered_features.shape", centered_features.shape)
    U, S, V = torch.pca_lowrank(centered_features, q=384)
    print("U.shape", U.shape)
    print("S.shape", S.shape)
    print("V.shape", V.shape)

    reduced_features = torch.matmul(centered_features, V[:, :384]) # (t_x+t_y)x(d)

    print("reduced_features.shape", reduced_features.shape)
    features = reduced_features.unsqueeze(0).permute(0, 2, 1) # Bx(d)x(t_x+t_y)
    print("features.shape", features.shape)

    features = features.squeeze(0)
    print("features.shape", features.shape)

    #use reaarange to reshape the tensor [d, n] back to [n, d]
    features = rearrange(features, 'n d -> d n')
    print("reshaped_features.shape", features.shape)


    #use reaarange to reshape the tensor  [n, d] to [batch_size, spatial_size, d] where n = batch_size*spatial_size
    features = rearrange(features, '(b s)d -> b s d', b=batch_size, s=spatial)
    print("reshaped_features.shape", features.shape)


# #randomly generate a tensor of size [256, 196, 784]

features = torch.rand(2, 196, 768)
#print floating point precision and floating point data type
print("features.dtype", features.dtype)
#chnage the data type to HalfTensor
features = features.half()
print("features.dtype", features.dtype)
#Change the data type to float32
#features = features.float()
print("features.dtype", features.dtype)
co_pca_dino_gvit(features)


#use rearrange to reshape the tensor back to its original shape


#concatenate features such as 256, 196, 768 become 256*192, 768

#co_pca_dino_gvit(features)

# import numpy as np
# from sklearn.decomposition import PCA

# # Generate a random tensor of size [256, 196, 768]
# tensor_size = (256, 196, 768)
# tensor = np.random.random(tensor_size)

# # Reshape the tensor to [256*196, 768]
# reshaped_tensor = tensor.reshape(-1, tensor_size[-1])

# # Perform PCA to reduce dimensionality
# n_components = tensor_size[-1] - tensor_size[-1]//2  # Keep half the original dimensions
# pca = PCA(n_components=n_components)
# transformed_tensor = pca.fit_transform(reshaped_tensor)

# # Inverse transform to get the reduced tensor
# reduced_tensor = pca.inverse_transform(transformed_tensor)

# # Reshape the reduced tensor back to its original shape
# reduced_tensor = reduced_tensor.reshape(tensor_size[:-1] + (n_components,))

# # Print the shape of the reduced tensor
# print("Reduced tensor shape:", reduced_tensor.shape)
