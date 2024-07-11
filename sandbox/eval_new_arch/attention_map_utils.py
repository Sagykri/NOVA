from sandbox.eval_new_arch.dino4cells.utils.extractor import ViTExtractor
import cv2
from sklearn.decomposition import PCA
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt 
%matplotlib inline

from sklearn.ensemble import RandomForestClassifier
import matplotlib.cm as cm

from sklearn.feature_selection import mutual_info_classif

def get_attn_maps(extractor, images, lbls=None, pc_indx_compare=[0,1], reducer=None, n_components=10):
    attn_maps = []
    
    features_processed = extractor.extract_descriptors(images.to('cuda')).detach().cpu().numpy()[:,0,...]
    
    attention_maps_flat = np.concatenate(features_processed)
    
    print(f"features_processed = {features_processed.shape}")
    
    
    
    
    if reducer is not None:
        # reducer = PCA(n_components=n_components, whiten=True)
        print(f"reducer: {reducer}")
        pca_descs_clean = reducer.fit_transform(attention_maps_flat)

        if lbls is not None:
            last = 0
            for i, l in enumerate(np.unique(lbls)):
                current = len(np.where(lbls==l)*features_processed.shape[1])
                indx = np.arange(i*last, i*last +  current)
                plt.scatter(pca_descs_clean[indx,pc_indx_compare[0]], pca_descs_clean[indx,pc_indx_compare[1]])
                last = current
            plt.show()
            
        print(f"pca_descs_clean: {pca_descs_clean.shape}")
        pca_descs_clean = pca_descs_clean.reshape(features_processed.shape[0], features_processed.shape[1], n_components)

        attention_maps_flat = np.concatenate(pca_descs_clean)
    # else:
    #     attention_maps_flat = np.concatenate(features_processed)

    ####################
    
    ############################### Fisher's Scores #########################
    
#     print(f"features_processed = {features_processed.shape}")
    
#     def get_labels():
#         labels = []
#         for i, l in enumerate(np.unique(lbls)):
#             l_len = len(np.where(lbls==l)[0])*features_processed.shape[1]
#             labels.extend([l]*l_len)
#         labels = np.asarray(labels)
#         return labels
    
#     print(f"attention_maps_flat = {attention_maps_flat.shape}")
#     full_labels = get_labels()
#     print(f"full_labels = {len(full_labels)}")
#     keys_class1 = attention_maps_flat[np.where(full_labels==0)[0]]
#     keys_class2 = attention_maps_flat[np.where(full_labels==1)[0]]
    
#     print(f"keys_class1={keys_class1.shape}")
    
#     # Calculate mean and variance for each class
#     mean_class1 = np.mean(keys_class1, axis=0)
#     mean_class2 = np.mean(keys_class2, axis=0)

#     variance_class1 = np.var(keys_class1, axis=0)
#     variance_class2 = np.var(keys_class2, axis=0)

#     # Calculate Fisher's score for each feature
#     fisher_scores = (mean_class1 - mean_class2) ** 2 / (variance_class1 + variance_class2)
#     print(fisher_scores.shape)
#     top_features_indices = np.argsort(fisher_scores)[-n_components:]
#     print(fisher_scores[top_features_indices])
#     print(f"top_features_indices={top_features_indices}")
    
#     # Select top features from the original keys
#     keys_class1_reduced = keys_class1[:, top_features_indices]
#     keys_class2_reduced = keys_class2[:, top_features_indices]
#     print(f"keys_class2_reduced = {keys_class2_reduced.shape}")
    
#     attention_maps_flat = np.concatenate([keys_class1_reduced, keys_class2_reduced])
#     print(f"attention_maps_flat = {attention_maps_flat.shape}")

    ############################### Mutual Information #########################
    
#     def get_labels():
#         labels = []
#         for i, l in enumerate(np.unique(lbls)):
#             l_len = len(np.where(lbls==l)[0])*features_processed.shape[1]
#             labels.extend([l]*l_len)
#         labels = np.asarray(labels)
#         return labels
    
#     full_labels = get_labels()

    
#     # Calculate mutual information
#     mutual_info = mutual_info_classif(attention_maps_flat, full_labels)

    
#     top_features_indices = np.argsort(mutual_info)[-n_components:]
#     print(mutual_info[top_features_indices])
   
#     # print(f"top_features_indices={top_features_indices}")
    
#     # Select top features from the original keys
# #     keys_class1_reduced = keys_class1[:, top_features_indices]
# #     keys_class2_reduced = keys_class2[:, top_features_indices]
# #     print(f"keys_class2_reduced = {keys_class2_reduced.shape}")
    
# #     attention_maps_flat = np.concatenate([keys_class1_reduced, keys_class2_reduced])
#     attention_maps_flat = attention_maps_flat[..., top_features_indices]
    # print(f"attention_maps_flat = {attention_maps_flat.shape}")

    ############################### RandomForest #########################
    
    # Train a Random Forest classifier
    clf = RandomForestClassifier(oob_score=True, random_state=42)
    lbls_formatted = []
    for i, l in enumerate(np.unique(lbls)):
        n = len(np.where(lbls==l)[0])*features_processed.shape[1]
        lbls_formatted.extend([l] * n)
        
    print(len(lbls_formatted), features_processed.shape)
    clf.fit(attention_maps_flat, lbls_formatted)
    
    print(f"oob score: {clf.oob_score_}")
    
    clf_for_pc_indx_compared = RandomForestClassifier(oob_score=True, random_state=42)
    clf_for_pc_indx_compared.fit(attention_maps_flat[...,pc_indx_compare], lbls_formatted)
    
    print(f"oob score for pc {pc_indx_compare}: {clf_for_pc_indx_compared.oob_score_}")
    
    
    # Get feature importances
    feature_importances = clf.feature_importances_
    sorted_indices = np.argsort(feature_importances)[::-1]
    top_k_indices = sorted_indices[:n_components]
    print(f"top {n_components} indices: {top_k_indices}")
    print(f"feature_importances in indices: {feature_importances[top_k_indices]}")
    
    
    attention_maps_flat = attention_maps_flat[:, top_k_indices]
    
    ########################################################################
    
    
    
    individual_pca_descs_clean = attention_maps_flat.reshape(features_processed.shape[0], features_processed.shape[1], -1)
    print(f"individual_pca_descs_clean shape: {individual_pca_descs_clean.shape}")
    
    ###################
    
    # individual_pca_descs_clean = pca_descs_clean.reshape(features_processed.shape[0], features_processed.shape[1], n_components)
    
    row_size, col_size = (int(features_processed.shape[1] ** 0.5),)*2
    
    for i in range(len(features_processed)):
        desc_img = individual_pca_descs_clean[i]
        desc_clean = np.stack([cv2.resize(desc_img[:,c].reshape(row_size, col_size), dsize=(100, 100), interpolation=cv2.INTER_CUBIC) for c in range(n_components)]).transpose(1,2,0)
        attn_maps.append(desc_clean)
    
    attn_maps = np.stack(attn_maps)
    
    # print(f"pca explained_variance_ratio_: {pca.explained_variance_ratio_}")
    # print(f"pca total explained: {np.sum(pca.explained_variance_ratio_)}%")
    
    return attn_maps


def get_extractor(model):
    extractor = ViTExtractor(
        model_type='vit_base_patch14_100',
        model=model,
        stride=7,
        device='cuda',
        load_size=100,
    )
    
    return extractor



def plt_attn_map(img, attn_map, pc_indx=0, alpha=0.7, img_cmap='gray', attn_cmap=None, overlay=False):
    if attn_cmap is None:        
        # Number of colors in the colormap
#         N = 256

#         # Get the rainbow colormap
#         rainbow_colors = cm.rainbow(np.linspace(0, 1, N))

#         # Add alpha values
#         alpha = np.linspace(0, 1, N)
#         rainbow_colors[:, 3] = alpha

#         # Create the LinearSegmentedColormap
#         attn_cmap = LinearSegmentedColormap.from_list('transparent_to_rainbow', rainbow_colors, N=N)

        
        attn_cmap = LinearSegmentedColormap.from_list('reds', [(0, 0, 0, 0), (1, 0, 0, 1)], N=256)

    
    # new_desc_clean = np.copy(attn_map)
        
    attn_map_min = attn_map.min(axis=(0,1))
    attn_map_max = attn_map.max(axis=(0,1))
    
    # new_desc_clean -= new_desc_clean.min(axis=(0,1))
    # new_desc_clean /= new_desc_clean.max(axis=(0,1))
    attn_map -= attn_map_min
    attn_map /= (attn_map_max - attn_map_min)
    
    if not overlay:
        fig, ax = plt.subplots(1,3)
        for i in range(2):
            ax[i].imshow(img[i,...], cmap=img_cmap)
            ax[i].axis('off')
        ax[2].imshow(attn_map[...,pc_indx], alpha=alpha, cmap=attn_cmap)
        ax[2].axis('off')
        plt.show()
    else:
        fig, ax = plt.subplots(1,4, figsize=(15,15))
        for i in range(4):
            ax[i].imshow(img[i%2,...], cmap=img_cmap)
            if i < 2:
                ax[i].imshow(attn_map[...,pc_indx], alpha=alpha, cmap=attn_cmap)
            ax[i].axis('off')
        plt.show()
    
def plt_attn_maps(imgs, attn_maps, pc_indxs=[0], alpha=0.7, img_cmap='gray', attn_cmap=None, overlay=False):
    n = len(imgs)
    for i in range(n):
        print(f"img: {i}")
        for pc in pc_indxs:
            print(f"pc={pc}")
            plt_attn_map(imgs[i], attn_maps[i], pc_indx=pc, alpha=alpha, img_cmap=img_cmap, attn_cmap=attn_cmap, overlay=overlay)