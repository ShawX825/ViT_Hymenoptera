import typing
import io
import os

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms

from models.modeling import VisionTransformer, CONFIGS
from utils.data_utils import HymenopteraDataset
# Test Image

transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
testset = HymenopteraDataset(transform=None, train=False)
imagenet_labels = {0:'Ants', 1:'Bees'}


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Prepare Model
config = CONFIGS["ViT-B_16"]
model = VisionTransformer(config, num_classes=2, zero_head=False, img_size=224, vis=True)
# model.load_from(np.load("attention_data/ViT-B_16-224.npz"))
checkpoint = torch.load('output/ants_bees_for_attn_checkpoint.bin',map_location=device)
model.to(device)
model.load_state_dict(checkpoint)
model.eval()

# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
# ])
# im = Image.open("attention_data/img.jpg")
# x = transform(im)
# idx = 0
# im, label = testset[idx]
for ii, (im, label) in enumerate(testset):
    x = transform_test(im)
    logits, att_mat = model(x.unsqueeze(0).to(device))

    att_mat = torch.stack(att_mat).squeeze(1)

    # Average the attention weights across all heads.
    att_mat = torch.mean(att_mat, dim=1).detach().cpu()
    logits = logits.detach().cpu()
    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = torch.eye(att_mat.size(1))
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size())
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])
        
    # Attention from the output token to the input space.
    v = joint_attentions[-1]
    grid_size = int(np.sqrt(aug_att_mat.size(-1)))
    mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
    mask = cv2.resize(mask / mask.max(), im.size)[..., np.newaxis]
    result = (mask * im).astype("uint8")

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))

    ax1.set_title('Original')
    ax2.set_title('Attention Map')
    _ = ax1.imshow(im)
    _ = ax2.imshow(result)



    probs = torch.nn.Softmax(dim=-1)(logits)
    top = torch.argsort(probs, dim=-1, descending=True)
    # print("Prediction Label and Attention Map!\n")
    # for idx in top[0]:
    #     print(f'{probs[0, idx.item()]:.5f} : {imagenet_labels[idx.item()]}', end='\n')
    txt = "Label: {}. Prediction: {}, with {} confidence".format(
            imagenet_labels[label],
            imagenet_labels[top[0][0].item()],
            round(probs[0, top[0][0].item()].item(),5)
            )
    fig.suptitle(txt, fontsize=12)
    plt.savefig("img/attn/{}_{}".format(label,ii),bbox_inches='tight',dpi=fig.dpi)