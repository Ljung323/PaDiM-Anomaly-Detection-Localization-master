import time
import random
from random import sample
import numpy as np
import pickle
from PIL import Image
from collections import OrderedDict
from torchvision import transforms as T
# from scipy.spatial.distance import mahalanobis
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torchvision.models import resnet18

def mahalanobis(u, v, VI):
    delta = u - v
    test = np.dot(delta, VI)
    m = np.dot(test, delta)
    return np.sqrt(m)

def main():
    # load model
    model = resnet18(pretrained=True, progress=True)
    t_d = 448
    d = 100

    model.to('cpu')
    model.eval()
    random.seed(1024)
    torch.manual_seed(1024)

    idx = torch.tensor(sample(range(0, t_d), d))

    # set model's intermediate outputs
    outputs = []

    def hook(module, input, output):
        outputs.append(output)

    model.layer1[-1].register_forward_hook(hook)
    model.layer2[-1].register_forward_hook(hook)
    model.layer3[-1].register_forward_hook(hook)

    test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])

    # extract train set features
    # train_feature_filepath = '/Users/erika/PaDiM-Anomaly-Detection-Localization-master/mvtec_result/temp_resnet18/train_bottle.pkl'
    # with open(train_feature_filepath, 'rb') as f:
    #     train_outputs = pickle.load(f)

    with open('/Users/erika/PaDiM-Anomaly-Detection-Localization-master/mean.pkl', 'rb') as f:
        loaded_mean = pickle.load(f)
    with open('/Users/erika/PaDiM-Anomaly-Detection-Localization-master/cov_inv.pkl', 'rb') as f:
        loaded_conv = pickle.load(f)


    x = Image.open("/Users/erika/Desktop/004のコピー.png").convert('RGB')
    start_time = time.time()
    transform = T.Compose([T.Resize(256, Image.ANTIALIAS),
                                      T.CenterCrop(224),
                                      T.ToTensor(),
                                      T.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])])
                                        
    x = transform(x)
    
    # model prediction
    with torch.no_grad():
        _ = model(x.unsqueeze(0).to('cpu'))

    # get intermediate layer outputs
    for k, v in zip(test_outputs.keys(), outputs):
        test_outputs[k].append(v.cpu().detach())

    # initialize hook outputs
    outputs = []
    for k, v in test_outputs.items():
        test_outputs[k] = torch.cat(v, 0)
    
    # Embedding concat
    embedding_vectors = test_outputs['layer1']

    for layer_name in ['layer2', 'layer3']:
        embedding_vectors = embedding_concat(embedding_vectors, test_outputs[layer_name])

    # randomly select d dimension
    embedding_vectors = torch.index_select(embedding_vectors, 1, idx)

    # calculate distance matrix
    B, C, H, W = embedding_vectors.size()
    embedding_vectors = embedding_vectors.view(B, C, H * W).numpy()
    dist_list = []

    for i in range(H * W):
        mean = loaded_mean[:, i]
        cov_inv = loaded_conv[:, :, i]
        dist = [mahalanobis(sample[:, i], mean.astype(np.float32), cov_inv.astype(np.float32)) for sample in embedding_vectors.astype(np.float32)]

        # print(dist)
        dist_list.append(dist)

    dist_list = np.array(dist_list).transpose(1, 0).reshape(B, H, W)

    # upsample
    dist_list = torch.tensor(dist_list)

    score_map = F.interpolate(dist_list.unsqueeze(1), size=x.size(2), mode='bilinear',
                                align_corners=False).squeeze().numpy()
    
    # apply gaussian smoothing on the score map
    # for i in range(score_map.shape[0]):
    #     score_map[i] = gaussian_filter(score_map[i], sigma=4)
    
    # Normalization
    max_score = score_map.max()
    min_score = score_map.min()
    scores = (score_map - min_score) / (max_score - min_score)

    print("---------")
    print(f"inference time: {time.time() - start_time} seconds")
    print("---------")

    plt.imshow(scores * 255)

    plt.show()


def embedding_concat(x, y):
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)

    return z


if __name__ == '__main__':
    main()
