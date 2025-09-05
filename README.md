# ✨CRM: Context-aware refinement module

### This repo provide training code for self-supervised depth estimation: dealing with transparent objects especially glass doors

This repo is the official project repository of the paper **Knowledge Distillation with Context-aware Refinement for Self-Supervised Depth Estimation in Transparent Environments** and is mainly used for providing training, inference code and visualization demo.,<br>
[ **CRM** ] - [ [Paper](**in submission now**😊) ]

<div align='left'>
<img src="https://github.com/Jmyeong/CRM/blob/main/images/Overview.png" alt="overview" width="800" />
</div>

## Overview
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Citation](#citation)

## Installation
-  
  ```bash
  pip install -r requirements.txt
  ```

## Preparation
-  
Before start our code, you should prepare mask images of the datasets.
We use Grounded-SAM (https://github.com/IDEA-Research/Grounded-Segment-Anything)
 ```bash

 ```

## Quick Start
***Let's first begin with some simple visualization demo with Sonata, our pre-trained PTv3 model:***
- **Visualization.** We provide the similarity heatmap and PCA visualization demo in the `demo` folder. You can run the following command to visualize the result:
  ```bash
  export PYTHONPATH=./
  python demo/0_pca.py
  python demo/1_similarity.py
  python demo/2_sem_seg.py  # linear probed head on ScanNet
  ```

<div align='left'>
<img src="https://raw.githubusercontent.com/pointcept/assets/main/sonata/demo.png" alt="teaser" width="800" />
</div>

***Then, here are the instruction to run inference on custom data with our Sonata:***

- **Data.** Organize your data in a dictionary with the following format:
  ```python
  # single point cloud
  point = {
    "coord": numpy.array,  # (N, 3)
    "color": numpy.array,  # (N, 3)
    "normal": numpy.array,  # (N, 3)
    "segment": numpy.array,  # (N,) optional
  }

  # batched point clouds

  # check the data structure of batched point clouds from here:
  # https://github.com/Pointcept/Pointcept#offset
  point = {
    "coord": numpy.array,  # (N, 3)
    "color": numpy.array,  # (N, 3)
    "normal": numpy.array,  # (N, 3)
    "batch": numpy.array,  # (N,) optional
    "segment": numpy.array,  # (N,) optional
  }
  ```
  One example of the data can be loaded by running the following command:
  ```python
  point = sonata.data.load("sample1")
  ```
- **Transform.** The data transform pipeline is shared as the one used in Pointcept codebase. You can use the following code to construct the transform pipeline:
  ```python
  config = [
      dict(type="CenterShift", apply_z=True),
      dict(
          type="GridSample",
          grid_size=0.02,
          hash_type="fnv",
          mode="train",
          return_grid_coord=True,
          return_inverse=True,
      ),
      dict(type="NormalizeColor"),
      dict(type="ToTensor"),
      dict(
          type="Collect",
          keys=("coord", "grid_coord", "color", "inverse"),
          feat_keys=("coord", "color", "normal"),
      ),
  ]
  transform = sonata.transform.Compose(config)
  ```
  The above default inference augmentation pipeline can also be acquired by running the following command:
  ```python
  transform = sonata.transform.default()
  ```
- **Model.** Load the pre-trained model by running the following command:
  ```python
  # Load the pre-trained model from Huggingface
  # supported models: "sonata"
  # ckpt is cached in ~/.cache/sonata/ckpt, and the path can be customized by setting 'download_root'
  model = sonata.model.load("sonata", repo_id="facebook/sonata").cuda()

  # or
  from sonata.model import PointTransformerV3
  model = PointTransformerV3.from_pretrained("facebook/sonata").cuda()

  # Load the pre-trained model from local path
  # assume the ckpt file is stored in the 'ckpt' folder
  model = sonata.model.load("ckpt/sonata.pth").cuda()

  # the ckpt file store the config and state_dict of pretrained model
  ```
  If *FlashAttention* is not available, load the pre-trained model with the following code:
  ```python
  custom_config = dict(
      enc_patch_size=[1024 for _ in range(5)],
      enable_flash=False,  # reduce patch size if necessary
  )
  model = sonata.load("sonata", repo_id="facebook/sonata", custom_config=custom_config).cuda()
  # or
  from sonata.model import PointTransformerV3
  model = PointTransformerV3.from_pretrained("facebook/sonata", **custom_config).cuda()
  ```
- **Inference.** Run the inference by running the following command:
  ```python
  point = transform(point)
  for key in point.keys():
      if isinstance(point[key], torch.Tensor):
          point[key] = point[key].cuda(non_blocking=True)
  point = model(point)
  ```
  As Sonata is a pre-trained **encoder-only** PTv3, the default output of the model is point cloud after hierarchical encoding. The encoded point feature can be mapping back to original scale with the following code:
  ```python
  for _ in range(2):
      assert "pooling_parent" in point.keys()
      assert "pooling_inverse" in point.keys()
      parent = point.pop("pooling_parent")
      inverse = point.pop("pooling_inverse")
      parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
      point = parent
  while "pooling_parent" in point.keys():
      assert "pooling_inverse" in point.keys()
      parent = point.pop("pooling_parent")
      inverse = point.pop("pooling_inverse")
      parent.feat = point.feat[inverse]
      point = parent
  ```
  Yet during data transformation, we operate `GridSampling` which makes the number of points feed into the network mismatch with the original point cloud. Using the following code to further map the feature back to the original point cloud:
  ```python
  feat = point.feat[point.inverse]
  ```

## Citation
If you find _Sonata_ useful to your research, please consider citing our works as an acknowledgment. (੭ˊ꒳​ˋ)੭✧
```bib
@inproceedings{wu2025sonata,
    title={Sonata: Self-Supervised Learning of Reliable Point Representations},
    author={Wu, Xiaoyang and DeTone, Daniel and Frost, Duncan and Shen, Tianwei and Xie, Chris and Yang, Nan and Engel, Jakob and Newcombe, Richard and Zhao, Hengshuang and Straub, Julian},
    booktitle={CVPR},
    year={2025}
}
```

```bib
@inproceedings{wu2024ptv3,
    title={Point Transformer V3: Simpler, Faster, Stronger},
    author={Wu, Xiaoyang and Jiang, Li and Wang, Peng-Shuai and Liu, Zhijian and Liu, Xihui and Qiao, Yu and Ouyang, Wanli and He, Tong and Zhao, Hengshuang},
    booktitle={CVPR},
    year={2024}
}
```
```bib
@inproceedings{wu2024ppt,
    title={Towards Large-scale 3D Representation Learning with Multi-dataset Point Prompt Training},
    author={Wu, Xiaoyang and Tian, Zhuotao and Wen, Xin and Peng, Bohao and Liu, Xihui and Yu, Kaicheng and Zhao, Hengshuang},
    booktitle={CVPR},
    year={2024}
}
```
```bib
@inproceedings{wu2023masked,
  title={Masked Scene Contrast: A Scalable Framework for Unsupervised 3D Representation Learning},
  author={Wu, Xiaoyang and Wen, Xin and Liu, Xihui and Zhao, Hengshuang},
  journal={CVPR},
  year={2023}
}
```
```bib
@inproceedings{wu2022ptv2,
    title={Point transformer V2: Grouped Vector Attention and Partition-based Pooling},
    author={Wu, Xiaoyang and Lao, Yixing and Jiang, Li and Liu, Xihui and Zhao, Hengshuang},
    booktitle={NeurIPS},
    year={2022}
}
```
```bib
@misc{pointcept2023,
    title={Pointcept: A Codebase for Point Cloud Perception Research},
    author={Pointcept Contributors},
    howpublished={\url{https://github.com/Pointcept/Pointcept}},
    year={2023}
}
```

## How to Contribute

We welcome contributions! Go to [CONTRIBUTING](./.github/CONTRIBUTING.md) and
our [CODE OF CONDUCT](./.github/CODE_OF_CONDUCT.md) for how to get started.

## License

- Sonata code is released by Meta under the [Apache 2.0 license](LICENSE);
- Sonata weight is released under the [CC-BY-NC 4.0 license](https://creativecommons.org/licenses/by-nc/4.0/deed.en) (restricted by NC of datasets like HM3D, ArkitScenes).