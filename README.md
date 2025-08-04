<div align="center">
  
# FreeSplatter: Pose-free Gaussian Splatting for Sparse-view 3D Reconstruction

<a href='https://bluestyle97.github.io/projects/freesplatter/'><img src='https://img.shields.io/badge/Project_Page-Website-red?logo=googlechrome&logoColor=white' alt='Project Page'></a>
<a href="https://arxiv.org/abs/2412.09573"><img src='https://img.shields.io/badge/arXiv-Paper-green?logo=arxiv&logoColor=white' alt='arXiv'></a>
<a href="https://huggingface.co/TencentARC/FreeSplatter"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Model_Card-Huggingface-orange"></a> 
<a href="https://huggingface.co/spaces/TencentARC/FreeSplatter"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Gradio%20Demo-Huggingface-orange"></a> <br>

**ICCV 2025**

</div>

---

This repo is the official implementation of FreeSplatter, a feed-forward framework capable of generating high-quality 3D Gaussians from uncalibrated sparse-view images and recovering their camera parameters in mere seconds.


https://github.com/user-attachments/assets/0c73b693-9428-46bd-843c-132434b9686f

# ⚙️ Installation

We recommend using `Python>=3.10`, `PyTorch>=2.1.0`, and `CUDA>=12.1`.
```bash
conda create --name freesplatter python=3.10
conda activate freesplatter
pip install -U pip

# Install PyTorch and xformers
# You may need to install another xformers version if you use a different PyTorch version
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
pip install xformers==0.0.27.post2

# Install other requirements
pip install -r requirements.txt
```

# 🤖 Pretrained Models

We provide the following pretrained models:

| Model | Description | #Params | Download |
| --- | --- | --- | --- |
| FreeSplatter-O | Object-level reconstruction model | 306M | [Download](https://huggingface.co/TencentARC/FreeSplatter/blob/main/freesplatter-object.safetensors) |
| FreeSplatter-O-2dgs | Object-level reconstruction model using [2DGS](https://surfsplatting.github.io/) (finetuned from FreeSplatter-O) | 306M | [Download](https://huggingface.co/TencentARC/FreeSplatter/blob/main/freesplatter-object-2dgs.safetensors) |
| FreeSplatter-S | Scene-level reconstruction model | 306M | [Download](https://huggingface.co/TencentARC/FreeSplatter/blob/main/freesplatter-scene.safetensors) |

# 💫 Inference

We recommand to start a gradio demo in your local machine, simply run:
```bash
python app.py
```

# ⚖️ License

FreeSplatter's code and models are licensed under the [Apache 2.0 License](LICENSE.txt) with additional restrictions to comply with Tencent's open-source policies. Besides, the libraries [Hunyuan3D-1](https://github.com/Tencent/Hunyuan3D-1) and [BRIAAI RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0) have their own non-commercial licenses.

# :books: Citation

If you find our work useful for your research or applications, please cite using this BibTeX:

```BibTeX
@article{xu2024freesplatter,
  title={FreeSplatter: Pose-free Gaussian Splatting for Sparse-view 3D Reconstruction},
  author={Xu, Jiale and Gao, Shenghua and Shan, Ying},
  journal={arXiv preprint arXiv:2412.09573},
  year={2024}
}
```
