# Blended Latent Diffusion [SIGGRAPH 2023]
<a href="https://omriavrahami.com/blended-latent-diffusion-page/"><img src="https://img.shields.io/static/v1?label=Project&message=Website&color=blue"></a> 
<a href="https://arxiv.org/abs/2206.02779"><img src="https://img.shields.io/badge/arXiv-2206.02779-b31b1b.svg"></a>
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
<a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch->=2.1.0-Red?logo=pytorch"></a>

<a href="https://omriavrahami.com/blended-latent-diffusion-page/"><img src="docs/teaser.png" /></a>

> <a href="https://omriavrahami.com/blended-latent-diffusion-page/">**Blended Latent Diffusion**</a>
>
> Omri Avrahami, Ohad Fried, Dani Lischinski
>
> Abstract: The tremendous progress in neural image generation, coupled with the emergence of seemingly omnipotent vision-language models has finally enabled text-based interfaces for creating and editing images. Handling *generic* images requires a diverse underlying generative model, hence the latest works utilize diffusion models, which were shown to surpass GANs in terms of diversity. One major drawback of diffusion models, however, is their relatively slow inference time. In this paper, we present an accelerated solution to the task of *local* text-driven editing of generic images, where the desired edits are confined to a user-provided mask. Our solution leverages a recent text-to-image Latent Diffusion Model (LDM), which speeds up diffusion by operating in a lower-dimensional latent space. We first convert the LDM into a local image editor by incorporating Blended Diffusion into it. Next we propose an optimization-based solution for the inherent inability of this LDM to accurately reconstruct images. Finally, we address the scenario of performing local edits using thin masks. We evaluate our method against the available baselines both qualitatively and quantitatively and demonstrate that in addition to being faster, our method achieves better precision than the baselines while mitigating some of their artifacts

<div>
  <img src="docs/object_editing.gif" width="200px"/>
  <img src="docs/new_object.gif" width="200px"/>
  <img src="docs/graffiti.gif" width="200px"/>
</div>

# Applications

### Background Editing
<img src="docs/applications/background_edit.png" />

### Text Generation
<img src="docs/applications/text.png" />

### Multiple Predictions
<img src="docs/applications/multiple_predictions.png" />

### Alter an Existing Object
<img src="docs/applications/object_edit.png" />

### Add a New Object
<img src="docs/applications/new_object.png" />

### Scribble Editing
<img src="docs/applications/scribble_edit.png" />

# Installation
Install the conda virtual environment:
```bash
$ conda env create -f environment.yaml
$ conda activate ldm
```

# Usage

## New :fire: - Stable Diffusion Implementation
You can use the newer Stable Diffusion implementation based on [Diffusers](https://github.com/huggingface/diffusers) library.
For that, you need to install PyTorch 2.1 and Diffusers via the following commands:
```bash
$ conda install pytorch==2.1.0 torchvision==0.16.0  pytorch-cuda=11.8 -c pytorch -c nvidia
$ pip install -U diffusers==0.19.3
```

* For using Stable Diffusion XL (requires a stronger GPU), use the following script:

```bash
$ python scripts/text_editing_SDXL.py --prompt "a stone" --init_image "inputs/img.png" --mask "inputs/mask.png"
```
You can use smaller `--batch_size` in order to save GPU memory.

* For using Stable Diffusion v2.1, use the following script:
```bash
$ python scripts/text_editing_SD2.py --prompt "a stone" --init_image "inputs/img.png" --mask "inputs/mask.png"
```

## Old - Latent Diffusion Model Implementation
For using the old implementation, based on the Latent Diffusion Model (LDM), you need first to download the pre-trained weights (5.7GB):
```bash
$ mkdir -p models/ldm/text2img-large/
$ wget -O models/ldm/text2img-large/model.ckpt https://ommer-lab.com/files/latent-diffusion/nitro/txt2img-f8-large/model.ckpt
```

If the above link is broken, you can use this [mirror link](https://huggingface.co/omriav/blended-latent-diffusion-ldm/resolve/main/model.ckpt?download=true).

Then, editing the image may require two steps:
### Step 1 - Generate initial predictions
```bash
$ python scripts/text_editing_LDM.py --prompt "a pink yarn ball" --init_image "inputs/img.png" --mask "inputs/mask.png"
```

The predictions will be saved in `outputs/edit_results/samples`.

You can use a larger batch size by specifying `--n_samples` to the maximum number that saturates your GPU.

### Step 2 (optional) - Reconstruct the original background
If you want to reconstruct the original image background, you can run the following:
```bash
$ python scripts/reconstruct.py --init_image "inputs/img.png" --mask "inputs/mask.png" --selected_indices 0 1
```

You can choose the specific image indices that you want to reconstruct. The results will be saved in `outputs/edit_results/samples/reconstructed_optimization`.

# Citation
If you find this project useful for your research, please cite the following:
```bibtex
@article{avrahami2023blendedlatent,
        author = {Avrahami, Omri and Fried, Ohad and Lischinski, Dani},
        title = {Blended Latent Diffusion},
        year = {2023},
        issue_date = {August 2023},
        publisher = {Association for Computing Machinery},
        address = {New York, NY, USA},
        volume = {42},
        number = {4},
        issn = {0730-0301},
        url = {https://doi.org/10.1145/3592450},
        doi = {10.1145/3592450},
        journal = {ACM Trans. Graph.},
        month = {jul},
        articleno = {149},
        numpages = {11},
        keywords = {zero-shot text-driven local image editing}
}

@InProceedings{Avrahami_2022_CVPR,
        author    = {Avrahami, Omri and Lischinski, Dani and Fried, Ohad},
        title     = {Blended Diffusion for Text-Driven Editing of Natural Images},
        booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
        month     = {June},
        year      = {2022},
        pages     = {18208-18218}
}
```

# Acknowledgements
This code is based on [Latent Diffusion Models](https://github.com/CompVis/latent-diffusion).
