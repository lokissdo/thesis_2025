import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn

import open_clip
from torchvision import transforms, models
from diffusers import DDIMScheduler, DiffusionPipeline

class MultiLabelModel(nn.Module):
    def __init__(self, model_name='resnet18', num_labels=40):
        super(MultiLabelModel, self).__init__()
        
        # Khởi tạo size dựa trên model_name
        if model_name == 'resnet18':
            self.size = 1024
            self.base_model = models.resnet18(pretrained=True)
            num_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Linear(num_features, num_labels)
        elif model_name in ['densenet121', 'vgg19']:
            self.size = 256
            if model_name == 'densenet121':
                self.base_model = models.densenet121(pretrained=True)
                num_features = self.base_model.classifier.in_features
                self.base_model.classifier = nn.Linear(num_features, num_labels)
            else:  # model_name == 'vgg19'
                self.base_model = models.vgg19(pretrained=True)
                num_features = self.base_model.classifier[6].in_features
                self.base_model.classifier[6] = nn.Linear(num_features, num_labels)
        elif model_name in ['resnet50', 'efficientNet']:
            self.size = 512
            if model_name == 'resnet50':
                self.base_model = models.resnet50(pretrained=True)
                num_features = self.base_model.fc.in_features
                # self.base_model.fc = nn.Linear(num_features, num_labels)
                self.base_model.fc = nn.Identity()  # Remove the original fully connected layer

                # Additional layers after the base model
                self.fc1 = nn.Linear(num_features, 1024)  # Add a new fully connected layer
                self.bn1 = nn.BatchNorm1d(1024)  # Batch normalization layer
                self.relu1 = nn.ReLU()  # ReLU activation
                self.drop1 = nn.Dropout(0.5)  # Dropout layer to prevent overfitting
        
                self.fc2 = nn.Linear(1024, 512)  # Another fully connected layer
                self.bn2 = nn.BatchNorm1d(512)  # Batch normalization
                self.relu2 = nn.ReLU()  # ReLU activation
        
                self.fc3 = nn.Linear(512, num_labels)  # Final layer for multi-label classification

            else:  # model_name == 'efficientnet'
                self.base_model = models.efficientnet_b0(pretrained=True)
                num_features = self.base_model.classifier[-1].in_features
                self.base_model.classifier[-1] = nn.Linear(num_features, num_labels)
        else:
            raise ValueError(f"Model '{model_name}' is not supported.")

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x = self.base_model(x)
        # x = self.sigmoid(x)  # Apply sigmoid activation for multi-label classification
        # return x
        x = self.base_model(x)  # Pass through ResNet50
        x = self.fc1(x)  # First fully connected layer
        x = self.bn1(x)  # Batch normalization
        x = self.relu1(x)  # ReLU activation
        x = self.drop1(x)  # Dropout layer

        x = self.fc2(x)  # Second fully connected layer
        x = self.bn2(x)  # Batch normalization
        x = self.relu2(x)  # ReLU activation

        x = self.fc3(x)  # Final layer
        x = self.sigmoid(x)  # Apply sigmoid activation

        return x        

class BlendedLatentDiffusion:
    def __init__(self):
        self.parse_args()
        self.device = self.args.device
        self.attr_idx = 31  # thuộc tính Smiling
        self.load_models()
        self.load_clip_model()
        self.load_classifier()
        self.setup_classifier_transform()

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--prompt", type=str, required=True)
        parser.add_argument("--init_image", type=str, required=True)
        parser.add_argument("--mask", type=str, required=True)
        parser.add_argument("--model_path", type=str, default="stabilityai/stable-diffusion-2-1-base")
        parser.add_argument("--batch_size", type=int, default=4)
        parser.add_argument("--blending_start_percentage", type=float, default=0.25)
        parser.add_argument("--device", type=str, default="cuda")
        parser.add_argument("--lora_path", type=str, default=None)
        parser.add_argument("--classifier_path", type=str, default="/kaggle/input/resnet50-multilabel-model-add-layers/resnet50_multilabel_model_add_layers.pth")
        parser.add_argument("--output_path", type=str, default="outputs/res.jpg")
        parser.add_argument("--output_path_1", type=str, default="outputs/res_1.jpg")
        parser.add_argument("--output_path_2", type=str, default="outputs/res_2.jpg")
        parser.add_argument("--output_path_3", type=str, default="outputs/res_3.jpg")
        parser.add_argument("--output_path_4", type=str, default="outputs/res_4.jpg")
        self.args = parser.parse_args()

    def load_models(self):
        pipe = DiffusionPipeline.from_pretrained(
            self.args.model_path,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
        if self.args.lora_path:
            pipe.load_lora_weights(self.args.lora_path)
        # pipe.load_lora_weights("/kaggle/input/sd2-fullface-model/trained_top1")
        self.vae = pipe.vae.to(self.device)
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder.to(self.device)
        self.unet = pipe.unet.to(self.device)
        self.scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )

    def load_clip_model(self):
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            'ViT-H-14', pretrained='laion2b_s32b_b79k'
        )
        self.clip_model = self.clip_model.to(self.device)
        self.clip_model.eval()

    def load_classifier(self):
        self.model = MultiLabelModel(model_name='resnet50', num_labels=40)
        weight_path = self.args.classifier_path
        state_dict = torch.load(weight_path, map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device).eval()

    def setup_classifier_transform(self):
        self.classifier_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

    @torch.no_grad()
    def edit_image(
        self,
        image_path,
        mask_path,
        prompts,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=5.0,
        generator=torch.Generator(device='cuda').manual_seed(42),
        blending_percentage=0.25,
        alpha=0.5,
    ):
        batch_size = len(prompts)

        # Prediction for the original image base on the classifier
        orig_img = Image.open(image_path).convert("RGB")
        x_orig = self.classifier_transform(orig_img).unsqueeze(0).to(self.device)
        orig_score = self.model(x_orig)[0, self.attr_idx].item()
        is_orig_near_zero = orig_score < 0.5

        # Load and resize the input image
        image = Image.open(image_path)
        image = image.resize((height, width), Image.BILINEAR)
        image = np.array(image)[:, :, :3]
        source_latents = self._image2latent(image)
        latent_mask, org_mask = self._read_mask(mask_path)

        text_input = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        max_length = text_input.input_ids.shape[-1]
        uncond_input = self.tokenizer(
            [""] * batch_size,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        latents = torch.randn(
            (batch_size, self.unet.in_channels, height // 8, width // 8),
            generator=generator,
            device=self.device,
            dtype=torch.float16,
        )

        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.scheduler.timesteps[
            int(len(self.scheduler.timesteps) * blending_percentage) :
        ]:
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep=t)

            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

            noise_source_latents = self.scheduler.add_noise(
                source_latents, torch.randn_like(latents), t
            )
            latents = latents * latent_mask + noise_source_latents * (1 - latent_mask)

        latents = 1 / 0.18215 * latents

        decoded_images = self.vae.decode(latents).sample

        images = (decoded_images / 2 + 0.5).clamp(0, 1)
        images = images.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (images * 255).round().astype(np.uint8)

        # Caculate CLIP scores
        text_tokens = open_clip.tokenize(prompts).to(self.device)
        text_features = self.clip_model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        images_tensor = torch.stack(
            [self.clip_preprocess(Image.fromarray(img)) for img in images]
        ).to(self.device)

        image_features = self.clip_model.encode_image(images_tensor)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        similarity = image_features @ text_features.T
        clip_scores = similarity[:, 0].tolist()

        # Calculate counterfactual class scores based on the classifier 
        class_scores_raw = []
        for img in images:
            img_pil = Image.fromarray(img)
            x = self.classifier_transform(img_pil).unsqueeze(0).to(self.device)
            with torch.no_grad():
                score = self.model(x)[0, self.attr_idx].item()
            class_scores_raw.append(score)

        class_scores_for_sort = [
            s if is_orig_near_zero else (1 - s) for s in class_scores_raw
        ]

        # Combine CLIP scores and class scores 
        final_scores = [
            alpha * c + (1 - alpha) * s for c, s in zip(clip_scores, class_scores_for_sort)
        ]

        # Sort images and scores based on final scores
        sorted_pairs = sorted(zip(final_scores, images), key=lambda x: x[0], reverse=True)
        sorted_scores, sorted_images = zip(*sorted_pairs)

        return list(sorted_images), list(sorted_scores)

    @torch.no_grad()
    def _image2latent(self, image):
        image = torch.from_numpy(image).float() / 127.5 - 1
        image = image.permute(2, 0, 1).unsqueeze(0).to(self.device)
        image = image.half()
        latents = self.vae.encode(image)["latent_dist"].mean
        latents = latents * 0.18215
        return latents

    def _read_mask(self, mask_path: str, dest_size=(64, 64)):
        org_mask = Image.open(mask_path).convert("L")
        mask = org_mask.resize(dest_size, Image.NEAREST)
        mask = np.array(mask) / 255
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = mask[np.newaxis, np.newaxis, ...]
        mask = torch.from_numpy(mask).half().to(self.device)
        return mask, org_mask

if __name__ == "__main__":
    bld = BlendedLatentDiffusion()
    results = bld.edit_image(
        bld.args.init_image,
        bld.args.mask,
        prompts=[bld.args.prompt] * bld.args.batch_size,
        blending_percentage=bld.args.blending_start_percentage,
    )
    results_flat = np.concatenate(results, axis=1)
    Image.fromarray(results_flat).save(bld.args.output_path)