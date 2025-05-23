#!/usr/bin/env python3
""" load ovis2-4B model and genaret description for the provided image"""
# import modules
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from PIL import Image
from transformers import AutoModelForCausalLM
from torchvision import transforms, models
from torch import nn
import os



class OvisCaptioner:
    """OvisCaptioner class"""

    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            "AIDC-AI/Ovis2-4B",
            torch_dtype=torch.bfloat16,
            multimodal_max_length=32768,
            trust_remote_code=True).cuda()
        self.model = torch.compile(self.model, mode="reduce-overhead")
        self.text_tokenizer = self.model.get_text_tokenizer()
        self.visual_tokenizer = self.model.get_visual_tokenizer()
        self.model_dtype = next(
            self.visual_tokenizer.backbone.parameters()).dtype
        self.device = next(self.visual_tokenizer.backbone.parameters()).device
        # from 1024 optimize it to 256 for first experiment now to 200
        self.gen_kwargs = dict(
            max_new_tokens=200,
            do_sample=False,
            top_p=None,
            top_k=None,
            temperature=None,
            repetition_penalty=None,
            eos_token_id=self.model.generation_config.eos_token_id,
            pad_token_id=self.text_tokenizer.pad_token_id,
            use_cache=True
        )
        # load mobilenetv3 large classifier
        self.scene_model = models.mobilenet_v3_large(pretrained=True)
        for param in self.scene_model.parameters():
            param.requires_grad = False
        self.scene_model.classifier[3] = nn.Linear(
            self.scene_model.classifier[3].in_features, 2)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(
            base_dir, "../img_type_classification/mobilenetv3_scene_classifier.pth")
        self.scene_model.load_state_dict(
            torch.load(model_path, map_location=self.device))
        self.scene_model = self.scene_model.to(self.device)
        self.scene_tf = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        self.scene_classes = ["meme", "non_meme"]

    def classify_image(self, img_path):
        """ Predict image type """
        img = Image.open(img_path).convert("RGB")
        x = self.scene_tf(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.scene_model(x)
            pred = output.argmax(dim=1).item()
        return self.scene_classes[pred]
    def describe_image(
            self,
            img_path,
            max_partition=9,
            text='Describe the image.', audio_path="caption_audio.mp3"):
        """ generate description for the image"""
        img_type = self.classify_image(img_path)
        # choose prompt based on type
        if img_type == "meme":
            # use the optimized prompt we got from experiment
            text = "Quickly describe this image. If there's a recognizable person, mention their name naturally in the description. If it's a meme, summarize the joke humorously. If neither is present, just describe the scene vividly without stating what's missing."
        else:
            text = "Describe this image carefully for a blind person."
        images = [Image.open(img_path)]
        query = f'<image>\n{text}'
        # preprocess input
        prompt, input_ids, pixel_values = self.model.preprocess_inputs(
            query, images, max_partition=max_partition)
        # set device and dtype
        pixel_values = pixel_values.to(
            dtype=self.model_dtype, device=self.device)
        visual_embeddings = self.visual_tokenizer.backbone(pixel_values)
        # attention mask
        attention_mask = torch.ne(input_ids, self.text_tokenizer.pad_token_id)
        input_ids = input_ids.unsqueeze(0).to(self.model.device)
        attention_mask = attention_mask.unsqueeze(0).to(self.model.device)

        # pixel values to expected format
        pixel_values = [pixel_values]

        # LLM caption generation
        # try to optimize by updating max_new_tokens and setting in to 256

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                **self.gen_kwargs)[0]
        output = self.text_tokenizer.decode(
            output_ids, skip_special_tokens=True)

        return {"type": img_type, "caption": output}
    
 