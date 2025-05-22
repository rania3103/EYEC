#!/usr/bin/env python
""" testing text to speech model with caption generated from the model ovis2-4B"""
import edge_tts
import asyncio
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))# import OvisCaptioner class
from app.app import OvisCaptioner
captioner = OvisCaptioner()

for i, img_path in enumerate(["data/memes_memotion/images/image_346.png", "data/quote/pinterest_1900024838573514.jpg", "data/memes_memotion/images/image_1.png"], 1):
    result = captioner.describe_image(img_path, audio_path=f"demo/caption{i}.mp3")
    print(f"Image {i} Caption: {result['caption']}")


