#!/usr/bin/env python
""" testing text to speech model with caption generated from the model ovis2-4B"""
import edge_tts
import asyncio

async def speak(text):
    communicate = edge_tts.Communicate(text, "en-US-JennyNeural")
    await communicate.save("tts/output.mp3")

asyncio.run(speak("Hello Rania, your AI assistant is speaking!"))



