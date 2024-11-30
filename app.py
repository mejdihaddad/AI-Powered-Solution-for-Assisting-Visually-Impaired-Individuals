import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from PIL import Image
import tempfile
from gtts import gTTS
import os
import base64
import io

os.environ["GOOGLE_API_KEY"] = ""

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

SCENE_PROMPT = """You are an advanced AI model designed to assist visually impaired individuals. 
When provided with an image, describe the scene in detail, including:
- Objects present in the image
- Colors and visual characteristics
- Actions or interactions
- Spatial relationships
- Any visible text or labels
Provide a comprehensive, descriptive narrative that helps someone visualize the scene."""

TEXT_PROMPT = """You are an expert at text extraction from images. 
Carefully identify and extract ALL visible text from the image, including:
- Labels
- Signs
- Documents
- Any written content

Provide the extracted text verbatim, preserving original formatting and line breaks."""

def encode(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def chain(prompt):
    template = ChatPromptTemplate.from_messages([
        ("system", prompt),
        ("human", [
            {"type": "text", "text": "{prompt}"},
            {"type": "image_url", "image_url": "{base64}"}
        ])
    ])
    
    return (
        RunnablePassthrough.assign(
            base64=lambda x: f"data:image/png;base64,{x['base64']}"
        )
        | template
        | model
        | StrOutputParser()
    )

def speak(text):
    temp = tempfile.mktemp(suffix='.mp3', prefix='speech_')
    
    tts = gTTS(text=text, lang='en')
    tts.save(temp)
    
    with open(temp, 'rb') as audio:
        bytes = audio.read()
    
    os.unlink(temp)
    
    return bytes

st.title("AI Powered Solution for Assisting Visually Impaired Individuals")

img = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

if img:
    image = Image.open(img)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    base64 = encode(image)
    
    st.subheader("Scene Description")
    scene_chain = chain(SCENE_PROMPT)
    scene = scene_chain.invoke({
        "prompt": SCENE_PROMPT,
        "base64": base64
    })
    st.write(scene)
    
    st.subheader("Text Extraction and Speech")
    text_chain = chain(TEXT_PROMPT)
    text = text_chain.invoke({
        "prompt": TEXT_PROMPT,
        "base64": base64
    })
    st.write("Extracted Text:", text)
    
    if st.button("Listen"):
        audio = speak(text)
        st.audio(audio, format="audio/mp3")