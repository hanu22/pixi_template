import logging
from pathlib import Path

import streamlit as st
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%d/%m/%y %H:%M:%S",
)

st.set_page_config(
    page_title="Project Name App",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Project Name")

img = Image.open(
    Path(
        "images",
        "ds_logo.png",
    )
)

with st.sidebar:
    st.image(img)

# streamlit run app.py
