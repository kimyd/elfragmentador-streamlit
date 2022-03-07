import logging
import os
import torch
import elfragmentador as ef
from elfragmentador.model import PepTransformerModel
from elfragmentador.visualization import SelfAttentionExplorer as SEA

import matplotlib.pyplot as plt
import numpy as np

import base64
from io import BytesIO

from matplotlib.figure import Figure
import seaborn as sns

import streamlit as st

import logging

logging.basicConfig(
    filename="app.log",
    filemode="w",
    level=logging.INFO,
    format="%(name)s - %(levelname)s - %(message)s",
)


def make_spec_fig(spectrum):
    # Generate the figure **without using pyplot**.
    fig = Figure(figsize=(8, 4))
    ax = fig.subplots()
    spectrum.plot(ax=ax)

    return fig

def make_heatmap_fig(attn_mat):
    fig = Figure(figsize=(10, 8))
    ax = fig.subplots()
    sns.heatmap(attn_mat, cmap = 'viridis', ax=ax)

    return fig, ax


st.title("ElFragmentador")

page_text = """

ElFragmentador implements a neural net that leverages the transformer architecture to predict peptide properties (retention time and fragmentation).

Currently the documentation lives here: [https://jspaezp.github.io/elfragmentador/](https://jspaezp.github.io/elfragmentador/)

Please check out [The Quickstart guide](https://jspaezp.github.io/elfragmentador/quickstart) for usage instructions on your local system.
Feel free to use this web app to check out how the model works.
"""

st.sidebar.markdown(page_text)

model = PepTransformerModel.load_from_checkpoint(ef.DEFAULT_CHECKPOINT)
model.eval()

sequence = st.sidebar.text_input("Peptide Sequence", value = 'MY[PHOSPHO]PEPTIDEK')
charge = st.sidebar.number_input('Charge', min_value = 1, max_value = 5, value = 3)
nce = st.sidebar.number_input('nce', min_value = 20.0, max_value = 40.0, value = 32.0, step = 0.1)


with torch.no_grad():
    with SEA(model) as sea:
        pred = model.predict_from_seq(
            sequence, charge=int(charge), nce=float(nce), as_spectrum=True
        )
    
        encoder_self_attn = []
        for i, _ in enumerate(model.encoder.transformer_encoder.layers):
            encoder_self_attn.append(sea.get_encoder_attn(i))

        decoder_self_attn = []
        for i, _ in enumerate(model.decoder.trans_decoder.layers):
            decoder_self_attn.append(sea.get_decoder_attn(i))


print(pred)

sptxt_text = pred.to_sptxt()
fig = make_spec_fig(pred)


st.pyplot(fig)
with st.expander("Encoder Self Attention Heatmaps"):
    for i, attn in enumerate(encoder_self_attn):
        plt, ax = make_heatmap_fig(attn)
        ax.set_title(f"Encoder layer {i} Self-Attention Weights")
        st.pyplot(plt)

with st.expander("Decoder Self Attention Heatmaps"):
    for i, attn in enumerate(decoder_self_attn):
        plt, ax = make_heatmap_fig(attn)
        ax.set_title(f"Decoder layer {i} Self-Attention Weights")
        st.pyplot(plt)

with st.expander(".sptxt Spectrum"):
    st.code(sptxt_text, language=None)


