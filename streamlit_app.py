import logging
import os
import torch
import elfragmentador as ef
from elfragmentador.model import PepTransformerModel
from elfragmentador.model.visualization import SelfAttentionExplorer as SEA

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
    sns.heatmap(attn_mat, cmap="viridis", ax=ax)

    return fig, ax


st.title("ElFragmentador")

page_text = """

ElFragmentador implements a neural net that leverages the transformer architecture to predict peptide properties (retention time and fragmentation).

Currently the documentation lives here: [https://jspaezp.github.io/elfragmentador/](https://jspaezp.github.io/elfragmentador/)

Please check out [The Quickstart guide](https://jspaezp.github.io/elfragmentador/quickstart) for usage instructions on your local system.
Feel free to use this web app to check out how the model works.

So far my internal metrics show that it works well for oxidized, phosphorylated, acetylated, tmt-6 and un-modified peptides of length 5-30 (tryptic and non-tryptic).


If you feel like it can be improved or want to let me know that it does not work in your data, feel free to open an issue in github!
https://github.com/jspaezp/elfragmentador/issues
"""

st.sidebar.markdown(page_text)


@st.cache_resource
def get_model():
    model = PepTransformerModel.load_from_checkpoint(ef.DEFAULT_CHECKPOINT)
    model.eval()
    return model


model = get_model()
sequence = st.sidebar.text_input(
    "Peptide Sequence",
    value="MY[U:21]PEPTIDEK/2",
    help=(
        "Peptide sequence, modifications are denoted with square brackets ([U:21] is"
        " phospho and [U:35] is oxidation), the charge is denoted with the integer"
        " after the '/''"
    ),
)
nce = st.sidebar.number_input(
    "nce", min_value=20.0, max_value=40.0, value=32.0, step=0.1
)

st.sidebar.markdown(
    """
Elfragmentador has been trained on several PTMs.

Number of sequences per modification
- 78316 [OXIDATION] U:35
- 4046 [PHOSPHO] U:21
- 27983 [GG] U:121
- 106394 [TMT6PLEX] U:737
-  66 [ACETYL] U:1
-  125 [METHYL] U:34
-  150 [DIMETHYL] U:36
-   53 [TRIMETHYL] U:37
-   39 [NITRO] U:354
"""
)


@st.cache_data
def predict_peptide(_model, nce, sequence):
    with torch.no_grad():
        with SEA(_model) as sea:
            pred = _model.predict_from_seq(sequence, nce=float(nce), as_spectrum=True)
            encoder_self_attn = []
            for i, _ in enumerate(model.main_model.encoder.encoder.layers):
                try:
                    encoder_self_attn.append(sea.get_encoder_attn(i))
                except IndexError:
                    print(f"Failed to get index {i} from the encoder")

            decoder_self_attn = []
            for i, _ in enumerate(model.main_model.decoder.trans_decoder.layers):
                try:
                    decoder_self_attn.append(sea.get_decoder_attn(i))
                except IndexError:
                    print(f"Failed to get index {i} from the decoder")

    return pred, encoder_self_attn, decoder_self_attn


pred, encoder_self_attn, decoder_self_attn = predict_peptide(model, nce, sequence)

print(pred)

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
