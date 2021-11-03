import logging
import os
import torch
import elfragmentador as ef
from elfragmentador.model import PepTransformerModel

import matplotlib.pyplot as plt
import numpy as np

import base64
from io import BytesIO

from matplotlib.figure import Figure
from matplotlib import pyplot as plt

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


st.title("ElFragmentador")

model = PepTransformerModel.load_from_checkpoint(ef.DEFAULT_CHECKPOINT)
model.eval()

sequence = st.text_input("Peptide Sequence", value = 'MYPEPTIDEK')
charge = st.number_input('Charge', min_value = 1, max_value = 5, value = 3)
nce = st.number_input('nce', min_value = 20.0, max_value = 40.0, value = 32.0, step = 0.1)


with torch.no_grad():
    pred = model.predict_from_seq(
        sequence, charge=int(charge), nce=float(nce), as_spectrum=True
    )

print(pred)

sptxt_text = pred.to_sptxt()
fig = make_spec_fig(pred)


st.pyplot(fig)
st.code(sptxt_text, language=None)


