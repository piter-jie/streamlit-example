from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
"""
# Welcome to Streamlit!

Edit `/streamlit_app.py` to customize this app to your heart's desire :heart:

If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
forums](https://discuss.streamlit.io).

In the meantime, below is an example of what you can do with just a few lines of code:
"""
import time
import numpy as np
import streamlit as st

# Streamlit widgets automatically run the script from top to bottom. Since
# this button is not connected to any other logic, it just causes a plain
# rerun.
upload_file = st.file_uploader("选择一个文件传入")
data = np.load(upload_file)
# 画信号波形图
figure, axes = plt.subplots()
axes.plot(data)
axes.set_title("Signal at Point A")
axes.set_xlabel("Time")
axes.set_ylabel("Amplitude")
figure.tight_layout()

st.write("Signal Plot at Point A")
st.pyplot(figure)
# plt.plot(data[:3000])
st.markdown('Streamlit Demo')
