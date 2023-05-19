from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
"""
# 信号故障类型诊断

"""
import time
import numpy as np
import streamlit as st

# Streamlit widgets automatically run the script from top to bottom. Since
# this button is not connected to any other logic, it just causes a plain
# rerun.
upload_file = st.file_uploader("选择一个测试文件传入",type=['.npy']) or 'default.npy'
# 读取文件为字节流
#file_bytes = upload_file.read()  

# 加载为numpy数组  
#data = np.load(io.BytesIO(file_bytes))  
data = np.load(upload_file)
# 画信号波形图
figure, axes = plt.subplots()
axes.plot(data[:3000])
axes.set_title("Signal at Point A")
axes.set_xlabel("Time")
axes.set_ylabel("Amplitude")
figure.tight_layout()

st.write("Signal Plot at Point A")
st.pyplot(figure)
# plt.plot(data[:3000])
st.markdown('Streamlit Demo')
