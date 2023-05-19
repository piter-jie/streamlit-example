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
#upload_file = st.file_uploader("选择一个测试文件传入",type=['.npy']) 
# 读取文件为字节流
#file_bytes = upload_file.read()
#if upload_file is None:
    #st.write('No file name provided')
#else:
    #data = np.load(upload_file)



upload_file = st.file_uploader(
    label = "选择一个测试文件传入",type=['.npy']
)

if upload_file is not None:
    # 不为空
    data = np.load(upload_file)
    st.success("上传文件成功！")
else:
    st.stop() # 退出      
# 加载为numpy数组  
#data = np.load(io.BytesIO(file_bytes))  
#data = np.load(upload_file)
# 画信号波形图
figure, axes = plt.subplots()
axes.plot(data[:3000])
axes.set_title("Signal at Point A")
axes.set_xlabel("Time")
axes.set_ylabel("Amplitude")
figure.tight_layout()

st.write("Signal Plot at Point A")
st.pyplot(figure)

#显示进度条
#st.write("10. st.progress()")
import time
st.write("正在检测故障类型")
# 添加placeholder
latest_iteration = st.empty()
bar = st.progress(0)
for i in range(100):
    # 更新进度条
    latest_iteration.text(f'Iteration {i+1}')
    bar.progress(i+1)
    time.sleep(0.1)
'运行结束!'
st.markdown('Streamlit Demo')
