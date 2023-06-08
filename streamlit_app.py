from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt

import time
import numpy as np
import streamlit as st

import cm
page = st.sidebar.selectbox("选择页面", ("在线故障诊断", "模型预测性能展示"))

# 根据选择的页面显示不同的内容
if page == "在线故障诊断":
    #st.subheader("页面 1")
    # 页面 1 的内容
    #st.write("这是页面 1 的内容")

    upload_file = st.file_uploader(
        label = "请选择一个测试文件传入",type=['.npy']
    )

    if upload_file is not None:
        # 不为空
        data = np.load(upload_file)
        st.success("上传文件成功！")
        #新加
        df = pd.DataFrame(data[:12000])
        buffer_size = 1000
        done = False
        last_rows = df.loc[0:buffer_size-1,:]

        placeholder = st.line_chart(last_rows)
       

        for i in range(buffer_size, len(df), buffer_size):
            if i + buffer_size > len(df):# 判断是否已读取df尾部
                done = True
                break
            new_rows = df.loc[i:i+buffer_size-1,:]
            last_rows = np.vstack((last_rows,new_rows))  
            placeholder.add_rows(new_rows)
            last_rows = last_rows[-buffer_size:]
            time.sleep(0.3)
        if not done:   
            for i in range(i, len(df), buffer_size):
                if i + buffer_size > len(df):  
                    break 
                new_rows = df.loc[i:i+buffer_size-1,:]
                last_rows = np.vstack((last_rows,new_rows))
                placeholder.add_rows(new_rows)
                last_rows = last_rows[-buffer_size:]
                time.sleep(0.3)
        
    else:
        st.stop() # 退出
    st.button("Re-run")
    # 加载为numpy数组  
    #data = np.load(io.BytesIO(file_bytes))  
    #data = np.load(upload_file)
    ## 画信号波形图
    #figure, axes = plt.subplots()
    #axes.plot(data[:3000])
    #axes.set_title("Signal at Point A")
    #axes.set_xlabel("Time")
    #axes.set_ylabel("Amplitude")
    #figure.tight_layout()

    #st.write("Signal Plot at Point A")
    #st.pyplot(figure)
    ##新的方法
    
    #修改版
   
    #st.button("Re-run")

# Streamlit widgets automatically run the script from top to bottom. Since
# this button is not connected to any other logic, it just causes a plain
# rerun.
    
  

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
        time.sleep(0.05)
    '运行结束!'
    st.write('故障类型检测结果是：')
    st.subheader('内圈故障')
#st.sidebar.selectbox("选择页面", ("在线故障诊断", "模型预测性能展示"))
elif page == "模型预测性能展示":
    st.subheader("页面 2")
    # 页面 2 的内容
    st.write("这是页面 2 的内容")
    st.write(cm.helloworld())
#st.sidebar.checkbox('Show status')
#page = st.sidebar.selectbox("选择页面", ("在线故障诊断", "模型预测性能展示"))
#st.sidebar.header('Settings')
