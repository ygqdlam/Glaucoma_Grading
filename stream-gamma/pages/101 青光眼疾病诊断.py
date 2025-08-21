import streamlit as st
import pandas as pd
import altair as alt
from urllib.error import URLError
import pages.grading.stream_test as segment
import pages.grading.stream_single_test as segment_single

import time
import numpy as np
import os
import cv2 as cv
from PIL import Image, ImageDraw
import torch
import tempfile
import tkinter as tk
from tkinter import filedialog
import ttkbootstrap as ttk
import sys
import multiprocessing

st.set_page_config(page_title="青光眼疾病诊断", page_icon="📊")

st.markdown("# 青光眼疾病诊断")
#st.sidebar.header("青光眼疾病诊断")

st.write(
    """选择一个文件夹，批量化青光眼疾病诊断"""
)
#def select_folder():
#    root = tk.Tk()
#    root.withdraw()
#    folder = filedialog.askdirectory()
#    root.destroy()
#    return folder

def select_folder_subprocess(queue):
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory()
    root.destroy()
    queue.put(folder)

def select_folder():
    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=select_folder_subprocess, args=(queue,))
    p.start()
    p.join()  # 等待子进程结束
    if not queue.empty():
        return queue.get()
    return None

if st.button("选择文件夹"):
    #root = ttk.Window(themename="superhero")  # 你可以换成 "cosmo" "flatly" "journal" 等主题
    #root.withdraw()  # 隐藏主窗口
    #folder_path = filedialog.askdirectory(title="请选择一个文件夹")
    #st.session_state.selected_folder = folder_path
    folder_path = select_folder()
    print("选择结果：", folder_path)
    stream_metrics = segment.predict(folder_path, st)
    print(stream_metrics)
    st.subheader('批量化输出结果')
    row0_cols = st.columns(2)
    row0_cols[0].markdown(f"**准确率(Acc)**")
    row0_cols[1].markdown(f"**AUC**")

    row1_cols = st.columns(2)
    row1_cols[0].markdown(stream_metrics[0])
    row1_cols[1].markdown(stream_metrics[2])


#if "selected_folder" in st.session_state:
#    st.write("选中文件夹：", st.session_state.selected_folder)

#if st.button("青光眼疾病诊断"):
#    stream_metrics = segment.predict(st.session_state.selected_folder, st)
#    print(stream_metrics)
#    st.subheader('Output Result')
#    row0_cols = st.columns(2)
#    row0_cols[0].markdown(f"**准确率(Acc)**")
#    row0_cols[1].markdown(f"**AUC**")

#    row1_cols = st.columns(2)
#    row1_cols[0].markdown(stream_metrics[0])
#    row1_cols[1].markdown(stream_metrics[2])





st.write(
    """输入一幅眼底图像，输出青光眼疾病诊断结果"""
)


img_file_buffer_segment = st.file_uploader("上传一张图像", type=['jpg','jpeg', 'png'], key=2)


# DEMO_IMAGE = "pages/Results/benign (2).png"
# if img_file_buffer_segment is not None:
#     img = cv.imdecode(np.fromstring(img_file_buffer_segment.read(), np.uint8), 1)
#     image = np.array(Image.open(img_file_buffer_segment))
#     file_name = img_file_buffer_segment.name
# else:
#     img = cv.imread(DEMO_IMAGE)
#     image = np.array(Image.open(DEMO_IMAGE))


if img_file_buffer_segment is not None:
    img = cv.imdecode(np.fromstring(img_file_buffer_segment.read(), np.uint8), 1)
    image = np.array(Image.open(img_file_buffer_segment))
    file_name = img_file_buffer_segment.name
else:
    st.stop()

st.text("上传图像")

cols = st.columns(2)
cols[0].image(image, clamp=True, channels='GRAY', use_container_width=True, caption="眼底图像")
#cols[1].image(img,channels='RBG', use_container_width=True)

# predict

stream_grading,pred = segment_single.predict(file_name, st)
st.subheader('诊断结果')


pred_values =  pred[["non", "early", "mid_advanced"]].iloc[0].astype(int).to_list()
# 模拟预测结果
pred_labels = ["non", "early", "mid_advanced"]
#pred_values = [0, 0, 1]  # 你的结果

# 第一行：显示类别名
row1 = st.columns(3)
for i, label in enumerate(pred_labels):
    row1[i].markdown(f"**{label}**")

# 第二行：显示0/对号
row2 = st.columns(3)
for i, val in enumerate(pred_values):
    if val == 1:
        row2[i].markdown("✅")  # 对号
    else:
        row2[i].markdown(" ")
















