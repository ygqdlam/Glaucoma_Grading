import streamlit as st
import pages.funds.stream_single_test as segment_single_seg
import pages.funds.stream_test as segment_seg

import time
import numpy as np
import os
import cv2 as cv
from PIL import Image, ImageDraw
import tempfile
import tkinter as tk
from tkinter import filedialog
import ttkbootstrap as ttk
import multiprocessing

st.set_page_config(page_title="青光眼视杯盘分割", page_icon="📈")


st.header("青光眼视杯盘分割")

st.write(
    """选择一个文件夹，批量化分割"""
)

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


#if st.button("选择文件夹"):
#    root = ttk.Window(themename="superhero")  # 你可以换成 "cosmo" "flatly" "journal" 等主题
#    root.withdraw()  # 隐藏主窗口
#    folder_path = filedialog.askdirectory(title="请选择一个文件夹")
#    st.session_state.selected_folder = folder_path

#if "selected_folder" in st.session_state:
#    st.write("选中文件夹：", st.session_state.selected_folder)

#if st.button("视杯盘分割"):
#    stream_metrics = segment_seg.predict(st.session_state.selected_folder, st)
#    print(stream_metrics)
#    st.subheader('Output Image')
#    row0_cols = st.columns(1)
#    row0_cols[0].markdown(f"**metrics为: {stream_metrics}**")

if st.button("选择文件夹"):
    folder_path = select_folder()
    stream_metrics = segment_seg.predict(folder_path, st)
    print(stream_metrics)
    st.subheader('批量化输出结果')
    row0_cols = st.columns(3)

    row0_cols[0].markdown(f"敏感度(sensitivity)")
    row0_cols[1].markdown(f"特异度(specificity)")
    row0_cols[2].markdown(f"杯盘比(cdr)")
    #row0_cols[3].markdown(f"相似率(dice)")


    row1_cols = st.columns(3)
    row1_cols[0].markdown(stream_metrics['macro_sensitivity'])
    row1_cols[1].markdown(stream_metrics['macro_specificity'])
    row1_cols[2].markdown(100-stream_metrics['macro_cdr'])
    #row1_cols[3].markdown(stream_metrics['macro_dice'])





st.write(
    """输入一幅眼底图像，输出视杯盘分割结果"""
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
# segment_seg.predict(file_name, st)


stream_mask,stream_metrics = segment_single_seg.predict(file_name, st)


st.subheader('分割结果')
row2_cols = st.columns(2)
row2_cols[0].image(stream_mask, clamp=True, channels='GRAY', use_container_width=True, caption="分割可视化结果")


row3_cols = st.columns(3)
row3_cols[0].markdown(f"敏感度(sensitivity)")
row3_cols[1].markdown(f"特异度(specificity)")
row3_cols[2].markdown(f"杯盘比(cdr)")
#row3_cols[3].markdown(f"相似率(dice)")


row4_cols = st.columns(3)
row4_cols[0].markdown(stream_metrics['macro_sensitivity'])
row4_cols[1].markdown(stream_metrics['macro_specificity'])
row4_cols[2].markdown(100-stream_metrics['macro_cdr'])
#row4_cols[3].markdown(stream_metrics['macro_dice'])
