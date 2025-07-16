import streamlit as st
import numpy as np
import pandas as pd
from SD import D2
from FD import D1
from sigmoids import sigmoid
from squashing import squashing  
from i_squashing import i_squashing 
from i_sigmoid import i_sigmoid
from IModPoly import IModPoly
from AsLS import baseline_als
from LPnorm import LPnorm
from wavelettransform import waveletlinear 
from fft import Smfft
from KalmanFiltering import Kalman
from ArithmeticAverage import MWA 
from meadianfiltering import MWM 
from SGfiltering import SGfilter 
import pywt
import copy

# 设置页面
st.set_page_config(layout="wide", page_title="拉曼光谱分析系统")
st.title("拉曼光谱分析系统")

# ===== 使用说明 - 移到顶部 =====
with st.expander("📌 使用指南（点击展开）", expanded=True):
    st.markdown("""
    **操作流程:**
    1. 📁 左侧上传光谱文件（TXT/CSV格式）
    2. ⚙️ 选择预处理方法（基线校正→数据变换→滤波→归一化）
    3. 🚀 点击"应用处理"按钮
    4. 📊 查看右侧处理结果
    5. 🔍 执行峰分析并导出结果

    **文件格式要求:**
    - 光谱文件：两列数据（波数+强度）
    - 支持多光谱同时处理

    **小技巧:**
    - 鼠标悬停在参数上可查看帮助提示
    - 点击图表可放大查看细节
    """)

# 初始化session状态
if 'raw_data' not in st.session_state:
    st.session_state.raw_data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'peaks' not in st.session_state:
    st.session_state.peaks = None

# 创建两列布局
col1, col2 = st.columns([1.2, 3])

with col1:
    # ===== 数据管理 =====
    with st.expander("📁 数据管理", expanded=True):
        uploaded_file = st.file_uploader("上传光谱文件", type=['txt', 'csv'])
        
        if uploaded_file:
            try:
                data = np.loadtxt(uploaded_file)
                x = data[:, 0]
                y = data[:, 1]
                st.session_state.raw_data = (x, y)
                st.success(f"数据加载成功！点数: {len(x)}")
            except Exception as e:
                st.error(f"文件加载失败: {str(e)}")

    # ===== 预处理设置 =====
    with st.expander("⚙️ 预处理设置", expanded=True):
        # 基线校准
        st.markdown("### 1. 基线校准")
        baseline_method = st.selectbox(
            "基线校准方法",
            ["无", "SD", "FD", "I-ModPoly", "AsLS"],
            key="baseline_method"
        )

        if baseline_method == "I-ModPoly":
            polyorder = st.slider("多项式阶数", 3, 10, 6, key="polyorder")
        elif baseline_method == "AsLS":
            lam = st.number_input("λ(平滑度)", value=1e7, format="%e", key="lam")
            p = st.slider("p(不对称性)", 0.01, 0.5, 0.1, key="p")

        # 数据变换
        st.markdown("### 2. 数据变换")
        transform_method = st.selectbox(
            "变换方法",
            ["无", "挤压函数(归一化版)", "挤压函数(原始版)", "Sigmoid(归一化版)", "Sigmoid(原始版)"],
            key="transform_method"
        )

        if transform_method == "Sigmoid(归一化版)":
            maxn = st.slider("归一化系数", 1, 20, 10, key="i_sigmoid_maxn")

        # 滤波处理
        st.markdown("### 3. 滤波处理")
        filter_method = st.selectbox(
            "滤波方法",
            ["无", "傅里叶滤波", "卡尔曼滤波", "移动平均", "中值滤波", "SG滤波", "小波滤波"],
            key="filter_method"
        )
        
        if filter_method == "傅里叶滤波":
            row_e = st.slider("截止频率", 1, 100, 51)
        elif filter_method == "卡尔曼滤波":
            R = st.number_input("噪声方差", value=0.0001, format="%f")
        elif filter_method in ["移动平均", "中值滤波"]:
            n = st.slider("窗口大小", 3, 21, 7, step=2)
        elif filter_method == "SG滤波":
            point = st.slider("窗口点数", 5, 31, 11, step=2)
            degree = st.slider("多项式阶数", 1, 5, 3)
        elif filter_method == "小波滤波":
            threshold = st.slider("阈值", 0.01, 1.0, 0.3, step=0.01)

        # 归一化
        st.markdown("### 4. 归一化")
        norm_method = st.selectbox(
            "归一化方法",
            ["无", "无穷大范数", "L10范数", "L4范数"],
            key="norm_method"
        )

        if st.button("🚀 应用处理", type="primary", use_container_width=True):
            if st.session_state.raw_data is None:
                st.warning("请先上传数据文件")
            else:
                # [原有处理逻辑保持不变...]
                pass

with col2:
    # ===== 结果展示区 =====
    st.markdown("## 📊 实时分析结果")
    
    # 系统状态卡片
    status_col1, status_col2 = st.columns(2)
    with status_col1:
        if st.session_state.raw_data:
            st.info(f"**数据信息**\n\n- 点数: {len(st.session_state.raw_data[0])}\n- 格式: {st.session_state.raw_data[1].shape}")
    with status_col2:
        if st.session_state.get('process_method'):
            st.success(f"**处理流程**\n\n{st.session_state.process_method}")

    # 光谱图表
    st.markdown("### 光谱可视化")
    if st.session_state.raw_data:
        # [原有图表代码保持不变...]
        pass
    else:
        st.info("等待数据上传...")

    # 分析结果
    st.markdown("### 🔍 峰分析结果")
    if st.session_state.peaks:
        # [原有峰分析代码保持不变...]
        pass

# 底部工具提示
st.caption("💡 提示：在任何步骤遇到问题，请参考顶部的使用指南")
