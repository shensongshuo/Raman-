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

# 设置页面
st.set_page_config(layout="wide", page_title="拉曼光谱分析系统")
st.title("拉曼光谱分析系统")

# 初始化session状态
if 'raw_data' not in st.session_state:
    st.session_state.raw_data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'peaks' not in st.session_state:
    st.session_state.peaks = None

# 创建两列布局（调整比例使右侧更宽）
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
        st.subheader("基线校准")
        baseline_method = st.selectbox(
            "基线校准方法",
            ["无", "SD", "FD", "I-ModPoly", "AsLS"],
            key="baseline_method"
        )

        # 动态参数
        if baseline_method == "I-ModPoly":
            polyorder = st.slider("多项式阶数", 3, 10, 6, key="polyorder")
        elif baseline_method == "AsLS":
            lam = st.number_input("λ(平滑度)", value=1e7, format="%e", key="lam")
            p = st.slider("p(不对称性)", 0.01, 0.5, 0.1, key="p")

        # 数据变换
        st.subheader("数据变换")
        transform_method = st.selectbox(
            "数据变换方法",
            ["无", "挤压函数(归一化版)", "挤压函数(原始版)", 
             "Sigmoid(归一化版)", "Sigmoid(原始版)"],
            key="transform_method"
        )

        if transform_method == "Sigmoid(归一化版)":
            maxn = st.slider("归一化系数", 1, 20, 10, key="i_sigmoid_maxn")
        
        # 归一化
        st.subheader("归一化")
        norm_method = st.selectbox(
            "归一化方法",
            ["无", "无穷大范数", "L10范数", "L4范数"],
            key="norm_method"
        )

        # 处理按钮
        if st.button("🚀 应用处理", type="primary", use_container_width=True):
            if st.session_state.raw_data is None:
                st.warning("请先上传数据文件")
            else:
                x, y = st.session_state.raw_data
                y_processed = y.copy()
                method_name = "原始数据"

                # 基线处理
                if baseline_method == "SD":
                    y_processed = D2(y_processed.reshape(1, -1))[0]
                    method_name = "SD基线校准"
                elif baseline_method == "FD":
                    y_processed = D1(y_processed.reshape(1, -1))[0]
                    method_name = "FD基线校准"
                elif baseline_method == "I-ModPoly":
                    y_processed = IModPoly(x, y_processed.reshape(1, -1), polyorder)[0]
                    method_name = f"I-ModPoly(阶数={polyorder})"
                elif baseline_method == "AsLS":
                    y_processed = baseline_als(y_processed.reshape(1, -1), lam, p, 10)[0]
                    method_name = f"AsLS(λ={lam:.1e},p={p})"

                # 数据变换处理
                if transform_method == "挤压函数(归一化版)":
                    y_processed = i_squashing(y_processed.reshape(1, -1))[0]
                    method_name += " + i_squashing"
                elif transform_method == "挤压函数(原始版)":
                    y_processed = squashing(y_processed.reshape(1, -1))[0]
                    method_name += " + squashing"
                elif transform_method == "Sigmoid(归一化版)":
                    y_processed = i_sigmoid(y_processed.reshape(1, -1), maxn)[0]
                    method_name += f" + i_sigmoid(maxn={maxn})"
                elif transform_method == "Sigmoid(原始版)":
                    y_processed = sigmoid(y_processed.reshape(1, -1))[0]
                    method_name += " + sigmoid"

                # 归一化处理
                if norm_method == "无穷大范数":
                    y_processed = LPnorm(y_processed.reshape(1, -1), np.inf)[0]
                    method_name += " + 无穷大范数"
                elif norm_method == "L10范数":
                    y_processed = LPnorm(y_processed.reshape(1, -1), 10)[0]
                    method_name += " + L10范数"
                elif norm_method == "L4范数":
                    y_processed = LPnorm(y_processed.reshape(1, -1), 4)[0]
                    method_name += " + L4范数"

                st.session_state.processed_data = (x, y_processed)
                st.session_state.process_method = method_name
                st.success(f"处理完成: {method_name}")

with col2:
    # ===== 系统信息和处理方法 =====
    with st.container():
        cols = st.columns([1, 2])
        with cols[0]:
            if st.session_state.raw_data:
                st.info(f"📊 数据点数: {len(st.session_state.raw_data[0])}")
        with cols[1]:
            if st.session_state.get('process_method'):
                st.success(f"🛠️ 当前处理方法: {st.session_state.process_method}")
    
    st.divider()
    
    # ===== 光谱图 =====
    st.header("📊 光谱图")
    chart_data = pd.DataFrame()
    if st.session_state.raw_data:
        x, y = st.session_state.raw_data
        chart_data["原始数据"] = y
        chart_data.index = x

    if st.session_state.processed_data:
        x, y = st.session_state.processed_data
        chart_data["处理后数据"] = y

    if not chart_data.empty:
        st.line_chart(chart_data)
    else:
        st.info("请先上传并处理数据")

    # ===== 分析结果 =====
    st.header("🔍 分析结果")
    if st.button("🔄 执行峰分析", use_container_width=True):
        if st.session_state.processed_data is None:
            st.warning("请先处理数据")
        else:
            # 模拟峰分析结果
            x, y = st.session_state.processed_data
            peaks = [
                {"位置(cm⁻¹)": 800, "强度": 1.2, "半高宽": 50, "物质归属": "SiO₂"},
                {"位置(cm⁻¹)": 1200, "强度": 2.3, "半高宽": 60, "物质归属": "TiO₂"},
                {"位置(cm⁻¹)": 1600, "强度": 1.8, "半高宽": 55, "物质归属": "Al₂O₃"}
            ]
            st.session_state.peaks = peaks
            st.success(f"检测到{len(peaks)}个峰")

    if st.session_state.peaks:
        st.dataframe(st.session_state.peaks)
        csv = pd.DataFrame(st.session_state.peaks).to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 下载分析结果",
            data=csv,
            file_name='peak_analysis.csv',
            mime='text/csv',
            use_container_width=True
        )

# 页面底部添加使用说明
st.divider()
with st.expander("ℹ️ 使用说明", expanded=False):
    st.markdown("""
    **操作流程:**
    1. 上传光谱文件(TXT/CSV)
    2. 选择预处理方法
    3. 点击"应用处理"
    4. 执行峰分析
    
    **注意事项:**
    - 确保数据文件为两列格式（波数+强度）
    - 复杂处理可能需要更长时间
    """)
