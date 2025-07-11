import streamlit as st
import numpy as np
import pandas as pd
from SD import D2
from FD import D1
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

# 创建两列布局
col1, col2 = st.columns([1, 3])

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
        baseline_method = st.selectbox(
            "基线校准方法",
            ["无", "SD", "FD", "I-ModPoly", "AsLS"]
        )

        # 动态参数
        if baseline_method == "I-ModPoly":
            polyorder = st.slider("多项式阶数", 3, 10, 6)
        elif baseline_method == "AsLS":
            lam = st.number_input("λ(平滑度)", value=1e7, format="%e")
            p = st.slider("p(不对称性)", 0.01, 0.5, 0.1)

        # 归一化
        norm_method = st.selectbox(
            "归一化方法",
            ["无", "无穷大范数", "L10范数", "L4范数"]
        )

        # 处理按钮
        if st.button("🚀 应用处理", type="primary"):
            if st.session_state.raw_data is None:
                st.warning("请先上传数据文件")
            else:
                x, y = st.session_state.raw_data
                y_processed = y.copy()

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
                else:
                    method_name = "未处理"

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
    # ===== 光谱图 =====
    st.header("📊 光谱图")

    # 创建图表
    chart_data = pd.DataFrame()
    if st.session_state.raw_data:
        x, y = st.session_state.raw_data
        chart_data["原始数据"] = y
        chart_data.index = x  # 使用x作为索引

    if st.session_state.processed_data:
        x, y = st.session_state.processed_data
        chart_data["处理后数据"] = y

    if not chart_data.empty:
        st.line_chart(chart_data)
    else:
        st.info("请先上传并处理数据")

    # ===== 分析结果 =====
    st.header("🔍 分析结果")

    if st.button("🔄 执行峰分析"):
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

        # 添加下载按钮
        csv = pd.DataFrame(st.session_state.peaks).to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 下载分析结果",
            data=csv,
            file_name='peak_analysis.csv',
            mime='text/csv'
        )

# 侧边栏添加信息
with st.sidebar:
    st.header("ℹ️ 系统信息")
    if st.session_state.raw_data:
        st.write(f"数据点数: {len(st.session_state.raw_data[0])}")
 
     if st.session_state.process_method:
        st.write(f"当前处理方法: {st.session_state.process_method}")

    st.divider()
    st.markdown("""
    **使用说明:**
    1. 上传光谱文件(TXT/CSV)
    2. 选择预处理方法
    3. 点击"应用处理"
    4. 执行峰分析
    """)
