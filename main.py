import streamlit as st
import numpy as np
import pandas as pd
from SD import D2
from FD import D1
from IModPoly import IModPoly
from AsLS import baseline_als
from LPnorm import LPnorm
import matplotlib.pyplot as plt
import matplotlib as mpl
import base64

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 设置页面
st.set_page_config(layout="wide", page_title="拉曼光谱分析系统")

# 自定义CSS样式 - 参考科学仪器公司设计风格
st.markdown("""
<style>
    /* 全局样式 */
    body {
        font-family: 'Inter', sans-serif;
    }
    .css-18e3th9 {
        padding: 0;
    }
    .css-1d391kg {
        border-radius: 0.5rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }
    .css-1d391kg:hover {
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
    }
    
    /* 顶部导航栏 */
    .header {
        background-color: #0F3460;
        color: white;
        padding: 1rem 2rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .logo {
        font-size: 1.5rem;
        font-weight: 600;
        display: flex;
        align-items: center;
    }
    .logo-icon {
        margin-right: 0.5rem;
    }
    .nav-links {
        display: flex;
        gap: 1.5rem;
    }
    .nav-link {
        color: rgba(255, 255, 255, 0.8);
        text-decoration: none;
        transition: color 0.3s;
    }
    .nav-link:hover {
        color: white;
    }
    
    /* 主内容区 */
    .main-content {
        padding: 2rem;
        background-color: #F8FAFC;
    }
    .section-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1E293B;
        margin-bottom: 1rem;
        border-bottom: 2px solid #E2E8F0;
        padding-bottom: 0.5rem;
    }
    
    /* 控制面板 */
    .control-panel {
        background-color: white;
        padding: 1.5rem;
        border-radius: 0.75rem;
    }
    .control-group {
        margin-bottom: 1.5rem;
    }
    .control-label {
        font-weight: 500;
        color: #334155;
        margin-bottom: 0.5rem;
    }
    
    /* 图表区域 */
    .chart-container {
        background-color: white;
        padding: 1.5rem;
        border-radius: 0.75rem;
    }
    
    /* 按钮样式 */
    .stButton>button {
        background-color: #165DFF;
        color: white;
        border-radius: 0.375rem;
        padding: 0.5rem 1.25rem;
        font-weight: 500;
        transition: all 0.2s;
        border: none;
    }
    .stButton>button:hover {
        background-color: #0E42D2;
        transform: translateY(-1px);
        box-shadow: 0 4px 6px rgba(22, 93, 255, 0.1);
    }
    .stButton>button:active {
        transform: translateY(0);
        box-shadow: none;
    }
    
    /* 表格样式 */
    .dataframe {
        border-collapse: collapse;
        width: 100%;
    }
    .dataframe th {
        background-color: #F1F5F9;
        color: #334155;
        padding: 0.75rem;
        text-align: left;
        font-weight: 600;
        border-bottom: 2px solid #E2E8F0;
    }
    .dataframe td {
        padding: 0.75rem;
        border-bottom: 1px solid #E2E8F0;
    }
    .dataframe tr:hover {
        background-color: #F8FAFC;
    }
    
    /* 侧边栏 */
    .sidebar .sidebar-content {
        background-color: #1E293B;
        color: white;
    }
    .sidebar-title {
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 1rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        padding-bottom: 0.5rem;
    }
    .sidebar-item {
        margin-bottom: 0.75rem;
    }
</style>
""", unsafe_allow_html=True)

# 页面标题和导航栏
st.markdown("""
<div class="header">
    <div class="logo">
        <span class="logo-icon">🔬</span>
        <span>SpectroAnalyzer</span>
    </div>
    <div class="nav-links">
        <a href="#" class="nav-link">主页</a>
        <a href="#" class="nav-link">分析工具</a>
        <a href="#" class="nav-link">帮助文档</a>
        <a href="#" class="nav-link">关于我们</a>
    </div>
</div>
""", unsafe_allow_html=True)

# 主内容区
st.markdown("<div class='main-content'>", unsafe_allow_html=True)

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
    st.markdown("<div class='control-panel'>", unsafe_allow_html=True)
    
    # ===== 数据管理 =====
    st.markdown("<h3 class='section-title'>📁 数据管理</h3>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("上传光谱文件", type=['txt', 'csv'], 
                                     help="支持TXT或CSV格式文件，第一列为波长，第二列为强度")
    
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
    st.markdown("<h3 class='section-title'>⚙️ 预处理设置</h3>", unsafe_allow_html=True)
    
    # 基线校准
    st.markdown("<div class='control-group'>", unsafe_allow_html=True)
    st.markdown("<label class='control-label'>基线校准方法</label>", unsafe_allow_html=True)
    baseline_method = st.selectbox(
        "",
        ["无", "SD", "FD", "I-ModPoly", "AsLS"],
        key="baseline_method",
        help="选择适合的基线校准方法去除背景干扰"
    )
    st.markdown("</div>", unsafe_allow_html=True)
    
    # 动态参数
    if baseline_method == "I-ModPoly":
        st.markdown("<div class='control-group'>", unsafe_allow_html=True)
        st.markdown("<label class='control-label'>多项式阶数</label>", unsafe_allow_html=True)
        polyorder = st.slider(
            "",
            3, 10, 6, 
            key="polyorder",
            help="多项式拟合的阶数，较高的值可以拟合更复杂的基线"
        )
        st.markdown("</div>", unsafe_allow_html=True)
    elif baseline_method == "AsLS":
        st.markdown("<div class='control-group'>", unsafe_allow_html=True)
        st.markdown("<label class='control-label'>λ(平滑度)</label>", unsafe_allow_html=True)
        lam = st.number_input(
            "",
            value=1e7, format="%e", 
            key="lam",
            help="控制基线平滑度的参数，值越大基线越平滑"
        )
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='control-group'>", unsafe_allow_html=True)
        st.markdown("<label class='control-label'>p(不对称性)</label>", unsafe_allow_html=True)
        p = st.slider(
            "",
            0.01, 0.5, 0.1, 
            key="p",
            help="控制拟合不对称性的参数，较小的值对负偏差更敏感"
        )
        st.markdown("</div>", unsafe_allow_html=True)
    
    # 归一化
    st.markdown("<div class='control-group'>", unsafe_allow_html=True)
    st.markdown("<label class='control-label'>归一化方法</label>", unsafe_allow_html=True)
    norm_method = st.selectbox(
        "",
        ["无", "无穷大范数", "L10范数", "L4范数"],
        key="norm_method",
        help="选择适合的归一化方法使光谱数据具有可比性"
    )
    st.markdown("</div>", unsafe_allow_html=True)
    
    # 处理按钮
    st.markdown("<div class='control-group'>", unsafe_allow_html=True)
    if st.button("🚀 应用处理", type="primary", help="点击执行所选的预处理方法"):
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
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    # ===== 光谱图 =====
    st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
    st.markdown("<h3 class='section-title'>📊 光谱图</h3>", unsafe_allow_html=True)
    
    # 创建图表
    if st.session_state.raw_data or st.session_state.processed_data:
        fig, ax = plt.subplots(figsize=(14, 7))
        
        if st.session_state.raw_data:
            x, y = st.session_state.raw_data
            ax.plot(x, y, label="原始数据", color='#4299e1', alpha=0.7, linewidth=2)
        
        if st.session_state.processed_data:
            x, y = st.session_state.processed_data
            ax.plot(x, y, label="处理后数据", color='#f56565', linewidth=2)
        
        ax.set_xlabel("波长 (cm⁻¹)", fontsize=14)
        ax.set_ylabel("强度", fontsize=14)
        ax.set_title("拉曼光谱数据", fontsize=16, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        # 美化图表
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        
        st.pyplot(fig)
    else:
        st.info("请先上传并处理数据")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # ===== 分析结果 =====
    st.markdown("<div class='chart-container' style='margin-top: 1.5rem;'>", unsafe_allow_html=True)
    st.markdown("<h3 class='section-title'>🔍 分析结果</h3>", unsafe_allow_html=True)
    
    if st.button("🔄 执行峰分析", help="对当前处理后的光谱数据进行峰检测和分析"):
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
        st.markdown("<div style='margin-top: 1rem;'>", unsafe_allow_html=True)
        
        # 美化表格显示
        df = pd.DataFrame(st.session_state.peaks)
        styled_df = df.style.format({
            '位置(cm⁻¹)': '{:.1f}',
            '强度': '{:.2f}',
            '半高宽': '{:.1f}'
        }).background_gradient(cmap='Blues', subset=['强度'])
        
        st.dataframe(styled_df, use_container_width=True)
        
        # 添加下载按钮
        st.markdown("<div style='margin-top: 1.5rem;'>", unsafe_allow_html=True)
        csv = pd.DataFrame(st.session_state.peaks).to_csv(sep='\t', na_rep='nan').encode('utf-8')
        st.download_button(
            label="📥 下载分析结果",
            data=csv,
            file_name='peak_analysis.tsv',
            mime='text/tab-separated-values',
            help="下载峰分析结果为TSV格式文件"
        )
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# 侧边栏添加信息
with st.sidebar:
    st.markdown("<h3 class='sidebar-title'>ℹ️ 系统信息</h3>", unsafe_allow_html=True)
    
    if st.session_state.raw_data:
        st.markdown(f"<div class='sidebar-item'>数据点数: {len(st.session_state.raw_data[0])}</div>", unsafe_allow_html=True)
    if st.session_state.get('process_method'):
        st.markdown(f"<div class='sidebar-item'>当前处理方法: {st.session_state.process_method}</div>", unsafe_allow_html=True)
    
    st.markdown("<div style='margin-top: 1.5rem; padding: 1rem; background-color: rgba(255, 255, 255, 0.05); border-radius: 0.5rem;'>", unsafe_allow_html=True)
    st.markdown("""
    **使用说明:**
    1. 上传光谱文件(TXT/CSV)
    2. 选择预处理方法
    3. 点击"应用处理"
    4. 执行峰分析
    """)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div style='margin-top: 2rem; text-align: center; color: rgba(255, 255, 255, 0.6); font-size: 0.875rem;'>", unsafe_allow_html=True)
    st.markdown("© 2025 SpectroAnalyzer™")
    st.markdown("版本 1.0.0")
    st.markdown("</div>", unsafe_allow_html=True)
