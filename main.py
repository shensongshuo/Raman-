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

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 设置页面
st.set_page_config(layout="wide", page_title="拉曼光谱分析系统")

# 自定义CSS样式
st.markdown("""
<style>
    .sidebar .sidebar-content {
        background-color: #f5f7fa;
        border-right: 1px solid #e1e4e8;
    }
    .css-18e3th9 {
        padding-top: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    .css-1d391kg {
        padding: 1.5rem;
        border-radius: 0.75rem;
        background-color: #ffffff;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
    }
    .css-1xarl3l {
        font-weight: 600;
        color: #2d3748;
    }
    .css-q8sbsg {
        margin-top: 1rem;
    }
    .css-12oz5g7 {
        max-width: 100%;
    }
    .css-163ttbj {
        margin-bottom: 1rem;
    }
    .css-10trblm {
        text-align: center;
    }
    .css-1v3fvcr {
        border-radius: 0.375rem;
    }
    .css-102b5pv {
        background-color: #4a5568;
        color: white;
        border-radius: 0.375rem;
        padding: 0.5rem 1rem;
    }
    .css-102b5pv:hover {
        background-color: #2d3748;
    }
    .css-12w0qpk {
        border-radius: 0.375rem;
    }
    .css-1offfwp {
        background-color: #f7fafc;
        border-radius: 0.375rem;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# 页面标题
st.markdown("<h1 style='text-align: center; color: #2d3748;'>拉曼光谱分析系统</h1>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center; color: #718096; margin-bottom: 2rem;'>专业的拉曼光谱数据处理与分析平台</div>", unsafe_allow_html=True)

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
    with st.container():
        st.markdown("<h3 class='css-1xarl3l'>📁 数据管理</h3>", unsafe_allow_html=True)
        st.markdown("<div class='css-163ttbj'></div>", unsafe_allow_html=True)
        
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
    with st.container():
        st.markdown("<h3 class='css-1xarl3l'>⚙️ 预处理设置</h3>", unsafe_allow_html=True)
        st.markdown("<div class='css-163ttbj'></div>", unsafe_allow_html=True)
        
        # 基线校准
        baseline_method = st.selectbox(
            "基线校准方法",
            ["无", "SD", "FD", "I-ModPoly", "AsLS"],
            help="选择适合的基线校准方法去除背景干扰"
        )
        
        # 动态参数
        if baseline_method == "I-ModPoly":
            polyorder = st.slider("多项式阶数", 3, 10, 6, 
                                 help="多项式拟合的阶数，较高的值可以拟合更复杂的基线")
        elif baseline_method == "AsLS":
            lam = st.number_input("λ(平滑度)", value=1e7, format="%e", 
                                 help="控制基线平滑度的参数，值越大基线越平滑")
            p = st.slider("p(不对称性)", 0.01, 0.5, 0.1, 
                         help="控制拟合不对称性的参数，较小的值对负偏差更敏感")
        
        # 归一化
        norm_method = st.selectbox(
            "归一化方法",
            ["无", "无穷大范数", "L10范数", "L4范数"],
            help="选择适合的归一化方法使光谱数据具有可比性"
        )
        
        # 处理按钮
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

with col2:
    # ===== 光谱图 =====
    with st.container():
        st.markdown("<h3 class='css-1xarl3l'>📊 光谱图</h3>", unsafe_allow_html=True)
        st.markdown("<div class='css-163ttbj'></div>", unsafe_allow_html=True)
        
        # 创建图表
        if st.session_state.raw_data or st.session_state.processed_data:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            if st.session_state.raw_data:
                x, y = st.session_state.raw_data
                ax.plot(x, y, label="原始数据", color='#4299e1', alpha=0.7, linewidth=2)
            
            if st.session_state.processed_data:
                x, y = st.session_state.processed_data
                ax.plot(x, y, label="处理后数据", color='#f56565', linewidth=2)
            
            ax.set_xlabel("波长 (cm⁻¹)", fontsize=12)
            ax.set_ylabel("强度", fontsize=12)
            ax.set_title("拉曼光谱数据", fontsize=14)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()
            
            st.pyplot(fig)
        else:
            st.info("请先上传并处理数据")
    
    # ===== 分析结果 =====
    with st.container():
        st.markdown("<h3 class='css-1xarl3l'>🔍 分析结果</h3>", unsafe_allow_html=True)
        st.markdown("<div class='css-163ttbj'></div>", unsafe_allow_html=True)
        
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
            st.markdown("<div class='css-q8sbsg'></div>", unsafe_allow_html=True)
            
            # 美化表格显示
            df = pd.DataFrame(st.session_state.peaks)
            styled_df = df.style.format({
                '位置(cm⁻¹)': '{:.1f}',
                '强度': '{:.2f}',
                '半高宽': '{:.1f}'
            }).background_gradient(cmap='Blues', subset=['强度'])
            
            st.dataframe(styled_df, use_container_width=True)
            
            # 添加下载按钮
            st.markdown("<div class='css-q8sbsg'></div>", unsafe_allow_html=True)
            csv = pd.DataFrame(st.session_state.peaks).to_csv(sep='\t', na_rep='nan').encode('utf-8')
            st.download_button(
                label="📥 下载分析结果",
                data=csv,
                file_name='peak_analysis.tsv',
                mime='text/tab-separated-values',
                help="下载峰分析结果为TSV格式文件"
            )

# 侧边栏添加信息
with st.sidebar:
    st.markdown("<h3 class='css-1xarl3l'>ℹ️ 系统信息</h3>", unsafe_allow_html=True)
    st.markdown("<div class='css-163ttbj'></div>", unsafe_allow_html=True)
    
    if st.session_state.raw_data:
        st.write(f"数据点数: {len(st.session_state.raw_data[0])}")
    if st.session_state.get('process_method'):
        st.write(f"当前处理方法: {st.session_state.process_method}")
    
    st.markdown("<div class='css-163ttbj'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='css-1offfwp'>
    **使用说明:**
    1. 上传光谱文件(TXT/CSV)
    2. 选择预处理方法
    3. 点击"应用处理"
    4. 执行峰分析
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='css-163ttbj'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; color: #718096; font-size: 0.875rem;'>
    V1.0 拉曼光谱分析系统
    </div>
    """, unsafe_allow_html=True)    
