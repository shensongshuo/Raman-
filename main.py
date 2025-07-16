# 在main.py顶部添加
from sklearn.linear_model import LinearRegression
def MSC(sdata):
    n = sdata.shape[0]  # 样本数量
    k = np.zeros(sdata.shape[0])
    b = np.zeros(sdata.shape[0])
 
    M = np.mean(sdata, axis=0)
 
    for i in range(n):
        y = sdata[i, :]
        y = y.reshape(-1, 1)
        M = M.reshape(-1, 1)
        model = LinearRegression()
        model.fit(M, y)
        k[i] = model.coef_
        b[i] = model.intercept_
#global spec_msc
    spec_msc = np.zeros_like(sdata)
    for i in range(n):
        bb = np.repeat(b[i], sdata.shape[1])
        kk = np.repeat(k[i], sdata.shape[1])
        temp = (sdata[i, :] - bb)/kk
        spec_msc[i, :] = temp
    return spec_msc
 pass



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
from MMnorm import MaMinorm 
# from MSCdef import MSC 
from SNV import standardization 
 




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
# ===== 数据变换 =====
 # 数据变换
        st.subheader("🔧数据变换1")
        transform_method = st.selectbox(
            "数据变换方法",
            ["无", "挤压函数(归一化版)", "挤压函数(原始版)", 
             "Sigmoid(归一化版)", "Sigmoid(原始版)"],
            key="transform_method",
            help="选择要应用的数据变换方法"
        )
        st.subheader("🔧 数据变换21（滤波处理）")
        filter_method = st.selectbox(
        "滤波方法",
        ["无", "傅里叶滤波(Smfft)", "卡尔曼滤波(KalmanF)", 
         "移动平均滤波(MWA)", "中值滤波(MWM)", 
         "Savitzky-Golay滤波(SG)", "小波滤波(wavelet)"],
        key="filter_method"
    )
        # 滤波参数动态设置
        if filter_method == "傅里叶滤波(Smfft)":
            row_e = st.slider("截止频率", 1, 100, 51, 
                         help="值越小滤波越强，保留的低频成分越多")
    
        elif filter_method == "卡尔曼滤波(KalmanF)":
            R = st.number_input("测量噪声方差(R)", value=0.0001, format="%f",
                          help="值越大滤波效果越平滑")
    
        elif filter_method in ["移动平均滤波(MWA)", "中值滤波(MWM)"]:
            n = st.slider("窗口大小", 3, 21, 7, step=2,
                     help="必须是奇数")
            iterations = st.slider("迭代次数", 1, 5, 1)
    
        elif filter_method == "Savitzky-Golay滤波(SG)":
            point = st.slider("窗口点数", 5, 31, 11, step=2)
            degree = st.slider("多项式阶数", 1, 5, 3)
    
        elif filter_method == "小波滤波(wavelet)":
            threshold = st.slider("阈值系数", 0.01, 1.0, 0.3, step=0.01,
                            help="值越大去噪越强")

# ===== 新增：数据变换方法3 =====
        st.subheader("📊 数据变换方法3（高级处理）")
        advanced_method = st.selectbox(
        "高级处理方法",
        ["无", "LP范数归一化", "最大最小归一化(MaMinorm)", 
         "多元散射校正(MSC)", "标准化(plotst)"],
        key="advanced_method"
    )
        if advanced_method == "LP范数归一化":
            lp_order = st.selectbox(
                "LP范数阶数",
                ["L1范数", "L2范数", "L4范数", "L10范数", "无穷大范数"],
                key="lp_order"
            )



        
        if transform_method == "Sigmoid(归一化版)":
            maxn = st.slider("归一化系数", 1, 20, 10, key="i_sigmoid_maxn", help="控制归一化程度，值越大归一化效果越强")
        
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

                # 在处理按钮部分添加滤波处理
                if filter_method == "傅里叶滤波(Smfft)":
                    y_processed = Smfft(y_processed, row_e)
                    method_name.append(f"Smfft(截止={row_e})")
    
                elif filter_method == "卡尔曼滤波(KalmanF)":
                    y_processed = KalmanF(y_processed, R)
                    method_name.append(f"KalmanF(R={R})")
    
                elif filter_method == "移动平均滤波(MWA)":
                    y_processed = MWA(y_processed, n, iterations)
                    method_name.append(f"MWA(窗口={n},迭代={iterations})")
    
                elif filter_method == "中值滤波(MWM)":
                    y_processed = MWM(y_processed, n, iterations)
                    method_name.append(f"MWM(窗口={n},迭代={iterations})")
    
                elif filter_method == "Savitzky-Golay滤波(SG)":
                    y_processed = SGfilter(y_processed, point, degree)
                    method_name.append(f"SG(点数={point},阶数={degree})")
    
                elif filter_method == "小波滤波(wavelet)":
                    y_processed = waveletlinear(y_processed, threshold)
                    method_name.append(f"小波(阈值={threshold})")
            # # 在处理按钮部分添加高级处理方法
            #     if advanced_method == "LP范数归一化":
            #         if lp_order == "L1范数":
            #             y_processed = LPnorm(y_processed.reshape(1, -1), 1)[0]  # 确保这里缩进 4 个空格或 1 个 Tab
            #     elif lp_order == "L2范数":
            #             y_processed = LPnorm(y_processed.reshape(1, -1), 2)[0]
            #     elif lp_order == "L4范数":
            #             y_processed = LPnorm(y_processed.reshape(1, -1), 4)[0]
            #     elif lp_order == "L10范数":
            #             y_processed = LPnorm(y_processed.reshape(1, -1), 10)[0]
            #     else:  # 无穷大范数
            #             y_processed = LPnorm(y_processed.reshape(1, -1), np.inf)[0]
            #     method_name += f" + LPnorm({lp_order})"  # 这行缩进应与 if 对齐

            #   elif advanced_method == "最大最小归一化(MaMinorm)":
            #       y_processed = MaMinorm(y_processed.reshape(1, -1))[0]
            #       method_name += " + MaMinorm"

            #   elif advanced_method == "多元散射校正(MSC)":
            #       y_processed = MSC(y_processed.reshape(1, -1))[0]
            #       method_name += " + MSC"

            #   elif advanced_method == "标准化(plotst)":
            #       y_processed = plotst(y_processed.reshape(1, -1))[0]
            #       method_name += " + 标准化"
            # 在处理按钮部分添加高级处理方法
                if advanced_method == "LP范数归一化":
                    lp_order = st.session_state.get('lp_order', "L1范数")  # 需要确保lp_order在之前有定义
                    if lp_order == "L1范数":
                        y_processed = LPnorm(y_processed.reshape(1, -1), 1)[0]
                    elif lp_order == "L2范数":
                        y_processed = LPnorm(y_processed.reshape(1, -1), 2)[0]
                    elif lp_order == "L4范数":
                        y_processed = LPnorm(y_processed.reshape(1, -1), 4)[0]
                    elif lp_order == "L10范数":
                        y_processed = LPnorm(y_processed.reshape(1, -1), 10)[0]
                    else:  # 无穷大范数
                        y_processed = LPnorm(y_processed.reshape(1, -1), np.inf)[0]
                    method_name += f" + LPnorm({lp_order})"

                elif advanced_method == "最大最小归一化(MaMinorm)":
                    y_processed = (y_processed - np.min(y_processed)) / (np.max(y_processed) - np.min(y_processed))
                    method_name += " + MaMinorm"
        
                elif advanced_method == "多元散射校正(MSC)":
                    y_processed = MSC(y_processed.reshape(1, -1))[0]
                    method_name += " + MSC"

                elif advanced_method == "标准化(plotst)":
                    y_processed = plotst(y_processed.reshape(1, -1))[0]
                    method_name += " + 标准化"



                
                
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
# 底部工具提示
st.caption("💡 提示：在任何步骤遇到问题，请参考顶部的使用指南")
