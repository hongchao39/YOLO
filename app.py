import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import os

# 类别名称
class_names = [
    'A','B','Bullseye','C','D','E','F','G','H','S','T','U','V','W','X','Y','Z',
    'circle','down','eight','five','four','left','nine','one','right',
    'seven','six','three','two','up'
]

# 页面配置
st.set_page_config(
    page_title="YOLO 卡片识别系统",
    page_icon="🎯",
    layout="wide"
)

# 标题
st.title("🎯 YOLO 卡片识别系统")
st.markdown("上传图片，识别字母、数字和符号卡片")

# 侧边栏配置
st.sidebar.header("⚙️ 检测设置")
confidence = st.sidebar.slider("置信度阈值", 0.0, 1.0, 0.3, 0.05)
st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 可识别类别")
st.sidebar.markdown("**字母**: A-Z (除I,J,K,L,M,N,O,P,Q,R)")
st.sidebar.markdown("**数字**: 1-9")
st.sidebar.markdown("**符号**: 靶心, 圆圈, 箭头")

# 加载模型（缓存）
@st.cache_resource
def load_model():
    return YOLO('bestL160epoch.pt')

try:
    model = load_model()
    st.sidebar.success("✅ 模型加载成功")
except Exception as e:
    st.error(f"❌ 模型加载失败: {e}")
    st.stop()

# 文件上传
uploaded_file = st.file_uploader(
    "📤 上传图片 (支持 JPG, PNG, JPEG)",
    type=['jpg', 'jpeg', 'png']
)

if uploaded_file is not None:
    # 读取图片
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    # 显示原图和结果
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 原始图片")
        st.image(image, use_container_width=True)
    
    # 运行检测
    with st.spinner("🔍 正在检测..."):
        results = model.predict(
            source=img_array,
            conf=confidence,
            save=False
        )
    
    # 绘制结果
    result_img = results[0].plot()  # 返回带标注的图片
    result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    
    with col2:
        st.subheader("🎯 检测结果")
        st.image(result_img_rgb, use_container_width=True)
    
    # 显示检测详情
    st.markdown("---")
    st.subheader("📊 检测详情")
    
    boxes = results[0].boxes
    if len(boxes) > 0:
        # 创建结果表格
        detection_data = []
        for i, box in enumerate(boxes):
            cls_idx = int(box.cls.cpu().numpy()[0])
            conf = float(box.conf.cpu().numpy()[0])
            class_name = class_names[cls_idx]
            
            detection_data.append({
                "序号": i + 1,
                "类别": class_name,
                "置信度": f"{conf:.2%}",
            })
        
        # 显示表格
        st.table(detection_data)
        
        # 统计信息
        st.success(f"✅ 共检测到 **{len(boxes)}** 个对象")
    else:
        st.warning("⚠️ 未检测到任何对象，尝试：\n- 降低置信度阈值\n- 使用包含实体卡片的图片\n- 确保图片清晰")

else:
    # 未上传图片时显示说明
    st.info("👆 请上传一张图片开始检测")
    
    # 显示示例
    st.markdown("---")
    st.subheader("💡 使用提示")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### ✅ 适合的图片")
        st.markdown("- 实体字母/数字卡片")
        st.markdown("- 清晰的照片")
        st.markdown("- 光线充足")
    
    with col2:
        st.markdown("#### ❌ 不适合的图片")
        st.markdown("- 电脑屏幕截图")
        st.markdown("- 模糊的照片")
        st.markdown("- 手写的字母")
    
    with col3:
        st.markdown("#### ⚙️ 调整建议")
        st.markdown("- 未检测到：降低置信度")
        st.markdown("- 误检太多：提高置信度")
        st.markdown("- 默认值：0.3")
