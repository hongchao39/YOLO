import gradio as gr
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os

# 类别名称
class_names = [
    'A','B','Bullseye','C','D','E','F','G','H','S','T','U','V','W','X','Y','Z',
    'circle','down','eight','five','four','left','nine','one','right',
    'seven','six','three','two','up'
]

# Load model
print("Loading model...")
model = YOLO('bestL160epoch.pt')
print("Model loaded successfully!")

def predict_image(image, confidence):
    """
    Prediction function
    Args:
        image: PIL Image or numpy array
        confidence: Confidence threshold (0-1)
    Returns:
        Annotated image, detection results text
    """
    if image is None:
        return None, "Please upload an image"
    
    # Run prediction
    results = model.predict(
        source=image,
        conf=confidence,
        save=False
    )
    
    # Get annotated image
    annotated_img = results[0].plot()
    
    # Get detection details
    boxes = results[0].boxes
    
    if len(boxes) > 0:
        # Generate detection results text
        result_text = f"✅ Detected {len(boxes)} object(s):\n\n"
        
        for i, box in enumerate(boxes):
            cls_idx = int(box.cls.cpu().numpy()[0])
            conf = float(box.conf.cpu().numpy()[0])
            class_name = class_names[cls_idx]
            
            result_text += f"{i+1}. {class_name} - Confidence: {conf:.2%}\n"
    else:
        result_text = "⚠️ No objects detected\n\nSuggestions:\n- Lower the confidence threshold\n- Use images with physical cards\n- Ensure the image is clear"
    
    return annotated_img, result_text


# Create Gradio interface
with gr.Blocks(title="YOLO Card Detection System") as demo:
    gr.Markdown("# 🎯 YOLO Card Detection System")
    gr.Markdown("Upload an image to detect letter, number, and symbol cards")
    
    with gr.Row():
        with gr.Column():
            # Input
            input_image = gr.Image(
                label="Upload Image",
                type="pil"
            )
            
            confidence_slider = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.3,
                step=0.05,
                label="Confidence Threshold"
            )
            
            submit_btn = gr.Button("🔍 Detect", variant="primary")
            
            # Examples
            gr.Examples(
                examples=[
                    ["test.jpg", 0.3],
                    ["test2.jpg", 0.3],
                ] if any([os.path.exists("test.jpg"), os.path.exists("test2.jpg")]) else [],
                inputs=[input_image, confidence_slider]
            )
        
        with gr.Column():
            # Output
            output_image = gr.Image(label="Detection Result")
            output_text = gr.Textbox(
                label="Detection Details",
                lines=10
            )
    
    # Detectable categories
    with gr.Accordion("📋 Detectable Card Categories (30 types)", open=False):
        gr.Markdown("""
        **Letter Cards (17):**
        A, B, C, D, E, F, G, H, S, T, U, V, W, X, Y, Z
        
        **Number Cards (9):**
        1 (one), 2 (two), 3 (three), 4 (four), 5 (five), 6 (six), 7 (seven), 8 (eight), 9 (nine)
        
        **Symbol Cards (4):**
        Bullseye, circle, up/down/left/right (arrow directions)
        """)
    
    with gr.Accordion("💡 Usage Tips", open=False):
        gr.Markdown("""
        **✅ Suitable Images:**
        - Physical letter/number cards
        - Clear photos
        - Good lighting
        
        **❌ Not Suitable:**
        - Computer screen screenshots
        - Blurry photos
        - Handwritten letters
        
        **⚙️ Adjustment Tips:**
        - Nothing detected: Lower confidence threshold
        - Too many false positives: Raise confidence threshold
        - Default value: 0.3
        """)
    
    # Bind event
    submit_btn.click(
        fn=predict_image,
        inputs=[input_image, confidence_slider],
        outputs=[output_image, output_text]
    )

# Launch app
if __name__ == "__main__":
    import os
    
    # Get host and port from environment variables (for Render deployment)
    host = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
    port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    
    demo.launch(
        share=False,  # Set to True to generate a public link
        server_name=host,
        server_port=port
    )
