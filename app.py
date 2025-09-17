import streamlit as st
from utils import predict_disease
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import os
import google.generativeai as genai

# -------------------------------
# Streamlit App Config
# -------------------------------
st.set_page_config(page_title="🌿 Agro Aid", layout="wide")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------------------------------
# Custom CSS Styling
# -------------------------------
st.markdown(
    """
    <style>
    body {
        background-color: #f4f9f4;
    }
    .main {
        background: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .stButton>button {
        border-radius: 10px;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.05);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# Sidebar Info
# -------------------------------
st.sidebar.title("ℹ️ About the App")
st.sidebar.info(
    """
    🌿 **Plant Disease Detection system - AGRO AID**  
    Upload a leaf image, and the AI model will:  
    - Detect the disease  
    - Show confidence score  
    - Provide treatment recommendations  
    - Generate a detailed PDF report  
    - Ask chatbot for farming queries  
    """
)
st.sidebar.success("✅ Powered by Deep Learning + Gemini AI")

# -------------------------------
# App Header
# -------------------------------
st.title("🌿 Plant Disease Detection & Recommendation")
st.write("Upload a leaf image to detect the disease and receive treatment advice.")

# -------------------------------
# File uploader
# -------------------------------
uploaded_file = st.file_uploader("📂 Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    temp_path = "temp.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    st.image(temp_path, caption="🌱 Uploaded Leaf Image", use_column_width=True)

    with st.spinner("🔍 Analyzing..."):
        try:
            disease, disease_type, recommendation, confidence = predict_disease(temp_path)

            st.markdown("### 🧾 Prediction Result")
            col1, col2 = st.columns(2)

            with col1:
                st.success(f"🌱 **Detected Disease:** {disease}")
                st.info(f"🧪 **Type:** {disease_type}")

            with col2:
                st.warning(f"📊 **Confidence:** {confidence:.2f}%")

            st.markdown("---")
            st.subheader("💡 Recommendation")
            st.success(recommendation)

            # -------------------------------
            # Generate PDF Report (with chatbot conversation)
            # -------------------------------
            def generate_pdf(disease, disease_type, confidence, recommendation, chat_history):
                pdf_path = "plant_disease_report.pdf"
                doc = SimpleDocTemplate(pdf_path, pagesize=A4)
                styles = getSampleStyleSheet()
                story = []

                story.append(Paragraph("🌿 Plant Disease Detection Report", styles["Title"]))
                story.append(Spacer(1, 20))

                story.append(Paragraph(f"<b>Disease:</b> {disease}", styles["Normal"]))
                story.append(Paragraph(f"<b>Disease Type:</b> {disease_type}", styles["Normal"]))
                story.append(Paragraph(f"<b>Confidence:</b> {confidence:.2f}%", styles["Normal"]))
                story.append(Spacer(1, 12))

                story.append(Paragraph("<b>Recommendation:</b>", styles["Heading2"]))
                story.append(Paragraph(recommendation, styles["Normal"]))
                story.append(Spacer(1, 20))

                if chat_history:
                    story.append(Paragraph("<b>Chatbot Conversation:</b>", styles["Heading2"]))
                    for q, a in chat_history:
                        story.append(Paragraph(f"🧑 You: {q}", styles["Normal"]))
                        story.append(Paragraph(f"🤖 Bot: {a}", styles["Normal"]))
                        story.append(Spacer(1, 6))

                doc.build(story)
                return pdf_path

            # Add PDF download button in sidebar
            st.sidebar.subheader("📥 Download Report")
            if st.sidebar.button("Generate PDF Report"):
                pdf_file = generate_pdf(disease, disease_type, confidence, recommendation, st.session_state.chat_history)
                with open(pdf_file, "rb") as file:
                    st.sidebar.download_button(
                        label="⬇️ Download Report",
                        data=file,
                        file_name="plant_disease_report.pdf",
                        mime="application/pdf"
                    )

        except Exception as e:
            st.error(f"❌ Error: {e}")

    os.remove(temp_path)

# -------------------------------
# Chatbot Assistant Section
# -------------------------------
st.markdown("---")
st.header("🤖 Farming Chatbot Assistant")

# Configure Gemini (API key in env)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

user_input = st.text_input("💬 Ask me anything about farming, crops, or plant diseases:")

if user_input:
    try:
        with st.spinner("🤔 Thinking... Generating response..."):
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(user_input)
            answer = response.text

            # Save to session history
            st.session_state.chat_history.append((user_input, answer))

            # Show in chat
            st.chat_message("user").markdown(user_input)
            st.chat_message("assistant").markdown(answer)

    except Exception as e:
        st.error(f"⚠️ Chatbot Error: {e}")
