
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from openai import OpenAI


# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Jeyad AIRBNB Estimator",
    page_icon="🏠",
    layout="wide"
)


# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------
st.markdown("""
<style>

    .stApp {
        background-color: #f7f8fb;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1100px;
    }

    .hero-box {
        background: linear-gradient(135deg, #ffffff, #f1f3f8);
        padding: 32px;
        border-radius: 20px;
        border: 1px solid #e5e7eb;
        box-shadow: 0px 8px 24px rgba(0,0,0,0.06);
        margin-bottom: 28px;
    }

    .hero-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .hero-subtitle {
        font-size: 18px;
        color: #60646c;
        margin-bottom: 0px;
    }

    .section-title {
        font-size: 24px;
        font-weight: 650;
        margin-top: 10px;
        margin-bottom: 18px;
    }

    .result-box {
        background: white;
        border-radius: 20px;
        padding: 28px;
        margin-top: 24px;
        text-align: center;
        border: 1px solid #e5e7eb;
        box-shadow: 0px 8px 24px rgba(0,0,0,0.07);
    }

    .result-label {
        font-size: 18px;
        color: #6b7280;
    }

    .result-price {
        font-size: 46px;
        font-weight: 750;
        margin-top: 4px;
    }

    div.stButton > button {
        width: 100%;
        height: 3.2rem;
        border-radius: 12px;
        font-size: 18px;
        font-weight: 600;
    }

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("rental_price_prediction_model_v1_0.joblib")


model = load_model()
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------
st.markdown("""
<div class="hero-box">
    <div class="hero-title">🏠 Jeyad AIRBNB Estimator</div>
    <div class="hero-subtitle">
        Estimate the expected rental price of an Airbnb property using Machine Learning.
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------
# INPUT SECTION
# ---------------------------------------------------
st.markdown(
    '<div class="section-title">Property Details</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2, gap="large")


with col1:

    room_type = st.selectbox(
        "Room Type",
        [
            "Entire home/apt",
            "Private room",
            "Shared room"
        ]
    )

    accommodates = st.number_input(
        "Number of Guests",
        min_value=1,
        step=1,
        value=2
    )

    bathrooms = st.number_input(
        "Bathrooms",
        min_value=0.0,
        step=0.5,
        value=1.0
    )

    bedrooms = st.number_input(
        "Bedrooms",
        min_value=0,
        step=1,
        value=1
    )

    beds = st.number_input(
        "Beds",
        min_value=0,
        step=1,
        value=1
    )


with col2:

    cancellation_policy = st.selectbox(
        "Cancellation Policy",
        [
            "strict",
            "flexible",
            "moderate"
        ]
    )

    cleaning_fee = st.selectbox(
        "Cleaning Fee Included?",
        [
            "True",
            "False"
        ]
    )

    instant_bookable = st.selectbox(
        "Instant Booking Available?",
        [
            "False",
            "True"
        ]
    )

    review_scores_rating = st.slider(
        "Review Score Rating",
        min_value=0.0,
        max_value=100.0,
        step=1.0,
        value=90.0
    )


# ---------------------------------------------------
# PREPARE INPUT DATA
# ---------------------------------------------------
input_data = pd.DataFrame([{
    "room_type": room_type,
    "accommodates": accommodates,
    "bathrooms": bathrooms,
    "cancellation_policy": cancellation_policy,
    "cleaning_fee": cleaning_fee,
    "instant_bookable": "f" if instant_bookable == "False" else "t",
    "review_scores_rating": review_scores_rating,
    "bedrooms": bedrooms,
    "beds": beds
}])


# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------
st.divider()

predict_button = st.button(
    "Estimate Rental Price",
    use_container_width=True
)


if predict_button:

    prediction_log = model.predict(input_data)[0]

    predicted_price = np.exp(prediction_log)
    st.session_state.predicted_price = predicted_price

    st.markdown(
        f"""
        <div class="result-box">
            <div class="result-label">Estimated Rental Price</div>
            <div class="result-price">${predicted_price:,.2f}</div>
            <div style="color:#6b7280; margin-top:8px;">
                Based on the property details entered above
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------------------------------------------------
# AI ASSISTANT
# ---------------------------------------------------

st.divider()

st.subheader("🤖 AI Pricing Assistant")

st.write(
    "Ask questions about the estimated rental price, "
    "Airbnb pricing, or how the property features may affect the price."
)

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

for message in st.session_state.chat_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
with st.form("ai_chat_form"):
    user_question = st.text_input(
        "Ask the AI Pricing Assistant",
        placeholder="Example: Why is my estimated price this amount?"
    )

    ask_button = st.form_submit_button(
        "🤖 Ask AI",
        use_container_width=True
    )

if ask_button and user_question:

    st.session_state.chat_messages.append({
        "role": "user",
        "content": user_question
    })

    with st.chat_message("user"):
        st.markdown(user_question)

    property_context = f"""
Current property:
Room type: {room_type}
Guests: {accommodates}
Bathrooms: {bathrooms}
Bedrooms: {bedrooms}
Beds: {beds}
Cancellation policy: {cancellation_policy}
Cleaning fee: {cleaning_fee}
Instant booking: {instant_bookable}
Review rating: {review_scores_rating}
"""

    if "predicted_price" in st.session_state:
        property_context += f"""
Current ML estimated rental price:
${st.session_state.predicted_price:.2f}
"""

    prompt = f"""
You are an AI pricing assistant for an Airbnb rental price estimator.

Help the user understand the estimated rental price and the property
features in clear, practical language.

The machine learning estimate is only a prediction and should not be
presented as a guaranteed market price.

{property_context}

User question:
{user_question}
"""

    try:
        response = client.responses.create(
            model="gpt-5.6-luna",
            input=prompt
        )

        assistant_answer = response.output_text

    except Exception as e:
        assistant_answer = (
            "The AI Pricing Assistant is temporarily unavailable. "
            "Please check the OpenAI API configuration."
        )

    with st.chat_message("assistant"):
        st.markdown(assistant_answer)

    st.session_state.chat_messages.append({
        "role": "assistant",
        "content": assistant_answer
    })
# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown("""
<br><br>
<div style="text-align:center; color:#9ca3af; font-size:13px;">
    Jeyad AIRBNB Estimator • Machine Learning Price Prediction
</div>
""", unsafe_allow_html=True)
