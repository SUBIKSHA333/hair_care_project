from datetime import datetime
import os
if not os.path.exists("hair_images"):
    os.makedirs("hair_images")

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from fpdf import FPDF


# 💬 Hair Doctor Chatbot Responses
symptom_responses = {
    "dandruff": "Use neem or tea tree oil shampoos. You can also apply lemon juice with coconut oil before wash.",
    "itchy": "Apply aloe vera gel or tea tree oil. Avoid harsh shampoos.",
    "dry": "Massage your scalp with warm coconut or almond oil 2–3 times a week.",
    "hair fall": "Include more protein, iron, and omega-3 in your diet. Try Ayurvedic oils like Bringraj.",
    "split ends": "Trim hair regularly and apply leave-in conditioners.",
    "greasy": "Use a gentle shampoo more frequently and avoid heavy conditioners.",
    "thinning": "Try scalp massage, biotin-rich foods, and avoid heat styling."
}

# 🧴 Smart Product Recommendations
recommendations = {
    "Low": {
        "shampoo": "Mild Herbal Shampoo (e.g., Himalaya Protein Shampoo)",
        "oil": "Coconut or Almond Oil",
        "tips": "Maintain a balanced diet, regular sleep, and avoid over-washing."
    },
    "Medium": {
        "shampoo": "Anti-hair fall Shampoo (e.g., Indulekha or Biotique Bio Kelp)",
        "oil": "Bringraj or Onion Oil",
        "tips": "Oil massage 2x/week, avoid chemical treatments, use silk pillowcase."
    },
    "High": {
        "shampoo": "Dermatologist-recommended shampoo (e.g., Ketoconazole-based)",
        "oil": "Ayurvedic oils (e.g., Kesh King, Bringadi)",
        "tips": "Consult a dermatologist. Use protein packs & iron-rich foods."
    }
}

# Load dataset
df = pd.read_csv("hair_fall_risk_dataset.csv")

# Encode categorical columns
gender_le = LabelEncoder()
shampoo_le = LabelEncoder()
diet_le = LabelEncoder()
water_le = LabelEncoder()
risk_le = LabelEncoder()

df["Gender"] = gender_le.fit_transform(df["Gender"])
df["Shampoo_Type"] = shampoo_le.fit_transform(df["Shampoo_Type"])
df["Diet_Quality"] = diet_le.fit_transform(df["Diet_Quality"])
df["Water_Type"] = water_le.fit_transform(df["Water_Type"])
df["Hair_Fall_Risk"] = risk_le.fit_transform(df["Hair_Fall_Risk"])

# Features and target
X = df.drop("Hair_Fall_Risk", axis=1)
y = df["Hair_Fall_Risk"]

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)

# --- Streamlit UI ---
st.title("\U0001f487‍♀️ Hair Fall Risk Predictor")
st.write("Enter your details to find your hair fall risk level.")

# --- User Name Input ---
user_name = st.text_input("\U0001f464 Enter your name (for history tracking):", key="username")
if user_name == "":
    st.warning("Please enter your name to proceed.")
    st.stop()

# Input fields
age = st.slider("Age", 15, 60, 25)
gender = st.selectbox("Gender", ["F", "M"])
sleep = st.slider("Sleep (hours/day)", 4, 10, 7)
stress = st.slider("Stress Level (1 = low, 10 = high)", 1, 10, 5)
wash_freq = st.slider("Hair Wash Frequency (times/week)", 1, 7, 3)
shampoo_type = st.selectbox("Shampoo Type", ["Herbal", "Chemical"])
diet_quality = st.selectbox("Diet Quality", ["Poor", "Average", "Good", "Excellent"])
water_type = st.selectbox("Water Type", ["Hard", "Soft"])

# Predict button
if st.button("Predict Hair Fall Risk"):
    user_input = pd.DataFrame({
        "Age": [age],
        "Gender": gender_le.transform([gender]),
        "Sleep": [sleep],
        "Stress": [stress],
        "Wash_Freq": [wash_freq],
        "Shampoo_Type": shampoo_le.transform([shampoo_type]),
        "Diet_Quality": diet_le.transform([diet_quality]),
        "Water_Type": water_le.transform([water_type])
    })

    prediction = model.predict(user_input)
    risk = risk_le.inverse_transform(prediction)[0]
    st.success(f"\U0001f9e0 Your predicted hair fall risk is: **{risk}**")

    # --- Save History to CSV ---
    history_data = {
        "Name": [user_name],
        "Date": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        "Age": [age],
        "Gender": [gender],
        "Sleep": [sleep],
        "Stress": [stress],
        "Wash_Freq": [wash_freq],
        "Shampoo_Type": [shampoo_type],
        "Diet_Quality": [diet_quality],
        "Water_Type": [water_type],
        "Risk_Level": [risk]
    }

    history_df = pd.DataFrame(history_data)

    if not os.path.exists("hair_history.csv"):
        history_df.to_csv("hair_history.csv", index=False)
    else:
        history_df.to_csv("hair_history.csv", mode='a', index=False, header=False)

    # Prediction confidence chart
    probs = model.predict_proba(user_input)[0]
    class_names = risk_le.inverse_transform([0, 1, 2])
    prob_df = pd.DataFrame({
        "Risk Level": class_names,
        "Probability": probs
    })
    st.subheader("\U0001f4ca Prediction Confidence")
    st.bar_chart(prob_df.set_index("Risk Level"))

    # Lifestyle bar chart
    input_data = {
        "Sleep": sleep,
        "Stress": stress,
        "Wash Frequency": wash_freq
    }
    st.subheader("\U0001f4c8 Your Lifestyle Inputs")
    st.bar_chart(pd.DataFrame.from_dict(input_data, orient='index', columns=["Value"]))
    st.markdown("---")
    st.subheader("📄 Download Your Report")

    if st.button("Download PDF Report"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        pdf.cell(200, 10, txt="Hair Fall Risk Report", ln=True, align='C')
        pdf.ln(10)

        pdf.cell(200, 10, txt=f"Name: {user_name}", ln=True)
        pdf.cell(200, 10, txt=f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
        pdf.ln(5)
        pdf.cell(200, 10, txt=f"Hair Fall Risk Level: {risk}", ln=True)
        pdf.cell(200, 10, txt=f"Sleep: {sleep} hrs/day", ln=True)
        pdf.cell(200, 10, txt=f"Stress: {stress} / 10", ln=True)
        pdf.cell(200, 10, txt=f"Wash Frequency: {wash_freq} times/week", ln=True)
        pdf.cell(200, 10, txt=f"Shampoo Type: {shampoo_type}", ln=True)
        pdf.cell(200, 10, txt=f"Diet Quality: {diet_quality}", ln=True)
        pdf.cell(200, 10, txt=f"Water Type: {water_type}", ln=True)
        pdf.ln(5)

        pdf.cell(200, 10, txt="Recommended Tips:", ln=True)
        tips = recommendations[risk]
        pdf.multi_cell(0, 10, txt=f"Shampoo: {tips['shampoo']}\nOil: {tips['oil']}\nCare Tips: {tips['tips']}")

        pdf.output("hair_risk_report.pdf")

        st.success("✅ Report saved as 'hair_risk_report.pdf' in your project folder.")


# --- Module 2: Hair Density Detector ---
st.markdown("---")
st.header("\U0001f4f7 Hair Density Detector")
st.write("Upload a scalp image to analyze your hair density.")

uploaded_image = st.file_uploader("Upload Scalp Image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    image = image.resize((300, 300))
    image_np = np.array(image)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)

    hair_pixels = cv2.countNonZero(thresh)
    total_pixels = thresh.size
    density_ratio = hair_pixels / total_pixels

    if density_ratio > 0.7:
        density = "High"
    elif density_ratio > 0.4:
        density = "Medium"
    else:
        density = "Low"

    st.subheader(f"\U0001f9e0 Detected Hair Density: {density} ({density_ratio:.2f})")
    st.image(image, caption="Uploaded Scalp Image", use_column_width=True)
    st.image(thresh, caption="Hair Detection Mask", channels="GRAY", use_column_width=True)
# --- MODULE 9: Image Upload & Tracking Over Time ---
st.markdown("---")
st.header("🗂️ Hair Image Tracker Over Time")
st.write("Upload your scalp image each time to track hair changes over time.")

if uploaded_image is not None:
    # Save image with username + timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{user_name}_{timestamp}.jpg"
    path = os.path.join("hair_images", filename)
    image.save(path)

    st.success(f"✅ Image saved as {filename}")
    st.info("You can view this in your 'hair_images' folder to compare later.")

    st.subheader("📂 Your Uploaded Images")

    import glob
    from PIL import Image as PILImage

    user_files = glob.glob(f"hair_images/{user_name}_*.jpg")

    if not user_files:
        st.info("You haven’t uploaded any images yet.")
    else:
        for file in sorted(user_files)[-5:]:  # Show latest 5
            img = PILImage.open(file)
            st.image(img, caption=os.path.basename(file), width=250)

# --- Module 3: Hair Doctor Chatbot ---
st.markdown("---")
st.header("\U0001f5e3️ Hair Doctor Chatbot")
st.write("Describe your hair/scalp problem, and I'll suggest a remedy!")

# Voice + Text Input
user_input = st.text_input("Type your hair concern or use the mic below 👇")

# 🎤 Voice Input
st.markdown("**Or Speak:** (Click below and allow mic access)")
voice_input = st.text_input("🎙️ Say a symptom (like dandruff, dry, etc.)", key="voice_input")

# Prefer voice if available
if voice_input:
    user_input = voice_input

if st.button("Get Advice"):
    response = "I'm sorry, I couldn't understand your problem. Please try using words like dandruff, dry, itchy, etc."
    for keyword in symptom_responses:
        if keyword in user_input.lower():
            response = symptom_responses[keyword]
            break
    st.success(response)

# --- Module 4: Hair Health Dashboard ---
st.markdown("---")
st.header("\U0001f4ca Hair Health Dashboard")
st.write("Here's a visual summary of your current inputs and risk prediction.")

if 'risk' in locals():
    st.subheader("\U0001f9fe Summary")
    st.markdown(f"**Hair Fall Risk:** {risk}")
    st.markdown(f"**Sleep:** {sleep} hrs/day  \n**Stress:** {stress} / 10  \n**Wash Frequency:** {wash_freq} times/week")

    input_data = {
        "Sleep": sleep,
        "Stress": stress,
        "Wash Freq": wash_freq
    }
    st.subheader("\U0001f4c8 Lifestyle Factors")
    st.bar_chart(pd.DataFrame.from_dict(input_data, orient='index', columns=["Value"]))

    fig, ax = plt.subplots()
    ax.pie(probs, labels=class_names, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.subheader("\U0001f9e0 Prediction Confidence")
    st.pyplot(fig)
else:
    st.info("\u26a0\ufe0f Make a prediction first to see your dashboard.")

# --- Module 5: Smart Recommender System ---
st.markdown("---")
st.header("\U0001f9f4 Smart Hair Care Recommender")
st.write("Based on your hair fall risk, here are personalized suggestions:")

if 'risk' in locals():
    rec = recommendations[risk]
    st.subheader("✅ Suggested Shampoo")
    st.info(rec["shampoo"])
    st.subheader("✅ Recommended Hair Oil")
    st.info(rec["oil"])
    st.subheader("✅ Hair Care Tips")
    st.success(rec["tips"])
else:
    st.info("⚠️ Make a prediction first to get recommendations.")

# --- Module 6: Hair History Tracker ---
st.markdown("---")
st.header("\U0001f4cb Hair History Tracker")
st.write("See all your past predictions and hair care inputs.")

if os.path.exists("hair_history.csv"):
    data = pd.read_csv("hair_history.csv")
    user_data = data[data["Name"] == user_name]

    if user_data.empty:
        st.info("No history found for your name.")
    else:
        st.dataframe(user_data.tail(10))

        avg_stress = round(user_data["Stress"].mean(), 1)
        avg_sleep = round(user_data["Sleep"].mean(), 1)

        st.subheader("\U0001f4ca Averages (Your Data)")
        st.write(f"🛌 Average Sleep: **{avg_sleep} hrs**")
        st.write(f"😰 Average Stress: **{avg_stress} / 10**")
else:
    st.info("No history yet. Make a prediction first.")



