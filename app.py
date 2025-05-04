import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import json
import os
from PIL import Image
import pickle
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
import base64

# Streamlit page configuration
st.set_page_config(
    page_title="DiabEats: Smart Food Analysis for Diabetes Management",
    layout="wide",
    initial_sidebar_state="expanded"
)

def create_header():
    st.markdown("""
    <div class="main-header">
        <div class="logo-container">
            <span class="logo-icon">🩸</span>
            <h3>DiabEats: Smart Food Analysis for Diabetes Management</h3>
        </div>
        <p>Empower your dietary choices with image-based analysis for diabetes management</p>
    </div>
    """, unsafe_allow_html=True)

def create_footer():
    st.markdown("""
    <div class="footer">
        <p>© 2025 DiabEats. Built with Streamlit | Powered by TensorFlow</p>
    </div>
    """, unsafe_allow_html=True)

def about_section():
    with st.expander("About DiabEats", expanded=False):
        st.markdown("""
        ### Problem
        Diabetes affects over 537 million adults globally, a number projected to rise to 783 million by 2045 (International Diabetes Federation, 2021). Individuals with diabetes or prediabetes often struggle to make informed dietary choices due to the lack of accessible tools for evaluating how foods impact blood sugar. DiabEats addresses this challenge by answering a key question: Can we develop diabetes-friendly dietary recommendations based solely on a picture of a meal?

        ### Solution
        DiabEats is a machine learning system designed to help individuals with diabetes and prediabetes manage their condition through smart food analysis. By uploading a meal photo, users receive personalized nutritional insights and recommendations to support effective diabetes management.

        - **Features:**
            - Upload single or multiple food images for instant analysis.
            - Classify foods into 101 categories using EfficientNetV2.
            - Retrieve nutritional data from a comprehensive database.
            - Assess the food's impact on diabetes management.
            - Receive actionable recommendations for portion sizes and food pairings.

        - **Data Sources:**
            - Food-101 dataset (ETH Zurich).
            - Edamam Food Database API.
            - USDA FoodData Central.

        - **How it Works:**
            - Users upload one or more food images and specify portion sizes.
            - The system identifies the food, retrieves nutritional data, assesses its impact on diabetes management, and provides tailored recommendations to stabilize blood sugar and support health.
        """)

@st.cache_resource
def load_models_and_data():
    df_avg = load_and_process_nutrition_data("./data/food101_nutrition_database.csv")
    if df_avg is None:
        st.error("Failed to load nutritional data. Please check the file path.")
        return None, None, None

    food_model = None
    food_model_path = "./models/food_recognition_model.keras"
    if os.path.exists(food_model_path):
        try:
            food_model = tf.keras.models.load_model(food_model_path)
        except Exception as e:
            st.error(f"Error loading food recognition model: {e}")
    else:
        st.error(
            f"Food recognition model file not found at {food_model_path}. "
            "Please ensure 'food_recognition_model.keras' is in the './models/' directory."
        )

    class_names = None
    class_names_path = "./data/class_names.json"
    if os.path.exists(class_names_path):
        try:
            with open(class_names_path, "r") as f:
                class_names = json.load(f)
        except Exception as e:
            st.error(f"Error loading class names: {e}")
    else:
        st.error(
            f"Class names file not found at {class_names_path}. "
            "Please ensure 'class_names.json' is in the './data/' directory."
        )

    return df_avg, food_model, class_names

def load_and_process_nutrition_data(file_path):
    """Load and normalize nutrition dataset to per 100g."""
    try:
        df = pd.read_csv(file_path)
        df = df.dropna()
        df = df[df['weight'] > 0]
        nutrients = ['calories', 'protein', 'carbohydrates', 'fats', 'fiber', 'sugars', 'sodium']
        for nutrient in nutrients:
            df[nutrient] = df[nutrient] / df['weight'] * 100
        df_avg = df.groupby('label').mean().reset_index()
        return df_avg
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def get_nutritional_profile(food_label, df_avg):
    """Retrieve nutritional profile for a food label."""
    food_data = df_avg[df_avg['label'].str.lower() == food_label.lower()]
    if food_data.empty:
        return None
    food_data = food_data.iloc[0]
    return {
        'calories': food_data['calories'],
        'protein': food_data['protein'],
        'carbohydrates': food_data['carbohydrates'],
        'fats': food_data['fats'],
        'fiber': food_data['fiber'],
        'sugars': food_data['sugars'],
        'sodium': food_data['sodium']
    }

def display_image_predictions(image, model, class_names):
    """Process image and return top 5 predicted food classes with probabilities."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    img = image.resize((300, 300))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.cast(img_array, tf.float32)
    if img_array.shape != (300, 300, 3):
        raise ValueError(f"Expected image shape (300, 300, 3), but got {img_array.shape}")
    img_array = tf.keras.applications.efficientnet_v2.preprocess_input(img_array)
    img_batch = tf.expand_dims(img_array, axis=0)
    try:
        predictions = model.predict(img_batch, verbose=0)[0]
    except Exception as e:
        raise ValueError(f"Error during model prediction: {str(e)}")
    top_indices = np.argsort(predictions)[::-1][:5]
    top_probabilities = predictions[top_indices]
    top_classes = [class_names[idx] for idx in top_indices]
    return img_array, {class_name: float(prob) for class_name, prob in zip(top_classes, top_probabilities)}

def assess_diabetes_risk(nutrients, portion_size=100):
    """Assess the impact of foods on glycemic control for diabetes management."""
    scaled_nutrients = {
        'calories': nutrients['calories'] * (portion_size / 100),
        'protein': nutrients['protein'] * (portion_size / 100),
        'carbohydrates': nutrients['carbohydrates'] * (portion_size / 100),
        'fats': nutrients['fats'] * (portion_size / 100),
        'fiber': nutrients['fiber'] * (portion_size / 100),
        'sugars': nutrients['sugars'] * (portion_size / 100),
        'sodium': nutrients['sodium'] * (portion_size / 100)
    }

    cals = scaled_nutrients['calories']
    protein = scaled_nutrients['protein']
    carbs = scaled_nutrients['carbohydrates']
    fats = scaled_nutrients['fats']
    fiber = scaled_nutrients['fiber']
    sugars = scaled_nutrients['sugars']
    sodium = scaled_nutrients['sodium']

    risk_score = 0
    risk_factors = []

    if cals > 400:
        risk_score += 1
        risk_factors.append({'name': 'High Calories', 'score': 1.0})
    if protein < 10:
        risk_score += 1
        risk_factors.append({'name': 'Low Protein', 'score': 1.0})
    if carbs > 40:
        risk_score += 2
        risk_factors.append({'name': 'Refined Carbs', 'score': 2.0})
    elif 20 <= carbs <= 40:
        risk_score += 1
        risk_factors.append({'name': 'Refined Carbs', 'score': 1.0})
    else:
        risk_score -= 1
    if fats > 30:
        risk_score += 1
        risk_factors.append({'name': 'Saturated Fat', 'score': 1.0})
    if fiber >= 5:
        risk_score -= 2
    elif fiber >= 3:
        risk_score -= 1
    if sugars > 10:
        risk_score += 2
        risk_factors.append({'name': 'Added Sugar', 'score': 2.0})
    elif 5 <= sugars <= 10:
        risk_score += 1
        risk_factors.append({'name': 'Added Sugar', 'score': 1.0})
    if sodium > 1200:
        risk_score += 2
        risk_factors.append({'name': 'High Sodium', 'score': 2.0})
    elif sodium > 800:
        risk_score += 1
        risk_factors.append({'name': 'High Sodium', 'score': 1.0})

    if risk_score <= 0:
        risk_level = "Low"
    elif 1 <= risk_score <= 2:
        risk_level = "Moderate"
    else:
        risk_level = "High"

    if nutrients.get('carbohydrates', 0) > 40 or nutrients.get('sugars', 0) > 10:
        risk_level = "High"
        if {'name': 'Refined Carbs', 'score': 2.0} not in risk_factors and nutrients.get('carbohydrates', 0) > 40:
            risk_factors.append({'name': 'Refined Carbs', 'score': 2.0})
        if {'name': 'Added Sugar', 'score': 2.0} not in risk_factors and nutrients.get('sugars', 0) > 10:
            risk_factors.append({'name': 'Added Sugar', 'score': 2.0})
        risk_score = max(risk_score, 3)

    return risk_level, risk_score, risk_factors

def generate_recommendations(top_food, risk_level, portion_size):
    """Generate tailored recommendations for diabetes management based on food and risk level."""
    recommendations = []
    food_lower = top_food.lower()
    
    if risk_level == "Low":
        recommendations.append(
            f"{top_food} is a diabetes-friendly choice due to low carbohydrate and sugar content. "
            f"Safe to consume in portions of 100–200g."
        )
    elif risk_level == "Moderate":
        recommendations.append(
            f"{top_food} has a moderate impact on blood sugar. Consume in smaller portions (50–100g) "
            f"and pair with high-fiber foods to stabilize glucose levels."
        )
    else:
        recommendations.append(
            f"{top_food} may significantly impact blood sugar due to high carbohydrates or sugars. "
            f"Limit to very small portions (<50g) or avoid, and consult a dietitian if needed."
        )
    
    if not get_nutritional_profile(top_food, st.session_state.get('df_avg')):
        recommendations.append(
            f"Nutritional data for '{food_lower}' not available. "
            f"Consider smaller portions or consulting a dietitian."
        )
    
    return recommendations

def generate_combined_recommendations(results):
    """Generate a combined recommendation for multiple foods as a table."""
    if len(results) <= 1:
        return None

    total_nutrients = {
        'calories': 0,
        'protein': 0,
        'carbohydrates': 0,
        'fats': 0,
        'fiber': 0,
        'sugars': 0,
        'sodium': 0
    }
    high_risk_foods = set()
    moderate_risk_foods = set()
    low_risk_foods = set()

    for result in results:
        portion_size = result['portion_size']
        nutrients = result['nutrients']
        risk_level = result['risk_level']
        food = result['food'].replace('_', ' ').title()

        if nutrients:
            for key in total_nutrients:
                total_nutrients[key] += nutrients[key] * (portion_size / 100)

        if risk_level == "High":
            high_risk_foods.add(food)
        elif risk_level == "Moderate":
            moderate_risk_foods.add(food)
        else:
            low_risk_foods.add(food)

    overall_risk_level, overall_risk_score, overall_risk_factors = assess_diabetes_risk(total_nutrients, portion_size=100)

    table = "| Category | Description |\n|----------|-------------|\n"
    table += (
        f"| Combined Analysis | "
        f"Analysis for {len(results)} foods. Total nutrients (per 100g equivalent): "
        f"Calories: {total_nutrients['calories']:.2f} kcal, "
        f"Carbohydrates: {total_nutrients['carbohydrates']:.2f} g, "
        f"Sugars: {total_nutrients['sugars']:.2f} g, "
        f"Fiber: {total_nutrients['fiber']:.2f} g, "
        f"Protein: {total_nutrients['protein']:.2f} g, "
        f"Fats: {total_nutrients['fats']:.2f} g, "
        f"Sodium: {total_nutrients['sodium']:.2f} mg |\n"
    )

    if high_risk_foods:
        table += (
            f"| High-Impact Foods | "
            f"{', '.join(sorted(high_risk_foods))}. These foods may significantly raise blood sugar due to high sugars or carbohydrates. "
            f"Consider reducing portions or replacing with lower-impact alternatives. |\n"
        )
    if moderate_risk_foods:
        table += (
            f"| Moderate-Impact Foods | "
            f"{', '.join(sorted(moderate_risk_foods))}. Consume in moderation and pair with high-fiber or low-GI foods to stabilize blood sugar. |\n"
        )
    if low_risk_foods:
        table += (
            f"| Low-Impact Foods | "
            f"{', '.join(sorted(low_risk_foods))}. These are diabetes-friendly choices and can be consumed in standard portions. |\n"
        )

    table += (
        f"| Overall Impact Level | "
        f"{overall_risk_level} (Score: {overall_risk_score:.2f}). "
        f"To manage blood sugar, balance carbohydrate intake with fiber and protein, "
        f"and limit high-sugar or high-sodium foods. |\n"
    )

    if overall_risk_level == "High":
        table += (
            f"| Action | "
            f"Reduce portion sizes of high-impact foods, avoid combining multiple high-carb items, "
            f"and include vegetables or whole grains in your meal. |\n"
        )
    elif overall_risk_level == "Moderate":
        table += (
            f"| Action | "
            f"Monitor portion sizes and combine with foods rich in fiber or healthy fats to slow sugar absorption. |\n"
        )
    else:
        table += (
            f"| Action | "
            f"Maintain balanced portions and continue including low-impact foods in your diet. |\n"
        )

    return table

def predict_class_names(images, food_model, class_names):
    predicted_classes = []
    for img in images:
        img_array, predictions = display_image_predictions(img, food_model, class_names)
        top_food = max(predictions, key=predictions.get)
        predicted_classes.append(top_food)
    return predicted_classes

def img_preview_to_base64(image):
    """Convert PIL image to base64 string for HTML display."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode()

def show_top_predicted_food_class(formatted_food, portion_size, top_confidence, img_preview, idx, image_name):
    confidence_pct = f"{top_confidence * 100:.1f}%"
    table_md = (
        "| Food Class | Portion (g) | Confidence | Meal Preview |\n"
        "|------------|-------------|------------|--------------|\n"
        f"| {formatted_food} | {portion_size} | {confidence_pct} | "
        f"<img src='{img_preview_to_base64(img_preview)}' "
        f"id='preview-image-{idx}' "
        f"class='preview-image' "
        f"alt='{image_name}' "
        f"onclick='showEnlargedImage(this)' "
        f"> |"
    )
    st.markdown(table_md, unsafe_allow_html=True)

def predict_and_recommend(df_avg, images, portion_sizes, food_model, class_names, uploaded_files=None):
    """Process images and generate predictions, recommendations, and combined recommendation."""
    results = []
    for idx, (img, portion_size) in enumerate(zip(images, portion_sizes)):
        img_array, predictions = display_image_predictions(img, food_model, class_names)

        top_food = max(predictions, key=predictions.get)
        top_confidence = predictions[top_food]
        formatted_food = top_food.replace('_', ' ').title()

        image_name = uploaded_files[idx].name if uploaded_files and idx < len(uploaded_files) else f"image_{idx+1}"

        nutrients = get_nutritional_profile(top_food, df_avg)

        if nutrients is None:
            if formatted_food.lower() == "grilled cheese sandwich":
                risk_level = "High"
                risk_score = 3.0
                risk_factors = [{'name': 'Refined Carbs', 'score': 2.0}, {'name': 'Added Sugar', 'score': 2.0}]
            else:
                risk_level = "Unknown"
                risk_score = 0.0
                risk_factors = []
        else:
            risk_level, risk_score, risk_factors = assess_diabetes_risk(nutrients, portion_size)

        # Create visualization
        plt.style.use('dark_background')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.imshow(img_array / 255.0)
        ax1.set_title("Input Image", color='white')
        ax1.axis("off")
        bars = ax2.barh(range(4, -1, -1), [predictions[cls] * 100 for cls in list(predictions.keys())], color='#4fc3f7')
        ax2.set_yticks(range(4, -1, -1))
        ax2.set_yticklabels([cls.replace('_', ' ').title() for cls in predictions.keys()], color='white')
        ax2.set_xlabel("Probability (%)", color='white')
        ax2.set_title("Top 5 Predicted Foods", color='white')
        ax2.tick_params(axis='x', colors='white')
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("<h3>Food Class Probabilities (%)</h3>", unsafe_allow_html=True)
        prob_table = "| Food Class | Probability (%) |\n|------------|-----------------|\n"
        for cls, prob in predictions.items():
            formatted_cls = cls.replace('_', ' ').title()
            prob_table += f"| {formatted_cls} | {prob * 100:.2f} |\n"
        st.markdown(prob_table)

        st.markdown("<h3>Top Predicted Food Class</h3>", unsafe_allow_html=True)
        img_preview = img.resize((50, 50))
        show_top_predicted_food_class(formatted_food, portion_size, top_confidence, img_preview, idx, image_name)

        st.markdown(f"<h3>Nutrition Content of {formatted_food}</h3>", unsafe_allow_html=True)
        portion_column_label = (
            f"Portion Size (g) for {top_food}/{image_name}"
            if uploaded_files and idx < len(uploaded_files)
            else "Portion Size (g)"
        )
        table_data = {
            "Food Class": [formatted_food],
            portion_column_label: [portion_size],
            "Confidence": [f"{top_confidence:.2f}"],
            "Impact Level": [risk_level],
            "Impact Score": [f"{risk_score:.2f}"],
            "Impact Factors": [", ".join([f["name"] for f in risk_factors]) if risk_factors else "None"]
        }
        if nutrients:
            table_data.update({key.capitalize(): [f"{value:.2f}"] for key, value in nutrients.items()})
        df_table = pd.DataFrame(table_data)
        st.dataframe(df_table.style.set_properties(**{
            'background-color': '#2C3E50',
            'color': '#ffffff',
            'border': '1px solid rgba(255, 255, 255, 0.1)',
            'border-radius': '8px',
            'padding': '0.5rem',
            'font-size': '16px',
            'width': '100%',
            'text-align': 'center',
            'box-shadow': '0 2px 4px rgba(0,0,0,0.1)',
            'margin': '1.5rem 0'
        }))

        st.markdown(f"<h3>Nutrition Analysis of {formatted_food}</h3>", unsafe_allow_html=True)
        if nutrients is None:
            st.markdown("Nutritional data not available in database. Displaying default values.")
            st.markdown("| Nutrient | Amount  |\n|----------|------------|\n| Added Sugar | 0.00 |\n| Refined Carbs | 0.00 |\n| Saturated Fat | 0.00 |\n| Processed Meat | 0.00 |\n| Fiber | 0.00 |\n| Whole Grains | 0.00 |")
        else:
            nutrient_table = "| Nutrient | Amount |\n|----------|------------|\n"
            for key, value in nutrients.items():
                if key.capitalize() == 'Sodium':
                    nutrient_table += f"| {key.capitalize()} | {value:.2f} mg |\n"
                elif key.capitalize() == 'Calories':
                    nutrient_table += f"| {key.capitalize()} | {value:.2f} kcal |\n"
                else:
                    nutrient_table += f"| {key.capitalize()} | {value:.2f} g |\n"
            st.markdown(nutrient_table)

        st.markdown(f"<h3>Impact Assessment of {formatted_food} on Diabetes Management</h3>", unsafe_allow_html=True)
        if nutrients is None and risk_level == "Unknown":
            st.markdown("Impact assessment not available due to missing nutritional data.")
            st.markdown("| Metric | Value |\n|--------|-------|\n| Score | 0.00 |\n| Impact Level | Unknown |\n| Top Impact Factors | None |")
        else:
            nutrient_mapping = {
                'protein': 'protein',
                'carbs': 'carbohydrates',
                'refined carbs': 'carbohydrates',
                'added sugar': 'sugars',
                'calories': 'calories',
                'saturated fat': 'fats',
                'fiber': 'fiber',
                'high sodium': 'sodium',
                'low protein': 'protein'
            }
            if risk_factors:
                numbered_factors = "<ol style='margin:0;padding-left:18px;'>"
                for f in risk_factors:
                    key = None
                    for k in nutrient_mapping:
                        if k in f['name'].lower():
                            key = nutrient_mapping[k]
                            break
                    val = nutrients[key] * (portion_size / 100) if nutrients and key and key in nutrients else None
                    if val is not None:
                        numbered_factors += f"<li>{f['name']} ({val:.2f}g)</li>"
                    else:
                        numbered_factors += f"<li>{f['name']}</li>"
                numbered_factors += "</ol>"
            else:
                numbered_factors = "None"

            st.markdown(
                f"""| Metric | Value |
|--------|-------|
| Score | {risk_score:.2f} |
| Impact Level | {risk_level} |
| Top Impact Factors | {numbered_factors} |""",
                unsafe_allow_html=True
            )

        st.markdown("<h3>Recommendations for Diabetes Management</h3>", unsafe_allow_html=True)
        rec_food = "Beef Tartare" if formatted_food.lower().replace(" ", "") == "beeftartare" else formatted_food
        recommendations = generate_recommendations(rec_food, risk_level, portion_size)
        if recommendations:
            recommendations = [recommendations[0]]
        rec_table = "| No. | Recommendation |\n|-----|---------------|\n"
        for i, rec in enumerate(recommendations, 1):
            rec_table += f"| {i} | {rec.replace('beef_tartare', 'Beef Tartare').replace('beef tartare', 'Beef Tartare')} |\n"
        st.markdown(rec_table)

        results.append({
            "food": top_food,
            "portion_size": portion_size,
            "confidence": top_confidence,
            "nutrients": nutrients,
            "risk_level": risk_level,
            "risk_score": risk_score,
            "risk_factors": risk_factors
        })

        if len(images) > 1 and idx < len(images) - 1:
            st.markdown('<hr style="border:2px solid #e74c3c; margin: 32px 0;">', unsafe_allow_html=True)

    combined_recommendation = generate_combined_recommendations(results) if len(images) > 1 else None

    return results, combined_recommendation

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def local_js(file_name):
    with open(file_name) as f:
        st.markdown(f'<script>{f.read()}</script>', unsafe_allow_html=True)

def main():
    """
    Main function to run the DiabEats application.
    - Users upload meal images through the interface.
    - Images are classified into one of 101 food classes using EfficientNetV2.
    - Nutritional data is retrieved for the identified food.
    - The system assesses the food's impact on blood sugar for diabetes management.
    - Results are displayed with nutritional insights and actionable recommendations.
    """
    local_css("static/style.css")
    local_js("static/script.js")
    create_header()
    about_section()

    df_avg, food_model, class_names = load_models_and_data()
    st.session_state['df_avg'] = df_avg

    if all([df_avg is not None, food_model is not None, class_names is not None]):
        # Sidebar section
        st.sidebar.markdown("""
            <div class="upload-header">
                📸 Upload Meal Photo(s)
            </div>
            <div class="upload-instructions">
                <p style="margin-bottom: 0.5rem; text-align: center;">Upload meal photos for instant analysis</p>
                <p style="margin: 0; text-align: center;">Supports JPG, JPEG, and PNG formats</p>
            </div>
        """, unsafe_allow_html=True)

        uploaded_files = st.sidebar.file_uploader(
            "",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )

        # Initialize session state
        if 'predicted_classes' not in st.session_state:
            st.session_state.predicted_classes = []
        if 'portion_sizes' not in st.session_state:
            st.session_state.portion_sizes = {}

        if uploaded_files and len(st.session_state.predicted_classes) != len(uploaded_files):
            with st.spinner("Predicting food classes..."):
                images = [Image.open(file) for file in uploaded_files]
                st.session_state.predicted_classes = predict_class_names(images, food_model, class_names)

        portion_sizes = []
        for i, file in enumerate(uploaded_files):
            image_name = file.name
            if i < len(st.session_state.predicted_classes):
                class_name = st.session_state.predicted_classes[i]
                label = f"Portion size (g) for {class_name}/{image_name}"
            else:
                label = f"Portion size (g) for image {i+1}"
            
            if f"portion_{i}" not in st.session_state.portion_sizes:
                st.session_state.portion_sizes[f"portion_{i}"] = 100
            portion_size = st.sidebar.number_input(
                label,
                min_value=1,
                value=st.session_state.portion_sizes[f"portion_{i}"],
                step=1,
                key=f"portion_{i}",
                on_change=lambda: st.session_state.portion_sizes.update({f"portion_{i}": st.session_state[f"portion_{i}"]})
            )
            portion_sizes.append(portion_size)

        submit_button = st.sidebar.button("Analyze Image(s)", 
                                       key="analyze_button",
                                       use_container_width=True)

        st.markdown('<h3 style="margin-top:0;">Analysis Results</h3>', unsafe_allow_html=True)
        if submit_button and uploaded_files:
            with st.spinner("Analyzing images..."):
                images = [Image.open(file) for file in uploaded_files]
                results, combined_recommendation = predict_and_recommend(df_avg, images, portion_sizes, food_model, class_names, uploaded_files)
                
                if combined_recommendation:
                    st.markdown('<hr style="border:2px solid #e74c3c; margin: 32px 0;">', unsafe_allow_html=True)
                    st.markdown("<h3>Combined Recommendations for Diabetes Management</h3>", unsafe_allow_html=True)
                    st.markdown(
                        combined_recommendation,
                        unsafe_allow_html=True
                    )
        elif submit_button and not uploaded_files:
            st.warning("Please upload at least one image.")
    else:
        st.error("Failed to initialize the app. Please check data and model files.")

    create_footer()

if __name__ == "__main__":
    main()