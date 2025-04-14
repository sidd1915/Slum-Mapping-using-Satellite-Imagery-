import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
import tifffile as tiff
import streamlit as st
import os
import rasterio
from shapely.geometry import box
import geopandas as gpd
from PIL import Image
import folium
import streamlit as st
from streamlit_folium import folium_static

# Must be the first Streamlit command
st.set_page_config(page_title="Mumbai Slum Mapping", layout="centered")


# ------------------------------ 
# Custom Loss Function 
# ------------------------------ 
def weighted_categorical_crossentropy(weights): 
    weights = tf.constant(weights) 
    def loss(y_true, y_pred): 
        y_true = tf.cast(y_true, tf.float32) 
        loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred) 
        weights_per_pixel = tf.reduce_sum(weights * y_true, axis=-1) 
        return tf.reduce_mean(loss * weights_per_pixel) 
    return loss

# ------------------------------ 
# Model Loader 
# ------------------------------ 
@st.cache_resource
def load_model_with_custom_loss():
    class_weights = [0.2, 0.2, 0.2, 0.4]  # background, vegetation, slum, water
    model = tf.keras.models.load_model(
        'model.h5',
        custom_objects={'loss': weighted_categorical_crossentropy(class_weights)}
    )
    return model

model = load_model_with_custom_loss()

# ------------------------------ 
# Class Info 
# ------------------------------ 
CLASS_NAMES = ['Background', 'Vegetation', 'Slum', 'Water']
CLASS_COLORS = [
    (50, 50, 50),       # Background: black
    (34, 139, 34),     # Vegetation: green
    (255, 0, 0),       # Slum: red
    (0, 120, 255)      # Water: blue
]

def decode_segmentation_mask(mask):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, color in enumerate(CLASS_COLORS):
        color_mask[mask == idx] = color
    return color_mask

# ------------------------------ 
# Resize Image with size check
# ------------------------------ 
def resize_image(image, target_size=(120, 120)):
    
    if image.dtype != np.uint8:
        temp_image = (255 * image).clip(0, 255).astype(np.uint8)
    else:
        temp_image = image

    # If grayscale, convert to RGB
    if len(temp_image.shape) == 2:
        temp_image = np.stack([temp_image]*3, axis=-1)

    # Drop alpha if 4 channels
    if temp_image.shape[-1] == 4:
        temp_image = temp_image[:, :, :3]

    # If already right size
    if temp_image.shape[:2] == target_size:
        return image

    # Resize with PIL
    pil_image = Image.fromarray(temp_image)
    pil_image = pil_image.resize(target_size, Image.BILINEAR)
    resized = np.array(pil_image)

    # Restore to normalized float if original was float
    if image.dtype != np.uint8:
        resized = resized / 255.0

    return resized


# ------------------------------ 
# Mapping Ground Truth Mask 
# ------------------------------ 
def map_ground_truth_mask(mask):
    mapped_mask = np.copy(mask)
    mapped_mask[mask == 0] = 0   # Background
    mapped_mask[mask == 1] = 1   # Vegetation
    mapped_mask[mask == 3] = 2   # Slum
    mapped_mask[mask == 6] = 3   # Water
    return mapped_mask


TIF_DIR = "assets/images"
MASK_DIR = "assets/masks"


features = []
crs = None  

for tif_file in os.listdir(TIF_DIR):
    if not tif_file.endswith(".tif"):
        continue

    tif_path = os.path.join(TIF_DIR, tif_file)
    base, _ = os.path.splitext(tif_file)  
    mask_file = f"{base}.png"   
    mask_file = tif_file.replace(".tif", ".png")
    mask_path = os.path.join(MASK_DIR, mask_file)

    if not os.path.exists(mask_path):
        print(f"Missing mask: {mask_file}")
        continue

    with rasterio.open(tif_path) as src:
        bounds = src.bounds
        if crs is None:
            crs = src.crs  
        tif_shape = src.read(1).shape

    mask_img = Image.open(mask_path).convert("L")
    mask = np.array(mask_img)

    if mask.shape != tif_shape:
        mask_img = mask_img.resize((tif_shape[1], tif_shape[0]))
        mask = np.array(mask_img)

    total_pixels = mask.size
    slum_density = np.sum(mask == 1) / total_pixels
    vegetation_density = np.sum(mask == 2) / total_pixels
    water_density = np.sum(mask == 3) / total_pixels

    features.append({
        "geometry": box(*bounds),
        "slum_density": slum_density,
        "vegetation_density": vegetation_density,
        "water_density": water_density,
        "filename": tif_file
    })


if features and crs:
    gdf = gpd.GeoDataFrame(features, crs=crs)
    gdf = gdf.to_crs(epsg=4326)
else:
    raise ValueError("No valid TIFF + PNG mask pairs were found.")



# ------------------------------ 
# Page: Info and Slum Mapping Demo 
# ------------------------------ 
page = st.sidebar.radio("Navigation", ["Project Info", "Slum Mapping Demo", "Mumbai Density Distribution Map"])

if page == "Project Info":
    st.title("Slum Mapping using Satellite Imagery")
    st.image("assets/example_overlay.png", caption="Drone captured photos in Mumbai", use_container_width=True)

    st.markdown("""  
    This app demonstrates slum area segmentation from high-resolution satellite imagery using Deep Learning.
    
    ### Key Highlights:
    - **City**: Mumbai
    - **Dataset Source**: Taken from Mendeley Data
    - **Model**: Custom U-Net trained on RGB images (Pleiades data)
    - **Classes**: Water, Vegetation, Slum, and Background
    - **Loss Function**: Weighted Categorical Crossentropy to tackle class imbalance
    
    """)

elif page == "Slum Mapping Demo":

    # --------------------- Utilities ----------------------

    def load_tif_image(uploaded_file):
        image = tiff.imread(uploaded_file)
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[0] == 3 and image.shape[-1] != 3:
            image = np.transpose(image, (1, 2, 0))
        image = image / 255.0  # Normalize
        return image

    def load_mask(uploaded_file):
        mask = Image.open(uploaded_file)
        return np.array(mask)

    def predict_mask(model, image):
        resized_image = resize_image(image)
        input_tensor = tf.convert_to_tensor(np.expand_dims(resized_image, axis=0), dtype=tf.float32)
        pred = model.predict(input_tensor)
        return np.argmax(pred[0], axis=-1)


    def display_custom_legend(class_colors, class_labels):
        st.markdown("### Legend")
        
        legend_html = ""
        
        for color, label in zip(class_colors, class_labels):
            color_hex = f"rgb({color[0]}, {color[1]}, {color[2]})"
            legend_html += f"""
            <div style='display:flex; align-items:center;'>
                <div style='width:20px;height:20px;background-color:{color_hex};margin-right:10px;'></div>
                <div>{label}</div>
            </div>
            """
        
        st.markdown(legend_html, unsafe_allow_html=True)

    class_colors = [(169, 169, 169),(34, 139, 34), (255, 69, 0), (0, 128, 255)]  # Improved colors
    class_labels = [ "Background","Vegetation","Slum","Water"]

    #display_custom_legend(class_colors, class_labels)

    def plot_all_three(input_image, pred_mask, true_mask=None, class_colors=None, label_values=None):
        """
        Displays input image, prediction mask (0–3 indices), and ground truth mask (0,1,3,6 values).

        Args:
            input_image: RGB image.
            pred_mask: Prediction mask with values 0–3 (indices).
            true_mask: Ground truth mask with values 0,1,3,6.
            class_colors: List of RGB tuples for each class.
            label_values: List of class label values [0,1,3,6] matching class_colors.
        """
        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(input_image, caption="Input Image", use_container_width=True)

        with col2:
            colored_pred = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
            for i, color in enumerate(class_colors):
                colored_pred[pred_mask == i] = color
            st.image(colored_pred, caption="Model Prediction", use_container_width=True)

        if true_mask is not None:
            with col3:
                colored_true = np.zeros((*true_mask.shape, 3), dtype=np.uint8)

                # Use a mapping from true class values to colors
                label_to_color = {label: color for label, color in zip(label_values, class_colors)}
                for label_value, color in label_to_color.items():
                    colored_true[true_mask == label_value] = color

                st.image(colored_true, caption="Ground Truth", use_container_width=True)


    label_values = [0, 1, 2, 3]  # Actual class values in ground truth

    # ------------------ Slum Area Stats ------------------
    def show_slum_area_stats(pred_mask, slum_class_index=2, pixel_resolution=0.5):
        slum_pixel_count = np.sum(pred_mask == slum_class_index)
        total_pixel_count = pred_mask.size
        slum_percentage = (slum_pixel_count / total_pixel_count) * 100
        slum_area_sqkm = slum_pixel_count * (pixel_resolution**2) / 1e6

        st.markdown("### 🧮 Slum Area Statistics")
        st.markdown(f"- **Slum Pixels**: `{slum_pixel_count}`")
        st.markdown(f"- **Total Pixels**: `{total_pixel_count}`")
        st.markdown(f"- **Slum Area Percentage**: `{slum_percentage:.2f}%`")
        st.markdown(f"- **Approx. Area (sq km)**: `{slum_area_sqkm:.6f}`")


    # ----------------- Streamlit UI -----------------------

    st.title("Slum Mapping Demo")

    # Initialize session state if not already done
    if 'uploaded_image' not in st.session_state:
        st.session_state['uploaded_image'] = None
    if 'uploaded_mask' not in st.session_state:
        st.session_state['uploaded_mask'] = None

    # File Uploads
    uploaded_image = st.file_uploader("Upload a .tif satellite image", type=["tif", "tiff"])
    uploaded_mask = st.file_uploader("Upload optional ground truth mask (PNG or TIFF)", type=["png", "tif", "tiff"])

    # Store the uploads in session state
    if uploaded_image:
        st.session_state['uploaded_image'] = uploaded_image
    if uploaded_mask:
        st.session_state['uploaded_mask'] = uploaded_mask

    # Load image and mask from session state
    if st.session_state['uploaded_image']:
        image = load_tif_image(st.session_state['uploaded_image'])
        pred_mask = predict_mask(model, image)

        true_mask = load_mask(st.session_state['uploaded_mask']) if st.session_state['uploaded_mask'] else None
        if true_mask is not None:
            true_mask = map_ground_truth_mask(true_mask)

        display_custom_legend(class_colors, class_labels)

        st.markdown("### Visualization")
        plot_all_three(image, pred_mask, true_mask, class_colors, label_values)

        st.markdown("### Heatmap Overlay for Slum Area")
        show_slum_area_stats(pred_mask, slum_class_index=2, pixel_resolution=0.5)


elif page == "Mumbai Density Distribution Map":
 
    st.title("Mumbai Land Use Distribution")

    m = folium.Map(location=[19.07, 72.87], zoom_start=11)

    def add_layer(name, column, color):
        layer = folium.FeatureGroup(name=name)
        for _, row in gdf.iterrows():
            value = row[column]
            if value > 0.01:
                opacity = min(0.9, value * 2)  # scale density to opacity
                folium.GeoJson(
                    row["geometry"].__geo_interface__,
                    style_function=lambda x, c=color, o=opacity: {
                        "fillColor": c,
                        "color": c,
                        "weight": 0.5,
                        "fillOpacity": o,
                    },
                    tooltip=f"{row['filename']}<br>{name} Density: {value:.2f}"
                ).add_to(layer)
        layer.add_to(m)

    # Add separate layers
    add_layer("Slum", "slum_density", "#d73027")
    add_layer("Vegetation", "vegetation_density", "#1a9850")
    add_layer("Water", "water_density", "#4575b4")

    folium.LayerControl().add_to(m)
    folium_static(m)
