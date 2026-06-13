import os
import gzip
import shutil
import time
import numpy as np
import streamlit as st
import nibabel as nib
import boto3
from botocore import UNSIGNED
from botocore.config import Config

st.set_page_config(page_title="DHARINI 3D Navigator", layout="centered", page_icon="🧠")
st.title("🧠 DHARINI Full Lifespan 3D Navigator")
st.caption("An interactive multi-specimen fly-through engine powered by IIT Madras and AWS Open Data.")
st.write("💡 **What to watch for in the animation:** Press *Play Loop* and compare Specimen 1 against Specimen 5. Notice how the internal dark chambers (ventricles) compress as the surrounding brain matter grows thicker, and look for the early physical folds starting to appear on the outer edge of the 24-week brain!")


# Configuration Profiles
BUCKET_NAME = "dharani-fetal-brain-atlas"
TMP_DIR = "/tmp/dharini_runtime"
os.makedirs(TMP_DIR, exist_ok=True)

SPECIMEN_MODELS = {
    "Specimen 1 (14 Weeks - FB34)": {
        "key": "data3d/FB34_nisl_128mpp_rgb_masked.nii.gz",
        "raw_nii": os.path.join(TMP_DIR, "FB34_volume.nii"),
        "gz": os.path.join(TMP_DIR, "FB34_volume.nii.gz"),
        "milestones": "Early second trimester baseline. Features completely smooth cortical surfaces and primitive ventricular chambers."
    },
    "Specimen 2 (17 Weeks - FB40)": {
        "key": "data3d/FB40_nisl_128mpp_rgb_masked.nii.gz",
        "raw_nii": os.path.join(TMP_DIR, "FB40_volume.nii"),
        "gz": os.path.join(TMP_DIR, "FB40_volume.nii.gz"),
        "milestones": "Initiation of rapid deep cellular migration. Axon pathways begin to bundle tightly around core subcortical regions."
    },
    "Specimen 3 (21 Weeks - FB62)": {
        "key": "data3d/FB62_nisl_128mpp_rgb_masked.nii.gz",
        "raw_nii": os.path.join(TMP_DIR, "FB62_volume.nii"),
        "gz": os.path.join(TMP_DIR, "FB62_volume.nii.gz"),
        "milestones": "Mid-development stratification. Clear layering of the cortical plate and subplate zones becomes highly visible."
    },
    "Specimen 4 (22 Weeks - FB74)": {
        "key": "data3d/FB74_nisl_128mpp_rgb_masked.nii.gz",
        "raw_nii": os.path.join(TMP_DIR, "FB74_volume.nii"),
        "gz": os.path.join(TMP_DIR, "FB74_volume.nii.gz"),
        "milestones": "Expansion of the cerebral cortex. The overall volumetric grid mass increases to accommodate growing neural pathways."
    },
    "Specimen 5 (24 Weeks - FB85)": {
        "key": "data3d/FB85_nisl_128mpp_rgb_masked.nii.gz",
        "raw_nii": os.path.join(TMP_DIR, "FB85_volume.nii"),
        "gz": os.path.join(TMP_DIR, "FB85_volume.nii.gz"),
        "milestones": "Late second trimester complexity. Visible formation of early physical sulci (folds) and a highly defined cerebellum layout."
    }
}

@st.cache_resource
def load_and_map_specimen(model_name):
    """Downloads and uncompresses the targeted NIfTI archive safely onto local disk space."""
    config = SPECIMEN_MODELS[model_name]
    raw_path = config["raw_nii"]
    gz_path = config["gz"]
    s3_key = config["key"]
    
    if not os.path.exists(raw_path):
        s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
        with st.spinner(f"📥 Downloading {model_name} from AWS (~105MB)..."):
            s3.download_file(BUCKET_NAME, s3_key, gz_path)
        
        with st.spinner("💥 Unpacking 3D volumetric matrix layers..."):
            with gzip.open(gz_path, 'rb') as f_in:
                with open(raw_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
                    
        if os.path.exists(gz_path):
            os.remove(gz_path)
            
    nii_obj = nib.load(raw_path, mmap='r')
    return nii_obj.dataobj

# Initialize Session States for the looping engine
if "play_loop" not in st.session_state:
    st.session_state.play_loop = False

# 1. Sidebar Control Center Elements
st.sidebar.header("🕹️ Configuration Menu")
chosen_specimen = st.sidebar.selectbox("Select Specimen Maturation", list(SPECIMEN_MODELS.keys()))

try:
    mmap_data = load_and_map_specimen(chosen_specimen)
    shape = mmap_data.shape
    
    # FIX: Grab only the 3rd index (Z-axis depth layers) from the shape tuple
    max_slices = int(shape[2])

    # Video Loop playback controls
    st.sidebar.subheader("🎬 Animation Player")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("▶️ Play Loop"):
            st.session_state.play_loop = True
    with col2:
        if st.button("⏹️ Pause"):
            st.session_state.play_loop = False

    # Interval pacing selector
    loop_speed = st.sidebar.slider("Playback Speed (Seconds/Frame)", 0.01, 0.5, 0.05, step=0.01)

    # Manual Navigation Sliders
    target_z = st.sidebar.slider("Manual Z-Index Position", 0, max_slices - 1, max_slices // 2)
    downsample = st.sidebar.slider("Pixel Detail Downsampling", 1, 4, 2)

    # 2. RUNTIME ANIMATION LOOP LOGIC (THE GIF EFFECT)
    if st.session_state.play_loop:
        center_slice = max_slices // 2
        start_frame = max(0, center_slice - 25)
        end_frame = min(max_slices, center_slice + 25)
        
        frame_placeholder = st.empty()
        
        while st.session_state.play_loop:
            for current_z in range(start_frame, end_frame):
                if not st.session_state.play_loop:
                    break
                    
                # Extract and process matrix arrays on the fly
                raw_slice = mmap_data[::downsample, ::downsample, current_z, 0, :].astype(np.float32)
                gray_calc = np.mean(raw_slice, axis=2)
                brain_mask = gray_calc > 5
                
                if np.any(brain_mask):
                    t_min, t_max = np.min(raw_slice[brain_mask]), np.max(raw_slice[brain_mask])
                    clipped = np.clip(raw_slice, t_min, t_max)
                    normalized = (255.0 * (clipped - t_min) / (t_max - t_min)).astype(np.uint8)
                    final_render = np.rot90(normalized, 1)
                else:
                    final_render = np.rot90(raw_slice, 1).astype(np.uint8)
                
                frame_placeholder.image(
                    final_render, 
                    caption=f"🎥 Cinematic Fly-Through | Active Slice: {current_z} / {max_slices}", 
                    use_container_width=True
                )
                time.sleep(loop_speed)
    else:
        # 3. STANDARD MANUAL CONTROL VIEWPORT
        raw_slice = mmap_data[::downsample, ::downsample, target_z, 0, :].astype(np.float32)
        gray_calc = np.mean(raw_slice, axis=2)
        brain_mask = gray_calc > 5
        
        if np.any(brain_mask):
            t_min, t_max = np.min(raw_slice[brain_mask]), np.max(raw_slice[brain_mask])
            clipped = np.clip(raw_slice, t_min, t_max)
            normalized = (255.0 * (clipped - t_min) / (t_max - t_min)).astype(np.uint8)
            final_render = np.rot90(normalized, 1)
        else:
            final_render = np.rot90(raw_slice, 1).astype(np.uint8)
            
        st.image(
            final_render,
            caption=f"Static Viewport | Z-index position: {target_z} (Resolution: {final_render.shape}x{final_render.shape})",
            use_container_width=True
        )

    # ==============================================================================
    # 4. DYNAMIC MEDICAL MILESTONE METADATA PANEL
    # ==============================================================================
    st.write("---")
    st.subheader("📋 Clinical Anatomical Observations")
    
    # Render descriptive information blocks
    st.info(SPECIMEN_MODELS[chosen_specimen]["milestones"])
    
    # Display structural data metrics cards
    metric_col1, metric_col2 = st.columns(2)
    with metric_col1:
        st.metric(label="Total Z-Axis Available Slices", value=f"{max_slices} layers")
    with metric_col2:
        st.metric(label="Original Native Resolution", value=f"{shape}x{shape}")

except Exception as e:
    st.error(f"Application exception encountered: {e}")
