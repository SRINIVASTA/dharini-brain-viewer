import streamlit as st
import numpy as np
import cv2
import boto3
from botocore import UNSIGNED
from botocore.config import Config

st.set_page_config(page_title="DHARINI Live Streamer", layout="centered", page_icon="🧠")
st.title("🧠 DHARINI Real-Time 3D Volume Slicer")
st.caption("Streaming exact voxel structures straight from the AWS Open Data Registry.")

BUCKET_NAME = "dharani-fetal-brain-atlas"
S3_KEY = "data3d/FB34_nisl_128mpp_rgb_masked.nii.gz"

# The real dimensions of the NIfTI array dataset: (Width, Height, Depth, 1, Channels)
X_DIM, Y_DIM, Z_DIM = 625, 625, 707
BYTES_PER_PIXEL = 3  # RGB data channels

@st.cache_resource
def get_s3():
    return boto3.client('s3', config=Config(signature_version=UNSIGNED))

# 1. UI Navigation Panel Elements
st.sidebar.header("🕹️ Slice Explorer")
target_z = st.sidebar.slider("Select Z-Index Position", 0, Z_DIM - 1, 279)
downsample = st.sidebar.slider("Pixel Detail Striding", 1, 4, 2)

@st.cache_data(show_spinner="Streaming true anatomical slice array from AWS...")
def stream_brain_slice(z_index):
    s3 = get_s3()
    
    # NIfTI-1 file header offset is exactly 348 bytes.
    nifti_header_offset = 348
    
    # Calculate the exact byte location of the selected 2D layer inside the 3D block
    slice_size_bytes = X_DIM * Y_DIM * BYTES_PER_PIXEL
    start_byte = nifti_header_offset + (z_index * slice_size_bytes)
    end_byte = start_byte + slice_size_bytes - 1
    
    # Execute HTTP Range Request to fetch only this specific slice's bytes (approx. 1.1MB)
    response = s3.get_object(
        Bucket=BUCKET_NAME,
        Key=S3_KEY,
        Range=f"bytes={start_byte}-{end_byte}"
    )
    
    # Read raw string bytes straight into an uncompressed NumPy array matrix
    raw_bytes = response['Body'].read()
    flat_array = np.frombuffer(raw_bytes, dtype=np.uint8)
    
    # Reshape vector straight into the standard tissue matrix grid
    slice_2d = flat_array.reshape((Y_DIM, X_DIM, BYTES_PER_PIXEL))
    return slice_2d

# 2. Extract and Process the Chosen Array Layer
try:
    raw_anatomy = stream_brain_slice(target_z)
    
    # Downsample matrix array to protect browser presentation memory
    processed_frame = raw_anatomy[::downsample, ::downsample, :]
    
    # --- COLAB MEDICAL CONTRAST-BOOST PIPELINE ---
    gray_calc = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2GRAY)
    brain_mask = gray_calc > 5  # Filter out dark, unmapped background voxels
    
    if np.any(brain_mask):
        # Localize exact maximum and minimum threshold vectors
        t_min = np.min(processed_frame[brain_mask])
        t_max = np.max(processed_frame[brain_mask])
        
        # Stretch visible tones across full 8-bit dynamic spectrum
        clipped = np.clip(processed_frame, t_min, t_max)
        normalized = (255.0 * (clipped - t_min) / (t_max - t_min)).astype(np.uint8)
        final_render = np.rot90(normalized, 1)
    else:
        final_render = np.rot90(processed_frame, 1)

    # 3. Output Render Result to Streamlit Canvas
    st.image(
        final_render, 
        caption=f"Anatomical Matrix view at Z-depth layer: {target_z}", 
        use_container_width=True
    )

except Exception as e:
    st.error(f"Failed to fetch cloud byte arrays: {e}")
    st.info("Tip: Ensure your GitHub repository contains boto3 and dependencies in requirements.txt.")
