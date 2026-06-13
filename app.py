import os
import gzip
import shutil
import numpy as np
import streamlit as st
import nibabel as nib
import boto3
from botocore import UNSIGNED
from botocore.config import Config

st.set_page_config(page_title="DHARINI Live Viewer", layout="centered", page_icon="🧠")
st.title("🧠 DHARINI Real-Time 3D Volume Slicer")
st.caption("Streaming 3D anatomical structures directly from the AWS Open Data Registry.")

# Production file configurations
BUCKET_NAME = "dharani-fetal-brain-atlas"
S3_KEY = "data3d/FB34_nisl_128mpp_rgb_masked.nii.gz"

TMP_DIR = "/tmp/dharini_runtime"
COMPRESSED_PATH = os.path.join(TMP_DIR, "FB34_volume.nii.gz")
UNCOMPRESSED_PATH = os.path.join(TMP_DIR, "FB34_volume.nii")

@st.cache_resource
def initialize_and_map_volume():
    """Downloads, decompresses, and memory-maps the 3D NIfTI file on the server disk."""
    if not os.path.exists(UNCOMPRESSED_PATH):
        os.makedirs(TMP_DIR, exist_ok=True)
        
        # 1. Download archive safely using an anonymous S3 hook
        s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
        with st.spinner("📥 Initializing 3D Dataset from AWS Open Data (approx. 107MB)..."):
            s3.download_file(BUCKET_NAME, S3_KEY, COMPRESSED_PATH)
        
        # 2. Decompress archive onto the local disk block space to save RAM
        with st.spinner("💥 Unpacking 3D volumetric matrix layers..."):
            with gzip.open(compressed_path if 'compressed_path' in locals() else COMPRESSED_PATH, 'rb') as f_in:
                with open(UNCOMPRESSED_PATH, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
                    
        # Remove compressed file to optimize local storage footprint
        if os.path.exists(COMPRESSED_PATH):
            os.remove(COMPRESSED_PATH)
            
    # 3. Memory-map the uncompressed NIfTI file (Uses 0MB RAM)
    nii_obj = nib.load(uncompressed_path if 'uncompressed_path' in locals() else UNCOMPRESSED_PATH, mmap='r')
    return nii_obj.dataobj

# Run the master cloud setup routine
try:
    mmap_data = initialize_and_map_volume()
    shape = mmap_data.shape
    
    # 4. Interactive Navigation Sidebar Controls
    st.sidebar.header("🕹️ Slice controls")
    # Axis 2 of this dataset corresponds to the true 707 depth layers
    target_z = st.sidebar.slider("Select Z-Index Depth", 0, shape[2] - 1, 279)
    downsample = st.sidebar.slider("Pixel Detail Striding", 1, 4, 2)
    
    # Extract EXACTLY ONE slice grid on demand
    # Pass 0 to the 4th axis index to drop the singleton dimension seamlessly
    raw_slice = mmap_data[::downsample, ::downsample, target_z, 0, :].astype(np.float32)
    
    # --- COLAB MEDICAL CONTRAST-BOOST PIPELINE ---
    gray_calc = np.mean(raw_slice, axis=2)
    brain_mask = gray_calc > 5  # Filter out background empty voids
    
    if np.any(brain_mask):
        t_min = np.min(raw_slice[brain_mask])
        t_max = np.max(raw_slice[brain_mask])
        
        # Stretch visible tones across full 8-bit dynamic canvas spectrum
        clipped = np.clip(raw_slice, t_min, t_max)
        normalized = (255.0 * (clipped - t_min) / (t_max - t_min)).astype(np.uint8)
        final_render = np.rot90(normalized, 1)
    else:
        final_render = np.rot90(raw_slice, 1).astype(np.uint8)
        
    # 5. Output Image Result to Canvas Container
    st.image(
        final_render,
        caption=f"Real anatomy viewport at Z-index position: {target_z} (Resolution: {final_render.shape[1]}x{final_render.shape[0]})",
        use_container_width=True
    )

except Exception as e:
    st.error(f"Application environment exception: {e}")
