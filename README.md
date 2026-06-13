# 🧠 DHARINI Full Lifespan 3D Fetal Brain Volume Navigator

An interactive, cloud-native 3D neuroimaging navigator designed to stream, reconstruct, and slice through high-resolution second-trimester human brain reconstructions from the **IIT Madras DHARINI Dataset**. 
---

## 🙏 Acknowledgements & Gratitude
* **To IIT Madras & The Sudha Gopalakrishnan Brain Centre**: Deepest gratitude is extended to the research engineers, neuroanatomists, and acquisition teams at IIT Madras. Their painstaking work in digitizing, aligning, and publishing these massive cellular-resolution datasets pushes the boundaries of global computational medicine.
* **To Sudha Murty & Philanthropic Visionaries**: Sincere thanks to **Sudha Murty** and the philanthropic foundations whose foundational support and vision for cutting-edge computing centers in India have made world-class, open-access neuroscience research facilities a reality. Their generosity empowers independent developers worldwide to interact with high-value scientific pipelines.

Built independently by **Srinivasta** using **Python, Streamlit, and GitHub**, this system connects anonymously to the **AWS Open Data Registry** to execute real-time multi-planar slice extraction with a zero-RAM server footprint.

---

## 🚀 Live Application
🔗 **[Launch the Interactive Dashboard Web App](https://streamlit.app)**

---

## ⚖️ Data Attribution & Project Disclaimer
* **Independent Development**: This software application is an independent open-source dashboard designed and architected entirely by **Srinivasta**. It is not financially sponsored by, affiliated with, or endorsed by IIT Madras or the Sudha Gopalakrishnan Brain Centre.
* **Data Source Credit**: The underlying 3D brain volumes utilized by this dashboard are sourced directly from the publicly accessible **DHARINI Fetal Brain Atlas** repository hosted on the AWS Registry of Open Data. Deep gratitude is extended to the original acquisition scientists for making their structural neuroimaging data open to the global research community.

---

## 🛠️ Key Architectural & Technical Capabilities

* **On-Demand Voxel Streaming**: Eliminates the need to host or download uncompressed multi-gigabyte files locally. The application leverages an anonymous, unsigned `boto3` configuration to download raw dataset layers directly from the cloud on demand.
* **Disk-Level Memory-Mapping (`mmap='r'`)**: To bypass Streamlit Cloud’s standard system RAM ceiling (<1GB), compressed `.gz` archives are unpacked directly into local NVMe disk blocks (`/tmp/`). `nibabel` then memory-maps the data structure, keeping the live runtime footprint under **50 Megabytes**.
* **Cinematic Fly-Through Engine**: Implements a frame-by-frame animation pipeline using Streamlit session state hooks (`st.session_state`), enabling a smooth, looping fly-through visualization across 891 sequential tissue slices.
* **Anatomical Contrast Boosting**: Features a custom pixel-intensity masking and normalization workflow (`NumPy`-based background suppression) that isolates brain matter and dynamically stretches the grayscale dynamic range to highlight cellular migration layers.
* **Institutional PDF Export Report**: Integrates an on-demand clinical dossier generator (`reportlab`) that captures the active high-contrast image matrix from the browser view and compiles it into a static, publication-ready PDF document complete with automated milestone metadata logs.

---

## 📊 Dataset Matrix Coverage (AWS Open Data)
The application dynamically routes data for all 5 official specimens from the open-source repository:
1. **Specimen 1 (14 Weeks - FB34)**: Early second-trimester baseline. Smooth cortical surfaces and primitive ventricular chambers.
2. **Specimen 2 (17 Weeks - FB40)**: Initiation of rapid deep cellular migration. Bundled axon pathways around core zones.
3. **Specimen 3 (21 Weeks - FB62)**: Mid-development stratification. Visual layering of the cortical plate and subplate zones.
4. **Specimen 4 (22 Weeks - FB74)**: Structural expansion of the cerebral cortex.
5. **Specimen 5 (24 Weeks - FB85)**: Late second-trimester complexity. Initial formation of physical sulci (folds) and highly defined cerebellum layout.

---

## 💻 Tech Stack & Dependencies

* **Frontend UI Framework**: Streamlit (with custom HTML/CSS responsive flexbox injection)
* **Medical Data Operations**: Nibabel (NIfTI-1 format parsing)
* **Mathematical Compute Engine**: NumPy (Array scaling, masking, matrix rotations)
* **Cloud Storage Integration**: Boto3 & Botocore (AWS S3 Unsigned SDK)
* **Document Compilation**: ReportLab (Vectorized PDF layout automation)

---

## 📥 How to Run Locally

### 1. Clone the Codebase
```bash
git clone https://github.com
cd dharini-brain-viewer
```

### 2. Install Project Requirements
Make sure you have Python 3.10+ active, then run:
```bash
pip install -r requirements.txt
```

### 3. Launch the Server App
```bash
streamlit run app.py
```
Your local browser tab will open automatically at `http://localhost:8501`.

---

## 📁 Repository Structure
```text
├── app.py              # Main application logic & user interface layout
├── requirements.txt    # Project runtime dependency checklist
└── README.md           # Professional portfolio presentation documentation
```

---

## 🔬 Scientific Impact
This application lowers the barrier to entry for studying human brain development. By creating an interactive, web-accessible "digital twin," neuroscientists, medical students, and AI researchers can analyze critical volumetric landmarks—such as ventricular compression, gyrification timelines, and tissue density gradients—directly from any device without requiring high-end graphics processing hardware.

---
*Developed and architected by **Srinivasta** as an open-source educational dashboard showcasing cloud data pipe optimization for high-throughput healthcare informatics.*
