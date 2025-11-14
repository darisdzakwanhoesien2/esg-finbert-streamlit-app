# app.py
import subprocess
import sys

# Always run Streamlit from this file (for Streamlit Cloud)
subprocess.run([
    sys.executable,
    "-m", "streamlit", "run",
    "dashboard/streamlit_app.py"
])
