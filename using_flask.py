from flask import Flask, render_template
import subprocess
import os

app = Flask(__name__)

# Path to your Streamlit script
STREAMLIT_APP_PATH = "path_to_streamlit_app.py"

@app.route("/")
def home():
    # Render an HTML template with an iframe to the Streamlit app
    return render_template("index.html")  # You'll create this template in Step 3

def run_streamlit():
    """Run the Streamlit app in a subprocess."""
    streamlit_command = f"streamlit run {STREAMLIT_APP_PATH} --server.port=8501"
    subprocess.Popen(streamlit_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

if __name__ == "__main__":
    # Run the Streamlit app as a subprocess
    run_streamlit()
    # Start Flask app
    app.run(debug=True, port=5000)
