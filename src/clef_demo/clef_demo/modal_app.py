import importlib.util
import subprocess
import shlex
import sys
import modal

app = modal.App("checkthat-streamlit-demo")

image = modal.Image.debian_slim(python_version="3.11").pip_install("streamlit", "modal")

def _resolve_streamlit_script_path() -> str:
    spec = importlib.util.find_spec("clef_demo.streamlit_app")
    if spec is None or spec.origin is None:
        raise RuntimeError("Unable to resolve module path for clef_demo.streamlit_app")
    return spec.origin


@app.function(
    image=image,
    timeout=60 * 60 * 1,
    memory=4096,
    scaledown_window=600,
)
@modal.concurrent(max_inputs=100)
# Increased startup_timeout to 60s. Modal's default is 5s, which can cause 
# 502 Bad Gateway errors if Streamlit takes a moment to initialize.
@modal.web_server(8501, startup_timeout=60) 
def serve_streamlit():
    script_path = _resolve_streamlit_script_path()
    
    # Safely wrap the path in quotes in case there are spaces in your folder names
    target = shlex.quote(script_path)
    
    # The command string is broken into a multi-line format for readability.
    # Critical proxy and payload configurations are injected as CLI flags here.
    cmd = (
        f"streamlit run {target} "
        "--server.port 8501 "
        "--server.address 0.0.0.0 "
        "--server.headless true "
        "--server.enableCORS false "
        "--server.enableXsrfProtection false "
        "--server.maxMessageSize 2 "
        "--server.enableWebsocketCompression false "
        "--browser.gatherUsageStats false"
    )
    
    # stdout and stderr are piped so you can see Streamlit's boot logs in your terminal
    subprocess.Popen(cmd, shell=True, stdout=sys.stdout, stderr=sys.stderr)