# Multi-Agent Coalition Formation System

The DIM framework for dynamic coalition formation in classrooms, built with Mesa framework and FastAPI. Students autonomously form groups based on skill complementarity, social preferences, and personality compatibility.


## 📋 Prerequisites

- Python 3.11 or higher
- pip (Python package manager)
- Modern web browser (Chrome, Firefox, Safari, or Edge)

## 🚀 Quick Start

### 1. Clone or Download the Project

```bash
cd /path/to/mas_project
```

### 2. Create Virtual Environment

```bash
python3 -m venv .venv
```

### 3. Activate Virtual Environment

**macOS/Linux:**
```bash
source .venv/bin/activate
```

**Windows:**
```bash
.venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Start the Backend API Server

```bash
python api_server.py
```

You should see:
```
INFO:     Started server process
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### 6. Open the Web Interface

Open `index.html` in your web browser:

**Option A - Direct:**
```bash
open index.html  # macOS
```

**Option B - Using Python's HTTP server (recommended):**

Open a **new terminal** (keep the API server running), activate the virtual environment, and run:
```bash
cd /path/to/mas_project
python -m http.server 8080
```

Then open your browser to: `http://localhost:8080`


