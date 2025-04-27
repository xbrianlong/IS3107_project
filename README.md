# IS3107_project

## Current Script command for extracting top songs for users and excluding global top songs
```
python -m src.data.data_extraction
```

## Under lightgcn/data, we need a `music` directory, in which we will store our intermediate data (e.g. train.txt, test.txt, etc.)

# Streamlit Dashboard Setup Guide

## Prerequisites
- Python 3.7 or higher
- pip (Python package installer)
- Git (for cloning the repository)

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/xbrianlong/IS3107_project.git
cd IS3107_project
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
```

### 3. Activate the virtual environment

**On Windows:**
```bash
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
source venv/bin/activate
```

### 4. Install the required dependencies
```bash
pip install -r requirements.txt
```

## Running the Dashboard

### 1. Navigate to the App directory
```bash
cd App
```

### 2. Run the application
```bash
streamlit run app.py
```

### 3. Access the dashboard
Open your web browser and go to:
```
http://localhost:8050
```
*(Note: The port may vary depending on your configuration)*