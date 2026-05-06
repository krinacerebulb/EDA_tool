# Auto EDA Platform

A polished, no-code exploratory data analysis platform that transforms raw datasets into interactive insights and downloadable reports within seconds.

---

## Overview

Auto EDA Platform is a self-service analytics application designed to simplify exploratory data analysis for both technical and non-technical users.

Upload a dataset and the platform automatically:

- Profiles the dataset
- Detects missing values and duplicates
- Identifies outliers
- Generates interactive visualizations
- Performs target-based analysis
- Detects time-series trends
- Generates plain-English insights
- Creates downloadable HTML reports

The platform eliminates repetitive manual EDA workflows and provides a fast, consistent, and user-friendly analytics experience.

---

## Features

- Upload CSV, Excel, JSON, and Parquet datasets
- Automated exploratory data analysis
- Smart data type detection
- Missing-value analysis
- Duplicate detection
- Outlier detection using IQR
- Interactive filtering system
- Correlation analysis
- Target-based analytics
- Time-series trend analysis
- Interactive Plotly visualizations
- Auto-generated insights
- Downloadable HTML reports
- No-code interface for non-technical users

---

## Technology Stack

| Layer           | Technology                  |
| --------------- | --------------------------- |
| Language        | Python 3.11+                |
| UI Framework    | Streamlit                   |
| Data Processing | Pandas, NumPy               |
| Visualization   | Plotly, Matplotlib, Seaborn |
| File Support    | openpyxl, xlrd, pyarrow     |
| Statistics      | SciPy                       |
| Templating      | Jinja2                      |

---

## Installation

### 1. Clone Repository

```bash
git clone <your-repository-url>
cd deploy
```

### 2. Create Virtual Environment

#### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

#### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run Application

```bash
streamlit run app.py
```

Application runs at:

```text
http://localhost:8501
```

---

## Deployment

### Streamlit Community Cloud

1. Push the project to GitHub
2. Open Streamlit Community Cloud
3. Create a new app
4. Select the repository
5. Set the main file path to:

```text
app.py
```

6. Click Deploy

---

## Usage Guide

### Upload Dataset

Upload supported file formats:

- CSV
- Excel
- JSON
- Parquet

using the sidebar uploader.

---

### Apply Filters

Use interactive sidebar filters to dynamically refine the dataset.

All analysis tabs update automatically based on filtered data.

---

### Explore Tabs

The platform includes:

- Overview
- Cleaning
- Statistics
- Visualizations
- Time Series
- Target Analysis
- Insights
- Report Generation

---

### Generate Reports

Generate downloadable HTML reports containing:

- Summary statistics
- Visualizations
- Dataset insights
- Data quality analysis

---

## Project Structure

```text
deploy/
├── app.py
├── modules/
│   ├── data_loader.py
│   ├── data_cleaning.py
│   ├── eda_analysis.py
│   ├── filters.py
│   ├── insights.py
│   ├── interactive_viz.py
│   ├── visualization.py
│   ├── target_analysis.py
│   ├── time_series.py
│   ├── type_detection.py
│   └── report_generator.py
├── utils/
├── ui/
├── assets/
├── .streamlit/
│   └── config.toml
├── requirements.txt
├── Dockerfile
├── Procfile
└── README.md
```

---

## Future Improvements

- AI-generated insights
- AutoML integration
- Advanced forecasting
- Multi-user collaboration
- Cloud database integrations
- Authentication system
- API support
- Dashboard export options

---

## License

This project is licensed under the MIT License.
