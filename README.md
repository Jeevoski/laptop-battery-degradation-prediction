# Laptop Battery Degradation Prediction

A machine learning and MLOps system to predict laptop battery degradation patterns, estimated remaining useful life (RUL), and capacity fade over time.

## 📊 Overview

Lithium-ion batteries in modern laptops degrade over time due to electrochemical aging, usage patterns, charge cycles, temperature, and depth of discharge. This project provides a robust, data-driven framework to model, analyze, and predict laptop battery health and degradation trajectories.

The goal is to proactively predict when a battery will reach its End-of-Life (EoL) or fall below acceptable capacity thresholds (typically 80% of design capacity), allowing users and IT administrators to plan replacements efficiently.

## 🔍 Core Features

- **Degradation Modeling**: Predicts remaining useful capacity and State-of-Health (SoH) based on charge cycles, thermal profiles, and age.
- **RUL (Remaining Useful Life) Forecast**: Time-series forecasting and regression models to estimate the remaining cycles before battery capacity drops below a critical threshold.
- **Interactive Reports**: Automated visualization of historical battery capacity fade and charge-discharge rates.
- **MLOps Pipeline**: Reproducible data preparation, model training, evaluation, and tracking (designed for easy expansion).

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Git

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Jeevoski/laptop-battery-degradation-prediction.git
   cd laptop-battery-degradation-prediction
   ```

2. **Create a virtual environment & install dependencies:**
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   
   pip install -r requirements.txt
   ```

## 📁 Repository Structure

```
├── collect_battery.py # CLI status collector (generates & parses powercfg report on Windows)
├── battery_history.csv # Aggregated CSV of battery status history entries over time
├── lap.py             # Core logic / entry point for the prediction modules
├── .gitignore         # Python-specific ignore configurations
└── README.md          # Project overview and documentation
```

## 📈 Future Roadmap

- [x] Integrate battery status collector CLI to extract battery report logs directly from OS (`powercfg /batteryreport` on Windows).
- [ ] Implement LSTM and Random Forest models for sequential capacity degradation prediction.
- [ ] Build a lightweight Streamlit dashboard for real-time local battery health tracking.
- [ ] Add model registry and experiment tracking.

## 🤝 Contributing

Contributions are welcome! Please feel free to open a Pull Request or create an Issue if you have feature suggestions, bug reports, or performance improvements.

## 📄 License

This project is licensed under the MIT License.
