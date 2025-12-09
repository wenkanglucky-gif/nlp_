"""
# Legal Text Classification API

## Overview
A production-ready Flask API for legal text classification using fine-tuned Legal-BERT model with comprehensive monitoring.

## Features
-  Fine-tuned Legal-BERT model for text classification
-  Real-time performance monitoring
-  Automatic anomaly detection and alerting
-  Auto-retraining mechanism
-  Interactive web demo interface
-  Full test coverage


## Installation

### 1. Clone the repository
```bash
git clone https://github.com/wenkanglucky-gif/nlp_
cd nlp_
```

### 2. Install dependencies
```bash
make install
# or manually:
pip install -r requirements.txt
```

### 3. Setup pre-commit hooks
```bash
pre-commit install
```

## Usage

### Start the API
```bash
make run
# or:
python app.py
```

### Make predictions
## Option 1: Use Web Demo (Recommended)
Open your browser and go to:
```
http://localhost:5000/demo
```
Type legal text in the box and click "Classify Text" to see prediction results in real-time.

## Option 2: Use API directly
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Your legal text here"}'
```

### View monitoring metrics
```bash
curl http://localhost:5000/metrics
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/demo` | GET | Interactive web demo interface |
| `/predict` | POST | Make prediction |
| `/metrics` | GET | View performance metrics |
| `/trigger_retrain` | POST | Trigger model retraining |

## Testing

### Run all tests
```bash
make test
```

### Run specific test
```bash
python -m pytest test_app.py::test_predict_endpoint_exists -v
```

## Code Quality

### Format code
```bash
make format
```

### Run linter
```bash
make lint
```

### Run all checks
```bash
make check
```

## Project Structure
```
nlp_/
├── app.py                 # Main Flask application
├── test_app.py            # Unit and integration tests
├── requirements.txt       # Python dependencies
├── Makefile              # Task automation
├── .gitignore            # Git ignore rules
├── .pre-commit-config.yaml  # Pre-commit hooks
├── README.md             # This file
├── saved_legalbert/          # Trained model 
└── monitor_logs/         # Monitoring data 
    ├── metrics.csv
    └── alerts.log
```

## Monitoring

The system logs:
- **Metrics**: timestamp, prediction, confidence, input_length → `metrics.csv`
- **Alerts**: Low confidence warnings → `alerts.log`
- **Auto-retrain**: Triggered when confidence < 0.40

## Development

### Setup development environment
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
make install
```

### Before committing
Pre-commit hooks automatically run:
- Code formatting (Black)
- Linting (Flake8)
- Basic validation

## Deployment

### Local deployment
```bash
python app.py
```

## Troubleshooting

**Issue**: Model not found  
**Solution**: Ensure `saved_legalbert/` directory exists with trained model files

**Issue**: Tests failing  
**Solution**: Run `make clean` then `make test`

**Issue**: Import errors  
**Solution**: Reinstall dependencies with `make install`


## Author
Aliya Khan
Wenkang Liu

## Contact
akhan053@uottawa.ca
wliu007@uottawa.ca
"""