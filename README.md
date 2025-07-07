# Euler's Method Differential Equation Solver

A web application built with Streamlit that solves differential equations of the form `dy/dx = ky` using Euler's method, where the rate of change is directly proportional to the amount present.

<!-- Add this badge AFTER you deploy to Streamlit Cloud -->
<!-- [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app) -->

## Features

- **Interactive Parameter Input**: Set proportionality constant (k), initial conditions (x₀, y₀), final x value, and step size
- **Real-time Visualization**: Compare numerical and analytical solutions with interactive plots
- **Step-by-Step Analysis**: View detailed calculations for each iteration
- **Error Analysis**: Comprehensive error metrics including absolute, relative, and RMS errors
- **Data Export**: Download results as CSV for further analysis

## Mathematical Background

The application solves differential equations of the form:
```
dy/dx = ky
```

Where:
- k is the proportionality constant
- The analytical solution is: y = y₀ × e^(k×(x-x₀))

Euler's method approximates the solution using: y_{n+1} = y_n + h × (dy/dx)_n

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/euler-method-solver.git
cd euler-method-solver
```

2. Install the required dependencies:
```bash
pip install streamlit numpy pandas plotly
```

3. Run the application:
```bash
streamlit run app.py
```

## For Streamlit Cloud Deployment

If you're deploying to Streamlit Cloud and get an installer error, rename `streamlit_requirements.txt` to `requirements.txt` in your GitHub repository.

## Usage

1. **Set Parameters**: Use the sidebar to input:
   - Proportionality constant (k)
   - Initial x and y values
   - Final x value
   - Step size (h)

2. **View Results**: Navigate through the tabs to see:
   - Visualization of numerical vs analytical solutions
   - Step-by-step calculations
   - Error analysis
   - Export options

3. **Export Data**: Download complete results as CSV for further analysis

## Dependencies

- streamlit>=1.28.0
- numpy>=1.24.0
- pandas>=2.0.0
- plotly>=5.15.0

## Deployment

### Streamlit Cloud
1. Push your code to GitHub
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your GitHub repository
4. Deploy directly from your repository

### Local Development
```bash
streamlit run app.py --server.port 5000
```

## Project Structure

```
euler-method-solver/
├── app.py              # Main Streamlit application
├── .streamlit/
│   └── config.toml     # Streamlit configuration
├── requirements.txt    # Python dependencies
└── README.md          # Project documentation
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is open source and available under the MIT License.
