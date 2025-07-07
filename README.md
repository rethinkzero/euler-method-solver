# Euler's Method Differential Equation Solver

A Streamlit web application that implements Euler's method for solving differential equations of the form dy/dx = ky.

## Features

- **Interactive Input**: Enter coefficient k, initial values (x₀, y₀), final x value, and step size
- **Step-by-Step Calculations**: View detailed calculations for each iteration
- **Graphical Visualization**: Compare Euler's method with analytical solution
- **Error Analysis**: See absolute error at each step
- **Download Results**: Export calculations as CSV file
- **Educational Content**: Built-in explanations of Euler's method

## Installation

1. Make sure you have Python 3.11 or later installed
2. Install the required dependencies:
   ```bash
   pip install streamlit numpy pandas plotly
   ```

## Usage

1. Run the application:
   ```bash
   streamlit run app.py
   ```

2. Open your web browser and go to `http://localhost:8501`

3. Enter your parameters:
   - **k**: Coefficient in the differential equation dy/dx = ky
   - **x₀**: Initial x value
   - **y₀**: Initial y value
   - **Final x**: The x value to calculate up to
   - **Step size (h)**: Smaller values give more accurate results

4. Click "Calculate Euler's Method" to see the results

## Example Problems

- **Exponential Growth**: k = 0.5, x₀ = 0, y₀ = 1, x_final = 2, h = 0.1
- **Exponential Decay**: k = -0.3, x₀ = 0, y₀ = 100, x_final = 5, h = 0.2
- **High Precision**: k = 1.0, x₀ = 0, y₀ = 1, x_final = 1, h = 0.01

## How It Works

Euler's method approximates the solution to a differential equation using the formula:
**y₍ₙ₊₁₎ = yₙ + h × f(xₙ, yₙ)**

For the specific equation dy/dx = ky, this becomes:
**y₍ₙ₊₁₎ = yₙ + h × k × yₙ**

The app compares this numerical solution with the analytical solution: **y = y₀ × e^(k(x-x₀))**

## Files

- `app.py`: Main Streamlit application
- `.streamlit/config.toml`: Streamlit configuration
- `pyproject.toml`: Project dependencies
- `README.md`: This file

## Dependencies

- streamlit
- numpy
- pandas
- plotly
- math (built-in)

Built with Streamlit • Educational tool for numerical methods
