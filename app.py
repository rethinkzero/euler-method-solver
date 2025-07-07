import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64

def euler_method(k, x0, y0, x_final, h):
    """
    Implement Euler's method for solving dy/dx = ky
    
    Args:
        k: proportionality constant
        x0: initial x value
        y0: initial y value
        x_final: final x value
        h: step size
    
    Returns:
        DataFrame with step-by-step calculations
    """
    # Calculate number of steps
    n_steps = int((x_final - x0) / h)
    
    # Initialize arrays
    x_values = np.zeros(n_steps + 1)
    y_values = np.zeros(n_steps + 1)
    dy_dx_values = np.zeros(n_steps + 1)
    
    # Set initial conditions
    x_values[0] = x0
    y_values[0] = y0
    dy_dx_values[0] = k * y0
    
    # Apply Euler's method
    for i in range(n_steps):
        x_values[i + 1] = x_values[i] + h
        y_values[i + 1] = y_values[i] + h * dy_dx_values[i]
        dy_dx_values[i + 1] = k * y_values[i + 1]
    
    # Create DataFrame for step-by-step display
    steps_df = pd.DataFrame({
        'Step': range(n_steps + 1),
        'x': x_values,
        'y': y_values,
        'dy/dx': dy_dx_values,
        'h * dy/dx': np.concatenate([h * dy_dx_values[:-1], [np.nan]])
    })
    
    return steps_df

def analytical_solution(k, x0, y0, x_values):
    """
    Calculate the analytical solution for dy/dx = ky
    The solution is y = y0 * exp(k * (x - x0))
    """
    return y0 * np.exp(k * (x_values - x0))

def calculate_errors(numerical_y, analytical_y):
    """
    Calculate absolute and relative errors
    """
    absolute_error = np.abs(numerical_y - analytical_y)
    relative_error = np.abs((numerical_y - analytical_y) / analytical_y) * 100
    return absolute_error, relative_error

def create_plot(steps_df, k, x0, y0):
    """
    Create interactive plot comparing numerical and analytical solutions
    """
    x_values = steps_df['x'].values
    y_numerical = steps_df['y'].values
    
    # Calculate analytical solution
    y_analytical = analytical_solution(k, x0, y0, x_values)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Numerical vs Analytical Solutions', 'Error Analysis'),
        vertical_spacing=0.12,
        specs=[[{"secondary_y": False}], [{"secondary_y": True}]]
    )
    
    # Main plot - solutions comparison
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=y_numerical,
            mode='lines+markers',
            name='Euler\'s Method (Numerical)',
            line=dict(color='blue', width=2),
            marker=dict(size=4)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=y_analytical,
            mode='lines',
            name='Analytical Solution',
            line=dict(color='red', width=2, dash='dash')
        ),
        row=1, col=1
    )
    
    # Error analysis
    abs_error, rel_error = calculate_errors(y_numerical, y_analytical)
    
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=abs_error,
            mode='lines+markers',
            name='Absolute Error',
            line=dict(color='green', width=2),
            marker=dict(size=3)
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=rel_error,
            mode='lines+markers',
            name='Relative Error (%)',
            line=dict(color='orange', width=2),
            marker=dict(size=3),
            yaxis='y4'
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title='Euler\'s Method: Differential Equation Solver',
        height=700,
        showlegend=True,
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)')
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="x", row=1, col=1)
    fig.update_yaxes(title_text="y", row=1, col=1)
    fig.update_xaxes(title_text="x", row=2, col=1)
    fig.update_yaxes(title_text="Absolute Error", row=2, col=1)
    fig.update_yaxes(title_text="Relative Error (%)", secondary_y=True, row=2, col=1)
    
    return fig, y_analytical, abs_error, rel_error

def export_results(steps_df, y_analytical, abs_error, rel_error):
    """
    Create downloadable CSV with all results
    """
    export_df = steps_df.copy()
    export_df['Analytical_y'] = y_analytical
    export_df['Absolute_Error'] = abs_error
    export_df['Relative_Error_percent'] = rel_error
    
    return export_df

def main():
    st.set_page_config(
        page_title="Euler's Method Solver",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("🔬 Euler's Method: Differential Equation Solver")
    st.markdown("""
    This application solves differential equations of the form **dy/dx = ky** using Euler's method,
    where the rate of change is directly proportional to the amount present.
    """)
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("📋 Input Parameters")
        
        # Input validation helper
        def validate_input(value, name, min_val=None, max_val=None):
            if min_val is not None and value < min_val:
                st.error(f"{name} must be >= {min_val}")
                return False
            if max_val is not None and value > max_val:
                st.error(f"{name} must be <= {max_val}")
                return False
            return True
        
        # Parameter inputs
        k = st.number_input(
            "Proportionality constant (k)",
            value=1.0,
            format="%.4f",
            help="The constant k in dy/dx = ky"
        )
        
        st.subheader("Initial Conditions")
        x0 = st.number_input(
            "Initial x value (x₀)",
            value=0.0,
            format="%.4f"
        )
        
        y0 = st.number_input(
            "Initial y value (y₀)",
            value=1.0,
            format="%.4f"
        )
        
        st.subheader("Solution Parameters")
        x_final = st.number_input(
            "Final x value",
            value=2.0,
            format="%.4f"
        )
        
        h = st.number_input(
            "Step size (h)",
            value=0.1,
            min_value=0.001,
            max_value=1.0,
            format="%.4f",
            help="Smaller step sizes give more accurate results"
        )
        
        # Validation
        valid_inputs = True
        if x_final <= x0:
            st.error("Final x value must be greater than initial x value")
            valid_inputs = False
        
        if h <= 0:
            st.error("Step size must be positive")
            valid_inputs = False
        
        if (x_final - x0) / h > 10000:
            st.error("Too many steps! Reduce step size or range.")
            valid_inputs = False
    
    # Main content area
    if valid_inputs:
        # Calculate solution
        try:
            steps_df = euler_method(k, x0, y0, x_final, h)
            
            # Create tabs for different views
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Visualization", "📋 Step-by-Step", "📈 Error Analysis", "💾 Export"])
            
            with tab1:
                st.subheader("Solution Visualization")
                
                # Create and display plot
                fig, y_analytical, abs_error, rel_error = create_plot(steps_df, k, x0, y0)
                st.plotly_chart(fig, use_container_width=True)
                
                # Display equation and solution
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"""
                    **Differential Equation:** dy/dx = {k}y
                    
                    **Initial Conditions:** x₀ = {x0}, y₀ = {y0}
                    
                    **Parameters:** Final x = {x_final}, Step size = {h}
                    """)
                
                with col2:
                    final_numerical = steps_df['y'].iloc[-1]
                    final_analytical = y_analytical[-1]
                    final_error = abs(final_numerical - final_analytical)
                    
                    st.success(f"""
                    **Final Results:**
                    
                    Numerical Solution: {final_numerical:.6f}
                    
                    Analytical Solution: {final_analytical:.6f}
                    
                    Absolute Error: {final_error:.6f}
                    
                    Relative Error: {(final_error/final_analytical)*100:.4f}%
                    """)
            
            with tab2:
                st.subheader("Step-by-Step Calculations")
                st.markdown("""
                Each step follows: **y_{n+1} = y_n + h × (dy/dx)_n**
                
                Where **(dy/dx)_n = k × y_n**
                """)
                
                # Display step-by-step table
                display_df = steps_df.copy()
                display_df = display_df.round(6)
                
                # Format the dataframe for better display
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True
                )
                
                st.caption(f"Total steps: {len(steps_df) - 1}")
            
            with tab3:
                st.subheader("Error Analysis")
                
                # Error statistics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Max Absolute Error",
                        f"{np.max(abs_error):.6f}",
                        help="Maximum absolute difference between numerical and analytical solutions"
                    )
                
                with col2:
                    st.metric(
                        "Max Relative Error",
                        f"{np.max(rel_error):.4f}%",
                        help="Maximum relative error as percentage"
                    )
                
                with col3:
                    st.metric(
                        "RMS Error",
                        f"{np.sqrt(np.mean(abs_error**2)):.6f}",
                        help="Root mean square error"
                    )
                
                # Error explanation
                st.markdown("""
                **Error Types:**
                - **Absolute Error**: |y_numerical - y_analytical|
                - **Relative Error**: |(y_numerical - y_analytical) / y_analytical| × 100%
                - **RMS Error**: √(mean(absolute_errors²))
                
                **Note:** Smaller step sizes generally reduce error but increase computation time.
                """)
            
            with tab4:
                st.subheader("Export Results")
                
                # Prepare export data
                export_df = export_results(steps_df, y_analytical, abs_error, rel_error)
                
                # Show preview
                st.write("**Data Preview:**")
                st.dataframe(export_df.head(10), use_container_width=True)
                
                # Download button
                csv_buffer = io.StringIO()
                export_df.to_csv(csv_buffer, index=False)
                csv_data = csv_buffer.getvalue()
                
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv_data,
                    file_name=f"euler_method_results_k{k}_h{h}.csv",
                    mime="text/csv",
                    help="Download complete results including numerical solution, analytical solution, and error analysis"
                )
                
                # Summary statistics
                st.subheader("Summary Statistics")
                summary_stats = {
                    "Parameter": ["k", "x₀", "y₀", "x_final", "h", "Steps"],
                    "Value": [k, x0, y0, x_final, h, len(steps_df) - 1]
                }
                
                st.table(pd.DataFrame(summary_stats))
        
        except Exception as e:
            st.error(f"An error occurred during calculation: {str(e)}")
            st.error("Please check your input parameters and try again.")
    
    else:
        st.warning("Please correct the input parameters in the sidebar to proceed.")
    
    # Footer with information
    st.markdown("---")
    st.markdown("""
    **About Euler's Method:**
    
    Euler's method is a first-order numerical procedure for solving ordinary differential equations 
    with a given initial value. For the equation dy/dx = ky, the analytical solution is 
    y = y₀ × e^(k×(x-x₀)).
    
    **Mathematical Background:**
    - The method approximates the solution using the slope at each point
    - Smaller step sizes improve accuracy but require more computation
    - The global error is proportional to the step size h
    """)

if __name__ == "__main__":
    main()
