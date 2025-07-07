# Deploying Your Euler's Method App to Streamlit Cloud

## Files Ready for Deployment

Your app is ready with these files:
- `app.py` - Main Streamlit application
- `streamlit_requirements.txt` - Dependencies list
- `.streamlit/config.toml` - Streamlit configuration
- `README.md` - Documentation

## Step-by-Step Deployment Process

### 1. Upload to GitHub

1. **Create a new GitHub repository**:
   - Go to [github.com](https://github.com) and sign in
   - Click "New repository"
   - Name it something like "euler-method-calculator"
   - Make it public (required for free Streamlit Cloud)

2. **Upload your files**:
   - Click "uploading an existing file"
   - Drag and drop these files:
     - `app.py`
     - `streamlit_requirements.txt`
     - `.streamlit/config.toml`
     - `README.md`
   - Commit the files

### 2. Deploy to Streamlit Cloud

1. **Go to Streamlit Cloud**:
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account

2. **Create new app**:
   - Click "New app"
   - Select your GitHub repository
   - Choose `app.py` as your main file
   - Click "Deploy"

3. **Configure dependencies**:
   - Streamlit Cloud will automatically detect `streamlit_requirements.txt`
   - If it doesn't, rename it to `requirements.txt` in GitHub

### 3. Access Your App

Once deployed (takes 2-5 minutes), you'll get a public URL like:
`https://your-username-euler-method-calculator-main-abc123.streamlit.app`

## Important Notes

- **File naming**: If deployment fails, rename `streamlit_requirements.txt` to `requirements.txt`
- **Repository must be public** for free tier
- **Automatic updates**: Any changes you push to GitHub will automatically update your app
- **Custom domain**: You can configure a custom domain in Streamlit Cloud settings

## Sharing Your App

Once deployed, anyone can:
- Visit the URL in any web browser
- Use the Euler's method calculator
- Enter different parameters and see results
- Download calculation results as CSV files
- Learn about Euler's method through the built-in explanations

Your app will be available 24/7 at no cost through Streamlit Cloud!
