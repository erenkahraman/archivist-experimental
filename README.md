# Archivist Pattern Analysis with Google Gemini

This project uses Google's Gemini Pro Vision API to analyze patterns in images. The pattern analysis is performed by Google's Gemini model, which provides detailed information about patterns, including:

- Primary pattern category
- Secondary pattern types
- Specific elements in the pattern
- Pattern layout and distribution
- Pattern density and scale

## Setup

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Set your Gemini API key:
   - Create a `.env` file in the root directory and add your API key:
     ```
     GEMINI_API_KEY=your_actual_api_key_here
     ```
   - Or set it as an environment variable:
     ```
     export GEMINI_API_KEY=your_actual_api_key_here
     ```
   - Or use the API endpoint to set it:
     ```
     POST /api/set-gemini-key
     {
       "api_key": "your_actual_api_key_here"
     }
     ```

3. Run the application:
   ```
   python -m src.app
   ```

## API Endpoints

- `POST /api/set-gemini-key`: Set or update the Gemini API key
- `POST /api/upload`: Upload an image for analysis
- `GET /api/images`: Get all processed images
- `POST /api/search`: Search for images based on patterns and other criteria

## How It Works

1. When an image is uploaded, it's sent to Google's Gemini API for pattern analysis
2. The API returns detailed information about the patterns in the image
3. The results are stored in the metadata and can be searched and filtered

## Security

### API Key Security

Never commit your actual Gemini API key to version control. The placeholder values in this repository (`your_gemini_api_key_here`) must be replaced with your actual API key when running the application, but should not be committed to the repository.

### Pre-commit Hook

A pre-commit hook is included to prevent accidentally committing API keys. To enable it:

1. Make sure the hook is executable:
   ```
   chmod +x .git/hooks/pre-commit
   ```

2. The hook will automatically check for API keys in staged files before committing.

### Environment Variables

- Use the `.env` file for local development (it's already in `.gitignore`)
- For production, set environment variables in your deployment environment
- The `.env.example` file shows what environment variables are needed, but doesn't contain actual secrets 