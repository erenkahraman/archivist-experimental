# Archivist Pattern Analysis with OpenAI

This project uses OpenAI's Vision API to analyze patterns in images. The pattern analysis is performed by OpenAI's GPT-4 Vision model, which provides detailed information about patterns, including:

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

2. Set your OpenAI API key:
   - Create a `.env` file in the root directory and add your API key:
     ```
     OPENAI_API_KEY=your_actual_api_key_here
     ```
   - Or set it as an environment variable:
     ```
     export OPENAI_API_KEY=your_actual_api_key_here
     ```
   - Or use the API endpoint to set it:
     ```
     POST /api/set-openai-key
     {
       "api_key": "your_actual_api_key_here"
     }
     ```

3. Run the application:
   ```
   python -m src.app
   ```

## API Endpoints

- `POST /api/set-openai-key`: Set or update the OpenAI API key
- `POST /api/upload`: Upload an image for analysis
- `GET /api/images`: Get all processed images
- `POST /api/search`: Search for images based on patterns and other criteria

## How It Works

1. When an image is uploaded, it's sent to OpenAI's Vision API for pattern analysis
2. The API returns detailed information about the patterns in the image
3. The results are stored in the metadata and can be searched and filtered

## Important Security Note

Never commit your actual OpenAI API key to version control. The placeholder values in this repository (`your_openai_api_key_here`) must be replaced with your actual API key when running the application, but should not be committed to the repository. 