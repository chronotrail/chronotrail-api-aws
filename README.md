# ChronoTrail API

Personal timeline data storage and natural language querying service built with FastAPI.

## Features

- Store location visits, text notes, photos, and voice recordings
- Natural language querying of personal timeline data
- AI-powered content processing (OCR, speech-to-text, image analysis)
- Vector-based semantic search
- OAuth authentication with Google and Apple
- Subscription-based usage tiers

## Tech Stack

- **API Framework**: FastAPI
- **Database**: PostgreSQL + AWS OpenSearch
- **File Storage**: AWS S3
- **AI Services**: AWS Bedrock, Textract, Transcribe, Rekognition
- **Authentication**: AWS Cognito
- **Package Management**: UV

## Development Setup

1. **Install UV** (Python package manager):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone and setup project**:
   ```bash
   git clone <repository-url>
   cd chronotrail-api
   ```

3. **Install dependencies**:
   ```bash
   uv sync
   ```

4. **Set up environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Run the development server**:
   ```bash
   # Using UV script
   uv run dev
   
   # Or directly with uvicorn
   uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

## Project Structure

```
app/
├── api/                 # API endpoints
│   └── v1/             # API version 1
├── core/               # Core configuration and utilities
├── db/                 # Database models and connections
├── models/             # Pydantic models
└── services/           # Business logic services
```

## API Documentation

Once running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Environment Variables

See `.env.example` for required environment variables.

## License

MIT License