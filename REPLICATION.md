# 🔄 Replication & Setup Guide

This guide provides step-by-step instructions to replicate the **NyayaSetu** Legal Advisory System environment from scratch.

## 📋 Prerequisites

Before starting, ensure you have the following installed on your system:

1.  **Python 3.10+**: [Download Python](https://www.python.org/downloads/)
2.  **Git**: [Download Git](https://git-scm.com/downloads)
3.  **Docker & Docker Compose** (Optional but recommended for local Qdrant): [Download Docker Desktop](https://www.docker.com/products/docker-desktop)
4.  **API Keys**:
    *   **Google Gemini API Key**: Get it from [Google AI Studio](https://aistudio.google.com/)
    *   **Qdrant Cloud API Key & URL** (If using Cloud): Get it from [Qdrant Cloud](https://cloud.qdrant.io/)

---

## 🛠️ Step-by-Step Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Start-End-G/Legal-Advisory-Multi-Agent-System.git
cd Legal-Advisory-Multi-Agent-System
```

*(Note: Replace the URL with your actual repository URL if different)*

### 2. Set Up a Virtual Environment

It is highly recommended to use a virtual environment to manage dependencies.

**Windows:**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

Install the required Python packages from `requirements.txt`.

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

1.  Create a file named `.env` in the root directory.
2.  Add the following variables to it:

```env
# Required: Google Gemini API Key
GOOGLE_API_KEY=your_google_api_key_here

# Optional: Qdrant Configuration (Defaults to localhost:6333 if not set)
# If using Qdrant Cloud:
QDRANT_URL=https://your-cluster-url.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key_here

# If using Local Qdrant (Docker):
# QDRANT_URL=http://localhost:6333
# QDRANT_API_KEY=
```

### 5. Setup Vector Database (Qdrant)

#### Option A: Using Qdrant Cloud (Recommended for Production)
Ensure your `.env` has the `QDRANT_URL` and `QDRANT_API_KEY` set.

#### Option B: Using Local Qdrant (Docker)
If you want to run Qdrant locally, use Docker Compose:

```bash
docker-compose up -d
```
This spins up a Qdrant instance at `http://localhost:6333`.

### 6. Initialize Collections

Run the setup script to create the necessary collections (`legal_knowledge`, `case_memory`, `evidence_vault`) in Qdrant.

```bash
python src/setup_qdrant.py
```
*Expected Output:* `✅ Created Collection: legal_knowledge` etc.

### 7. Ingest Legal Data

Download and ingest the Indian Laws dataset into the vector database. This may take some time depending on your internet connection.

```bash
python src/ingest_data.py
```
*Expected Output:* `🚀 SUCCESS: Full Indian Legal Code Ingested!`

---

## 🏃‍♂️ Running the Application

Once the setup is complete, you can launch the Streamlit application.

```bash
streamlit run src/app.py
```

*   The application should open automatically in your browser at `http://localhost:8501`.
*   If it doesn't, click the URL displayed in the terminal.

---

## 🔍 Troubleshooting

*   **`ModuleNotFoundError`**: Ensure your virtual environment is activated and you have installed requirements (`pip install -r requirements.txt`).
*   **Qdrant Connection Error**:
    *   Check if `QDRANT_URL` in `.env` is correct.
    *   If running locally, ensure Docker container is running (`docker ps`).
    *   If using Cloud, check your API Key and internet connection.
*   **Google API Error**: Verify your `GOOGLE_API_KEY` is valid and has access to Gemini Flash models.
