# Synthetic Persona Generation for Finland

This project implements a data pipeline and interactive dashboard that generates a statistically representative synthetic dataset of the Finnish population. It combines real demographic statistics with a Large Language Model (LLM) to simulate how different demographic segments might theoretically respond to user-defined prompts.

## The Idea
This project is an adaptation of NVIDIA's *Nemotron-Personas* concept, applied to Finnish open data. The core premise is that LLMs output generic answers unless heavily contextualized. By using real demographic data from Statistics Finland to create a foundational "skeleton" for a synthetic person, and then using an LLM to enrich that skeleton with a localized narrative, we can build a dataset that reflects actual population distributions. The resulting interactive tool allows users to test how variables like age, region, and occupation might influence responses to policy or lifestyle questions.

## What This Project Practices
This repository serves as a practical exercise in building compound AI systems and data engineering pipelines. The primary skills and concepts demonstrated include:

* **Statistical Data Processing:** Using Pandas and NumPy to clean, weight, and sample marginal distributions from public government datasets.
* **Deterministic Imputation:** Applying logical edit rules during the sampling phase to prevent demographic contradictions (e.g., ensuring a 0-year-old cannot be classified as a pensioner with a university degree).
* **LLM Orchestration:** Designing strict system prompts to enforce JSON schema compliance, handling API rate limits, and grounding the LLM with real-world context (like official name registries) to minimize hallucination.
* **Interactive Data Visualization:** Building a frontend with Streamlit and Folium to filter datasets, trigger real-time LLM inferences, and aggregate sentiment scores onto a geospatial map of Finnish municipalities.

## Pipeline Architecture

The system is structured into four distinct phases:

1.  **Data Ingestion:** Gathers marginal demographic distributions (age, gender, education, occupation, income) from the Statistics Finland (Tilastokeskus) API and cultural naming data from the Digital and Population Data Services Agency (DVV).
2.  **Skeleton Generation (`phase2_generate.py`):** Uses weighted probabilities to generate base demographic profiles. It applies post-sampling corrections to maintain logical dependencies between age, education, and employment status.
3.  **Narrative Enrichment (`phase3_enrich.py`):** Passes the demographic skeletons to the Gemini API. The LLM returns a structured JSON object containing a culturally appropriate name, hobbies, primary transport, and life goals based on the provided parameters.
4.  **Spatial Simulation (`app.py`):** A Streamlit application where users can input questions. The app filters the synthetic personas, passes the user's question to the LLM from the perspective of the filtered personas, and plots the resulting sentiment scores (-1.0 to 1.0) on a map.

## Setup and Usage

**Prerequisites**
* Python 3.10+
* A Google Gemini API Key

**Installation**
```bash
git clone [https://github.com/yourusername/finnish-synthetic-personas.git](https://github.com/yourusername/finnish-synthetic-personas.git)
cd finnish-synthetic-personas
pip install -r requirements.txt


**Configuration**
Create a .env file in the root directory and add your API key:

Code snippet
GEMINI_API_KEY=your_actual_api_key_here
Execution

```bash
# 1. Generate the base demographic dataset
python src/phase2_generate.py

# 2. Enrich the base dataset via the LLM (run a small batch to test)
python src/phase3_enrich.py --limit 10

# 3. Start the local Streamlit dashboard
streamlit run app.py
