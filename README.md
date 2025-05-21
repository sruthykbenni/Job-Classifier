# Job Classification and Recommendation using Unsupervised Learning

## Overview
This project is a Job Recommendation System that clusters scraped job listings based on required skills using unsupervised learning. It allows users to select job categories of interest and get personalized job alerts. The system is deployed using Streamlit, offering an interactive interface for job tracking and recommendation.

## Features
- **Web Scraping:** Extracts job details from Karkidi.com using BeautifulSoup.
- **Skill-Based Clustering:** Groups jobs using KMeans clustering on TF-IDF features of skills.
- **Real-Time Classification:** Classifies new job listings into pre-defined clusters.
- **User Interest Matching:** Filters jobs based on the user’s preferred job categories.
- **Interactive Web Application:** Streamlit-based interface for real-time recommendations and data download.
- **Automated Daily Update:** Supports scheduled job scraping and classification (manual or cron-enabled).

## Technologies Used
- Python
- BeautifulSoup (for web scraping)
- Scikit-learn (for TF-IDF and clustering)
- Pandas & NumPy
- Streamlit
- Pickle (for model and vectorizer storage)

## Installation
To run this project locally, follow these steps:

1. Clone the Repository
```
git clone https://github.com/sruthykbenni/job-recommendation-system.git
cd job-recommendation-system
```
3. Create a Virtual Environment (Optional but Recommended)
```
python -m venv env
source env/bin/activate  # On macOS/Linux
env\Scripts\activate  # On Windows
```
4. Install Dependencies
```
pip install -r requirements.txt
```
5. Run the Streamlit App
```
streamlit run app.py
```

## Dataset
Job listings are scraped directly from https://www.karkidi.com. Each job entry includes:
- Title
- Company
- Location
- Experience
- Skills
- Summary

The data is stored in a structured format for processing and classification.

## Model Training
The pipeline follows these steps:

1. Data Preprocessing: Lowercasing, handling missing values in skill text.
2. TF-IDF Vectorization: Converts skill text into feature vectors using TfidfVectorizer.
3. Normalization: TF-IDF vectors are normalized using L2 norm.
4. KMeans Clustering: Groups job listings into 5 clusters based on skill similarity.
5. Cluster Labeling: Human-readable labels assigned to each cluster after inspection.
6. Model Persistence: Saves trained KMeans model and vectorizer with pickle for reuse.

## Deployment
The project is deployed as a web application using Streamlit. Features include:
- Dropdown to select preferred job categories.
- Button to run daily job check and see new listings.
- Download filtered jobs as CSV.
- Real-time classification of newly scraped jobs.

You can access the live demo here:
[🔗 Live App](https://job-classifier-ffasuixadrgziuorfymcup.streamlit.app/)

## Usage
1. Open the web app.
2. Select your preferred job categories (e.g., Data Science & ML).
3. Click "Run Daily Job Check" to see matching jobs.
4. Download the results for offline access.

## Future Improvements
- Add email notification system for new job alerts.
- Integrate a database to track seen jobs and avoid duplicates.
- Enable user registration and login for persistent preferences.
- Add automatic daily scraping using cron or APScheduler.
- Support more job sites (e.g., LinkedIn, Indeed).

## Contributing
Feel free to contribute! Fork the repository, make your changes, and submit a pull request. Let’s build a smarter job recommendation engine together.
