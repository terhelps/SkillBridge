"""

Concept and purpose of model:

1. Extracting relevant data and cleaning data from raw data in csv file
2. Using ollama to group words into cluster
3. Using Kmeans to find clusters
4. Using ollama to label each clusters to a suitable course name
5. Provide summary output on clusters needed. 

To use this model: 

    1. Ensure that ollama is installed and running
        - Enter 'ollama list' in terminal
        - If llama3 is not inside, enter 'ollama pull llama3'
        - Enter 'ollama serve'
    2. In a separate terminal, run course_model.py and wait for 5mins. 

"""

import time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import ollama

# Load CSV
from pathlib import Path
DATA_PATH = Path(__file__).parent.parent / 'data' / 'jobscope.csv'
jobscope_df = pd.read_csv(DATA_PATH)
print(jobscope_df.head())

# Identify target col (May change depending on data)
TEXT_COL = 'Job_Description'
texts = jobscope_df[TEXT_COL].dropna().tolist()

# Creating model
model_id = 'llama3'
def extract_skills(jobtexts, model):
    prompt_1 = f"""
    From the following job description, extract only the specific technical and operational skills required. 
    Return them as a comma separated list with no explanation.
    
    Job Description: {jobtexts}
    """

    response = ollama.generate(model=model, prompt=prompt_1)
    return response['response']

# Generating response from ollama
print("\nExtracting skills from job scopes...")
timestart = time.time()
jobscope_df['extracted_skills'] = jobscope_df[TEXT_COL].apply(
    lambda x: extract_skills(str(x), model_id)
)
endtime = time.time()
print("Skills extracted in" ,int(endtime - timestart), "seconds")
print(jobscope_df['extracted_skills'].head())

#Elbow method
vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
X = vectorizer.fit_transform(jobscope_df['extracted_skills'])

inertia = []
silhouette_scores = []
k_range = range(2, 10)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    inertia.append(km.inertia_)
    silhouette_scores.append(silhouette_score(X, km.labels_))

#Pick cluster
best_k = k_range[silhouette_scores.index(max(silhouette_scores))]
print(f"\nOptimal number of clusters: {best_k}")

km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
jobscope_df['cluster'] = km_final.fit_predict(X)

#Labeling cluster with ollama
def label_cluster(cluster_texts, model):
    sample = "\n".join(cluster_texts[:5])  # Send up to 5 samples to Ollama
    prompt = f"""
    These job descriptions have been grouped together by similarity.
    What is the common skill theme they share?
    Suggest a short course name (max 6 words) for this group.
    Return only the course name, nothing else.
    
    Job Descriptions:
    {sample}
    """
    response = ollama.generate(model=model, prompt=prompt)
    return response['response'].strip()

print("\nLabelling clusters...")
cluster_labels = {}
for cluster_id in range(best_k):
    cluster_texts = jobscope_df[jobscope_df['cluster'] == cluster_id][TEXT_COL].tolist()
    label = label_cluster(cluster_texts, model_id)
    cluster_labels[cluster_id] = label
    print(f"Cluster {cluster_id}: {label}")

jobscope_df['course_name'] = jobscope_df['cluster'].map(cluster_labels)

# Summary
summary = jobscope_df.groupby(['cluster', 'course_name']).size().reset_index(name='job_count')
print("\n── Suggested Courses ──────────────────────────────────────")
print(summary.to_string(index=False))