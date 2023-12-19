import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
from fastapi import FastAPI
import requests

app = FastAPI()


origins = [
    "http://127.0.0.1:8000",
    "http://localhost:5173",
   
    "http://localhost:5000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


response = requests.get('https://workshala-7v7q.onrender.com/internshipData')
data = response.json()  
df = pd.DataFrame(data)
df_copy1=df
df_copy1['description_new'] = df_copy1['description'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)


vectorizer = TfidfVectorizer()
matrix = vectorizer.fit_transform(df_copy1['description_new'])



cosine_sim = cosine_similarity(matrix , matrix)

def get_guide(cases):
    selected_description = ', '.join(cases)
     

    user_vector = vectorizer.transform([selected_description])
    
    cosine_sim_with_selected_description = cosine_similarity(user_vector, matrix)
    

    sim_scores = list(enumerate(cosine_sim_with_selected_description[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    indices = [score[0] for score in sim_scores[:5]]
    recommendations = []
    for i in indices:
       recommendations.append({"steps" : df['description'].iloc[i] })
    
    return recommendations



@app.get("/guide/{cases}")
def recommendation_func(cases : str):
    recommendations = get_guide([cases])
    return recommendations













    