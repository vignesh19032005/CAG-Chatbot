import requests
from django.shortcuts import render
from django.http import JsonResponse
import redis
import json
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from django.views.decorators.csrf import ensure_csrf_cookie


API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
HEADERS = {
    "Authorization": f"Bearer USE_YOUR_HUGGINGFACE_API_KEY",  # Replace with your API Key
    "Content-Type": "application/json"
}

r = redis.Redis(host='localhost', port=6379, db=0)

embedder = SentenceTransformer('all-MiniLM-L6-v2')

def ask_mistral(question):
    payload = {"inputs": question}
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    return response.json()[0]['generated_text']

def check_cache(question):
    question_embedding = embedder.encode([question])[0]
    cached_answer = r.get(str(question_embedding))  
    if cached_answer:
        return cached_answer.decode("utf-8")  
    else:
        return None

def save_to_cache(question, answer):
    question_embedding = embedder.encode([question])[0]
    r.set(str(question_embedding), answer)

def ask_question(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        user_question = data.get('question')

        cached_answer = check_cache(user_question)

        if cached_answer:
            return JsonResponse({"answer": cached_answer})
        else:
            model_answer = ask_mistral(user_question)
            save_to_cache(user_question, model_answer)
            return JsonResponse({"answer": model_answer})

@ensure_csrf_cookie
def index(request):
    return render(request, 'chatbot/index.html')

def chatbot_ask(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    
    try:
        data = json.loads(request.body)
        message = data.get('message')
        if not message:
            return JsonResponse({'error': 'Message is required'}, status=400)
        
        response = ask_mistral(message)
        return JsonResponse({'message': response})
        
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
