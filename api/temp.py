from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import csv
import os
import requests
import json
from rest_framework.response import Response



#get law related dictionary from file
law_related_answers = {}
with open(os.path.join(os.path.dirname(__file__), 'law.csv'), 'r') as f:
    reader = csv.reader(f)
    try:
        law_related_answers = dict(reader)
    except ValueError as e:
        print(f"Error: {e}")
        for row in reader:
            print(row)


#read law_related_questions from csv file
law_related_questions = []
with open(os.path.join(os.path.dirname(__file__), 'law.csv'), 'r') as f:
    reader = csv.reader(f)
    law_related_questions = list(reader)
law_related_questions = [item for sublist in law_related_questions for item in sublist]


# Marking the questions as related to law
law_labels = [1] * len(law_related_questions)
     


#get non-law questions from the file
non_law_questions = []
with open(os.path.join(os.path.dirname(__file__), 'non_law_questions.csv'), 'r') as f:
    reader = csv.reader(f)
    non_law_questions = list(reader)
non_law_questions = [item for sublist in non_law_questions for item in sublist]


# Marking non-law questions
non_law_labels = [0] * len(non_law_questions)
def is_law_related(papa):
# Combining law and non-law questions and labels
    all_questions = law_related_questions + non_law_questions
    all_labels = law_labels + non_law_labels

    # Vectorizing the text data
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(all_questions)

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, all_labels, test_size=0.2, random_state=42)

    # Initializing the Logistic Regression model
    model = LogisticRegression()

    # Training the model
    model.fit(X_train, y_train)

    # User prompt input
    question = papa

    # Vectorizing the user prompt
    user_prompt_vectorized = vectorizer.transform([question])

    # Predicting the probability of being related to law
    probability = model.predict_proba(user_prompt_vectorized)[0, 1]

    # Adjusting the decision threshold to 0.5 (default)
    threshold = 0.5

    # Displaying the result
    if probability > threshold:
        return True
        
    else:
        return False

vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(law_related_questions)

def trainer(question):
    
    prompt="""
    I will give you a question, and you will show your great analytical skills
    first of all you just need to analyze the question or any other text which is given to you after the word
    PAPA, if the word or sentence is anyhow not related to law then you will say NO, and only NO, you do not
    need to provide any clarifications that why it is not a law related prompt,etc,and if it is
    related then you will simply produce YES as the answer, and only YES, you do not need to provide any clarifications,
    so your prompt is PAPA:"""

    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key=AIzaSyB_QcnBt09KT5Wj5p80Dfk9F9IOzDcd-Sg"

    headers = {
    'Content-Type': 'application/json',
}

    data = {
    "contents": [
        {
            "parts": [
                {
                    "text": prompt+question
                }
            ]
        }
    ]
}

    response = requests.post(url, headers=headers, data=json.dumps(data))
    data=response.json()
    data=data['candidates'][0]['content']['parts'][0]['text']
    
    if data=="NO":
        #append the question to non_law_questions.csv in new line
        if question not in non_law_questions:
            with open(os.path.join(os.path.dirname(__file__), 'non_law_questions.csv'), 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([question])

        return "I can't provide info regarding this topic"
    else:
        #train the model and return Thank you
        prompt="""
        Now remeber very carefully that your answer should not exceed 2 lines and please do not use bullet points or commas or any other punctuation marks, you only simply answer as if you are giving a short reply in maximum 2 lines and you will not cross this barrier at any cost, now your question is:
        """
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key=AIzaSyB_QcnBt09KT5Wj5p80Dfk9F9IOzDcd-Sg"
        headers = {
        'Content-Type': 'application/json',

        }
        data = {
    "contents": [
        {
            "parts": [
                {
                    "text": prompt+question
                }
            ]
        }
    ]
}      
        response = requests.post(url, headers=headers, data=json.dumps(data))
        data=response.json()
        
        answer=data['candidates'][0]['content']['parts'][0]['text']
        question = question.replace('\n', ' ').replace('\r', '')
        answer = answer.replace('\n', ' ').replace('\r', '')

    # Write to the CSV file
        with open(os.path.join(os.path.dirname(__file__), 'law.csv'), 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([question, f'"{answer}"'])
    #Write to the question file
        file_path = os.path.join(os.path.dirname(__file__), 'law_related_questions.csv')

# Read existing content
        existing_content = []
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            existing_content = list(reader)

# Find the last entry that ends with a comma
        last_entry_index = len(existing_content) - 1

# Ensure the list is not empty before checking indices
        if last_entry_index >= 0:
            while last_entry_index >= 0:
                if existing_content[last_entry_index] and existing_content[last_entry_index][-1].endswith(''):
                    break
                last_entry_index -= 1

    # Modify the existing content
        if last_entry_index >= 0:
            existing_content[last_entry_index].append(question)

        # Write the modified content back to the file
            with open(file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(existing_content)
        
        #was related to law
        return "Sorry, I don't have an answer, I'll learn from Gemini and will answer you next time"




def get_answer(question):
    user_vector = vectorizer.transform([question])

    # Calculate cosine similarity with each question
    similarities = cosine_similarity(user_vector, question_vectors).flatten()

    # Find the index of the most similar question
    max_similarity_index = similarities.argmax()

    # Check if similarity exceeds a threshold
    similarity_threshold = 0.5  # Adjust as needed
    if similarities[max_similarity_index] > similarity_threshold:
        matching_question = law_related_questions[max_similarity_index]
        answer = law_related_answers.get(matching_question, None)

        if answer is None:
    # Call the trainer function when key is not found
            return str(trainer(matching_question))
        else:
    # Use the answer if the key is found
            if '"' in answer:
                answer = answer.replace('"', '')
            return answer
    else:
        return "I'm  sorry, I don't understand. Please rephrase your question."

