# generate_response.py
import os
import logging
import google.generativeai as genai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

genai.configure(api_key=os.getenv("PALM_API_KEY"))

def generate_response(query, exercises, exercise_embeddings, query_embedding):
    if not exercises or not exercise_embeddings:
        return "No relevant exercises found."

    similarity_scores = cosine_similarity([query_embedding], exercise_embeddings)[0]
    adjusted_exercises = sorted(zip(exercises, similarity_scores), key=lambda x: x[1], reverse=True)

    input_text = f"User query: {query}\n\nHere are some recommended exercises:\n"
    for exercise, _ in adjusted_exercises[:3]:
        name = exercise.get("name", "Unnamed Exercise").strip()
        prep = exercise.get("preparation", "")
        exec_ = exercise.get("execution", "")
        target = exercise.get("target_muscles", "")
        main = exercise.get("main_muscle", "")
        diff = exercise.get("difficulty", "")

        input_text += f"- {name.title()}\n"
        input_text += f"  Preparation: {prep}\n"
        input_text += f"  Execution: {exec_}\n"
        input_text += f"  Target Muscles: {target}\n"
        input_text += f"  Main Muscle: {main}\n"
        input_text += f"  Difficulty: {diff}\n\n"

    input_text += (
        "Please provide a helpful response considering the following user preferences:\n"
        "1. User wants exercises suited for their training goal.\n"
        "2. Consider exercise difficulty, muscle groups, and any specific requests.\n"
        "3. Provide rationale for the chosen exercises and explain why others might be less suitable.\n"
        "4. Suggest variations or complementary exercises if applicable.\n"
        "Please limit your response to approximately 200 words."
    )

    logger.info("Input text sent to LLM:")
    logger.info(input_text)

    try:
        model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")
        response = model.generate_content(
            contents=[input_text],
            generation_config={"temperature": 0.7, "max_output_tokens": 300}
        )

        if response and response.candidates:
            logger.info("Raw response from model:")
            logger.info(response)
            candidate = response.candidates[0]
            generated_text = ''.join(part.text for part in candidate.content.parts)
            return generated_text.strip()
        else:
            logger.warning("No response received from model.")
            return "No response received from the model."
    except Exception as e:
        logger.error(f"Error calling Gemini LLM: {e}")
        return "Error generating response."
