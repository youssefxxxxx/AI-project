# preprocess_exercises.py
import os
import re
from dotenv import load_dotenv
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import chromadb

# Load environment variables
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
client = MongoClient(MONGO_URI)
db = client["fitness-tracker"]  # * your DB name
exercises_collection = db["workout"]  #  your collection name

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
client_chroma = chromadb.PersistentClient(path=os.getenv("CHROMA_DB_PATH", "./chroma_db_exercises"))
collection = client_chroma.get_or_create_collection(name="exercises_embeddings")

def normalize_text(text):
    """
    Normalize text by removing extra spaces and non-alphanumeric chars.
    """
    text = text.strip().lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s,]', '', text)
    return text

def vectorize_text(name, preparation, execution, target_muscles, main_muscle, difficulty):
    # Create a single text string from all fields
    text = (
        f"{name}. {preparation}. {execution}. "
        f"Target muscles: {target_muscles}. Main muscle: {main_muscle}. Difficulty: {difficulty}"
    )
    return model.encode([text])[0]#for embeding (encode)

def process_batch(batch):
    ids = []
    embeddings = []
    metadatas = []

    for exercise in batch:
        try:
            name = exercise.get("Exercise Name", "Unnamed Exercise").strip()
            preparation = exercise.get("Preparation", "")
            execution = exercise.get("Execution", "")
            target_muscles = exercise.get("Target_Muscles", "")
            main_muscle = exercise.get("Main_muscle", "")
            difficulty = exercise.get("Difficulty (1-5)", "")

            # Normalize fields
            name = normalize_text(name)
            preparation = normalize_text(preparation)
            execution = normalize_text(execution)
            target_muscles = normalize_text(target_muscles)
            main_muscle = normalize_text(main_muscle)
            difficulty = str(difficulty)

            if not name or not execution:
                continue

            embedding = vectorize_text(name, preparation, execution, target_muscles, main_muscle, difficulty)

            ids.append(str(exercise["_id"]))
            metadatas.append({
                "name": name,
                "preparation": preparation,
                "execution": execution,
                "target_muscles": target_muscles,
                "main_muscle": main_muscle,
                "difficulty": difficulty
            })
            embeddings.append(embedding)
        except Exception as e:
            print(f"Error processing exercise {exercise.get('_id')}: {e}")

    if ids:
        collection.add(
            documents=["" for _ in ids],  # We don't need the original text as doc
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas
        )

def main():
    LIMIT = int(os.getenv("EXERCISE_LIMIT", 5000))
    exercises = list(exercises_collection.find({}, {
        "_id": 1, "Exercise Name": 1, "Preparation": 1, "Execution": 1,
        "Target_Muscles": 1, "Main_muscle": 1, "Difficulty (1-5)": 1
    }).limit(LIMIT))

    batch_size = int(os.getenv("BATCH_SIZE", 100))
    total_batches = (len(exercises) + batch_size - 1) // batch_size

    for i in range(0, len(exercises), batch_size):
        batch = exercises[i:i+batch_size]
        print(f"Processing batch {i // batch_size + 1}/{total_batches}...")
        try:
            process_batch(batch)
            print(f"Finished batch {i // batch_size + 1}/{total_batches}")
        except Exception as e:
            print(f"Error processing batch {i // batch_size + 1}: {e}")

    print("All batches processed successfully!")

if __name__ == "__main__":
    main()
