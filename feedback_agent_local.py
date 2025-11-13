import pandas as pd
from sentence_transformers import SentenceTransformer, util
from gpt4all import GPT4All
import os

class FeedbackAgentLocal:
    def __init__(self, feedback_csv, model_filename="DeepSeek-R1-Distill-Qwen-7B-Q3_K_M.gguf"):
        print("Init!!")

        project_root = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(project_root, model_filename)
        print(model_path, project_root)

        # Check if model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        # Load feedback CSV
        self.df = pd.read_csv(feedback_csv)

        # Load sentence embedding model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.df['embedding'] = self.df['Text'].apply(
            lambda x: self.model.encode(x, convert_to_tensor=True)
        )

        # Load GPT model locally
        print(f"Loading local model from {model_path} ...")
        self.gpt_model = GPT4All(model_path, model_type="gguf", allow_download=False)
        print("Model loaded successfully ✅")

    # Find most relevant feedbacks
    def get_relevant_feedbacks(self, question, top_k=5):
        q_emb = self.model.encode(question, convert_to_tensor=True)
        scores = [float(util.cos_sim(q_emb, e)) for e in self.df['embedding']]
        self.df['score'] = scores
        top_feedbacks = self.df.sort_values(by='score', ascending=False).head(top_k)
        return top_feedbacks['Text'].tolist()

    # Generate an answer using the local DeepSeek model
    def answer_question(self, question, top_k=5):
        relevant_feedbacks = self.get_relevant_feedbacks(question, top_k)
        context_text = "\n".join(relevant_feedbacks)
        prompt = f"""
You are a feedback analysis assistant.
Below are user feedback messages:

{context_text}

Question: {question}

Provide a clear, concise answer based on the feedbacks.
"""
        answer = self.gpt_model.generate(prompt, max_tokens=300)
        return answer


# Example run
if __name__ == "__main__":
    agent = FeedbackAgentLocal("feedback.csv", "./DeepSeek-R1-Distill-Qwen-7B-Q3_K_M.gguf")
    q = "מה הנושאים המרכזיים במשובים בעלי ציון נמוך מ-3?"
    print(agent.answer_question(q))