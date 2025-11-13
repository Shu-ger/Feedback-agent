from fastapi import FastAPI, Request
from feedback_agent_local import FeedbackAgentLocal
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
agent = FeedbackAgentLocal("feedback.csv")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:4200"] for Angular
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/ask")
async def ask(request: Request):
    try:
        data = await request.json()
        question = data.get("question")
        if not question:
            return {"answer": "לא נשלחה שאלה."}

        # Replace this with your GPT agent call
        # answer = f"תשובה לדוגמה עבור: {question}"
        answer = agent.answer_question(question)
        return {"answer": answer}

    except Exception as e:
        return {"answer": f"שגיאה בקריאת הנתונים: {str(e)}"}
# def ask_agent(question: str = Query(...)):
#     print("in")
#     answer = agent.answer_question(question)
#     return {"question": question, "answer": answer}

# להרצה:
# uvicorn api_server_local:app --reload
