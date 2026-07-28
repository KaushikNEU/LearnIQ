import json
from openai import AsyncOpenAI
from app.config import get_settings
from app.agents.retriever import run_retriever

settings = get_settings()

GENERATE_PROMPT = """You are an expert quiz creator for academic courses.

Generate {num_questions} quiz questions based strictly on the provided context.
Mix question types: multiple choice (MCQ) and short answer.

Return ONLY valid JSON in this exact format, no other text:
{{
  "questions": [
    {{
      "id": 1,
      "type": "mcq",
      "question": "...",
      "options": ["A. ...", "B. ...", "C. ...", "D. ..."],
      "correct_answer": "A",
      "explanation": "Brief explanation of why this is correct"
    }},
    {{
      "id": 2,
      "type": "short_answer",
      "question": "...",
      "sample_answer": "A model answer for grading reference",
      "key_points": ["point 1", "point 2"]
    }}
  ]
}}"""

EVALUATE_PROMPT = """You are a fair and constructive academic grader.

Evaluate the student's answer against the question and model answer.
Return ONLY valid JSON:
{{
  "score": "correct|partial|incorrect",
  "percentage": 0-100,
  "feedback": "Specific, constructive feedback explaining what was right and wrong",
  "correct_answer": "The full correct answer for reference"
}}"""

async def generate_quiz(subject: str, topic: str, num_questions: int = 5) -> dict:
    client = AsyncOpenAI(api_key=settings.openai_api_key)

    retrieval = await run_retriever(topic, subject, n_results=8)
    context = retrieval.get("context", "")
    sources = retrieval.get("sources", [])

    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": GENERATE_PROMPT.format(num_questions=num_questions)
            },
            {
                "role": "user",
                "content": f"Course context:\n{context}\n\nGenerate {num_questions} questions about: {topic}"
            }
        ],
        temperature=0.7,
        response_format={"type": "json_object"}
    )

    raw = response.choices[0].message.content
    questions = json.loads(raw)

    return {
        "subject": subject,
        "topic": topic,
        "questions": questions.get("questions", []),
        "sources": sources
    }

async def evaluate_answer(question: str, student_answer: str, sample_answer: str) -> dict:
    client = AsyncOpenAI(api_key=settings.openai_api_key)

    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": EVALUATE_PROMPT},
            {
                "role": "user",
                "content": f"Question: {question}\nModel answer: {sample_answer}\nStudent answer: {student_answer}"
            }
        ],
        temperature=0.1,
        response_format={"type": "json_object"}
    )

    return json.loads(response.choices[0].message.content)