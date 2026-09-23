import ollama
import json
import re


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(question, context):

    prompt = f"""
You are an AI assistant that answers questions about a PDF document.

IMPORTANT RULES:

1. Use ONLY the information provided in the document context.
2. Do NOT use outside knowledge.
3. Do NOT invent information.
4. If the answer cannot be found in the context, say:
   "I could not find the answer in the document."
5. Give a clear and easy-to-understand answer.
6. If appropriate, use bullet points.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    try:

        response = ollama.chat(
            model="llama3.2:3b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response["message"]["content"].strip()

    except Exception as error:

        print("Failed to generate answer:", error)

        return (
            "Sorry, I could not generate an answer. "
            "Please make sure Ollama is running."
        )


# ============================================================
# GENERATE RELEVANT FOLLOW-UP QUESTIONS
# ============================================================

def generate_suggested_questions(
    question,
    answer,
    context
):

    prompt = f"""
You are helping a user explore information from a PDF document.

Generate exactly 4 useful follow-up questions.

IMPORTANT RULES:

1. Questions MUST be based ONLY on the document context.
2. Questions MUST be relevant to the current answer.
3. Questions MUST be different from the current question.
4. Questions should explore other information from the document.
5. Do NOT invent topics that are not present in the document.
6. Keep questions short and easy to understand.
7. Do NOT provide answers.
8. Return ONLY a valid JSON array.
9. The JSON array must contain exactly 4 questions.

CURRENT QUESTION:
{question}

CURRENT ANSWER:
{answer}

DOCUMENT CONTEXT:
{context}

RETURN FORMAT:

[
    "Question 1",
    "Question 2",
    "Question 3",
    "Question 4"
]
"""

    try:

        response = ollama.chat(
            model="llama3.2:3b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        content = response["message"]["content"].strip()

        print("\n==============================")
        print("Suggested Questions Response:")
        print(content)
        print("==============================\n")


        # ----------------------------------------------------
        # Remove markdown code fences
        # ----------------------------------------------------

        content = re.sub(
            r"```json\s*",
            "",
            content,
            flags=re.IGNORECASE
        )

        content = re.sub(
            r"```\s*$",
            "",
            content
        )

        content = content.strip()


        # ----------------------------------------------------
        # Try to find JSON array if Llama adds extra text
        # ----------------------------------------------------

        match = re.search(
            r"\[[\s\S]*\]",
            content
        )

        if match:

            content = match.group(0)


        # ----------------------------------------------------
        # Convert JSON string to Python list
        # ----------------------------------------------------

        questions = json.loads(content)


        # ----------------------------------------------------
        # Make sure result is a list
        # ----------------------------------------------------

        if not isinstance(questions, list):

            print(
                "Suggested questions response "
                "is not a list."
            )

            return []


        # ----------------------------------------------------
        # Keep only valid strings
        # ----------------------------------------------------

        cleaned_questions = []

        for q in questions:

            if isinstance(q, str):

                q = q.strip()

                if q:

                    cleaned_questions.append(q)


        # ----------------------------------------------------
        # Remove duplicate questions
        # ----------------------------------------------------

        unique_questions = []

        for q in cleaned_questions:

            if q not in unique_questions:

                unique_questions.append(q)


        # ----------------------------------------------------
        # Return maximum 4 questions
        # ----------------------------------------------------

        return unique_questions[:4]


    except json.JSONDecodeError as error:

        print(
            "Could not parse suggested questions JSON:"
        )

        print(error)

        print(
            "Raw response:"
        )

        print(content)

        return []


    except Exception as error:

        print(
            "Failed to generate suggested questions:"
        )

        print(error)

        return []