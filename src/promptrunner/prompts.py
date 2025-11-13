NLI_BASIC_PROMPT = """
Read the following Context and Statement (introduced respectively by the [CONTEXT] and [STATEMENT] tags):
[CONTEXT] {}
[STATEMENT] {}

Choose one or more from the following:
If you feel uncertain and you feel that multiple options apply, choose them all instead, even though it might feel contradictory.
Assuming the context is true, the statement:
true: is most likely to be true
either: can be either true or false
false: is most likely false
"""

NLI_SYSTEM_PROMPT = """
Respond only with the label, in all lowercase letters. 
"""

LIVENLI_SYSTEM_PROMPT = """
You are an expert at natural language inference tasks. 
Always provide your responses as valid JSON that can be parsed by Python's json.loads() function.

Your response format should be:
{
    "classification": {
        "label": "...",
        "explanation": "..."
    },
    "highlights": {
        "context": "...",
        "statement": "..."
    }
}

For classification labels:
- If selecting multiple classes, sort them alphabetically and join with hyphens (e.g., "either-false-true")

For highlights:
- Mark highlighted segments with '<<' at the beginning and '>>' at the end
- Only highlight the most important words/phrases mentioned in explanations

CRITICAL: Respond with ONLY the raw JSON object. Do not use markdown formatting, code blocks, or any wrapper text.

Do NOT wrap your response in:
- ```json ... ```
- ``` ... ```
- Any markdown formatting
- Any explanatory text before or after the JSON
"""

LIVENLI_TASK_PROMPT = """
Read the following Context and Statement (introduced respectively by the [CONTEXT] and [STATEMENT] tags):
[CONTEXT] {}
[STATEMENT] {}

Choose one or more from the following:
If you feel uncertain and you feel that multiple options apply, choose them all instead, even though it might feel contradictory.
Assuming the context is true, the statement:
true: is most likely to be true
either: can be either true or false
false: is most likely false

Explain, in a few sentences, why you chose your answer.
If you chose more than one option, elaborate in wich circumstances each option is possible.
Explain all the options you chose.
Your explanation should include new information and refer to specific parts of the sentences. It should NOT simply repeat the sentences.
Avoid "The context and statement means the same/opposite thing". Specify which part of the context and statement means the same/opposite thing.
Avoid "Just because X doesn't mean Y". Say under what circumstances X does not mean Y, or say that X can mean Y or Z.
Avoid "The statement is ambiguous/it's not clear what it means". Elaborate what the possible meanings are and why it is ambiguous.

Highlight the words in the Context and Statement that are relevant to your explanations. 
Your explanations should refer to specific words/parts of sentences. 
Highlight those words and phrases that your explanations mentioned. 
Only highlight the words that are most important for the explanations.
"""

NEGATION_PROMPT = """
Negate the following statement. 
Ensure that it makes logical sense. Output the negated statement and nothing else. \
If not possible to negate, output "[ERROR]".:

[STATEMENT] {}
"""

CONTRADICTION_PROMPT = """
Rephrase the following statement, ensuring that it entirely contradicts the original. \
Output the rephrased statement and nothing else. \
If not possible to rephrase, output "[ERROR]".:

[STATEMENT] {}
"""
