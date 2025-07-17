from datasets import load_dataset
from logger import logger

def test_dataset_download():
    logger.info("Loading datasets...")
    preference = load_dataset("nvidia/HelpSteer2", data_dir="preference")['train']
    # disagreements = load_dataset("nvidia/HelpSteer2", data_dir="disagreements")['train']

    preference = preference.map(
        lambda x: {
            "query": x["prompt"],
            "output_a": x["response_1"],
            "output_b": x["response_2"],
            "label": x['preference_strength'],
        },
        remove_columns=['split', 'prompt', 'response_1', 'response_2', 'preference_strength', 'preference_statement', 'preference_elaboration', 'three_most_similar_preferences', 'all_preferences_unprocessed'],
        desc="Processing preference dataset"
    )

def test_llm_judge():
    from semi_automated_dataset_creation.LLMJudge import Judge

    judge = Judge()
    judge.load_model("prometheus-eval/prometheus-7b-v2.0","meta-llama/Llama-2-7b-chat-hf")  # Example model, replace with actual model name

    instruction = "What is the capital of France?"
    response = "The capital of France is Marseille."
    reference_answer = "Paris."
    criteria_description = "Single-fact factual Q&A accuracy."
    score1_description = "Completely incorrect or nonsensical answer."
    score2_description = "Mostly incorrect; mentions the correct country but wrong capital."
    score3_description = "Partially correct; vague or hedged (e.g., lists multiple cities, one of which is correct)."
    score4_description = "Almost correct; minor inaccuracy or formatting issue."
    score5_description = "Fully correct and precise; exactly matches reference answer."

    input_text = f"""
###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: \"Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)\"
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Reference Answer (Score 5):
{reference_answer}

###Score Rubrics:
[{criteria_description}]
Score 1: {score1_description}
Score 2: {score2_description}
Score 3: {score3_description}
Score 4: {score4_description}
Score 5: {score5_description}

###Feedback: 
"""
    output_text = judge.judge(input_text)
    print(f"Input: {input_text}\nOutput: {output_text}")

test_llm_judge()