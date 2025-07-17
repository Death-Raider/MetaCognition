from LLMJudge import Judge

judge = Judge()
judge.load_model("prometheus-eval/prometheus-7b-v2.0","meta-llama/Llama-2-7b-chat-hf")  # Example model, replace with actual model name

instruction = "Solve: What is 15 * 3 + 6?"
response = "First, multiply 15 by 3 to get 45. Then add 6 to get 51. So the answer is 51."
reference_answer = "First compute 15 * 3 = 45. Then 45 + 6 = 51. Final answer: 51."
criteria_description = "Evaluates quality of multi-step reasoning: (1) Meta-Cognitive Planning – does it outline how it will approach the problem?, (2) Strategic Reasoning – are the chosen steps correct and in good order?, (3) Tactical Execution – are calculations and operations accurate?"

score1_description = "Completely incorrect reasoning or operations; no clear structure or math steps."
score2_description = "Vague or confusing plan; has step errors or flawed calculations."
score3_description = "Reasoning follows some structure, but has one significant flaw or missing logic."
score4_description = "Mostly correct, clear plan and steps, but one minor error in either reasoning or execution."
score5_description = "Completely correct and clear step-by-step reasoning with accurate computation."

input_text = f"""
### Task Description:
You are given an instruction (question), a response to evaluate, a reference answer (that earns a score of 5), and a score rubric focused on evaluating hierarchical reasoning processes.
Your job is to:
1. Write detailed feedback that assesses the response based on the three levels of reasoning:
   - Meta-Cognitive Planning (how well the model outlines its plan of attack)
   - Strategic Reasoning (are the chosen steps valid and well ordered?)
   - Tactical Execution (are calculations or specific steps accurate?)
2. Then, assign a final score (1 to 5) that reflects the overall reasoning quality as per the rubric.
3. Output format: "Feedback: (your detailed analysis here) [RESULT] (an integer between 1 and 5)"
4. Do not write any additional explanation.

### Instruction to Evaluate:
{instruction}

### Response to Evaluate:
{response}

### Reference Answer (Score 5):
{reference_answer}

### Score Rubric:
[{criteria_description}]
Score 1: {score1_description}
Score 2: {score2_description}
Score 3: {score3_description}
Score 4: {score4_description}
Score 5: {score5_description}

### Feedback:
"""

output_text = judge.judge(input_text)
print(f"Input: {input_text}\nOutput: {output_text}")