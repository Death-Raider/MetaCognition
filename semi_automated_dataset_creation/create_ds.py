from openai import OpenAI
import os
from datasets import load_dataset
import json

### Load Judge
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
# client = OpenAI(api_key="YOUR_API_KEY")

### Load Instructions
with open("semi_automated_dataset_creation/instructions.txt", "r") as f:
    instructions = f.read()

def Judge(data_entry):
    dict_data = json.dumps(data_entry)
    response = client.responses.create(
        model="gpt-4.1", #  "o4-mini"
        instructions=instructions,
        input=dict_data,
    )
    output_text = response.output_text
    output_json: dict = json.loads(output_text)
    return {
        "query": data_entry["query"],
        "output_a": data_entry["output_a"],
        "M_a": output_json.get("M_a",""),
        "T_a": output_json.get("T_a",""),
        "A_a": output_json.get("A_a",""),
        "S_a": output_json.get("S_a",""),
        "output_b": data_entry["output_b"],
        "M_b": output_json.get("M_b",""),
        "T_b": output_json.get("T_b",""),
        "A_b": output_json.get("A_b",""),
        "S_b": output_json.get("S_b",""),

        "label": data_entry["label"],
    }



### Load Dataset
dummy_data = load_dataset("nvidia/HelpSteer2", data_dir="preference")['train']
preference = dummy_data.map(
    lambda x: {
        "query": x["prompt"],
        "output_a": x["response_1"],
        "output_b": x["response_2"],
        "label": x['preference_strength'],
    },
    remove_columns=['split', 'prompt', 'response_1', 'response_2', 'preference_strength', 'preference_statement', 'preference_elaboration', 'three_most_similar_preferences', 'all_preferences_unprocessed'],
    desc="Processing preference dataset"
)


new_dataset = preference.map(Judge, desc="Applying Judge to dataset")


print(response.output_text)