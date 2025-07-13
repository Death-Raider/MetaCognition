from datasets import load_dataset

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
