from torch.utils.data import Dataset
class PreferenceDataLoader(Dataset):
    """
    A class to load and preprocess preference learning data.
    """
    def __init__(self, data, strat):
        self.data = data
        self.samples = [self.preprocess(d, strat) for d in data]

    def preprocess(self, example, strat):
        prompt = self.build_prompt(example['query'], strat)
        chosen = example['output_a'] if example['label'] == 0 else example['output_b']
        rejected = example['output_b'] if example['label'] == 0 else example['output_a']
        return {
            'prompt': prompt,
            'chosen': chosen,
            'rejected': rejected
        }
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
    
    def build_prompt(self, query: str, strat: str) -> str:
        """
        return meta prompt from given stratergy
        """
        if strat == "Chain of Thought":
            FIXED_META_PROMPT = "Let's solve this step-by-step:"
            return f"{FIXED_META_PROMPT}\n\nQuestion: {query}\nAnswer:"
        
        elif strat == "Self Verify":
            FIXED_META_PROMPT = "Let's verify each step:"
            return f"{FIXED_META_PROMPT}\n\nQuestion: {query}\nAnswer:"
        
        else:
            return f"Question: {query}\nAnswer:"