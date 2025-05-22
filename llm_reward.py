# Install these packages if you haven't:
# pip install transformers torch accelerate

import numpy as np
import re
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline


class llmExaminer:
    def __init__(self, model_name="google/flan-t5-large", verbose=False):
        self.verbose = verbose
        self.pipe = self.load_llm(model_name)

    def load_llm(self, model_name):
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name, device_map="auto", torch_dtype="auto")
        pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
        return pipe

    def evaluate(self, state, action, base_reward):
        prompt = self.build_prompt(state, action, base_reward)
        response = self.pipe(prompt, max_new_tokens=150, do_sample=False)[0]['generated_text']
        if self.verbose:
            print("Prompt:\n", prompt)
            print("\nLLM Response:\n", response)
        return self.parse_response(response)

    def build_prompt(self, state, action, base_reward):
        uav_a = state[0:3]
        uav_b = state[3:6]
        aoi = state[6:11]
        energy_a = state[11]
        energy_b = state[12]
        move_a = action[0:2]
        move_b = action[2:4]
        target_iotd = int(np.argmax(action[4:9]))
        aoi_target = aoi[target_iotd]

        prompt = f"""
Evaluate the following UAV mission decision with a score from 0 to 1 and a short comment.

State:
- UAV A position: {uav_a}
- UAV B position (jammer): {uav_b}
- AoI values (for 5 IoTDs): {aoi}
- Energy used by UAV A: {energy_a}
- Energy used by UAV B: {energy_b}

Action taken:
- UAV A movement: {move_a}
- UAV B movement: {move_b}
- IoTD targeted: #{target_iotd} (AoI: {aoi_target})

Base reward from environment: {base_reward}

Respond in the format:
Score: <value>
Comment: <your reasoning>
"""
        return prompt.strip()

    def parse_response(self, text):
        score_match = re.search(r"Score:\s*([0-9.]+)", text)
        comment_match = re.search(r"Comment:\s*(.+)", text, re.DOTALL)

        score = float(score_match.group(1)) if score_match else 0.0
        comment = comment_match.group(1).strip() if comment_match else "No comment found."
        return score, comment


# Example usage:
if __name__ == "__main__":
    examiner = llmExaminer(verbose=True)

    # Dummy state and action vectors
    state = [1, 1, 1, 2, 2, 2, 0.3, 0.4, 0.5, 0.2, 0.6, 10, 15]
    action = [0, 1, 1, 0, 0, 0, 1, 0, 0]  # Last 5 for IoTD one-hot target
    base_reward = 0.65

    score, comment = examiner.evaluate(state, action, base_reward)
    print("\nFinal Evaluation:")
    print(f"Score: {score}")
    print(f"Comment: {comment}")
