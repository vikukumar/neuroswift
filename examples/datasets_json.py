import json
import random
import os
import string

def generate_random_word(length):
    return ''.join(random.choices(string.ascii_lowercase, k=length))

def generate_gibberish_sentence(word_count):
    return ' '.join(generate_random_word(random.randint(3, 8)) for _ in range(word_count)) + "."

def generate_random_dataset(filename="examples/data/neuroswift_qa.jsonl", target_lines=50000):
    print(f"Generating {target_lines} lines of random JSONL data...")
    
    # Banks of realistic-looking educational triggers to mix with random data
    subjects = ["React", "Python backend", "MLOps", "Generative AI", "Next.js 16.1.1", "authentication protocols"]
    actions = ["explain", "debug", "write a script for", "simplify", "analyze"]
    
    lines_written = 0
    
    with open(filename, "w", encoding="utf-8") as f:
        for _ in range(target_lines):
            # Randomly pick a subject and action to form a semi-structured question
            subject = random.choice(subjects)
            action = random.choice(actions)
            
            # Combine structured triggers with completely random gibberish to test token variance
            user_question = f"Can you {action} {subject}? {generate_gibberish_sentence(random.randint(5, 15))}"
            
            # Generate a random length response
            assistant_response = generate_gibberish_sentence(random.randint(20, 50))
            
            # Format as standard ChatML JSON
            qa_pair = {
                "messages": [
                    {"role": "system", "content": "You are Jasika, an AI tutor. Answer the student's question accurately."},
                    {"role": "user", "content": user_question},
                    {"role": "assistant", "content": assistant_response}
                ]
            }
            
            f.write(json.dumps(qa_pair) + "\n")
            lines_written += 1
            
            if lines_written % 10000 == 0:
                print(f"Progress: {lines_written} / {target_lines} JSON lines created...")

    print(f"\nSuccess! '{filename}' generated.")
    print(f"File Size: {os.path.getsize(filename) / (1024 * 1024):.2f} MB")

if __name__ == "__main__":
    # You can change this number to generate 100,000+ lines instantly
    generate_random_dataset(target_lines=500000)