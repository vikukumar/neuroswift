"""
examples/web_rag_assistant.py
=============================
God-Level Demonstration: Web-Search Grounded Assistant (V3).

This script uses Auto-Intelligence V3 to answer questions about real-time events 
by scraping DuckDuckGo results automatically.
"""
import torch
from neuroswift.omni import NeuroSwiftAssistant
from neuroswift.layers import auto_device
from pathlib import Path

def main():
    model_dir = Path("artifacts/neuroswift-tiny")
    if not (model_dir / "model.safetensors").exists():
        print(f"Error: No model found at {model_dir}. Please run 'python -m neuroswift train' first.")
        return

    device = auto_device()
    print(f"Loading Auto-Intelligence V3 Assistant on {device} …")
    
    assistant = NeuroSwiftAssistant.from_pretrained(model_dir, device=device)
    
    # Real-time intelligence interactive loop
    print("\n--- NeuroSwift Web-Grounded Assistant ---")
    print("Type 'quit' to exit. Try asking about current world events.")
    
    while True:
        try:
            query = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
            
        if query.lower() in {"quit", "exit", "bye"}:
            break
            
        # Enable live web search for this query via Auto-Intelligence
        print(f"Synthesizing answer from local RAG + Live Web Intelligence...")
            
        result = assistant.answer(
            query,
            retrieve_k=2,
            max_new_tokens=100,
            temperature=0.3, # Use lower temperature for factuality
            use_ssi=True # Enable Selective State Injection for RAG grounding
        )
        
        print(f"\nNeuroSwift [{result['answer_source']}]: {result['answer']}")

if __name__ == "__main__":
    main()
