"""
examples/web_rag_assistant.py
=============================
God-Level Demonstration: Web-Search Grounded Assistant.

This script uses WebSearchRAG to answer questions about real-time events 
by scraping DuckDuckGo results and using them as context for NeuroSwift.
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
    print(f"Loading God-Level Assistant on {device} …")
    
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
            
        # Enable live web search for this query
        print(f"Searching the web for: {query} ...")
        web_hits = assistant.rag.web_query(query)
        
        if web_hits:
            print(f"Found {len(web_hits)} live results. Synthesizing answer...")
            
        result = assistant.answer(
            query,
            retrieve_k=2,
            max_new_tokens=100,
            temperature=0.3 # Use lower temperature for factuality
        )
        
        print(f"\nNeuroSwift [{result['answer_source']}]: {result['answer']}")

if __name__ == "__main__":
    main()
