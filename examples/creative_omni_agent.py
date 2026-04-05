"""
examples/creative_omni_agent.py
================================
God-Level Demonstration: Multimodal Omni Generation (V3).

This script uses the Invincible V3 Alpha (God-Mode) architecture to generate 
synchronized media (Image, Audio, Video) from a single text prompt.
"""
import torch
from neuroswift.omni import NeuroSwiftOmni
from neuroswift.tokenizer import load_tokenizer
from neuroswift.layers import auto_device
from pathlib import Path

def main():
    model_dir = Path("artifacts/neuroswift-omni")
    if not (model_dir / "model.safetensors").exists():
        print(f"Error: No Omni model found at {model_dir}. Please run 'python -m neuroswift train-omni' first.")
        return

    device = auto_device()
    print(f"Loading Invincible V3 Alpha (God-Mode) Omni Model on {device} …")
    print("Architecture: Dynamic Depth Scaling (DDS) + Multi-Head Latent Attention (MLA)")
    
    model = NeuroSwiftOmni.from_pretrained(model_dir, device=device)
    tokenizer = load_tokenizer(model_dir)
    
    # Prompt for multimodal creation
    prompt = "A high-fidelity cinematic landscape of Mars with futuristic domes."
    print(f"\n--- NeuroSwift Multimodal Creative Agent ---\nPrompt: {prompt}")
    
    prompt_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    out_dir = Path("artifacts/creative_outputs")
    
    # Unified generation flow
    for modality in ["image", "audio", "video"]:
        print(f"Generating {modality} …")
        artifact = model.generate_artifact(
            prompt_ids,
            prompt_text=prompt,
            modality=modality,
            out_dir=out_dir
        )
        print(f"  Artifact refined via Neural Pixel Refinement (NPR) -> {out_dir}/{modality}")
        
    print(f"\nGeneration complete! All creative artifacts are in: {out_dir}")

if __name__ == "__main__":
    main()
