"use client"

import { Zap, Image as ImageIcon, Music, Film, Terminal } from "lucide-react"

export default function MultimodalPage() {
  return (
    <div className="prose prose-invert max-w-none">
      <h1 className="text-4xl font-black tracking-tight mb-4">Multimodal Synthesis</h1>
      <p className="text-xl text-muted-foreground mb-10">
        NeuroSwift 1.0.0 is native-multimodal. It doesn't just use "heads" — it projects all data into a shared **Omni-Latent Space** for seamless cross-modality generation.
      </p>

      <section className="mb-16">
        <h2 className="flex items-center gap-3 text-2xl font-bold">
          <Terminal className="w-6 h-6 text-primary" /> Text Generation
        </h2>
        <p>Ultra-efficient text synthesis using SSD-State persistence.</p>
        <pre className="bg-black/50 p-6 rounded-xl border border-white/5">
          <code>{`# Generate text via Python API
from neuroswift import NeuroSwiftLM

model = NeuroSwiftLM.from_pretrained("./artifacts/neuroswift-1.0.0")
output = model.generate("Describe the future of AI", max_new_tokens=100)
print(output)`}</code>
        </pre>
      </section>

      <section id="image" className="mb-16 border-t border-white/5 pt-16">
        <h2 className="flex items-center gap-3 text-2xl font-bold">
          <ImageIcon className="w-6 h-6 text-primary" /> Image Synthesis (NPR)
        </h2>
        <p>
          Leveraging **Neural Pixel Refinement (NPR)**, NeuroSwift 1.0.0 generates high-fidelity images in a single auto-regressive pass.
        </p>
        <pre className="bg-black/50 p-6 rounded-xl border border-white/5">
          <code>{`# Generate high-res image
python -m neuroswift generate-omni --prompt "A futuristic silicon-blue city" --modality image`}</code>
        </pre>
      </section>

      <section id="audio" className="mb-16 border-t border-white/5 pt-16">
        <h2 className="flex items-center gap-3 text-2xl font-bold">
          <Music className="w-6 h-6 text-primary" /> Audio Synthesis (Waveform)
        </h2>
        <p>Direct waveform generation without the need for spectrogram conversion.</p>
        <pre className="bg-black/50 p-6 rounded-xl border border-white/5">
          <code>{`# Generate speech/audio
python -m neuroswift generate-omni --prompt "The voice of the machine" --modality audio`}</code>
        </pre>
      </section>

      <section id="video" className="mb-16 border-t border-white/5 pt-16">
        <h2 className="flex items-center gap-3 text-2xl font-bold">
          <Film className="w-6 h-6 text-primary" /> Video Synthesis (Temporal-SSD)
        </h2>
        <p>Temporal state persistence allows for flicker-free video generation with perfect logic continuity.</p>
        <pre className="bg-black/50 p-6 rounded-xl border border-white/5">
          <code>{`# Generate video artifact
python -m neuroswift generate-omni --prompt "Data flowing like a river" --modality video`}</code>
        </pre>
      </section>

      <div className="glass p-8 rounded-2xl border border-primary/20 bg-primary/5 italic text-sm text-primary/80">
        "बुद्धिः क्षिप्रतरा स्वभावात्" – Intelligence is Naturally Swift. <br />
        Built for the world stage by VIKLM Researchers.
      </div>
    </div>
  )
}
