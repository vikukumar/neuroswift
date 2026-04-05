"use client"

import { Database, Cloud, Zap, ArrowRight, Layers } from "lucide-react"

export default function TrainingPage() {
  return (
    <div className="prose prose-invert max-w-none">
      <h1 className="text-4xl font-black tracking-tight mb-4">Training NeuralSwift 1.0.0</h1>
      <p className="text-xl text-muted-foreground mb-10">
        NeuroSwift 1.0.0 achieves world-leading throughput via **Absolute Parallelism**. Ingest 5GB of raw data in under 10 minutes.
      </p>

      <section className="mb-16">
        <h2 className="flex items-center gap-3 text-2xl font-bold">
          <Database className="w-6 h-6 text-primary" /> Local Ingestion
        </h2>
        <p>Auto-detects 20+ file formats including PDF, Log, Code, and Excel.</p>
        <pre className="bg-black/50 p-6 rounded-xl border border-white/5">
          <code>{`# Local folder auto-train
python -m neuroswift train --data-dir "./my_local_data" --epochs 10`}</code>
        </pre>
      </section>

      <section className="mb-16 border-t border-white/5 pt-16">
        <h2 className="flex items-center gap-3 text-2xl font-bold">
          <Cloud className="w-6 h-6 text-primary" /> Remote (HF & Kaggle)
        </h2>
        <p>Parallel batch fetching with robust caching for billion-token datasets.</p>
        <pre className="bg-black/50 p-6 rounded-xl border border-white/5">
          <code>{`# Ingest multiple repositories simultaneously
python -m neuroswift train \\
  --hf-dataset "user/repo1,user/repo2" \\
  --kaggle-dataset "user/dataset1"`}</code>
        </pre>
      </section>

      <section className="mb-16 border-t border-white/5 pt-16">
        <h2 className="flex items-center gap-3 text-2xl font-bold">
          <Zap className="w-6 h-6 text-primary" /> Performance Optimization
        </h2>
        <p>Leverage the GIL-Bypass Engine for 10x throughput.</p>
        <ul className="grid md:grid-cols-2 gap-4 list-none p-0">
          <li className="glass p-6 rounded-2xl border border-white/10">
            <h4 className="font-bold mb-2">Process-Level Normalize</h4>
            <p className="text-sm text-muted-foreground">Distributed Regex and SHA-256 deduplication across all CPU cores.</p>
          </li>
          <li className="glass p-6 rounded-2xl border border-white/10">
            <h4 className="font-bold mb-2">Mixed Precision (AMP)</h4>
            <p className="text-sm text-muted-foreground">Doubles GPU training speed via FP16/BF16 tensor core acceleration.</p>
          </li>
        </ul>
      </section>
      
      <div className="glass p-8 rounded-2xl border border-primary/20 bg-primary/5 text-center mt-12">
        <p className="text-lg font-bold mb-2">"न्यूरोस्विफ्ट: विचार की गति से बुद्धिमत्ता।"</p>
        <p className="text-sm text-primary/70 italic text-center">— VIKLM Researchers</p>
      </div>
    </div>
  )
}
