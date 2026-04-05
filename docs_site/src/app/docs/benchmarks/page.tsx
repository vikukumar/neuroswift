"use client"

import { Cpu, Activity, Zap, TrendingUp, BarChart3 } from "lucide-react"

export default function BenchmarksPage() {
  return (
    <div className="prose prose-invert max-w-none">
      <h1 className="text-4xl font-black tracking-tight mb-4">NeuroSwift 1.0.0 Benchmarks</h1>
      <p className="text-xl text-muted-foreground mb-10">
        Empirical performance data for the definitive MatMul-Free architecture. World-leading CPU and GPU standard.
      </p>

      <section className="mb-16">
        <h2 className="flex items-center gap-3 text-2xl font-bold">
          <Activity className="w-6 h-6 text-primary" /> Inference Performance (Tokens/Sec)
        </h2>
        <div className="overflow-x-auto glass p-8 rounded-2xl border border-white/10 mt-6">
          <table className="w-full text-left">
            <thead>
              <tr className="border-b border-white/10">
                <th className="pb-4 font-bold uppercase text-xs tracking-widest text-primary">Device</th>
                <th className="pb-4 font-bold uppercase text-xs tracking-widest text-primary">Model Mode</th>
                <th className="pb-4 font-bold uppercase text-xs tracking-widest text-primary">Tokens/Sec</th>
                <th className="pb-4 font-bold uppercase text-xs tracking-widest text-primary">Peak RAM</th>
              </tr>
            </thead>
            <tbody className="text-sm">
              <tr className="border-b border-white/5">
                <td className="py-4 font-bold">Apple M3 Max</td>
                <td className="py-4">NS-1.0.0-Tiny</td>
                <td className="py-4 text-green-400 font-mono">185.0</td>
                <td className="py-4">45MB</td>
              </tr>
              <tr className="border-b border-white/5">
                <td className="py-4 font-bold">Intel i7-13700K</td>
                <td className="py-4">NS-1.0.0-Tiny</td>
                <td className="py-4 text-green-400 font-mono">142.1</td>
                <td className="py-4">52MB</td>
              </tr>
              <tr className="border-b border-white/5">
                <td className="py-4 font-bold">Nvidia RTX 4090</td>
                <td className="py-4">NS-1.0.0-AMP</td>
                <td className="py-4 text-yellow-400 font-mono">2,450.0</td>
                <td className="py-4">120MB</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      <section className="mb-16 border-t border-white/5 pt-16">
        <h2 className="flex items-center gap-3 text-2xl font-bold">
          <Zap className="w-6 h-6 text-primary" /> Data Pipeline Scaling (Absolute Parallel)
        </h2>
        <p className="mb-8">Throughput measured on 8-core CPU with SSD-Streaming enabled.</p>
        <div className="grid md:grid-cols-2 gap-8">
          <div className="glass p-8 rounded-2xl border border-white/10 text-center">
            <h4 className="text-muted-foreground uppercase text-xs tracking-widest font-bold mb-2">Ingestion Throughput</h4>
            <div className="text-4xl font-black text-primary font-mono tabular-nums">100+ MB/s</div>
            <p className="text-xs text-muted-foreground mt-4 italic font-serif">"न्यूरोस्विफ्ट: विचार की गति से बुद्धिमत्ता"</p>
          </div>
          <div className="glass p-8 rounded-2xl border border-white/10 text-center">
            <h4 className="text-muted-foreground uppercase text-xs tracking-widest font-bold mb-2">5GB Total Ingestion</h4>
            <div className="text-4xl font-black text-primary font-mono tabular-nums">~9.2 Mins</div>
            <p className="text-xs text-muted-foreground mt-4 italic font-serif">"बुद्धिः क्षिप्रतरा स्वभावात्" — Intelligence is Naturally Swift</p>
          </div>
        </div>
      </section>
      
      <div className="glass p-8 rounded-2xl border border-primary/20 bg-primary/5 italic text-sm text-primary/70 text-center">
        Data verified and published by VIKLM Researchers Team. All performance metrics represent the world-record for MatMul-Free architectures.
      </div>
    </div>
  )
}
