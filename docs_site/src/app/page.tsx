"use client"

import { Navbar } from "@/components/navbar"
import { ArrowRight, Zap, Globe, Package, Cpu, ShieldCheck } from "lucide-react"
import Link from "next/link"
import { motion } from "framer-motion"

export default function Home() {
  return (
    <div className="flex flex-col min-h-screen">
      <Navbar />
      
      <main className="flex-grow">
        {/* Hero Section */}
        <section className="relative pt-24 pb-32 overflow-hidden">
          <div className="container mx-auto px-4 relative z-10 text-center">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
              className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full glass border border-white/20 text-sm font-medium mb-8"
            >
              <Zap className="w-4 h-4 text-primary" />
              <span>NeuroSwift 1.0.0 Global Standard Released</span>
            </motion.div>
            
            <motion.h1
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.1 }}
              className="text-6xl md:text-8xl font-black tracking-tighter mb-6 bg-clip-text text-transparent bg-gradient-to-r from-foreground via-primary to-foreground"
            >
              Intelligence at the <br /> Speed of Thought.
            </motion.h1>

            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.5, delay: 0.2 }}
              className="mb-8"
            >
              <p className="text-2xl font-serif text-primary/80 italic mb-2">"बुद्धिः क्षिप्रतरा स्वभावात्"</p>
              <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
                <span className="text-foreground font-medium">न्यूरोस्विफ्ट:</span> विचार की गति से बुद्धिमत्ता। <br />
                The world's most advanced MatMul-Free AI architecture for VIKLM Researchers.
              </p>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.3 }}
              className="flex items-center justify-center gap-4 mb-20"
            >
              <Link href="/docs" className="bg-primary hover:bg-primary/90 text-primary-foreground px-8 py-4 rounded-xl font-bold flex items-center gap-2 shadow-xl shadow-primary/20 transition-all">
                Get Started <ArrowRight className="w-4 h-4" />
              </Link>
              <Link href="https://github.com/vikukumar/neuroswift" target="_blank" className="glass hover:bg-white/10 px-8 py-4 rounded-xl font-bold transition-all">
                View on GitHub
              </Link>
            </motion.div>

            {/* Performance Benchmarks Card */}
            <motion.div
              initial={{ opacity: 0, y: 40 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.4 }}
              className="max-w-5xl mx-auto glass-card p-12 rounded-[2rem] relative group"
            >
              <div className="grid md:grid-cols-3 gap-12 text-left">
                <div>
                  <h3 className="text-4xl font-black text-primary mb-2">142.1</h3>
                  <p className="text-muted-foreground uppercase text-xs tracking-widest font-bold">Tokens/Sec (Intel i7)</p>
                  <div className="w-12 h-1 bg-primary mt-4 rounded-full" />
                </div>
                <div>
                  <h3 className="text-4xl font-black text-primary mb-2">2,450.0</h3>
                  <p className="text-muted-foreground uppercase text-xs tracking-widest font-bold">Tokens/Sec (RTX 4090)</p>
                  <div className="w-12 h-1 bg-primary mt-4 rounded-full" />
                </div>
                <div>
                  <h3 className="text-4xl font-black font-mono text-primary mb-2">5GB / 10M</h3>
                  <p className="text-muted-foreground uppercase text-xs tracking-widest font-bold">Parallel Ingestion Scaling</p>
                  <div className="w-12 h-1 bg-primary mt-4 rounded-full" />
                </div>
              </div>
            </motion.div>
          </div>
        </section>

        {/* Feature Grid */}
        <section className="container mx-auto px-4 py-32 border-t border-white/5">
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            <FeatureCard 
              icon={<Cpu className="w-6 h-6" />}
              title="MatMul-Free"
              desc="Addition-only inference bypasses floating-point bottlenecks for 10x throughput on standard hardware."
            />
            <FeatureCard 
              icon={<Globe className="w-6 h-6" />}
              title="Omni-Source"
              desc="Native parallel integration with Hugging Face, Kaggle, Local Folders, and Web RAG."
            />
            <FeatureCard 
              icon={<ShieldCheck className="w-6 h-6" />}
              title="God-Mode Stable"
              desc="Deterministic state-based reasoning with perfect grammar and logic persistence."
            />
          </div>
        </section>
      </main>

      <footer className="py-12 border-t border-white/5 bg-background">
        <div className="container mx-auto px-4 text-center">
          <p className="text-muted-foreground text-sm">
            © 2026 Vikash Kumar & VIKLM Researchers. All Rights Reserved.
          </p>
          <div className="mt-4 text-xs font-serif italic text-primary/50">
            बुद्धिः क्षिप्रतरा स्वभावात् — Intelligence is Naturally Swift
          </div>
        </div>
      </footer>
    </div>
  )
}

function FeatureCard({ icon, title, desc }: { icon: React.ReactNode, title: string, desc: string }) {
  return (
    <div className="glass hover:bg-white/5 p-8 rounded-2xl border border-white/10 transition-all duration-300 group">
      <div className="w-12 h-12 rounded-xl bg-primary/20 flex items-center justify-center text-primary mb-6 group-hover:scale-110 transition-transform">
        {icon}
      </div>
      <h3 className="text-xl font-bold mb-4">{title}</h3>
      <p className="text-muted-foreground leading-relaxed">{desc}</p>
    </div>
  )
}
