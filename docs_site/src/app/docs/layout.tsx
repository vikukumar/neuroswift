import { Navbar } from "@/components/navbar"
import { ScrollArea } from "@/components/ui/scroll-area"
import Link from "next/link"

export default function DocsLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <div className="flex flex-col min-h-screen">
      <Navbar />
      <div className="container mx-auto px-4 flex-grow flex gap-8 py-10">
        <aside className="hidden lg:block w-64 shrink-0">
          <div className="sticky top-24">
            <h4 className="font-bold mb-4 uppercase text-xs tracking-widest text-primary">Get Started</h4>
            <nav className="flex flex-col gap-2 text-sm">
              <Link href="/docs" className="hover:text-primary transition-colors">Architecture Overview</Link>
              <Link href="/docs/installation" className="hover:text-primary transition-colors">Installation</Link>
            </nav>

            <h4 className="font-bold mt-8 mb-4 uppercase text-xs tracking-widest text-primary">Training</h4>
            <nav className="flex flex-col gap-2 text-sm">
              <Link href="/docs/training" className="hover:text-primary transition-colors">Local Data Ingestion</Link>
              <Link href="/docs/hf-kaggle" className="hover:text-primary transition-colors">HF & Kaggle Integration</Link>
              <Link href="/docs/parallelism" className="hover:text-primary transition-colors">Extreme Parallelism</Link>
            </nav>

            <h4 className="font-bold mt-8 mb-4 uppercase text-xs tracking-widest text-primary">Generative Omni</h4>
            <nav className="flex flex-col gap-2 text-sm">
              <Link href="/docs/multimodal" className="hover:text-primary transition-colors">Generating Text</Link>
              <Link href="/docs/multimodal#image" className="hover:text-primary transition-colors">Generating Image</Link>
              <Link href="/docs/multimodal#audio" className="hover:text-primary transition-colors">Generating Audio</Link>
              <Link href="/docs/multimodal#video" className="hover:text-primary transition-colors">Generating Video</Link>
            </nav>

            <h4 className="font-bold mt-8 mb-4 uppercase text-xs tracking-widest text-primary">Brand Identity</h4>
            <nav className="flex flex-col gap-2 text-sm">
              <Link href="/docs/branding" className="hover:text-primary transition-colors">Visual Assets Gallery</Link>
              <Link href="/docs/benchmarks" className="hover:text-primary transition-colors">SOTA Comparisons</Link>
            </nav>
          </div>
        </aside>
        
        <main className="flex-grow max-w-4xl glass-card p-10 rounded-3xl overflow-hidden">
          {children}
        </main>
      </div>
    </div>
  )
}
