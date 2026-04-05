"use client"

import Link from "next/link"
import Image from "next/image"
import { useTheme } from "next-themes"
import { Sun, Moon, Github, Search, Menu } from "lucide-react"

export function Navbar() {
  const { setTheme, theme } = useTheme()

  return (
    <header className="sticky top-0 z-50 w-full glass dark:glass-dark border-b transition-all">
      <div className="container mx-auto px-4 flex h-16 items-center justify-between">
        <div className="flex items-center gap-6">
          <Link href="/" className="flex items-center gap-2">
            <Image src="/logo.png" alt="NeuroSwift" width={32} height={32} className="rounded-sm" />
            <span className="font-bold text-xl tracking-tighter text-primary">NeuroSwift <span className="text-foreground/60 text-sm">1.0.0</span></span>
          </Link>
          <nav className="hidden md:flex items-center gap-6 text-sm font-medium">
            <Link href="/docs" className="hover:text-primary transition-colors">Documentation</Link>
            <Link href="/docs/training" className="hover:text-primary transition-colors">Training</Link>
            <Link href="/docs/multimodal" className="hover:text-primary transition-colors">Omni-Ingestion</Link>
            <Link href="/docs/benchmarks" className="hover:text-primary transition-colors">Benchmarks</Link>
          </nav>
        </div>

        <div className="flex items-center gap-4">
          <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-full bg-foreground/5 dark:bg-white/5 border border-white/10 text-xs text-muted-foreground cursor-pointer hover:bg-foreground/10 transition-all">
            <Search className="w-3 h-3" />
            <span>Search docs...</span>
            <kbd className="ml-2 font-sans opacity-50">⌘K</kbd>
          </div>
          
          <button
            onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
            className="p-2 rounded-full hover:bg-foreground/5 transition-colors"
          >
            {theme === "dark" ? <Sun className="w-5 h-5 text-yellow-400" /> : <Moon className="w-5 h-5" />}
          </button>
          
          <Link href="https://github.com/vikukumar/neuroswift" target="_blank" className="p-2 rounded-full hover:bg-foreground/5 transition-colors">
            <Github className="w-5 h-5" />
          </Link>

          <button className="md:hidden p-2">
            <Menu className="w-5 h-5" />
          </button>
        </div>
      </div>
    </header>
  )
}
