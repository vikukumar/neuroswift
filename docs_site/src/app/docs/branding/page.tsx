"use client"

import Image from "next/image"

export default function BrandingPage() {
  const assets = [
    { name: "GitHub Banner", path: "/branding/github_banner.png", ratio: "2:1" },
    { name: "4K Desktop Wallpaper", path: "/branding/desktop_4k.png", ratio: "16:9" },
    { name: "MacBook Mockup", path: "/branding/macbook_mockup.png", ratio: "16:10" },
    { name: "LinkedIn Header", path: "/branding/linkedin_header.png", ratio: "4:1" },
    { name: "Twitter/X Header", path: "/branding/twitter_header.png", ratio: "3:1" },
    { name: "Ternary Logic Abstract", path: "/branding/ternary_abstract.png", ratio: "1:1" },
    { name: "DDS Thinking Visual", path: "/branding/dds_visual.png", ratio: "16:9" },
    { name: "iPhone 16 Pro Mockup", path: "/branding/iphone_mockup.png", ratio: "9:16" },
    { name: "Mobile Neural Mesh", path: "/branding/mobile_wallpaper.png", ratio: "9:16" },
    { name: "App Store Icon", path: "/branding/app_icon.png", ratio: "1:1" },
    { name: "Insta Feature Post", path: "/branding/insta_post.png", ratio: "1:1" },
    { name: "YouTube Channel Art", path: "/branding/youtube_banner.png", ratio: "16:9" },
    { name: "Ultra-Wide Showcase", path: "/branding/ultrawide_banner.png", ratio: "21:9" },
    { name: "Tablet UI Mockup", path: "/branding/tablet_mockup.png", ratio: "4:3" },
    { name: "Vertical Insta Story", path: "/branding/insta_story.png", ratio: "9:16" },
  ]

  return (
    <div className="prose prose-invert max-w-none">
      <h1 className="text-4xl font-black tracking-tight mb-4 text-primary">Visual Identity Gallery</h1>
      <p className="text-xl text-muted-foreground mb-10">
        The definitive world-standard branding ecosystem for **NeuroSwift 1.0.0** and **VIKLM Researchers**. Download these high-fidelity assets for consistent brand dominance.
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        {assets.map((asset, i) => (
          <div key={i} className="glass p-4 rounded-2xl border border-white/10 group hover:bg-white/5 transition-all">
            <div className="relative aspect-video rounded-xl overflow-hidden mb-4 border border-white/5">
              <img 
                src={asset.path} 
                alt={asset.name} 
                className="object-cover w-full h-full group-hover:scale-105 transition-transform duration-500"
              />
            </div>
            <div className="flex justify-between items-center">
              <div>
                <h4 className="font-bold text-sm m-0">{asset.name}</h4>
                <p className="text-[10px] text-muted-foreground uppercase tracking-widest m-0">{asset.ratio} Standard</p>
              </div>
              <a 
                href={asset.path} 
                download 
                className="bg-primary/20 hover:bg-primary/30 text-primary px-3 py-1 rounded-full text-[10px] font-bold transition-all"
              >
                DOWNLOAD
              </a>
            </div>
          </div>
        ))}
      </div>

      <div className="mt-20 glass p-8 rounded-2xl border border-primary/20 bg-primary/5 text-center">
        <p className="text-sm italic text-primary/70 mb-4">"बुद्धिः क्षिप्रतरा स्वभावात्" — Intelligence is Naturally Swift</p>
        <p className="text-xs text-muted-foreground">© 2026 Vikash Kumar & VIKLM Researchers</p>
      </div>
    </div>
  )
}
