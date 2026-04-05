import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { ThemeProvider } from "@/components/theme-provider";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "NeuroSwift 1.0.0 | VIKLM Researchers",
  description: "The definitive High-Performance AI Architecture. 10x throughput, GIL-bypass, and absolute parallel power for the world's fastest intelligence.",
  openGraph: {
    title: "NeuroSwift 1.0.0 | VIKLM Researchers",
    description: "Intelligence at the Speed of Thought.",
    images: ["/og.png"],
    url: "https://neuroswift.viklm.ai",
    siteName: "NeuroSwift",
  },
  twitter: {
    card: "summary_large_image",
    title: "NeuroSwift 1.0.0",
    description: "Extreme Speed AI Architecture.",
    images: ["/og.png"],
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={`${inter.className} bg-mesh min-h-screen bg-fixed`}>
        <ThemeProvider
          attribute="class"
          defaultTheme="system"
          enableSystem
          disableTransitionOnChange
        >
          <div className="relative flex min-h-screen flex-col">
            {children}
          </div>
        </ThemeProvider>
      </body>
    </html>
  );
}
