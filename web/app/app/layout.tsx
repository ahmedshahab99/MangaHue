import type { Metadata } from "next";
import { IBM_Plex_Mono, Manrope } from "next/font/google";
import "./globals.css";

const sans = Manrope({
  subsets: ["latin"],
  variable: "--font-sans",
  display: "swap",
});

const mono = IBM_Plex_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
  weight: ["400", "500", "600"],
  display: "swap",
});

export const metadata: Metadata = {
  title: {
    default: "MangaHue — AI manga colorization with consistent characters",
    template: "%s · MangaHue",
  },
  description:
    "Colorize manga pages and whole chapters in seconds: ink, screentones and speech bubbles preserved, character colors kept consistent, exports to PNG and CBZ.",
  icons: {
    icon: "/icon.svg",
  },
  openGraph: {
    type: "website",
    title: "MangaHue — AI manga colorization",
    description:
      "Upload a page or paste a chapter link. Studio-grade color, consistent characters, PNG and CBZ export.",
  },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className="dark">
      <body className={`${sans.variable} ${mono.variable} antialiased`}>{children}</body>
    </html>
  );
}
