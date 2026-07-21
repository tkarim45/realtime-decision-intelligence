import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const sans = Geist({ variable: "--font-sans", subsets: ["latin"] });
const mono = Geist_Mono({ variable: "--font-mono", subsets: ["latin"] });

export const metadata: Metadata = {
  title: "realtime-decision-intelligence",
  description:
    "Detects incidents in service telemetry, works out which ones are worth acting on, and picks the remediation.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className={`${sans.variable} ${mono.variable} bg-slate-950 text-slate-200 antialiased`}>
        {children}
      </body>
    </html>
  );
}
