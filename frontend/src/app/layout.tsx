import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "SignalAgents Dashboard",
  description: "Portfolio-grade brand image dashboard running alongside Streamlit",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
