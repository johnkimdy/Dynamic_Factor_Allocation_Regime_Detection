import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Helix Factor Strategy | Active Rebalancing Dashboard",
  description: "Visualize dynamic factor allocation and rebalancing across periods",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
