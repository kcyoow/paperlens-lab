import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "PaperLens Lab",
  description:
    "A source-linked paper reader that turns highlighted ideas into runnable experiments.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-surface-secondary text-text-primary antialiased">
        {children}
      </body>
    </html>
  );
}
