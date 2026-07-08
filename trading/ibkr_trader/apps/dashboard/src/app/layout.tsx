import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Trading Dashboard',
  description: 'Live algorithmic trading monitor',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-[#0a0b0d] text-gray-100 font-mono antialiased">
        {children}
      </body>
    </html>
  );
}
