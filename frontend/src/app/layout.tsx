import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'TR–RU Bilateral Trade Forecasting & Intelligence Platform',
  description: 'Research platform forecasting monthly bilateral trade between Türkiye and Russia.',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="tr">
      <body>
        {children}
      </body>
    </html>
  );
}
