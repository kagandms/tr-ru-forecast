import type { Metadata } from 'next';

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
      <body style={{ fontFamily: 'system-ui, -apple-system, sans-serif', margin: 0, padding: 0 }}>
        {children}
      </body>
    </html>
  );
}
