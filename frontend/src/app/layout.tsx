import type { Metadata } from 'next';
import './globals.css';
import { LocaleProvider } from '@/locales';

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
        <LocaleProvider>{children}</LocaleProvider>
      </body>
    </html>
  );
}
