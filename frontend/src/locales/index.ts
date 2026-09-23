/**
 * Localization configuration placeholder for TR / EN / RU support.
 * Implemented in accordance with PROJECT_SPEC.md Section 7.
 */

export const SUPPORTED_LOCALES = ['tr', 'en', 'ru'] as const;
export type Locale = (typeof SUPPORTED_LOCALES)[number];

export const DEFAULT_LOCALE: Locale = 'tr';

export interface LocaleStrings {
  title: string;
  description: string;
  status: string;
  milestone: string;
}

export const strings: Record<Locale, LocaleStrings> = {
  tr: {
    title: 'TR–RU İkili Ticaret Tahmin & İstihbarat Platformu',
    description: 'Türkiye ve Rusya arasındaki aylık dış ticaret akımlarını analiz eden tahmin platformu.',
    status: 'Sistem Temeli Aktif',
    milestone: 'M0 — Repository & Engineering Foundation',
  },
  en: {
    title: 'TR–RU Bilateral Trade Forecasting & Intelligence Platform',
    description: 'Forecasting platform analyzing monthly bilateral trade flows between Türkiye and Russia.',
    status: 'Engineering Foundation Active',
    milestone: 'M0 — Repository & Engineering Foundation',
  },
  ru: {
    title: 'Платформа прогнозирования и анализа двусторонней торговли ТР–РФ',
    description: 'Платформа прогнозирования ежемесячных торговых потоков между Турцией и Россией.',
    status: 'Инженерный фундамент активен',
    milestone: 'M0 — Repository & Engineering Foundation',
  },
};
