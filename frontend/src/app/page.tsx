import { DEFAULT_LOCALE, strings } from '@/locales';

export default function Home() {
  const content = strings[DEFAULT_LOCALE];

  return (
    <main style={{ padding: '3rem 2rem', maxWidth: '800px', margin: '0 auto' }}>
      <header style={{ borderBottom: '1px solid #e5e7eb', paddingBottom: '1.5rem', marginBottom: '2rem' }}>
        <h1 style={{ fontSize: '1.875rem', fontWeight: 700, color: '#111827', margin: 0 }}>
          {content.title}
        </h1>
        <p style={{ color: '#4b5563', marginTop: '0.5rem', fontSize: '1rem' }}>
          {content.description}
        </p>
      </header>

      <section style={{ backgroundColor: '#f9fafb', border: '1px solid #e5e7eb', borderRadius: '8px', padding: '1.5rem' }}>
        <h2 style={{ fontSize: '1.25rem', fontWeight: 600, color: '#1f2937', marginTop: 0 }}>
          {content.milestone}
        </h2>
        <p style={{ color: '#059669', fontWeight: 500 }}>
          ● {content.status}
        </p>
        <p style={{ color: '#6b7280', fontSize: '0.875rem', lineHeight: 1.6 }}>
          Frontend skeleton initialized. Localization architecture (TR / EN / RU) configured.
          Dashboard modules will be implemented in Milestone 13 in accordance with Specification Pack v1.0.
        </p>
      </section>
    </main>
  );
}
