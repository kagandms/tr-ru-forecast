import { mkdir, writeFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import sharp from 'sharp';

const WIDTH = 1080;
const HEIGHT = 1350;
const OUTPUT_DIRECTORY = resolve(process.cwd(), '../docs/research/linkedin-carousel');
const COLORS = {
  paper: '#f4f2eb',
  panel: '#fbfaf6',
  ink: '#1f292c',
  muted: '#626d70',
  line: '#cbd0cb',
  green: '#245b52',
  orange: '#ad5c3c',
  paleGreen: '#e5ebe3',
  paleOrange: '#efe5dc',
};

function escapeXml(value) {
  return value.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
}

function text(x, y, value, options = {}) {
  const {
    size = 28,
    color = COLORS.ink,
    family = 'Arial, Helvetica, sans-serif',
    weight = 400,
    spacing = 0,
    anchor = 'start',
  } = options;
  return `<text x="${x}" y="${y}" fill="${color}" font-family="${family}" font-size="${size}" font-weight="${weight}" letter-spacing="${spacing}" text-anchor="${anchor}">${escapeXml(value)}</text>`;
}

function rule(x1, y1, x2, y2, color = COLORS.line, width = 1) {
  return `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${width}" />`;
}

function rect(x, y, width, height, fill, stroke = 'none') {
  return `<rect x="${x}" y="${y}" width="${width}" height="${height}" fill="${fill}" stroke="${stroke}" />`;
}

function multiText(x, y, lines, options = {}) {
  const lineHeight = options.lineHeight ?? (options.size ?? 28) * 1.35;
  return lines.map((line, index) => text(x, y + index * lineHeight, line, options)).join('');
}

function frame(slideNumber, content) {
  const header = [
    rect(60, 48, 48, 36, COLORS.ink),
    text(84, 72, 'TR/RU', { size: 12, color: COLORS.paper, family: 'monospace', weight: 700, anchor: 'middle' }),
    text(126, 72, 'BILATERAL TRADE RESEARCH', { size: 14, color: COLORS.muted, family: 'monospace', weight: 600, spacing: 1 }),
    rule(60, 105, 1020, 105, COLORS.ink, 2),
  ].join('');
  const footer = [
    rule(60, 1270, 1020, 1270, COLORS.ink, 2),
    text(60, 1310, 'TR–RU FORECAST · RESEARCH PROJECT', { size: 12, color: COLORS.muted, family: 'monospace', weight: 600, spacing: 0.6 }),
    text(1020, 1310, `${String(slideNumber).padStart(2, '0')} / 07`, { size: 12, color: COLORS.muted, family: 'monospace', weight: 600, anchor: 'end' }),
  ].join('');
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${WIDTH}" height="${HEIGHT}" viewBox="0 0 ${WIDTH} ${HEIGHT}">${rect(0, 0, WIDTH, HEIGHT, COLORS.paper)}${header}${content}${footer}</svg>`;
}

const slides = [
  [
    '01-cover',
    [
      text(60, 190, 'A MONTHLY VIEW OF BILATERAL TRADE', { size: 15, color: COLORS.green, family: 'monospace', weight: 700, spacing: 1.2 }),
      multiText(60, 330, ['Türkiye–Russia', 'Trade Forecasting'], { size: 72, family: 'Georgia, Times New Roman, serif', lineHeight: 86 }),
      text(60, 525, 'Forecasts, model comparison and mirror statistics', { size: 26, color: COLORS.muted }),
      rule(60, 630, 1020, 630, COLORS.ink, 2),
      rect(60, 690, 455, 130, COLORS.panel, COLORS.line),
      rect(565, 690, 455, 130, COLORS.panel, COLORS.line),
      text(88, 740, 'TR → RU', { size: 28, family: 'monospace', weight: 700, color: COLORS.green }),
      text(88, 780, 'Türkiye exports to Russia', { size: 22 }),
      text(592, 740, 'RU → TR', { size: 28, family: 'monospace', weight: 700, color: COLORS.orange }),
      text(592, 780, 'Türkiye imports from Russia', { size: 22 }),
      multiText(60, 940, ['A research and decision-support platform', 'for 14 monthly trade series.'], { size: 30, lineHeight: 42 }),
      text(60, 1120, '1, 3 AND 6-MONTH FORECAST HORIZONS', { size: 16, color: COLORS.muted, family: 'monospace', weight: 600, spacing: 0.8 }),
    ].join(''),
  ],
  [
    '02-research-question',
    [
      text(60, 180, '01 / THE QUESTION', { size: 15, color: COLORS.green, family: 'monospace', weight: 700, spacing: 1 }),
      multiText(60, 300, ['Can forecasting models', 'improve monthly trade', 'forecasts?'], { size: 58, family: 'Georgia, Times New Roman, serif', lineHeight: 72 }),
      text(60, 555, 'The project compares model results with a simple seasonal baseline.', { size: 23, color: COLORS.muted }),
      rule(60, 630, 1020, 630, COLORS.ink, 2),
      text(60, 720, 'PROJECT SCOPE', { size: 14, color: COLORS.muted, family: 'monospace', weight: 700, spacing: 1 }),
      text(60, 835, '14', { size: 80, family: 'Georgia, Times New Roman, serif' }),
      text(60, 885, 'monthly series', { size: 21, color: COLORS.muted }),
      rule(350, 755, 350, 930),
      text(410, 835, '2', { size: 80, family: 'Georgia, Times New Roman, serif' }),
      text(410, 885, 'trade directions', { size: 21, color: COLORS.muted }),
      rule(700, 755, 700, 930),
      text(760, 835, '3', { size: 80, family: 'Georgia, Times New Roman, serif' }),
      text(760, 885, 'forecast horizons', { size: 21, color: COLORS.muted }),
      text(60, 1035, 'h = 1 month     h = 3 months     h = 6 months', { size: 19, color: COLORS.green, family: 'monospace', weight: 600 }),
      multiText(60, 1130, ['Both forecast targets use Türkiye-reported trade data.', 'Russia-reported records are kept for comparison.'], { size: 19, color: COLORS.muted, lineHeight: 29 }),
    ].join(''),
  ],
  [
    '03-project-map',
    [
      text(60, 180, '02 / HOW IT WORKS', { size: 15, color: COLORS.green, family: 'monospace', weight: 700, spacing: 1 }),
      text(60, 300, 'From data to dashboard', { size: 55, family: 'Georgia, Times New Roman, serif' }),
      text(60, 365, 'A clear path from source records to research results.', { size: 22, color: COLORS.muted }),
      rule(60, 420, 1020, 420, COLORS.ink, 2),
      ...[
        ['01', 'UN COMTRADE', 'Monthly Türkiye–Russia trade records'],
        ['02', 'DATA PREPARATION', 'Quality checks, mirror pairs and model features'],
        ['03', 'MODEL COMPARISON', 'Four model families · time-based testing'],
        ['04', 'FASTAPI', 'Read-only access to verified results'],
        ['05', 'NEXT.JS DASHBOARD', 'Explore forecasts in Turkish, English and Russian'],
      ].flatMap((row, index) => {
        const y = 500 + index * 135;
        return [
          text(66, y + 34, row[0], { size: 15, color: COLORS.green, family: 'monospace', weight: 700 }),
          text(145, y + 34, row[1], { size: 18, family: 'monospace', weight: 700 }),
          text(145, y + 72, row[2], { size: 20, color: COLORS.muted }),
          rule(145, y + 100, 1020, y + 100),
          ...(index < 4 ? [rule(83, y + 44, 83, y + 125, COLORS.green, 2)] : []),
        ];
      }),
    ].join(''),
  ],
  [
    '04-model-families',
    [
      text(60, 180, '03 / THE MODELS', { size: 15, color: COLORS.green, family: 'monospace', weight: 700, spacing: 1 }),
      text(60, 300, 'Four ways to forecast', { size: 55, family: 'Georgia, Times New Roman, serif' }),
      text(60, 365, 'Each method is tested on the same time-based schedule.', { size: 22, color: COLORS.muted }),
      rule(60, 420, 1020, 420, COLORS.ink, 2),
      ...[
        ['01', 'SEASONAL NAIVE', 'Uses the value from the same month last year.'],
        ['02', 'SARIMA', 'Learns patterns from the trade series history.'],
        ['03', 'SARIMAX', 'Adds economic indicators available at the forecast origin.'],
        ['04', 'XGBOOST', 'Learns more complex patterns from prepared features.'],
      ].flatMap((row, index) => {
        const y = 525 + index * 155;
        return [
          text(65, y, row[0], { size: 14, color: COLORS.green, family: 'monospace', weight: 700 }),
          text(145, y, row[1], { size: 20, family: 'monospace', weight: 700 }),
          text(145, y + 43, row[2], { size: 21, color: COLORS.muted }),
          rule(145, y + 80, 1020, y + 80),
        ];
      }),
      multiText(60, 1185, ['A champion is selected separately for each', 'trade series and forecast horizon.'], { size: 19, color: COLORS.green, weight: 600, lineHeight: 28 }),
    ].join(''),
  ],
  [
    '05-results',
    [
      text(60, 180, '04 / THE RESULT', { size: 15, color: COLORS.green, family: 'monospace', weight: 700, spacing: 1 }),
      text(60, 300, 'How did the forecasts perform?', { size: 50, family: 'Georgia, Times New Roman, serif' }),
      text(60, 365, 'Average error on the 2025 holdout. Lower sMAPE is better.', { size: 21, color: COLORS.muted }),
      rule(60, 420, 1020, 420, COLORS.ink, 2),
      text(60, 530, 'OPERATIONAL MODEL SYSTEM', { size: 15, family: 'monospace', weight: 700 }),
      rect(60, 555, 480, 66, COLORS.green),
      text(570, 602, '26.17%', { size: 37, family: 'Georgia, Times New Roman, serif', color: COLORS.green }),
      text(60, 700, 'SEASONAL NAIVE BASELINE', { size: 15, family: 'monospace', weight: 700 }),
      rect(60, 725, 690, 66, '#727b78'),
      text(780, 772, '37.50%', { size: 37, family: 'Georgia, Times New Roman, serif', color: COLORS.ink }),
      rule(60, 870, 1020, 870, COLORS.ink, 2),
      text(60, 980, '36 / 42', { size: 64, family: 'Georgia, Times New Roman, serif' }),
      multiText(340, 955, ['tasks had lower average error', 'than the Seasonal Naive baseline.'], { size: 23, color: COLORS.muted, lineHeight: 34 }),
      rect(60, 1080, 960, 112, '#eeede6'),
      multiText(82, 1124, ['This is a retrospective 2025 evaluation using latest-vintage data', 'and fixed lag rules. It is not a real-time performance test.'], { size: 17, color: COLORS.ink, lineHeight: 26 }),
    ].join(''),
  ],
  [
    '06-dashboard-terms',
    [
      text(60, 180, '05 / A QUICK GUIDE', { size: 15, color: COLORS.green, family: 'monospace', weight: 700, spacing: 1 }),
      text(60, 300, 'Four terms to know', { size: 55, family: 'Georgia, Times New Roman, serif' }),
      rule(60, 355, 1020, 355, COLORS.ink, 2),
      ...[
        ['CHAMPION', 'The model selected for one trade series and horizon.'],
        ['HOLDOUT', 'Data saved for the final model test. Here, it is 2025.'],
        ['MIRROR STATISTICS', 'A comparison of Türkiye and Russia trade reports.'],
        ['sMAPE', 'A forecast error score. A lower score is better.'],
      ].flatMap((row, index) => {
        const y = 470 + index * 170;
        return [
          text(60, y, row[0], { size: 16, color: COLORS.green, family: 'monospace', weight: 700, spacing: 0.5 }),
          text(60, y + 48, row[1], { size: 22 }),
          rule(60, y + 88, 1020, y + 88),
        ];
      }),
      text(60, 1195, 'TR / EN / RU DASHBOARD', { size: 16, color: COLORS.muted, family: 'monospace', weight: 700 }),
    ].join(''),
  ],
  [
    '07-tools-and-scope',
    [
      text(60, 180, '06 / THE PLATFORM', { size: 15, color: COLORS.green, family: 'monospace', weight: 700, spacing: 1 }),
      multiText(60, 300, ['Research methods,', 'built into a tool.'], { size: 56, family: 'Georgia, Times New Roman, serif', lineHeight: 70 }),
      rule(60, 390, 1020, 390, COLORS.ink, 2),
      text(60, 485, 'TECHNOLOGIES', { size: 15, color: COLORS.muted, family: 'monospace', weight: 700, spacing: 1 }),
      multiText(60, 550, ['Python · pandas · statsmodels · XGBoost · SHAP', 'FastAPI · Next.js · React · TypeScript'], { size: 22, lineHeight: 38 }),
      rule(60, 665, 1020, 665),
      text(60, 755, 'EXPLORE THE DASHBOARD', { size: 15, color: COLORS.muted, family: 'monospace', weight: 700, spacing: 1 }),
      text(60, 815, 'frontend-ten-psi-12.vercel.app', { size: 28, color: COLORS.green, family: 'monospace', weight: 600 }),
      rect(60, 925, 960, 205, COLORS.panel, COLORS.line),
      text(84, 970, 'SCOPE NOTE', { size: 14, color: COLORS.orange, family: 'monospace', weight: 700, spacing: 0.8 }),
      multiText(84, 1020, ['The dashboard presents a retrospective holdout, not live forecasts.', 'Source-rights review for some macro inputs is still in progress.'], { size: 19, color: COLORS.ink, lineHeight: 31 }),
      text(60, 1195, 'BUILT FOR REPRODUCIBILITY AND TRANSPARENT COMPARISON', { size: 14, color: COLORS.muted, family: 'monospace', weight: 600, spacing: 0.5 }),
    ].join(''),
  ],
];

await mkdir(OUTPUT_DIRECTORY, { recursive: true });
for (let index = 0; index < slides.length; index += 1) {
  const [name, content] = slides[index];
  const svg = frame(index + 1, content);
  const svgPath = join(OUTPUT_DIRECTORY, `${name}.svg`);
  const pngPath = join(OUTPUT_DIRECTORY, `${name}.png`);
  await writeFile(svgPath, svg, 'utf8');
  await sharp(Buffer.from(svg)).png().toFile(pngPath);
}
