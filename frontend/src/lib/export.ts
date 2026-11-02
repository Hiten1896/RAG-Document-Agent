import type { MindMapResponse, SourceMetadata } from '@/lib/api';

export interface ExportEntry {
  question: string;
  answer: string;
  sources: SourceMetadata[];
  evidence?: string[];
}

export interface ExportImage {
  dataUrl: string;
  width: number;
  height: number;
}

export interface ExportDocument {
  title: string;
  fileName: string;
  entries: ExportEntry[];
  mindMapImage?: ExportImage;
}

export function sanitizeExportFilename(value: string): string {
  const safe = value
    .normalize('NFKD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/[^a-zA-Z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 64);
  return safe || 'document-chat';
}

export function plainText(markdown: string): string {
  return markdown
    .replace(/```[^\n]*\n?/g, '')
    .replace(/`([^`]+)`/g, '$1')
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
    .replace(/^#{1,6}\s+/gm, '')
    .replace(/(\*\*|__)(.*?)\1/g, '$2')
    .replace(/(\*|_)(.*?)\1/g, '$2')
    .replace(/^\s*[-*+]\s+/gm, '• ')
    .trim();
}

function escapeXml(value: string): string {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&apos;');
}

export async function createMindMapImage(
  mindMap: MindMapResponse,
): Promise<ExportImage> {
  const points: Array<{
    title: string;
    source: string | null;
    x: number;
    y: number;
    parent: number | null;
  }> = [];
  let leafRow = 0;
  const placeNode = (
    title: string,
    source: string | null,
    children: MindMapResponse['children'],
    depth: number,
    parent: number | null,
  ): number => {
    const index = points.length;
    points.push({ title, source, x: 36 + depth * 220, y: 0, parent });
    const childIndices = children.map((child) =>
      placeNode(child.title, child.source, child.children, depth + 1, index),
    );
    if (childIndices.length) {
      points[index].y =
        childIndices.reduce((sum, childIndex) => sum + points[childIndex].y, 0) /
        childIndices.length;
    } else {
      points[index].y = 34 + leafRow * 46;
      leafRow += 1;
    }
    return index;
  };
  placeNode(mindMap.title, null, mindMap.children, 0, null);

  const width = 960;
  const height = Math.max(100, 68 + leafRow * 46);
  const links = points
    .filter((point) => point.parent !== null)
    .map((point) => {
      const parent = points[point.parent as number];
      return `<path d="M ${parent.x + 8} ${parent.y} C ${parent.x + 90} ${parent.y}, ${point.x - 90} ${point.y}, ${point.x - 8} ${point.y}" fill="none" stroke="#94a3b8" stroke-width="2"/>`;
    })
    .join('');
  const nodes = points
    .map((point, index) => {
      const label = escapeXml(plainText(point.title).slice(0, 56));
      const source = point.source ? escapeXml(plainText(point.source).slice(0, 40)) : '';
      return `<g><circle cx="${point.x}" cy="${point.y}" r="6" fill="${index === 0 ? '#4f46e5' : '#6366f1'}"/><text x="${point.x + 12}" y="${point.y + 4}" font-family="Arial,sans-serif" font-size="15" font-weight="${index === 0 ? 700 : 500}" fill="#0f172a">${label}</text>${source ? `<text x="${point.x + 12}" y="${point.y + 19}" font-family="Arial,sans-serif" font-size="11" fill="#64748b">${source}</text>` : ''}</g>`;
    })
    .join('');
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}"><rect width="100%" height="100%" fill="#ffffff"/>${links}${nodes}</svg>`;
  const image = new Image();
  const objectUrl = URL.createObjectURL(
    new Blob([svg], { type: 'image/svg+xml;charset=utf-8' }),
  );
  try {
    image.src = objectUrl;
    await image.decode();
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext('2d');
    if (!context) throw new Error('Could not render the mind map image.');
    context.drawImage(image, 0, 0);
    return { dataUrl: canvas.toDataURL('image/png'), width, height };
  } finally {
    URL.revokeObjectURL(objectUrl);
  }
}

export async function downloadWordDocument(data: ExportDocument): Promise<void> {
  const { Document, HeadingLevel, ImageRun, Packer, Paragraph } = await import('docx');
  const children: InstanceType<typeof Paragraph>[] = [
    new Paragraph({ text: data.title, heading: HeadingLevel.TITLE }),
  ];

  data.entries.forEach((entry, index) => {
    children.push(
      new Paragraph({
        text: `Question ${index + 1}`,
        heading: HeadingLevel.HEADING_2,
      }),
      ...plainText(entry.question)
        .split(/\r?\n/)
        .map((line) => new Paragraph({ text: line })),
      new Paragraph({ text: 'Answer', heading: HeadingLevel.HEADING_3 }),
      ...plainText(entry.answer)
        .split(/\r?\n/)
        .map((line) => new Paragraph({ text: line })),
    );

    if (entry.sources.length) {
      children.push(new Paragraph({ text: 'Sources', heading: HeadingLevel.HEADING_3 }));
      entry.sources.forEach((source) => {
        const location = source.page
          ? `page ${source.page}`
          : source.location || 'document';
        children.push(
          new Paragraph({
            text: `${source.source || 'Document'} (${location})`,
            bullet: { level: 0 },
          }),
        );
      });
    }
    if (entry.evidence?.length) {
      children.push(
        new Paragraph({ text: 'Retrieved evidence', heading: HeadingLevel.HEADING_3 }),
        ...entry.evidence.map((text) => new Paragraph({ text: plainText(text) })),
      );
    }
  });

  if (data.mindMapImage) {
    const encoded = data.mindMapImage.dataUrl.split(',')[1];
    if (!encoded) throw new Error('Could not prepare the mind map image.');
    const binary = atob(encoded);
    const bytes = Uint8Array.from(binary, (character) => character.charCodeAt(0));
    const imageWidth = 600;
    const imageHeight = Math.min(
      900,
      (data.mindMapImage.height / data.mindMapImage.width) * imageWidth,
    );
    children.push(
      new Paragraph({ text: 'Mind map', heading: HeadingLevel.HEADING_2 }),
      new Paragraph({
        children: [
          new ImageRun({
            data: bytes,
            type: 'png',
            transformation: { width: imageWidth, height: imageHeight },
          }),
        ],
      }),
    );
  }

  const document = new Document({ sections: [{ children }] });
  const blob = await Packer.toBlob(document);
  const url = URL.createObjectURL(blob);
  const link = window.document.createElement('a');
  link.href = url;
  link.download = `${data.fileName}.docx`;
  link.click();
  URL.revokeObjectURL(url);
}
