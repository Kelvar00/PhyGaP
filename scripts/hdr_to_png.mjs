import { promises as fs } from 'node:fs';
import path from 'node:path';
import { deflateSync } from 'node:zlib';
import { fileURLToPath } from 'node:url';

import { FloatType } from 'three';
import { RGBELoader } from 'three/examples/jsm/loaders/RGBELoader.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const envmapDir = path.resolve(__dirname, '../public/envmaps');
const loader = new RGBELoader().setDataType(FloatType);

const crcTable = new Uint32Array(256);
for (let i = 0; i < 256; i += 1) {
  let c = i;
  for (let j = 0; j < 8; j += 1) {
    c = (c & 1) ? (0xedb88320 ^ (c >>> 1)) : (c >>> 1);
  }
  crcTable[i] = c >>> 0;
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function linearToSrgb(value) {
  if (value <= 0.0031308) {
    return 12.92 * value;
  }
  return 1.055 * Math.pow(value, 1 / 2.4) - 0.055;
}

function encodeChunk(type, data = Buffer.alloc(0)) {
  const typeBuffer = Buffer.from(type, 'ascii');
  const lengthBuffer = Buffer.alloc(4);
  lengthBuffer.writeUInt32BE(data.length, 0);

  const crcBuffer = Buffer.concat([typeBuffer, data]);
  let crc = 0xffffffff;
  for (const byte of crcBuffer) {
    crc = crcTable[(crc ^ byte) & 0xff] ^ (crc >>> 8);
  }
  crc = (crc ^ 0xffffffff) >>> 0;

  const crcOut = Buffer.alloc(4);
  crcOut.writeUInt32BE(crc, 0);

  return Buffer.concat([lengthBuffer, typeBuffer, data, crcOut]);
}

function encodePng(width, height, rgbaBytes) {
  const signature = Buffer.from([
    0x89, 0x50, 0x4e, 0x47,
    0x0d, 0x0a, 0x1a, 0x0a,
  ]);

  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0);
  ihdr.writeUInt32BE(height, 4);
  ihdr[8] = 8;
  ihdr[9] = 6;
  ihdr[10] = 0;
  ihdr[11] = 0;
  ihdr[12] = 0;

  const srgb = Buffer.from([0]);
  const raw = Buffer.alloc((width * 4 + 1) * height);

  for (let y = 0; y < height; y += 1) {
    const rowOffset = y * (width * 4 + 1);
    raw[rowOffset] = 0;
    rgbaBytes.copy(raw, rowOffset + 1, y * width * 4, (y + 1) * width * 4);
  }

  const compressed = deflateSync(raw, { level: 9 });

  return Buffer.concat([
    signature,
    encodeChunk('IHDR', ihdr),
    encodeChunk('sRGB', srgb),
    encodeChunk('IDAT', compressed),
    encodeChunk('IEND'),
  ]);
}

function hdrToSrgbBytes(floatData, width, height) {
  const rgbaBytes = Buffer.alloc(width * height * 4);

  for (let pixelIndex = 0; pixelIndex < width * height; pixelIndex += 1) {
    const srcOffset = pixelIndex * 4;
    const dstOffset = pixelIndex * 4;

    const linearR = Math.max(0, floatData[srcOffset]);
    const linearG = Math.max(0, floatData[srcOffset + 1]);
    const linearB = Math.max(0, floatData[srcOffset + 2]);

    const mappedR = linearToSrgb(clamp(linearR, 0, 1));
    const mappedG = linearToSrgb(clamp(linearG, 0, 1));
    const mappedB = linearToSrgb(clamp(linearB, 0, 1));

    rgbaBytes[dstOffset] = Math.round(clamp(mappedR, 0, 1) * 255);
    rgbaBytes[dstOffset + 1] = Math.round(clamp(mappedG, 0, 1) * 255);
    rgbaBytes[dstOffset + 2] = Math.round(clamp(mappedB, 0, 1) * 255);
    rgbaBytes[dstOffset + 3] = 255;
  }

  return rgbaBytes;
}

async function convertHdrFile(fileName) {
  const inputPath = path.join(envmapDir, fileName);
  const outputPath = path.join(envmapDir, `${path.parse(fileName).name}.png`);

  const fileBuffer = await fs.readFile(inputPath);
  const uint8 = new Uint8Array(
    fileBuffer.buffer,
    fileBuffer.byteOffset,
    fileBuffer.byteLength
  );

  const parsed = loader.parse(uint8);
  const rgbaBytes = hdrToSrgbBytes(parsed.data, parsed.width, parsed.height);
  const pngBuffer = encodePng(parsed.width, parsed.height, rgbaBytes);

  await fs.writeFile(outputPath, pngBuffer);
  console.log(`[OK] ${fileName} -> ${path.basename(outputPath)} (${parsed.width}x${parsed.height})`);
}

async function main() {
  const files = await fs.readdir(envmapDir);
  const hdrFiles = files.filter((file) => file.toLowerCase().endsWith('.hdr')).sort();

  if (hdrFiles.length === 0) {
    throw new Error(`No HDR files found in ${envmapDir}`);
  }

  for (const hdrFile of hdrFiles) {
    await convertHdrFile(hdrFile);
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
