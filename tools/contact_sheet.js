// 64px 인물 이미지 여러 장을 한 장으로 모아 검수용 시트를 만든다.
// node tools/contact_sheet.js out.png img1.png img2.png ...   (4배 확대, 한 줄 4장, 회색 배경)
const fs = require('fs'), zlib = require('zlib');

function decode(fn) {
  const b = fs.readFileSync(fn);
  let o = 8, ihdr, idat = [], plte, trns;
  while (o < b.length) {
    const l = b.readUInt32BE(o), t = b.toString('ascii', o + 4, o + 8), d = b.slice(o + 8, o + 8 + l);
    if (t === 'IHDR') ihdr = d; else if (t === 'IDAT') idat.push(d); else if (t === 'PLTE') plte = d; else if (t === 'tRNS') trns = d;
    o += 12 + l;
  }
  const w = ihdr.readUInt32BE(0), h = ihdr.readUInt32BE(4), ct = ihdr[9], ch = { 0: 1, 2: 3, 3: 1, 4: 2, 6: 4 }[ct];
  const raw = zlib.inflateSync(Buffer.concat(idat)), st = w * ch, px = Buffer.alloc(h * st);
  let p = 0;
  for (let y = 0; y < h; y++) {
    const f = raw[p++];
    for (let x = 0; x < st; x++) {
      const a = x >= ch ? px[y * st + x - ch] : 0, up = y ? px[(y - 1) * st + x] : 0, c = (x >= ch && y) ? px[(y - 1) * st + x - ch] : 0;
      let v = raw[p++];
      if (f === 1) v += a; else if (f === 2) v += up; else if (f === 3) v += (a + up) >> 1;
      else if (f === 4) { const q = a + up - c, pa = Math.abs(q - a), pb = Math.abs(q - up), pc = Math.abs(q - c); v += (pa <= pb && pa <= pc) ? a : (pb <= pc ? up : c); }
      px[y * st + x] = v & 255;
    }
  }
  const rgba = Buffer.alloc(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    let r, g, bl, al = 255;
    if (ct === 6) [r, g, bl, al] = px.slice(i * 4, i * 4 + 4);
    else if (ct === 2) [r, g, bl] = px.slice(i * 3, i * 3 + 3);
    else if (ct === 3) { const k = px[i]; [r, g, bl] = plte.slice(k * 3, k * 3 + 3); al = trns && k < trns.length ? trns[k] : 255; }
    else if (ct === 0) r = g = bl = px[i];
    else { r = g = bl = px[i * 2]; al = px[i * 2 + 1]; }
    rgba.set([r, g, bl, al], i * 4);
  }
  return { w, h, rgba };
}

function crc32(buf) {
  let c, crc = 0xffffffff;
  for (let n = 0; n < buf.length; n++) {
    c = (crc ^ buf[n]) & 0xff;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    crc = (crc >>> 8) ^ c;
  }
  return (crc ^ 0xffffffff) >>> 0;
}
function chunk(type, data) {
  const len = Buffer.alloc(4); len.writeUInt32BE(data.length);
  const td = Buffer.concat([Buffer.from(type), data]);
  const crc = Buffer.alloc(4); crc.writeUInt32BE(crc32(td));
  return Buffer.concat([len, td, crc]);
}

const [out, ...files] = process.argv.slice(2);
const S = 4, CELL = 64 * S + 8, COLS = Math.min(4, files.length), ROWS = Math.ceil(files.length / COLS);
const W = COLS * CELL, H = ROWS * CELL, img = Buffer.alloc(W * H * 3, 0xc8);
files.forEach((fn, k) => {
  const { w, rgba } = decode(fn), s = (64 * S) / w;
  const ox = (k % COLS) * CELL + 4, oy = Math.floor(k / COLS) * CELL + 4;
  for (let y = 0; y < 64 * S; y++) for (let x = 0; x < 64 * S; x++) {
    const i = (Math.floor(y / s) * w + Math.floor(x / s)) * 4, a = rgba[i + 3] / 255, j = ((oy + y) * W + ox + x) * 3;
    for (let c = 0; c < 3; c++) img[j + c] = Math.round(rgba[i + c] * a + img[j + c] * (1 - a));
  }
});
const rows = Buffer.alloc(H * (W * 3 + 1));
for (let y = 0; y < H; y++) img.copy(rows, y * (W * 3 + 1) + 1, y * W * 3, (y + 1) * W * 3);
const ihdr = Buffer.alloc(13); ihdr.writeUInt32BE(W, 0); ihdr.writeUInt32BE(H, 4); ihdr[8] = 8; ihdr[9] = 2;
fs.writeFileSync(out, Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk('IHDR', ihdr), chunk('IDAT', zlib.deflateSync(rows)), chunk('IEND', Buffer.alloc(0))]));
console.log(out, W + 'x' + H, files.map((f, k) => (k + 1) + ':' + f.split(/[\\/]/).pop()).join(' '));
