// 인물 도트 이미지 규격 검사: node tools/check_portrait.js file1.png [file2.png ...]
// 검사: 크기, RGBA, 반투명 픽셀, 색 수, 위아래 여백, 가로 범위, 512px 파일의 8배 격자 일치
const fs = require('fs'), zlib = require('zlib');

function decode(fn) {
  const b = fs.readFileSync(fn);
  let o = 8, ihdr, idat = [], plte, trns;
  while (o < b.length) {
    const l = b.readUInt32BE(o), t = b.toString('ascii', o + 4, o + 8), d = b.slice(o + 8, o + 8 + l);
    if (t === 'IHDR') ihdr = d; else if (t === 'IDAT') idat.push(d); else if (t === 'PLTE') plte = d; else if (t === 'tRNS') trns = d;
    o += 12 + l;
  }
  const w = ihdr.readUInt32BE(0), h = ihdr.readUInt32BE(4), bd = ihdr[8], ct = ihdr[9];
  const ch = { 0: 1, 2: 3, 3: 1, 4: 2, 6: 4 }[ct];
  if (bd !== 8) return { w, h, ct, err: 'bitdepth ' + bd + ' 미지원' };
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
  // RGBA로 통일
  const rgba = Buffer.alloc(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    let r, g, bl, al = 255;
    if (ct === 6) { [r, g, bl, al] = px.slice(i * 4, i * 4 + 4); }
    else if (ct === 2) { [r, g, bl] = px.slice(i * 3, i * 3 + 3); }
    else if (ct === 3) { const k = px[i]; [r, g, bl] = plte.slice(k * 3, k * 3 + 3); al = trns && k < trns.length ? trns[k] : 255; }
    else if (ct === 0) { r = g = bl = px[i]; }
    else { r = g = bl = px[i * 2]; al = px[i * 2 + 1]; }
    rgba.set([r, g, bl, al], i * 4);
  }
  return { w, h, ct, rgba };
}

for (const fn of process.argv.slice(2)) {
  const d = decode(fn);
  if (d.err) { console.log(fn, d.err); continue; }
  const { w, h, ct, rgba } = d;
  const issues = [];
  const s = w / 64;
  if (w !== h || !Number.isInteger(s)) issues.push(`크기 ${w}x${h} (64의 정수배 정사각형 아님)`);
  const hasAlpha = ct === 6 || ct === 4 || ct === 3;
  let semi = 0, opaqueCorner = rgba[3] === 255;
  const cols = new Set();
  let mnx = w, mxx = -1, mny = h, mxy = -1;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = (y * w + x) * 4, a = rgba[i + 3];
    if (a > 0 && a < 255) semi++;
    if (a > 0) {
      cols.add(rgba.slice(i, i + 3).toString('hex'));
      mnx = Math.min(mnx, x); mxx = Math.max(mxx, x); mny = Math.min(mny, y); mxy = Math.max(mxy, y);
    }
  }
  if (!hasAlpha || opaqueCorner) issues.push('배경 투명 아님');
  if (semi) issues.push(`반투명 픽셀 ${semi}개`);
  if (cols.size > 24) issues.push(`색 ${cols.size}개 (24 초과)`);
  let grid = 0;
  if (Number.isInteger(s) && s > 1) for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const j = ((Math.floor(y / s) * s) * w + Math.floor(x / s) * s) * 4, i = (y * w + x) * 4;
    if (rgba.readUInt32BE(i) !== rgba.readUInt32BE(j)) grid++;
  }
  if (grid) issues.push(`8배 격자 불일치 ${grid}px (도트가 아니거나 흐림)`);
  const u = v => Math.round(v / (s || 1));
  const top = u(mny), bottom = u(h - 1 - mxy);
  if (top < 2) issues.push(`위 여백 ${top}px`);
  if (bottom < 2) issues.push(`아래 여백 ${bottom}px`);
  console.log(`${fn}: ${w}x${h} ${hasAlpha ? 'alpha' : 'RGB'} 색${cols.size} 범위 x${u(mnx)}-${u(mxx)} y${top}-${u(mxy)} → ${issues.length ? '문제: ' + issues.join(', ') : '통과'}`);
}
