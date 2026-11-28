// 필요한 인물만 뽑아 보기: node tools/extract.js id1 id2 ... [--hints] [--conf]
//   기본: figures.json 항목만 출력
//   --hints: hints.json의 해당 힌트도 출력
//   --conf : confusable_ids 인물의 요약(이름·시대·국가·역할·시기)도 출력
// 큰 파일(figures.json·hints.json)을 통째로 읽지 않으려고 만든 도구.
const fs = require('fs');
const path = require('path');
const root = path.join(__dirname, '..');
const read = (p) => JSON.parse(fs.readFileSync(path.join(root, p), 'utf8'));
const args = process.argv.slice(2);
const ids = args.filter((a) => !a.startsWith('--'));
const withHints = args.includes('--hints');
const withConf = args.includes('--conf');
if (!ids.length) { console.error('사용법: node tools/extract.js id1 id2 ... [--hints] [--conf]'); process.exit(1); }
const figures = read('content/figures.json');
const byId = new Map(figures.map((f) => [f.id, f]));
const hints = withHints ? read('content/hints.json') : null;
const out = {};
for (const id of ids) {
  const f = byId.get(id);
  if (!f) { out[id] = '미등록 id'; continue; }
  const e = { figure: f };
  if (withHints) e.hints = hints[id] || '힌트 없음';
  if (withConf) e.confusables = (f.confusable_ids || []).map((c) => {
    const g = byId.get(c);
    return g ? { id: c, name: g.name, era: g.era, state: g.state, roles: g.roles, active_period: g.active_period } : { id: c, name: '미등록 id' };
  });
  out[id] = e;
}
console.log(JSON.stringify(out, null, 2));
