// 콘텐츠 기계 검사: node tools/validate.js
// figures.json 형식·허용 값, hints.json 형식, 힌트 속 이름·별칭 노출을 검사한다.
// 허용 값은 content/schema.md와 같아야 한다. schema.md를 바꾸면 여기도 바꾼다.
const fs = require('fs');
const path = require('path');
const root = path.join(__dirname, '..');
const read = (p) => JSON.parse(fs.readFileSync(path.join(root, p), 'utf8'));

const ERA = ["선사","고조선","삼국","남북국","고려","조선 전기","조선 후기","개항기","일제강점기","현대"];
const STATE = ["고조선","부여","고구려","백제","신라","가야","발해","후백제","태봉","고려","조선","대한제국","대한민국 임시정부","해당 없음","대한민국","북한","외국","기타"];
const ROLE = ["국왕","왕족","문신·관료","무신·장수","학자·사상가","종교인","예술가·문인","개혁가·혁명가","독립운동가","사회운동가","정치인","기타"];
const FIELD = ["정치","외교","군사","경제","사상·종교","학문","문화·예술","과학기술","사회운동","독립운동"];
const ID_RE = /^[a-z0-9]+(-[a-z0-9]+)*$/;

const errors = [], warnings = [];
const figures = read('content/figures.json');
const ids = new Set(), names = new Map();

for (const p of figures) {
  const t = p.id;
  if (!ID_RE.test(t)) errors.push(`${t}: id 형식`);
  if (ids.has(t)) errors.push(`${t}: id 중복`);
  ids.add(t);
  if (!ERA.includes(p.era)) errors.push(`${t}: era '${p.era}'`);
  if (!STATE.includes(p.state)) errors.push(`${t}: state '${p.state}'`);
  if (p.state === '외국') errors.push(`${t}: 외국인은 수록 제외`);
  for (const [k, L] of [['roles', ROLE], ['fields', FIELD]]) {
    const a = p[k] || [];
    if (!(a.length >= 1 && a.length <= 3)) errors.push(`${t}: ${k} 개수 ${a.length}`);
    if (new Set(a).size !== a.length) errors.push(`${t}: ${k} 중복`);
    for (const x of a) if (!L.includes(x)) errors.push(`${t}: ${k} '${x}'`);
  }
  const ap = p.active_period || {};
  for (const y of [ap.start, ap.end]) if (y !== null && (!Number.isInteger(y) || y === 0)) errors.push(`${t}: 연도 '${y}'`);
  if (ap.start != null && ap.end != null && ap.start > ap.end) errors.push(`${t}: start > end`);
  if (!Array.isArray(p.achievements) || p.achievements.length < 2 || p.achievements.length > 5) errors.push(`${t}: achievements 개수 ${(p.achievements || []).length} (2~5)`);
  if (!Number.isInteger(p.exam_count)) errors.push(`${t}: exam_count`);
  if (!Array.isArray(p.sources) || p.sources.length < 1) errors.push(`${t}: sources 없음`);
  else if (p.sources.every((s) => !s.url)) warnings.push(`${t}: 확인된 source url 없음`);
  if ((p.confusable_ids || []).includes(t)) errors.push(`${t}: confusable 자기 참조`);
  if ((p.aliases || []).includes(p.name)) errors.push(`${t}: 별칭에 대표 이름 중복`);
  for (const n of [p.name, ...(p.aliases || [])]) {
    if (names.has(n)) errors.push(`별칭 충돌 '${n}': ${names.get(n)} / ${t}`);
    names.set(n, t);
  }
}
const unregistered = new Set();
for (const p of figures) for (const c of p.confusable_ids || []) {
  if (!ID_RE.test(c)) errors.push(`${p.id}: confusable id 형식 '${c}'`);
  if (!ids.has(c)) unregistered.add(c);
  else {
    const o = figures.find((x) => x.id === c);
    if (!(o.confusable_ids || []).includes(p.id)) warnings.push(`confusable 단방향: ${p.id} → ${c}`);
  }
}

let hintCount = 0;
if (fs.existsSync(path.join(root, 'content/hints.json'))) {
  const hints = read('content/hints.json');
  for (const id of Object.keys(hints)) if (!ids.has(id)) errors.push(`hints: 미등록 id '${id}'`);
  for (const p of figures) {
    const h = hints[p.id];
    if (!h) { warnings.push(`${p.id}: 힌트 없음`); continue; }
    hintCount++;
    if (h.length !== 5) errors.push(`${p.id}: 힌트 ${h.length}개`);
    h.forEach((x, i) => {
      if (x.level !== i + 1) errors.push(`${p.id}: 힌트 ${i + 1} level=${x.level}`);
      if (!x.text) errors.push(`${p.id}: 힌트 ${i + 1} text 없음`);
      if (!Array.isArray(x.sources) || !x.sources.length || x.sources.some((s) => !s.title)) errors.push(`${p.id}: 힌트 ${i + 1} sources 없음`);
      else if (x.sources.every((s) => !s.url)) warnings.push(`${p.id}: 힌트 ${i + 1} 확인된 url 없음`);
      // 두 글자 이상 이름·별칭이 힌트에 들어가면 오류
      for (const n of [p.name, ...(p.aliases || [])]) {
        if (n.length >= 2 && x.text && x.text.includes(n)) errors.push(`${p.id}: 힌트 ${i + 1}에 이름·별칭 '${n}'`);
      }
    });
  }
}

console.log(`인물 ${figures.length}명, 힌트 있는 인물 ${hintCount}명`);
console.log(`오류 ${errors.length}건`); errors.forEach((e) => console.log('  ✗ ' + e));
console.log(`경고 ${warnings.length}건`); warnings.forEach((w) => console.log('  ! ' + w));
console.log(`미등록 confusable id ${unregistered.size}개: ${[...unregistered].join(', ')}`);
process.exit(errors.length ? 1 : 0);
