// ============================================================
// Configuração — troque pela URL da sua API quando publicar
// ============================================================
const API_BASE_URL = 'http://127.0.0.1:8000';

// ============================================================
// Estado
// ============================================================
let METRICS = null;
let PATIENTS = [];
let activeModel = 'lr';
let activeFilter = 'todos';
let searchTerm = '';
let sortKey = 'id';
let sortAsc = true;
let expandedId = null;

const MODEL_ORDER = ['lr', 'lda', 'qda', 'nb'];

// ============================================================
// Fetch helpers
// ============================================================
async function fetchJSON(path){
  const res = await fetch(`${API_BASE_URL}${path}`);
  if(!res.ok) throw new Error(`${path} -> ${res.status}`);
  return res.json();
}

function getPath(obj, path){
  return path.split('.').reduce((o,k)=>o?.[k], obj);
}

// ============================================================
// Boot
// ============================================================
async function boot(){
  document.getElementById('offlineUrl').textContent = API_BASE_URL;
  try{
    const [metrics, patientsResp] = await Promise.all([
      fetchJSON('/api/metrics'),
      fetchJSON('/api/patients?page=1&page_size=500'),
    ]);
    METRICS = metrics;
    PATIENTS = patientsResp.results;
    activeModel = METRICS.best_model;

    document.getElementById('statusDot').className = 'status-dot online';
    document.getElementById('statusText').textContent = 'api conectada';
    document.getElementById('loadingState').style.display = 'none';
    document.getElementById('mainContent').style.display = 'block';

    renderAll();
  }catch(err){
    console.error(err);
    document.getElementById('statusDot').className = 'status-dot offline';
    document.getElementById('statusText').textContent = 'api offline';
    document.getElementById('loadingState').style.display = 'none';
    document.getElementById('offlineBanner').classList.add('show');
  }
}

// ============================================================
// Render: cabeçalho / badge do melhor modelo
// ============================================================
function renderBestBadge(){
  const best = METRICS.models[METRICS.best_model];
  document.getElementById('bestBadge').innerHTML = `
    <div class="label">Melhor modelo · recall</div>
    <div class="model-name">${best.label}</div>
    <div class="reasoning">${(best.recall*100).toFixed(1)}% de recall — menor taxa de falsos negativos entre os 4</div>
  `;
}

// ============================================================
// Render: tabs de modelo
// ============================================================
function renderTabs(){
  const wrap = document.getElementById('modelTabs');
  wrap.innerHTML = MODEL_ORDER.map(key=>{
    const m = METRICS.models[key];
    return `
      <button class="model-tab ${key===activeModel?'active':''}" data-model="${key}">
        <span>${m.label}</span>
        <span class="tab-metric">acc ${(m.accuracy*100).toFixed(1)}%</span>
      </button>
    `;
  }).join('');

  wrap.querySelectorAll('.model-tab').forEach(btn=>{
    btn.addEventListener('click', ()=>{
      activeModel = btn.dataset.model;
      renderAll();
    });
  });
}

// ============================================================
// Render: matriz de confusão
// ============================================================
function renderConfusionMatrix(){
  const cm = METRICS.models[activeModel].confusion_matrix;
  const total = cm.tn + cm.fp + cm.fn + cm.tp;
  const intensity = (v)=> 0.12 + 0.55 * (v/total);

  const cells = [
    {v: cm.tn, l:'Verdadeiro\nBenigno', color: 'teal', alpha: intensity(cm.tn)},
    {v: cm.fp, l:'Falso\nMaligno',      color: 'amber', alpha: intensity(cm.fp)},
    {v: cm.fn, l:'Falso\nBenigno',      color: 'amber', alpha: intensity(cm.fn)},
    {v: cm.tp, l:'Verdadeiro\nMaligno', color: 'teal', alpha: intensity(cm.tp)},
  ];

  document.getElementById('cmGrid').innerHTML = cells.map(c=>{
    const base = c.color === 'teal' ? '79,168,143' : '217,142,59';
    return `
      <div class="cm-cell" style="background: rgba(${base}, ${c.alpha}); border: 1px solid rgba(${base},0.4);">
        <div class="v">${c.v}</div>
        <div class="l">${c.l.replace('\n','<br>')}</div>
      </div>
    `;
  }).join('');
}

// ============================================================
// Render: readout de métricas
// ============================================================
function renderReadout(){
  const m = METRICS.models[activeModel];
  const rows = [
    {label:'Acurácia', value: m.accuracy},
    {label:'Precisão', value: m.precision},
    {label:'Recall', value: m.recall},
    {label:'F1-Score', value: m.f1},
    {label:'AUC-ROC', value: m.roc_auc},
  ];
  document.getElementById('readoutList').innerHTML = rows.map(r=>`
    <div class="readout-row">
      <div class="readout-label">${r.label}</div>
      <div class="readout-bar-track"><div class="readout-bar-fill" style="width:${(r.value*100).toFixed(1)}%"></div></div>
      <div class="readout-value">${(r.value*100).toFixed(1)}%</div>
    </div>
  `).join('');
}

// ============================================================
// Render: comparação de recall entre modelos
// ============================================================
function renderRecallCompare(){
  const bestKey = Object.keys(METRICS.models).reduce((a,b)=>
    METRICS.models[a].recall > METRICS.models[b].recall ? a : b
  );
  document.getElementById('recallCompare').innerHTML = MODEL_ORDER.map(key=>{
    const m = METRICS.models[key];
    const isBest = key === bestKey;
    return `
      <div class="rc-row">
        <div class="rc-label">${m.label}</div>
        <div class="rc-track"><div class="rc-fill ${isBest?'is-best':''}" style="width:${(m.recall*100).toFixed(1)}%"></div></div>
        <div class="rc-value">${(m.recall*100).toFixed(1)}%</div>
      </div>
    `;
  }).join('');
}

// ============================================================
// Render: tabela de pacientes
// ============================================================
function getFilteredSortedPatients(){
  let list = [...PATIENTS];

  if(activeFilter === 'benigno') list = list.filter(p=>p.actual==='benigno');
  if(activeFilter === 'maligno') list = list.filter(p=>p.actual==='maligno');
  if(activeFilter === 'erros') list = list.filter(p=>!p.predictions[activeModel].correct);

  if(searchTerm){
    list = list.filter(p=> String(p.id).includes(searchTerm));
  }

  list.sort((a,b)=>{
    const va = getPath(a, sortKey), vb = getPath(b, sortKey);
    if(va < vb) return sortAsc ? -1 : 1;
    if(va > vb) return sortAsc ? 1 : -1;
    return 0;
  });

  return list;
}

function renderTable(){
  const list = getFilteredSortedPatients();
  document.getElementById('tableCount').textContent = `${list.length} de ${PATIENTS.length} casos`;

  const rowsHtml = list.map(p=>{
    const pred = p.predictions[activeModel];
    const isExpanded = p.id === expandedId;
    const mainRow = `
      <tr data-id="${p.id}" class="${isExpanded?'expanded':''}">
        <td class="mono-cell">#${p.id}</td>
        <td class="mono-cell">${p.features.radius_mean}</td>
        <td class="mono-cell">${p.features.texture_mean}</td>
        <td class="mono-cell">${p.features.area_mean}</td>
        <td class="mono-cell">${p.features.concavity_mean}</td>
        <td><span class="badge ${p.actual}">${p.actual}</span></td>
        <td>
          <div class="pred-cell">
            <span class="badge ${pred.label}">${pred.label}</span>
            <div class="prob-bar-track"><div class="prob-bar-fill ${pred.label}" style="width:${(pred.probability*100).toFixed(0)}%"></div></div>
          </div>
        </td>
        <td>${pred.correct ? '<span class="hit-icon">·</span>' : '<span class="miss-icon">✕ erro</span>'}</td>
      </tr>
    `;

    if(!isExpanded) return mainRow;

    const detailChips = MODEL_ORDER.map(key=>{
      const pr = p.predictions[key];
      const label = METRICS.models[key].label;
      return `
        <div class="detail-chip">
          <div class="chip-model">${label}</div>
          <div class="chip-pred">
            <span class="badge ${pr.label}">${pr.label}</span>
            ${pr.correct ? '' : '<span class="miss-icon">✕</span>'}
          </div>
          <div class="prob-bar-track"><div class="prob-bar-fill ${pr.label}" style="width:${(pr.probability*100).toFixed(0)}%"></div></div>
          <div class="chip-prob" style="margin-top:6px;">${(pr.probability*100).toFixed(1)}% confiança</div>
        </div>
      `;
    }).join('');

    return mainRow + `
      <tr class="detail-row"><td colspan="8">
        <div class="detail-drawer">${detailChips}</div>
      </td></tr>
    `;
  }).join('');

  document.getElementById('tableBody').innerHTML = rowsHtml;

  document.querySelectorAll('#tableBody tr[data-id]').forEach(row=>{
    row.addEventListener('click', ()=>{
      const id = Number(row.dataset.id);
      expandedId = expandedId === id ? null : id;
      renderTable();
    });
  });
}

// ============================================================
// Render geral
// ============================================================
function renderAll(){
  renderBestBadge();
  renderTabs();
  renderConfusionMatrix();
  renderReadout();
  renderRecallCompare();
  renderTable();
}

// ============================================================
// Listeners estáticos
// ============================================================
document.querySelectorAll('.filter-pill').forEach(btn=>{
  btn.addEventListener('click', ()=>{
    document.querySelectorAll('.filter-pill').forEach(b=>b.classList.remove('active'));
    btn.classList.add('active');
    activeFilter = btn.dataset.filter;
    renderTable();
  });
});

document.getElementById('searchInput').addEventListener('input', (e)=>{
  searchTerm = e.target.value.trim();
  renderTable();
});

document.addEventListener('click', (e)=>{
  const th = e.target.closest('th[data-sort]');
  if(!th) return;
  const key = th.dataset.sort;
  if(sortKey === key){ sortAsc = !sortAsc; } else { sortKey = key; sortAsc = true; }
  document.querySelectorAll('th[data-sort]').forEach(el=>el.classList.remove('sorted','asc'));
  th.classList.add('sorted');
  if(sortAsc) th.classList.add('asc');
  renderTable();
});

boot();