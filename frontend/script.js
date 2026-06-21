const API_BASE_URL = '';

let METRICS = null;
let PATIENTS = [];
let CORRELATION = null;
let DISTRIBUTIONS = null;
let activeModel = 'lr';
let activeFilter = 'todos';
let searchTerm = '';
let sortKey = 'id';
let sortAsc = true;
let expandedId = null;
let activePage = 'avaliacao';

const MODEL_ORDER = ['lr', 'lda', 'qda', 'nb'];

async function fetchJSON(path){
  const res = await fetch(`${API_BASE_URL}${path}`);
  if(!res.ok) throw new Error(`${path} -> ${res.status}`);
  return res.json();
}

function getPath(obj, path){
  return path.split('.').reduce((o,k)=>o?.[k], obj);
}

function getConfidence(pred){
  return pred.label === 'maligno' ? pred.probability : 1 - pred.probability;
}

async function boot(){
  document.getElementById('offlineUrl').textContent = API_BASE_URL || window.location.origin;
  try{
    const [metrics, patientsResp, correlation, distributions] = await Promise.all([
      fetchJSON('/api/metrics'),
      fetchJSON('/api/patients?page=1&page_size=500'),
      fetchJSON('/api/analysis/correlation'),
      fetchJSON('/api/analysis/distributions'),
    ]);
    METRICS = metrics;
    PATIENTS = patientsResp.results;
    CORRELATION = correlation;
    DISTRIBUTIONS = distributions;
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

function renderBestBadge(){
  const best = METRICS.models[METRICS.best_model];
  document.getElementById('bestBadge').innerHTML = `
    <div class="label">Melhor modelo · recall</div>
    <div class="model-name">${best.label}</div>
    <div class="reasoning">${(best.recall*100).toFixed(1)}% de recall — menor taxa de falsos negativos entre os 4</div>
  `;
}

function renderTabsInto(containerId){
  const wrap = document.getElementById(containerId);
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

function renderTabs(){
  renderTabsInto('modelTabs');
  renderTabsInto('modelTabsAnalises');
}

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
          </div>
        </td>
        <td>${pred.correct ? '<span class="hit-icon">·</span>' : '<span class="miss-icon">✕ erro</span>'}</td>
      </tr>
    `;

    if(!isExpanded) return mainRow;

    const detailChips = MODEL_ORDER.map(key=>{
      const pr = p.predictions[key];
      const label = METRICS.models[key].label;
      const confidence = getConfidence(pr);
      return `
        <div class="detail-chip">
          <div class="chip-model">${label}</div>
          <div class="chip-pred">
            <span class="badge ${pr.label}">${pr.label}</span>
            ${pr.correct ? '' : '<span class="miss-icon">✕</span>'}
          </div>
          <div class="prob-bar-track"><div class="prob-bar-fill ${pr.label}" style="width:${(confidence*100).toFixed(0)}%"></div></div>
          <div class="chip-prob" style="margin-top:6px;">${(confidence*100).toFixed(1)}% confiança</div>
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

function corrColor(v){
  const neutral = [28,34,43];
  const pos = [217,142,59];
  const neg = [79,168,143];
  const t = Math.min(Math.abs(v),1);
  const base = v >= 0 ? pos : neg;
  const rgb = neutral.map((c,i)=> Math.round(c + (base[i]-c)*t));
  return `rgb(${rgb.join(',')})`;
}

function shortFeatureName(f){
  return f.replace('_mean','').replace('_',' ');
}

function renderCorrelationHeatmap(){
  const features = CORRELATION.features;
  const matrix = CORRELATION.matrix;
  const n = features.length;

  let html = `<div class="corr-grid" style="grid-template-columns: 110px repeat(${n}, 56px);">`;
  html += `<div class="corr-corner"></div>`;
  features.forEach(f=>{
    html += `<div class="corr-label-x">${shortFeatureName(f)}</div>`;
  });

  features.forEach((rowFeature, i)=>{
    html += `<div class="corr-label-y">${shortFeatureName(rowFeature)}</div>`;
    features.forEach((colFeature, j)=>{
      const v = matrix[i][j];
      html += `<div class="corr-cell" style="background:${corrColor(v)};">${v.toFixed(2)}</div>`;
    });
  });

  html += `</div>`;
  document.getElementById('corrHeatmap').innerHTML = html;
}

function buildAreaPath(xs, ys, maxY, width, height){
  const n = xs.length;
  const points = ys.map((y,i)=>{
    const px = (i/(n-1))*width;
    const py = height - (y/maxY)*height;
    return [px, py];
  });
  let d = `M0,${height} `;
  points.forEach(([px,py])=>{ d += `L${px.toFixed(1)},${py.toFixed(1)} `; });
  d += `L${width},${height} Z`;
  return d;
}

function renderDistributions(){
  const features = DISTRIBUTIONS.features;
  const width = 200, height = 90;

  const html = features.map(feature=>{
    const b = DISTRIBUTIONS.benigno[feature];
    const m = DISTRIBUTIONS.maligno[feature];
    const maxY = Math.max(...b.y, ...m.y);

    const pathB = buildAreaPath(b.x, b.y, maxY, width, height);
    const pathM = buildAreaPath(m.x, m.y, maxY, width, height);

    return `
      <div class="dist-cell">
        <div class="dist-title">${shortFeatureName(feature)}</div>
        <svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">
          <path d="${pathB}" fill="rgba(79,168,143,0.45)" stroke="#4FA88F" stroke-width="1.2"></path>
          <path d="${pathM}" fill="rgba(217,142,59,0.4)" stroke="#D98E3B" stroke-width="1.2"></path>
        </svg>
      </div>
    `;
  }).join('');

  document.getElementById('distGrid').innerHTML = html;
}

function renderNotebookConfusionMatrix(){
  const cm = METRICS.models[activeModel].confusion_matrix;
  const total = cm.tn + cm.fp + cm.fn + cm.tp;
  const intensity = (v)=> 0.12 + 0.55 * (v/total);

  const cells = [
    {key:'corner', el:'<div class="nb-cm-corner"></div>'},
    {key:'colMaligno', el:'<div class="nb-cm-col-label">Maligno</div>'},
    {key:'colBenigno', el:'<div class="nb-cm-col-label">Benigno</div>'},
    {key:'rowMaligno', el:'<div class="nb-cm-row-label">Maligno</div>'},
    {key:'tp', v:cm.tp, l:'Verdadeiro Positivo', color:'teal'},
    {key:'fn', v:cm.fn, l:'Falso Negativo', color:'amber'},
    {key:'rowBenigno', el:'<div class="nb-cm-row-label">Benigno</div>'},
    {key:'fp', v:cm.fp, l:'Falso Positivo', color:'amber'},
    {key:'tn', v:cm.tn, l:'Verdadeiro Negativo', color:'teal'},
  ];

  const html = cells.map(c=>{
    if(c.el) return c.el;
    const base = c.color === 'teal' ? '79,168,143' : '217,142,59';
    return `
      <div class="nb-cm-cell" style="background: rgba(${base}, ${intensity(c.v)}); border: 1px solid rgba(${base},0.4);">
        <div class="v">${c.v}</div>
        <div class="l">${c.l}</div>
      </div>
    `;
  }).join('');

  document.getElementById('nbCmGrid').innerHTML = html;
}

function renderAll(){
  renderBestBadge();
  renderTabs();
  renderReadout();
  renderRecallCompare();
  renderTable();
  renderCorrelationHeatmap();
  renderDistributions();
  renderNotebookConfusionMatrix();
}

document.querySelectorAll('.page-tab').forEach(btn=>{
  btn.addEventListener('click', ()=>{
    document.querySelectorAll('.page-tab').forEach(b=>b.classList.remove('active'));
    btn.classList.add('active');
    activePage = btn.dataset.page;
    document.getElementById('page-avaliacao').style.display = activePage === 'avaliacao' ? 'block' : 'none';
    document.getElementById('page-analises').style.display = activePage === 'analises' ? 'block' : 'none';
  });
});

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