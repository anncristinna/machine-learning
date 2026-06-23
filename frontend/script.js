const API_BASE_URL = '';

let METRICS        = null;
let METRICS_TUNED  = null;
let PATIENTS       = [];
let CORRELATION    = null;
let DISTRIBUTIONS  = null;
let BOXPLOTS       = null;
let PCA_BOUNDARY   = null;
let activeModel    = 'lr';
let activeFilter   = 'todos';
let searchTerm     = '';
let sortKey        = 'id';
let sortAsc        = true;
let expandedId     = null;
let activePage     = 'exploratoria';
let pcaRendered    = false;

const MODEL_ORDER = ['lr', 'lda', 'qda', 'nb'];
const PAGES       = ['exploratoria', 'resultados', 'etica'];

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
    const [metrics, metricsTuned, patientsResp, correlation, distributions, boxplots, pcaBoundary] = await Promise.all([
      fetchJSON('/api/metrics'),
      fetchJSON('/api/metrics/tuned'),
      fetchJSON('/api/patients?page=1&page_size=500'),
      fetchJSON('/api/analysis/correlation'),
      fetchJSON('/api/analysis/distributions'),
      fetchJSON('/api/analysis/boxplots'),
      fetchJSON('/api/analysis/pca_boundary'),
    ]);
    METRICS       = metrics;
    METRICS_TUNED = metricsTuned;
    PATIENTS      = patientsResp.results;
    CORRELATION   = correlation;
    DISTRIBUTIONS = distributions;
    BOXPLOTS      = boxplots;
    PCA_BOUNDARY  = pcaBoundary;
    activeModel   = METRICS.best_model;

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
    btn.addEventListener('click', ()=>{ activeModel = btn.dataset.model; renderAll(); });
  });
}

function renderReadout(){
  const m = METRICS.models[activeModel];
  const rows = [
    {label:'Acurácia', value:m.accuracy},
    {label:'Precisão', value:m.precision},
    {label:'Recall',   value:m.recall},
    {label:'F1-Score', value:m.f1},
    {label:'AUC-ROC',  value:m.roc_auc},
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
    return `
      <div class="rc-row">
        <div class="rc-label">${m.label}</div>
        <div class="rc-track"><div class="rc-fill ${key===bestKey?'is-best':''}" style="width:${(m.recall*100).toFixed(1)}%"></div></div>
        <div class="rc-value">${(m.recall*100).toFixed(1)}%</div>
      </div>
    `;
  }).join('');
}

function buildCmHtml(cm, containerId){
  const total = cm.tn + cm.fp + cm.fn + cm.tp;
  const intensity = v => 0.12 + 0.55 * (v/total);
  const cells = [
    {el:'<div class="nb-cm-corner"></div>'},
    {el:'<div class="nb-cm-col-label">Maligno</div>'},
    {el:'<div class="nb-cm-col-label">Benigno</div>'},
    {el:'<div class="nb-cm-row-label">Maligno</div>'},
    {v:cm.tp, l:'Verdadeiro Positivo', color:'teal'},
    {v:cm.fn, l:'Falso Negativo',      color:'amber'},
    {el:'<div class="nb-cm-row-label">Benigno</div>'},
    {v:cm.fp, l:'Falso Positivo',      color:'amber'},
    {v:cm.tn, l:'Verdadeiro Negativo', color:'teal'},
  ];
  document.getElementById(containerId).innerHTML = cells.map(c=>{
    if(c.el) return c.el;
    const base = c.color==='teal' ? '79,168,143' : '217,142,59';
    return `
      <div class="nb-cm-cell" style="background:rgba(${base},${intensity(c.v)});border:1px solid rgba(${base},0.4);">
        <div class="v">${c.v}</div>
        <div class="l">${c.l}</div>
      </div>
    `;
  }).join('');
}

function renderNotebookCM(){
  buildCmHtml(METRICS.models[activeModel].confusion_matrix, 'nbCmGrid');
}

function renderTunedSection(){
  const mt = METRICS_TUNED.models[activeModel];
  const mb = METRICS.models[activeModel];

  const params = mt.best_params;
  document.getElementById('tunedParams').innerHTML = `
    <div class="tuned-params">
      ${Object.entries(params).map(([k,v])=>`
        <div class="tuned-param">
          <span class="param-key">${k}</span>
          <span class="param-val">${v === null ? 'null' : (typeof v === 'number' && v < 0.001 ? v.toExponential(1) : v)}</span>
        </div>
      `).join('')}
    </div>
  `;

  const rows = [
    {label:'Acurácia',  before:mb.accuracy,  after:mt.accuracy},
    {label:'Precisão',  before:mb.precision, after:mt.precision},
    {label:'Recall',    before:mb.recall,    after:mt.recall},
    {label:'F1-Score',  before:mb.f1,        after:mt.f1},
    {label:'AUC-ROC',   before:mb.roc_auc,   after:mt.roc_auc},
  ];

  document.getElementById('tunedMetricsCompare').innerHTML = `
    <table class="compare-table">
      <thead>
        <tr>
          <th>Métrica</th>
          <th>Baseline</th>
          <th>Tuned</th>
          <th>Δ</th>
        </tr>
      </thead>
      <tbody>
        ${rows.map(r=>{
          const delta = ((r.after - r.before) * 100);
          const sign  = delta >= 0 ? '+' : '';
          const cls   = Math.abs(delta) < 0.05 ? 'delta-neu' : delta > 0 ? 'delta-pos' : 'delta-neg';
          return `
            <tr>
              <td class="col-metric">${r.label}</td>
              <td class="col-before">${(r.before*100).toFixed(1)}%</td>
              <td class="col-after">${(r.after*100).toFixed(1)}%</td>
              <td class="${cls}">${sign}${delta.toFixed(1)}%</td>
            </tr>
          `;
        }).join('')}
      </tbody>
    </table>
  `;

  buildCmHtml(mt.confusion_matrix, 'tunedCmGrid');
}

function getFilteredSortedPatients(){
  let list = [...PATIENTS];
  if(activeFilter==='benigno') list = list.filter(p=>p.actual==='benigno');
  if(activeFilter==='maligno') list = list.filter(p=>p.actual==='maligno');
  if(activeFilter==='erros')   list = list.filter(p=>!p.predictions[activeModel].correct);
  if(searchTerm) list = list.filter(p=>String(p.id).includes(searchTerm));
  list.sort((a,b)=>{
    const va=getPath(a,sortKey), vb=getPath(b,sortKey);
    if(va<vb) return sortAsc?-1:1;
    if(va>vb) return sortAsc?1:-1;
    return 0;
  });
  return list;
}

function renderTable(){
  const list = getFilteredSortedPatients();
  document.getElementById('tableCount').textContent = `${list.length} de ${PATIENTS.length} casos`;
  document.getElementById('tableBody').innerHTML = list.map(p=>{
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
        <td><div class="pred-cell"><span class="badge ${pred.label}">${pred.label}</span></div></td>
        <td>${pred.correct?'<span class="hit-icon">·</span>':'<span class="miss-icon">✕ erro</span>'}</td>
      </tr>
    `;
    if(!isExpanded) return mainRow;
    const chips = MODEL_ORDER.map(key=>{
      const pr   = p.predictions[key];
      const conf = getConfidence(pr);
      return `
        <div class="detail-chip">
          <div class="chip-model">${METRICS.models[key].label}</div>
          <div class="chip-pred">
            <span class="badge ${pr.label}">${pr.label}</span>
            ${pr.correct?'':'<span class="miss-icon">✕</span>'}
          </div>
          <div class="prob-bar-track"><div class="prob-bar-fill ${pr.label}" style="width:${(conf*100).toFixed(0)}%"></div></div>
          <div class="chip-prob" style="margin-top:6px;">${(conf*100).toFixed(1)}% confiança</div>
        </div>
      `;
    }).join('');
    return mainRow + `<tr class="detail-row"><td colspan="8"><div class="detail-drawer">${chips}</div></td></tr>`;
  }).join('');
  document.querySelectorAll('#tableBody tr[data-id]').forEach(row=>{
    row.addEventListener('click', ()=>{
      const id = Number(row.dataset.id);
      expandedId = expandedId===id ? null : id;
      renderTable();
    });
  });
}

function corrColor(v){
  const neutral=[28,34,43], pos=[217,142,59], neg=[79,168,143];
  const t=Math.min(Math.abs(v),1), base=v>=0?pos:neg;
  return `rgb(${neutral.map((c,i)=>Math.round(c+(base[i]-c)*t)).join(',')})`;
}

function shortName(f){
  return f.replace('_mean','').replace('_se',' se').replace('_worst',' w').replace(/_/g,' ');
}

function renderCorrelationHeatmap(){
  const {features, matrix} = CORRELATION;
  const n = features.length;
  let html = `<div class="corr-grid" style="grid-template-columns:110px repeat(${n},56px);">`;
  html += `<div class="corr-corner"></div>`;
  features.forEach(f=>{ html += `<div class="corr-label-x">${shortName(f)}</div>`; });
  features.forEach((rf,i)=>{
    html += `<div class="corr-label-y">${shortName(rf)}</div>`;
    features.forEach((_,j)=>{
      const v=matrix[i][j];
      html += `<div class="corr-cell" style="background:${corrColor(v)};">${v.toFixed(2)}</div>`;
    });
  });
  document.getElementById('corrHeatmap').innerHTML = html + `</div>`;
}

function buildAreaPath(xs,ys,maxY,W,H){
  let d=`M0,${H} `;
  ys.forEach((y,i)=>{ d+=`L${((i/(xs.length-1))*W).toFixed(1)},${(H-(y/maxY)*H).toFixed(1)} `; });
  return d+`L${W},${H} Z`;
}

function renderDistributions(){
  const {features}=DISTRIBUTIONS; const W=200,H=90;
  document.getElementById('distGrid').innerHTML = features.map(f=>{
    const b=DISTRIBUTIONS.benigno[f], m=DISTRIBUTIONS.maligno[f];
    const maxY=Math.max(...b.y,...m.y);
    return `
      <div class="dist-cell">
        <div class="dist-title">${shortName(f)}</div>
        <svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">
          <path d="${buildAreaPath(b.x,b.y,maxY,W,H)}" fill="rgba(79,168,143,0.45)" stroke="#4FA88F" stroke-width="1.2"/>
          <path d="${buildAreaPath(m.x,m.y,maxY,W,H)}" fill="rgba(217,142,59,0.4)" stroke="#D98E3B" stroke-width="1.2"/>
        </svg>
      </div>
    `;
  }).join('');
}

function renderClassDistribution(){
  const bal=METRICS.class_balance_train;
  const maxV=Math.max(bal.benigno,bal.maligno);
  const W=300,H=160,barW=80,maxH=100,padB=35,padL=40;
  const hB=(bal.benigno/maxV)*maxH, hM=(bal.maligno/maxV)*maxH;
  const xB=padL+20, xM=padL+20+barW+30;
  document.getElementById('classDistChart').innerHTML = `
    <svg viewBox="0 0 ${W} ${H}" style="max-width:320px;">
      <rect x="${xB}" y="${H-padB-hB}" width="${barW}" height="${hB}" fill="rgba(79,168,143,0.6)" rx="3"/>
      <rect x="${xM}" y="${H-padB-hM}" width="${barW}" height="${hM}" fill="rgba(217,142,59,0.6)" rx="3"/>
      <text x="${xB+barW/2}" y="${H-padB-hB-6}" text-anchor="middle" font-family="IBM Plex Mono,monospace" font-size="14" fill="#E8E6DE" font-weight="600">${bal.benigno}</text>
      <text x="${xM+barW/2}" y="${H-padB-hM-6}" text-anchor="middle" font-family="IBM Plex Mono,monospace" font-size="14" fill="#E8E6DE" font-weight="600">${bal.maligno}</text>
      <text x="${xB+barW/2}" y="${H-10}" text-anchor="middle" font-family="IBM Plex Mono,monospace" font-size="10" fill="#8A93A3">Benigno (0)</text>
      <text x="${xM+barW/2}" y="${H-10}" text-anchor="middle" font-family="IBM Plex Mono,monospace" font-size="10" fill="#8A93A3">Maligno (1)</text>
      <line x1="${padL-5}" y1="${H-padB}" x2="${W-20}" y2="${H-padB}" stroke="#2A323D" stroke-width="1"/>
    </svg>
  `;
}

function renderBoxPlots(){
  const {features, stats}=BOXPLOTS;
  document.getElementById('boxPlotGrid').innerHTML = features.map(f=>{
    const s=stats[f];
    const all=[s.whisker_lo,s.q1,s.median,s.q3,s.whisker_hi,...s.outliers];
    const vMin=Math.min(...all),vMax=Math.max(...all),range=vMax-vMin||1;
    const W=70,H=108,top=6,bot=98,h=bot-top,cx=35,bw=22;
    const toY=v=>bot-((v-vMin)/range)*h;
    const ywlo=toY(s.whisker_lo),yq1=toY(s.q1),ymed=toY(s.median),yq3=toY(s.q3),ywhi=toY(s.whisker_hi);
    let svg='';
    svg+=`<line x1="${cx}" y1="${ywlo}" x2="${cx}" y2="${yq1}" stroke="#4FA88F" stroke-width="1.2"/>`;
    svg+=`<line x1="${cx}" y1="${yq3}" x2="${cx}" y2="${ywhi}" stroke="#4FA88F" stroke-width="1.2"/>`;
    svg+=`<line x1="${cx-6}" y1="${ywlo}" x2="${cx+6}" y2="${ywlo}" stroke="#4FA88F" stroke-width="1.2"/>`;
    svg+=`<line x1="${cx-6}" y1="${ywhi}" x2="${cx+6}" y2="${ywhi}" stroke="#4FA88F" stroke-width="1.2"/>`;
    svg+=`<rect x="${cx-bw/2}" y="${yq3}" width="${bw}" height="${yq1-yq3}" fill="rgba(79,168,143,0.2)" stroke="#4FA88F" stroke-width="1.2" rx="2"/>`;
    svg+=`<line x1="${cx-bw/2}" y1="${ymed}" x2="${cx+bw/2}" y2="${ymed}" stroke="#D98E3B" stroke-width="2"/>`;
    s.outliers.forEach(v=>{ svg+=`<circle cx="${cx}" cy="${toY(v)}" r="1.8" fill="none" stroke="#C75C5C" stroke-width="1"/>`; });
    return `
      <div class="boxplot-card">
        <div class="bp-title">${shortName(f)}</div>
        <svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}">${svg}</svg>
        <div class="bp-median">${s.median.toFixed(2)}</div>
      </div>
    `;
  }).join('');
}

function renderPCABoundary(){
  const canvas=document.getElementById('pcaCanvas');
  if(!canvas||!PCA_BOUNDARY) return;
  const ctx=canvas.getContext('2d');
  const CW=canvas.width, CH=canvas.height;
  const {grid,points,variance_explained}=PCA_BOUNDARY;
  const size=grid.size, preds=grid.predictions;
  ctx.clearRect(0,0,CW,CH);
  ctx.fillStyle='#0E1116'; ctx.fillRect(0,0,CW,CH);
  const cW=CW/size, cH=CH/size;
  for(let row=0;row<size;row++){
    for(let col=0;col<size;col++){
      ctx.fillStyle=parseInt(preds[row*size+col])===1?'rgba(217,142,59,0.28)':'rgba(79,168,143,0.28)';
      ctx.fillRect(col*cW,row*cH,cW+0.5,cH+0.5);
    }
  }
  const xToC=x=>(x-grid.x_min)/(grid.x_max-grid.x_min)*CW;
  const yToC=y=>CH-(y-grid.y_min)/(grid.y_max-grid.y_min)*CH;
  points.forEach(p=>{
    ctx.beginPath(); ctx.arc(xToC(p.x),yToC(p.y),3,0,Math.PI*2);
    ctx.fillStyle=p.label===1?'#D98E3B':'#4FA88F';
    ctx.strokeStyle='rgba(14,17,22,0.7)'; ctx.lineWidth=0.8;
    ctx.fill(); ctx.stroke();
  });
  ctx.fillStyle='rgba(14,17,22,0.55)'; ctx.fillRect(0,CH-38,230,38);
  ctx.fillStyle='#8A93A3'; ctx.font='11px "IBM Plex Mono",monospace';
  ctx.fillText(`CP1: ${variance_explained[0]}% da variância explicada`,10,CH-20);
  ctx.fillText(`CP2: ${variance_explained[1]}% da variância explicada`,10,CH-6);
  pcaRendered=true;
}

function renderExplanations(){
  document.getElementById('explHyperparams').innerHTML = `
    <div class="expl-grid">
      <div class="expl-card">
        <div class="expl-model">LDA</div>
        <div class="expl-subtitle">shrinkage + solver</div>
        <div class="expl-body">O LDA confia em uma matriz de covariância. Com colunas muito correlacionadas, essa matriz fica instável. O <strong>shrinkage</strong> "encolhe" estimativas extremas em direção à média, forçando estabilidade matemática. O <strong>solver</strong> padrão (svd) não suporta shrinkage — é necessário usar <em>lsqr</em> ou <em>eigen</em>.</div>
      </div>
      <div class="expl-card">
        <div class="expl-model">QDA</div>
        <div class="expl-subtitle">reg_param</div>
        <div class="expl-body">O QDA calcula uma matriz de covariância <em>por classe</em>, tornando-o mais frágil a ruídos. O <strong>reg_param</strong> injeta uma fração da matriz identidade (Σ + λI) ao cálculo, adicionando estabilidade e evitando overfitting em dados com alta colinearidade.</div>
      </div>
      <div class="expl-card">
        <div class="expl-model">Naive Bayes</div>
        <div class="expl-subtitle">var_smoothing</div>
        <div class="expl-body">A equação gaussiana divide pela variância (σ²). Se uma feature tiver variância próxima de zero, ocorre divisão por zero. O <strong>var_smoothing</strong> adiciona uma fatia mínima da maior variância encontrada a todas as features, prevenindo o colapso matemático.</div>
      </div>
    </div>
  `;
  document.getElementById('explCrossval').innerHTML = `
    <div class="crossval-block">
      <p>Na construção de modelos de ML, a validação atua como um "simulado" antes da prova final. O conjunto de <strong>Treino</strong> ensina os padrões, o conjunto de <strong>Teste</strong> avalia a performance final, e o conjunto de <strong>Validação</strong> serve para testar e ajustar hiperparâmetros de forma segura.</p>
      <div class="crossval-steps">
        <div class="crossval-step"><span class="step-num">K-Fold</span><span>O conjunto de treinamento é fatiado em <strong>k pedaços</strong>. O modelo é treinado em k−1 pedaços e validado no restante, repetindo até que todos os fragmentos tenham sido usados para validar pelo menos uma vez. O resultado final é a <strong>média do desempenho em todas as rodadas</strong>.</span></div>
        <div class="crossval-step"><span class="step-num">Grid</span><span>O <strong>GridSearchCV</strong> combina Validação Cruzada com busca exaustiva de hiperparâmetros. Para 3 combinações testadas com 5 folds, são executados <strong>15 treinamentos individuais</strong>. O algoritmo declara a combinação com maior recall como a campeã e reconstrói o modelo final com ela.</span></div>
        <div class="crossval-step"><span class="step-num">Recall</span><span>A métrica norteadora do GridSearchCV é o <strong>Recall</strong>. O modelo base pode ter excelente acurácia, mas deixar passar casos malignos críticos (Falsos Negativos). A validação cruzada permite forçar o algoritmo a minimizar esses erros sem ferir o rigor metodológico.</span></div>
      </div>
      <div class="crossval-vs">
        <div class="vs-card vs-bad">
          <div class="vs-label">✕ Sem validação</div>
          <p>O modelo é treinado e julgado diretamente na base de teste. Qualquer ajuste após ver o resultado comete <strong>Vazamento de Dados</strong> — o modelo decora a prova final e falha com novos pacientes no mundo real.</p>
        </div>
        <div class="vs-card vs-good">
          <div class="vs-label">✓ Com validação cruzada</div>
          <p>Os ajustes são feitos em uma "caixa de areia". A base de teste permanece como um <strong>cofre inviolável</strong>, acessada estritamente uma única vez no final para atestar o desempenho real do algoritmo.</p>
        </div>
      </div>
    </div>
  `;
}

function renderEthics(){
  const sections = [
    {
      num:'Pilar 01', title:'O Peso Assimétrico dos Erros',
      body:`Um <strong>Falso Positivo</strong> (classificar um tumor benigno como maligno) gera ansiedade severa e submete a paciente a biópsias desnecessárias; contudo, o erro é rapidamente corrigido por exames subsequentes.<br><br>Um <strong>Falso Negativo</strong> (classificar um tumor maligno como benigno) é um erro catastrófico. Ele priva a paciente do tratamento precoce, permitindo a progressão silenciosa da doença. Sob a ótica ética do princípio da <strong>não-maleficência</strong>, justifica-se a escolha do grupo em priorizar a otimização do Recall no GridSearchCV.`
    },
    {
      num:'Pilar 02', title:'IA como Suporte à Decisão, não como Substituta Humana',
      body:`Nenhum modelo estatístico deve ser interpretado como um <strong>diagnóstico final autônomo</strong>. Eticamente, estes sistemas devem atuar como ferramentas de <strong>suporte à decisão clínica</strong> (segunda opinião), alertando o corpo médico para casos de alto risco. A validação e a palavra final devem permanecer sob a <strong>responsabilidade do médico especialista</strong>.`
    },
    {
      num:'Pilar 03', title:'Viés de Dados, Representatividade e Generalização',
      body:`O dataset foi coletado na <strong>década de 1990</strong> em uma região específica dos Estados Unidos. Fatores genéticos e perfis epidemiológicos variam entre populações — como a <strong>população brasileira</strong>. Utilizar este modelo universalmente sem validação local violaria o princípio da <strong>equidade na saúde</strong>.`
    },
  ];
  document.getElementById('ethicsContent').innerHTML = sections.map(s=>`
    <div class="ethics-block">
      <div class="ethics-num">${s.num}</div>
      <div class="ethics-title">${s.title}</div>
      <div class="ethics-body">${s.body}</div>
    </div>
  `).join('');
}

function renderAll(){
  renderBestBadge();
  renderTabs();
  renderReadout();
  renderRecallCompare();
  renderNotebookCM();
  renderTunedSection();
  renderExplanations();
  renderTable();
  renderCorrelationHeatmap();
  renderDistributions();
  renderClassDistribution();
  renderBoxPlots();
  renderEthics();
  if(activePage==='resultados' && !pcaRendered) renderPCABoundary();
}

document.querySelectorAll('.page-tab').forEach(btn=>{
  btn.addEventListener('click', ()=>{
    document.querySelectorAll('.page-tab').forEach(b=>b.classList.remove('active'));
    btn.classList.add('active');
    activePage = btn.dataset.page;
    PAGES.forEach(id=>{
      document.getElementById(`page-${id}`).style.display = activePage===id ? 'block' : 'none';
    });
    if(activePage==='resultados' && !pcaRendered) renderPCABoundary();
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

document.getElementById('searchInput').addEventListener('input', e=>{
  searchTerm = e.target.value.trim();
  renderTable();
});

document.addEventListener('click', e=>{
  const th = e.target.closest('th[data-sort]');
  if(!th) return;
  const key = th.dataset.sort;
  if(sortKey===key){ sortAsc=!sortAsc; } else { sortKey=key; sortAsc=true; }
  document.querySelectorAll('th[data-sort]').forEach(el=>el.classList.remove('sorted','asc'));
  th.classList.add('sorted');
  if(sortAsc) th.classList.add('asc');
  renderTable();
});

boot();