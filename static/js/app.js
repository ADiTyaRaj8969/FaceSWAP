/* -- State ---------------------------------------------------------------- */
const state = {
  sourceMode:    'upload',   // 'upload' | 'camera'
  sourceFile:    null,       // File object (upload mode)
  sourceB64:     null,       // base64 string (camera mode)
  targetFile:    null,       // File object
  cameraStream:  null,
  cameraCaptured: false,
  lastResultB64: null,
  lastResultFilename: null,
};

/* -- Mode switch ---------------------------------------------------------- */
function switchMode(mode) {
  state.sourceMode = mode;

  const btnUpload = document.getElementById('btn-upload-src');
  const btnCamera = document.getElementById('btn-camera-src');
  const zoneUpload = document.getElementById('zone-upload');
  const zoneCam    = document.getElementById('zone-camera');

  if (mode === 'upload') {
    btnUpload.classList.add('active');
    btnCamera.classList.remove('active');
    zoneUpload.classList.remove('hidden');
    zoneCam.classList.add('hidden');
    stopCamera();
  } else {
    btnCamera.classList.add('active');
    btnUpload.classList.remove('active');
    zoneCam.classList.remove('hidden');
    zoneUpload.classList.add('hidden');
  }

  clearSource();
}

/* -- File loading --------------------------------------------------------- */
function loadFile(event, which) {
  const file = event.target.files[0];
  if (!file) return;
  if (which === 'source') setSourceFile(file);
  else setTargetFile(file);
}

function handleDrop(event, which) {
  event.preventDefault();
  dragLeave(event.currentTarget);
  const file = event.dataTransfer.files[0];
  if (!file) return;
  if (which === 'source') setSourceFile(file);
  else setTargetFile(file);
}

function dragEnter(el) { el.classList.add('drag-over'); }
function dragLeave(el) { el.classList.remove('drag-over'); }

function setSourceFile(file) {
  if (!file.type.startsWith('image/')) return showError('Only image files are supported.');
  if (file.size > 50 * 1024 * 1024) return showError('Image exceeds 50 MB.');
  state.sourceFile = file;
  state.sourceB64  = null;
  showPreview('source', file);
  autoDetectFaces('source', file, null);
}

function setTargetFile(file) {
  if (!file.type.startsWith('image/')) return showError('Only image files are supported.');
  if (file.size > 50 * 1024 * 1024) return showError('Image exceeds 50 MB.');
  state.targetFile = file;
  showPreview('target', file);
  autoDetectFaces('target', file, null);
}

function showPreview(which, file) {
  const url = URL.createObjectURL(file);
  document.getElementById(`img-${which}-preview`).src = url;
  document.getElementById(`preview-${which}`).classList.remove('hidden');
  if (which === 'source') {
    document.getElementById('zone-upload').classList.add('hidden');
  } else {
    document.getElementById(`drop-${which}`).classList.add('hidden');
  }
}

function clearSource() {
  state.sourceFile    = null;
  state.sourceB64     = null;
  state.cameraCaptured = false;
  document.getElementById('preview-source').classList.add('hidden');
  document.getElementById('img-source-preview').src = '';
  document.getElementById('src-face-info').textContent = '';
  if (state.sourceMode === 'upload') {
    document.getElementById('zone-upload').classList.remove('hidden');
    document.getElementById('file-source').value = '';
  }
}

function clearTarget() {
  state.targetFile = null;
  document.getElementById('preview-target').classList.add('hidden');
  document.getElementById('img-target-preview').src = '';
  document.getElementById('tgt-face-info').textContent = '';
  document.getElementById('drop-target').classList.remove('hidden');
  document.getElementById('file-target').value = '';
}

/* -- Face auto-detect (quick preview feedback) ---------------------------- */
async function autoDetectFaces(which, file, b64) {
  const infoEl = document.getElementById(`${which === 'source' ? 'src' : 'tgt'}-face-info`);
  infoEl.textContent = 'Detecting faces…';

  const fd = new FormData();
  if (file) fd.append('image', file);

  try {
    let resp, data;
    if (file) {
      resp = await fetch('/api/detect', { method: 'POST', body: fd });
    } else {
      resp = await fetch('/api/detect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: b64 }),
      });
    }
    data = await resp.json();
    if (data.ok) {
      const n = data.faces;
      infoEl.textContent = n > 0
        ? `✓ ${n} face${n > 1 ? 's' : ''} detected`
        : '⚠ No face detected - please try a clearer image';
      infoEl.style.color = n > 0 ? '#10b981' : '#f59e0b';
      // update preview thumbnail with bounding boxes
      document.getElementById(`img-${which}-preview`).src = data.thumbnail;
    } else {
      infoEl.textContent = data.error || 'Detection failed';
    }
  } catch {
    infoEl.textContent = 'Detection unavailable';
  }
}

/* -- Camera --------------------------------------------------------------- */
async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
    });
    state.cameraStream = stream;
    const video = document.getElementById('camera-video');
    video.srcObject = stream;

    document.getElementById('btn-start-cam').classList.add('hidden');
    document.getElementById('btn-capture').classList.remove('hidden');
    document.getElementById('camera-overlay').style.display = 'flex';
  } catch (err) {
    showError('Camera access denied or unavailable. Please use the Upload option.');
    console.error(err);
  }
}

function capturePhoto() {
  const video  = document.getElementById('camera-video');
  const canvas = document.getElementById('camera-canvas');
  canvas.width  = video.videoWidth  || 640;
  canvas.height = video.videoHeight || 480;

  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  const b64 = canvas.toDataURL('image/jpeg', 0.92);
  state.sourceB64     = b64;
  state.sourceFile    = null;
  state.cameraCaptured = true;

  // Show preview
  document.getElementById('img-source-preview').src = b64;
  document.getElementById('preview-source').classList.remove('hidden');
  document.getElementById('btn-capture').classList.add('hidden');
  document.getElementById('btn-retake').classList.remove('hidden');
  document.getElementById('camera-overlay').style.display = 'none';

  stopCamera();
  autoDetectFaces('source', null, b64);
}

function retakePhoto() {
  clearSource();
  document.getElementById('btn-retake').classList.add('hidden');
  document.getElementById('btn-start-cam').classList.remove('hidden');
  document.getElementById('camera-overlay').style.display = 'none';
}

function stopCamera() {
  if (state.cameraStream) {
    state.cameraStream.getTracks().forEach(t => t.stop());
    state.cameraStream = null;
  }
  const video = document.getElementById('camera-video');
  video.srcObject = null;
}

/* -- Settings toggle ------------------------------------------------------ */
function toggleSettings() {
  const grid  = document.getElementById('settings-grid');
  const arrow = document.getElementById('settings-arrow');
  const open  = !grid.classList.contains('hidden');
  grid.classList.toggle('hidden', open);
  arrow.classList.toggle('open', !open);
}

function updateSlider(name) {
  const map = { blend: 'sl-blend', tone: 'sl-tone', hair: 'sl-hair', neck: 'sl-neck' };
  const valMap = { blend: 'val-blend', tone: 'val-tone', hair: 'val-hair', neck: 'val-neck' };
  const v = document.getElementById(map[name]).value;
  document.getElementById(valMap[name]).textContent = v;
}

/* -- Progress simulation --------------------------------------------------- */
const STAGES = [
  [10,  'Detecting faces…'],
  [20,  'Extracting landmarks…'],
  [35,  'Segmenting hair & neck…'],
  [50,  'Running deep swap model…'],
  [65,  'Matching skin tones…'],
  [75,  'Blending hair-to-neck…'],
  [85,  'Applying Laplacian blend…'],
  [93,  'Harmonising colours…'],
  [98,  'Computing quality metrics…'],
];

let _progressTimer = null;
let _stageIdx = 0;

function startProgressSimulation() {
  _stageIdx = 0;
  setProgress(5, 'Initialising pipeline…');
  _progressTimer = setInterval(() => {
    if (_stageIdx < STAGES.length) {
      const [pct, msg] = STAGES[_stageIdx++];
      setProgress(pct, msg);
    }
  }, 1100);
}

function stopProgressSimulation() {
  clearInterval(_progressTimer);
  _progressTimer = null;
}

function setProgress(pct, msg) {
  document.getElementById('progress-fill').style.width = pct + '%';
  document.getElementById('progress-label').textContent = msg;
}

/* -- Swap ----------------------------------------------------------------- */
async function runSwap() {
  // Validate
  const hasSource = state.sourceFile || state.sourceB64;
  if (!hasSource)      return showError('Please provide a source face (upload or camera).');
  if (!state.targetFile) return showError('Please upload a target face image.');

  // UI: disable swap button, show progress
  const btn = document.getElementById('btn-swap');
  btn.disabled = true;
  btn.textContent = 'Processing…';
  document.getElementById('progress-wrap').classList.remove('hidden');
  document.getElementById('results').classList.add('hidden');
  startProgressSimulation();

  try {
    const fd = new FormData();

    if (state.sourceFile) {
      fd.append('source_file', state.sourceFile);
    } else {
      fd.append('source_b64', state.sourceB64);
    }
    fd.append('target_file', state.targetFile);
    fd.append('blend_strength',  document.getElementById('sl-blend').value);
    fd.append('tone_match',      document.getElementById('sl-tone').value);
    fd.append('hair_preserve',   document.getElementById('sl-hair').value);
    fd.append('neck_blend',      document.getElementById('sl-neck').value);

    const resp = await fetch('/api/swap', { method: 'POST', body: fd });
    const data = await resp.json();

    stopProgressSimulation();

    if (!data.ok) {
      showError(data.error || 'Swap failed.');
      resetSwapBtn();
      return;
    }

    setProgress(100, 'Done!');
    setTimeout(() => showResults(data), 400);

  } catch (err) {
    stopProgressSimulation();
    showError('Network error: ' + err.message);
    resetSwapBtn();
  }
}

function resetSwapBtn() {
  const btn = document.getElementById('btn-swap');
  btn.disabled = false;
  btn.innerHTML = `<svg viewBox="0 0 24 24"><polyline points="17 1 21 5 17 9"/><path d="M3 11V9a4 4 0 0 1 4-4h14"/><polyline points="7 23 3 19 7 15"/><path d="M21 13v2a4 4 0 0 1-4 4H3"/></svg> Swap Faces`;
  document.getElementById('progress-wrap').classList.add('hidden');
}

/* -- Show results --------------------------------------------------------- */
function showResults(data) {
  // Images
  const srcURL = state.sourceFile
    ? URL.createObjectURL(state.sourceFile)
    : state.sourceB64;
  document.getElementById('res-source').src  = srcURL;
  document.getElementById('res-target').src  = URL.createObjectURL(state.targetFile);
  document.getElementById('res-result').src  = data.result_image;

  // Quality metrics
  const q = data.quality || {};
  setMetric('m-align',   q.alignment   != null ? q.alignment.toFixed(1) + '/100'  : '-');
  setMetric('m-blend',   q.blend       != null ? q.blend.toFixed(1)     + '/100'  : '-');
  setMetric('m-de',      data.delta_e  != null ? data.delta_e.toFixed(2)           : '-');
  setMetric('m-natural', q.naturalness != null ? q.naturalness.toFixed(1)+ '/100' : '-');

  // Tone chips
  const st = data.src_tone || {};
  const tt = data.tgt_tone || {};
  const de = data.delta_e  || 0;
  document.getElementById('tone-row').innerHTML = `
    <span class="tone-chip"><strong>Source:</strong> ${st.category || '-'} · ${st.undertone || '-'}</span>
    <span class="tone-chip"><strong>Target:</strong> ${tt.category || '-'} · ${tt.undertone || '-'}</span>
    <span class="tone-chip" style="color:${de < 10 ? '#10b981' : de < 20 ? '#f59e0b' : '#ef4444'}">
      <strong>ΔE</strong> ${de.toFixed(2)} - ${de < 10 ? 'Excellent match' : de < 15 ? 'Good match' : de < 20 ? 'Moderate mismatch' : 'High mismatch'}
    </span>`;

  // Download link
  state.lastResultB64 = data.result_image;
  const downloadBtn = document.getElementById('btn-download');
  downloadBtn.href = data.result_image;
  downloadBtn.download = 'face_swap_result.png';

  // Show section
  document.getElementById('results').classList.remove('hidden');

  // Restore swap btn
  resetSwapBtn();

  // Scroll to results
  setTimeout(() => {
    document.getElementById('results').scrollIntoView({ behavior: 'smooth', block: 'start' });
  }, 100);
}

function setMetric(id, value) {
  document.querySelector(`#${id} .metric-card__value`).textContent = value;
}

/* -- Reset ---------------------------------------------------------------- */
function resetAll() {
  clearSource();
  clearTarget();
  document.getElementById('results').classList.add('hidden');
  document.getElementById('progress-wrap').classList.add('hidden');
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

/* -- Error toast ---------------------------------------------------------- */
let _toastTimer = null;
function showError(msg) {
  const toast = document.getElementById('toast');
  document.getElementById('toast-msg').textContent = msg;
  toast.classList.remove('hidden');
  clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => toast.classList.add('hidden'), 5000);
}
