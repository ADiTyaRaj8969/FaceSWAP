import { useEffect, useRef, useState } from 'react';
import { useNavigate, Link }    from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import TiltedCard    from '../components/ui/TiltedCard';
import SpotlightCard from '../components/ui/SpotlightCard';

const MAX_MB    = 50;
const MAX_BYTES = MAX_MB * 1024 * 1024;

/* Toast */
function Toast({ msg, onClose }) {
  return (
    <AnimatePresence>
      {msg && (
        <motion.div
          initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
          className="fixed bottom-5 left-1/2 -translate-x-1/2 z-50 flex items-center gap-3 bg-red-600 text-white px-4 sm:px-5 py-3 rounded-xl shadow-xl text-sm w-[calc(100%-2rem)] sm:w-auto max-w-sm sm:max-w-md"
        >
          <span className="shrink-0">&#x2715;</span>
          <span className="flex-1 font-medium">{msg}</span>
          <button onClick={onClose} className="shrink-0 opacity-70 hover:opacity-100 transition-opacity">&#x2715;</button>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

/* Progress bar */
function ProgressBar({ pct, label }) {
  return (
    <div className="bg-white border border-border shadow-sm rounded-xl p-4 sm:p-5 flex flex-col gap-3">
      <div className="h-1.5 bg-bg3 rounded-full overflow-hidden">
        <motion.div className="h-full rounded-full bg-teal"
          animate={{ width: `${pct}%` }} transition={{ duration: 0.35 }} />
      </div>
      <p className="text-xs text-navy-light text-center font-medium">{label}</p>
    </div>
  );
}

/* Drop zone */
function DropZone({ onFile, disabled }) {
  const inputRef = useRef();
  const [dragging, setDragging] = useState(false);

  const handle = (file) => {
    if (!file) return;
    if (!file.type.startsWith('image/')) return;
    if (file.size > MAX_BYTES) { alert(`Max ${MAX_MB} MB`); return; }
    onFile(file);
  };

  return (
    <div
      onClick={() => !disabled && inputRef.current?.click()}
      onDrop={e => { e.preventDefault(); setDragging(false); handle(e.dataTransfer.files[0]); }}
      onDragOver={e => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      className={`border-2 border-dashed rounded-xl p-6 sm:p-10 flex flex-col items-center gap-3 cursor-pointer transition-all duration-200
        ${dragging ? 'border-teal bg-teal/5' : 'border-border2 hover:border-teal/50 hover:bg-teal/[0.02]'}
        ${disabled ? 'opacity-40 pointer-events-none' : ''}`}
    >
      <svg className="w-9 h-9 sm:w-10 sm:h-10 text-slate" fill="none" stroke="currentColor" strokeWidth={1.5} viewBox="0 0 24 24">
        <rect x="3" y="3" width="18" height="18" rx="2"/>
        <circle cx="8.5" cy="8.5" r="1.5"/>
        <polyline points="21 15 16 10 5 21"/>
      </svg>
      <p className="text-navy text-sm font-medium text-center">
        Drag and drop or <span className="text-teal underline">browse</span>
      </p>
      <p className="text-slate text-xs">JPG · PNG · WEBP · max {MAX_MB} MB</p>
      <input ref={inputRef} type="file" accept="image/*" hidden onChange={e => handle(e.target.files[0])} />
    </div>
  );
}

/* Image preview */
function ImagePreview({ file, b64, onClear, infoEl }) {
  const src = b64 || (file ? URL.createObjectURL(file) : null);
  if (!src) return null;
  return (
    <div className="relative rounded-xl overflow-hidden border border-border bg-bg3">
      <img src={src} alt="preview" className="w-full max-h-72 sm:max-h-96 object-contain block mx-auto bg-white" />
      {infoEl && (
        <div className="px-3 py-2 bg-white border-t border-border text-xs text-navy-light font-medium min-h-[32px] text-center">
          {infoEl}
        </div>
      )}
      <button onClick={onClear}
        className="absolute top-2 right-2 bg-white/90 backdrop-blur-sm text-red-600 font-medium px-2.5 py-1 rounded shadow text-xs hover:bg-red-50 hover:text-red-700 transition-colors">
        Clear
      </button>
    </div>
  );
}

const STAGES = [
  [10,'Detecting faces...'], [20,'Extracting landmarks...'], [35,'Segmenting hair and neck...'],
  [50,'Running deep swap model...'], [65,'Matching skin tones...'], [75,'Blending hair to neck...'],
  [85,'Applying Laplacian blend...'], [93,'Harmonising colours...'], [98,'Quality metrics...'],
];

export default function AppPage() {
  const [srcMode, setSrcMode] = useState('upload');
  const [srcFile, setSrcFile] = useState(null);
  const [srcB64,  setSrcB64]  = useState(null);
  const [srcInfo, setSrcInfo] = useState('');

  const [tgtMode, setTgtMode] = useState('upload');
  const [tgtFile, setTgtFile] = useState(null);
  const [tgtB64,  setTgtB64]  = useState(null);
  const [tgtInfo, setTgtInfo] = useState('');

  const videoRef    = useRef(null);
  const canvasRef   = useRef(null);
  const tgtVideoRef = useRef(null);
  const tgtCanvasRef = useRef(null);
  const [stream,       setStream]       = useState(null);
  const [camActive,    setCamActive]    = useState(false);
  const [tgtStream,    setTgtStream]    = useState(null);
  const [tgtCamActive, setTgtCamActive] = useState(false);

  const [swapping, setSwapping] = useState(false);
  const [progress, setProgress] = useState(0);
  const [pLabel,   setPLabel]   = useState('');
  const [result,   setResult]   = useState(null);
  const [toast,    setToast]    = useState('');

  const detectFaces = async (file, b64) => {
    if (file) {
      const fd = new FormData();
      fd.append('image', file);
      const r = await fetch('/api/detect', { method: 'POST', body: fd });
      return r.json();
    }
    const r = await fetch('/api/detect', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: b64 }),
    });
    return r.json();
  };

  const onSrcFile = async (file) => {
    setSrcFile(file); setSrcB64(null); setSrcInfo('Detecting...');
    try { const d = await detectFaces(file, null); setSrcInfo(d.faces > 0 ? `${d.faces} face(s) detected` : 'No face detected'); }
    catch { setSrcInfo(''); }
  };

  const onTgtFile = async (file) => {
    setTgtFile(file); setTgtB64(null); setTgtInfo('Detecting...');
    try { const d = await detectFaces(file, null); setTgtInfo(d.faces > 0 ? `${d.faces} face(s) detected` : 'No face detected'); }
    catch { setTgtInfo(''); }
  };

  const startTgtCamera = async () => {
    try {
      const s = await _getCamStream();
      setTgtStream(s); setTgtCamActive(true);
    } catch (e) { setToast(_camError(e)); }
  };

  const captureTgt = async () => {
    const v = tgtVideoRef.current, c = tgtCanvasRef.current;
    if (!v || !c) return;
    c.width = v.videoWidth; c.height = v.videoHeight;
    c.getContext('2d').drawImage(v, 0, 0);
    const b64 = c.toDataURL('image/jpeg', 0.92);
    setTgtB64(b64); setTgtFile(null); setTgtCamActive(false);
    tgtStream?.getTracks().forEach(t => t.stop()); setTgtStream(null);
    setTgtInfo('Detecting...');
    try { const d = await detectFaces(null, b64); setTgtInfo(d.faces > 0 ? `${d.faces} face(s) detected` : 'No face detected'); }
    catch { setTgtInfo(''); }
  };

  const stopTgtCamera = () => { tgtStream?.getTracks().forEach(t => t.stop()); setTgtStream(null); setTgtCamActive(false); };

  useEffect(() => {
    if (camActive && stream && videoRef.current) {
      videoRef.current.srcObject = stream;
      videoRef.current.play?.().catch(() => {});
    }
  }, [camActive, stream]);

  useEffect(() => {
    if (tgtCamActive && tgtStream && tgtVideoRef.current) {
      tgtVideoRef.current.srcObject = tgtStream;
      tgtVideoRef.current.play?.().catch(() => {});
    }
  }, [tgtCamActive, tgtStream]);

  const _camError = (e) => {
    if (!navigator.mediaDevices?.getUserMedia)
      return 'Camera not available here (needs HTTPS or localhost). Use Upload.';
    switch (e?.name) {
      case 'NotAllowedError':  return 'Camera permission blocked — click the camera icon in the address bar, Allow, then retry. Or use Upload.';
      case 'NotFoundError':    return 'No camera found — use Upload instead.';
      case 'NotReadableError': return 'Camera is busy in another app — close it and retry.';
      default:                 return `Camera error (${e?.name || 'unknown'}) — use Upload instead.`;
    }
  };

  const _getCamStream = async () => {
    try {
      return await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, facingMode: 'user' } });
    } catch (e) {
      if (e?.name === 'NotAllowedError') throw e;
      return await navigator.mediaDevices.getUserMedia({ video: true });
    }
  };

  const startCamera = async () => {
    try {
      const s = await _getCamStream();
      setStream(s); setCamActive(true);
    } catch (e) { setToast(_camError(e)); }
  };

  const capture = async () => {
    const v = videoRef.current, c = canvasRef.current;
    if (!v || !c) return;
    c.width = v.videoWidth; c.height = v.videoHeight;
    c.getContext('2d').drawImage(v, 0, 0);
    const b64 = c.toDataURL('image/jpeg', 0.92);
    setSrcB64(b64); setSrcFile(null); setCamActive(false);
    stream?.getTracks().forEach(t => t.stop()); setStream(null);
    setSrcInfo('Detecting...');
    try { const d = await detectFaces(null, b64); setSrcInfo(d.faces > 0 ? `${d.faces} face(s) detected` : 'No face detected'); }
    catch { setSrcInfo(''); }
  };

  const stopCamera = () => { stream?.getTracks().forEach(t => t.stop()); setStream(null); setCamActive(false); };

  const runSwap = async () => {
    if (!srcFile && !srcB64)       return setToast('Please provide a source face.');
    if (!tgtFile && !tgtB64)       return setToast('Please provide a target face.');
    setSwapping(true); setResult(null); setProgress(5); setPLabel('Initialising...');

    let idx = 0;
    const timer = setInterval(() => {
      if (idx < STAGES.length) { const [p,l] = STAGES[idx++]; setProgress(p); setPLabel(l); }
    }, 1100);

    const fd = new FormData();
    if (srcFile) fd.append('source_file', srcFile); else fd.append('source_b64', srcB64);
    if (tgtFile) fd.append('target_file', tgtFile); else fd.append('target_b64', tgtB64);

    try {
      const resp = await fetch('/api/swap', {
        method: 'POST', body: fd
      });
      const text = await resp.text();
      let data;
      try { data = JSON.parse(text); }
      catch { throw new Error(`Server error ${resp.status}: ${text.slice(0, 120)}`); }
      clearInterval(timer);
      if (!data.ok) { setToast(data.error || 'Swap failed.'); setSwapping(false); return; }
      setProgress(100); setPLabel('Done!');
      setTimeout(() => { setResult(data); setSwapping(false); }, 400);
    } catch (e) {
      clearInterval(timer);
      setToast('Network error: ' + e.message);
      setSwapping(false);
    }
  };

  const reset = () => {
    setSrcFile(null); setSrcB64(null); setSrcInfo('');
    setTgtFile(null); setTgtB64(null);  setTgtInfo('');
    setResult(null); setProgress(0);
    stopCamera(); stopTgtCamera();
  };

  return (
    <div className="min-h-screen bg-bg font-sans text-navy">

      {/* HEADER — using ict_logo.png colored version since background is light */}
      <header className="sticky top-0 z-40 glass border-b border-border shadow-sm">
        <div className="max-w-5xl mx-auto px-4 sm:px-6 h-14 sm:h-16 flex items-center justify-between gap-3">
          <Link to="/" className="flex items-center gap-2.5 shrink-0 hover:opacity-80 transition-opacity">
            <img src="/logos/ict_logo.png" alt="ICT" className="h-8 sm:h-10 w-auto object-contain" />
            <div className="hidden sm:block w-px h-6 bg-border" />
            <div className="hidden sm:flex flex-col leading-none">
              <span className="text-navy font-extrabold tracking-tight text-sm">DeepFace Studio</span>
              <span className="text-slate text-[9px] tracking-wider uppercase font-medium">Marwadi University</span>
            </div>
          </Link>
          <div className="flex items-center gap-2 sm:gap-3 min-w-0">
            <button onClick={() => navigate('/')}
              className="flex items-center gap-1.5 text-xs text-navy border border-border bg-white rounded-md px-2.5 sm:px-3 py-1.5 hover:bg-bg3 transition-colors whitespace-nowrap shrink-0 font-medium shadow-sm">
              <svg className="w-3.5 h-3.5 shrink-0" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>
                <polyline points="9 22 9 12 15 12 15 22"/>
              </svg>
              Home
            </button>
          </div>
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-4 sm:px-6 py-6 sm:py-8 flex flex-col gap-5 sm:gap-7">

        {/* NOTICE */}
        <div className="flex items-start sm:items-center gap-2 bg-yellow-50 border border-yellow-200 rounded-lg px-3 sm:px-4 py-2.5 text-yellow-800 text-xs font-medium leading-relaxed">
          <span className="shrink-0 mt-0.5 sm:mt-0 text-yellow-600">&#9888;</span>
          <span>Use responsibly. Only process images you own or have explicit consent to use.</span>
        </div>

        {/* INPUT PANELS */}
        <div className="grid grid-cols-1 md:grid-cols-[1fr_32px_1fr] items-start gap-4 md:gap-0">

          {/* SOURCE */}
          <SpotlightCard className="bg-white border border-border shadow-sm rounded-2xl p-4 sm:p-6 flex flex-col gap-4">
            <div>
              <h2 className="text-navy font-bold flex items-center gap-2 text-sm sm:text-base">
                <span className="w-5 h-5 sm:w-6 sm:h-6 bg-teal rounded-full flex items-center justify-center text-xs font-extrabold text-white shrink-0 shadow-sm">1</span>
                Source Face
              </h2>
              <p className="text-slate text-xs mt-1 font-medium">Your face — upload or use camera</p>
            </div>

            {/* mode toggle */}
            <div className="flex bg-bg3 border border-border rounded-lg p-1 gap-1">
              {['upload', 'camera'].map(m => (
                <button key={m} onClick={() => setSrcMode(m)}
                  className={`flex-1 flex items-center justify-center gap-1.5 sm:gap-2 py-2 rounded-md text-xs sm:text-sm font-semibold transition-all duration-200 ${srcMode === m ? 'bg-white text-teal shadow-sm border border-border/50' : 'text-slate hover:text-navy'}`}>
                  {m === 'upload'
                    ? <svg className="w-3.5 h-3.5 sm:w-4 sm:h-4 shrink-0" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
                    : <svg className="w-3.5 h-3.5 sm:w-4 sm:h-4 shrink-0" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24"><path d="M23 7l-7 5 7 5V7z"/><rect x="1" y="5" width="15" height="14" rx="2" ry="2"/></svg>}
                  {m.charAt(0).toUpperCase() + m.slice(1)}
                </button>
              ))}
            </div>

            {srcMode === 'upload' && !srcFile && !srcB64 && <DropZone onFile={onSrcFile} />}

            {srcMode === 'camera' && !srcB64 && (
              <div className="flex flex-col gap-3">
                <div className="relative rounded-xl overflow-hidden bg-black aspect-[4/3] shadow-inner">
                  <video ref={videoRef} autoPlay playsInline muted className="w-full h-full object-cover" />
                  <canvas ref={canvasRef} className="hidden" />
                  {camActive && (
                    <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                      <div className="w-28 h-28 sm:w-32 sm:h-32 border-2 border-teal/60 rounded-full animate-pulse-ring shadow-[0_0_0_9999px_rgba(0,0,0,0.3)]" />
                    </div>
                  )}
                </div>
                <div className="flex gap-2 justify-center flex-wrap">
                  {!camActive && (
                    <button onClick={startCamera} className="flex items-center gap-2 bg-white border border-border text-navy shadow-sm px-4 py-2 rounded-lg text-xs sm:text-sm font-semibold hover:border-teal/50 transition-colors">
                      Start Camera
                    </button>
                  )}
                  {camActive && (
                    <button onClick={capture} className="flex items-center gap-2 bg-teal text-white px-4 sm:px-5 py-2 rounded-lg text-xs sm:text-sm font-bold shadow hover:bg-teal-light transition-colors">
                      Capture
                    </button>
                  )}
                  {camActive && (
                    <button onClick={stopCamera} className="text-xs sm:text-sm text-slate border border-border bg-white shadow-sm px-3 py-2 rounded-lg hover:text-navy transition-colors font-medium">
                      Cancel
                    </button>
                  )}
                </div>
              </div>
            )}

            {(srcFile || srcB64) && (
              <ImagePreview file={srcFile} b64={srcB64} infoEl={srcInfo} onClear={() => { setSrcFile(null); setSrcB64(null); setSrcInfo(''); }} />
            )}
          </SpotlightCard>

          {/* arrow */}
          <div className="hidden md:flex items-center justify-center pt-24 text-border2">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24"><line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/></svg>
          </div>

          <div className="md:hidden flex items-center gap-3">
            <div className="flex-1 h-px bg-border" />
            <span className="text-slate text-xs font-bold tracking-widest">THEN</span>
            <div className="flex-1 h-px bg-border" />
          </div>

          {/* TARGET */}
          <SpotlightCard className="bg-white border border-border shadow-sm rounded-2xl p-4 sm:p-6 flex flex-col gap-4">
            <div>
              <h2 className="text-navy font-bold flex items-center gap-2 text-sm sm:text-base">
                <span className="w-5 h-5 sm:w-6 sm:h-6 bg-teal rounded-full flex items-center justify-center text-xs font-extrabold text-white shrink-0 shadow-sm">2</span>
                Target Face
              </h2>
              <p className="text-slate text-xs mt-1 font-medium">The body or background to swap your face onto</p>
            </div>

            {/* mode toggle */}
            <div className="flex bg-bg3 border border-border rounded-lg p-1 gap-1">
              {['upload', 'camera'].map(m => (
                <button key={m} onClick={() => setTgtMode(m)}
                  className={`flex-1 flex items-center justify-center gap-1.5 sm:gap-2 py-2 rounded-md text-xs sm:text-sm font-semibold transition-all duration-200 ${tgtMode === m ? 'bg-white text-teal shadow-sm border border-border/50' : 'text-slate hover:text-navy'}`}>
                  {m === 'upload'
                    ? <svg className="w-3.5 h-3.5 sm:w-4 sm:h-4 shrink-0" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
                    : <svg className="w-3.5 h-3.5 sm:w-4 sm:h-4 shrink-0" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24"><path d="M23 7l-7 5 7 5V7z"/><rect x="1" y="5" width="15" height="14" rx="2" ry="2"/></svg>}
                  {m.charAt(0).toUpperCase() + m.slice(1)}
                </button>
              ))}
            </div>

            {tgtMode === 'upload' && !tgtFile && !tgtB64 && <DropZone onFile={onTgtFile} />}

            {tgtMode === 'camera' && !tgtB64 && (
              <div className="flex flex-col gap-3">
                <div className="relative rounded-xl overflow-hidden bg-black aspect-[4/3] shadow-inner">
                  <video ref={tgtVideoRef} autoPlay playsInline muted className="w-full h-full object-cover" />
                  <canvas ref={tgtCanvasRef} className="hidden" />
                  {tgtCamActive && (
                    <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                      <div className="w-28 h-28 sm:w-32 sm:h-32 border-2 border-teal/60 rounded-full animate-pulse-ring shadow-[0_0_0_9999px_rgba(0,0,0,0.3)]" />
                    </div>
                  )}
                </div>
                <div className="flex gap-2 justify-center flex-wrap">
                  {!tgtCamActive && (
                    <button onClick={startTgtCamera} className="flex items-center gap-2 bg-white border border-border text-navy shadow-sm px-4 py-2 rounded-lg text-xs sm:text-sm font-semibold hover:border-teal/50 transition-colors">
                      Start Camera
                    </button>
                  )}
                  {tgtCamActive && (
                    <button onClick={captureTgt} className="flex items-center gap-2 bg-teal text-white px-4 sm:px-5 py-2 rounded-lg text-xs sm:text-sm font-bold shadow hover:bg-teal-light transition-colors">
                      Capture
                    </button>
                  )}
                  {tgtCamActive && (
                    <button onClick={stopTgtCamera} className="text-xs sm:text-sm text-slate border border-border bg-white shadow-sm px-3 py-2 rounded-lg hover:text-navy transition-colors font-medium">
                      Cancel
                    </button>
                  )}
                </div>
              </div>
            )}

            {(tgtFile || tgtB64) && (
              <ImagePreview file={tgtFile} b64={tgtB64} infoEl={tgtInfo} onClear={() => { setTgtFile(null); setTgtB64(null); setTgtInfo(''); }} />
            )}
          </SpotlightCard>
        </div>


        {/* SWAP BUTTON */}
        <div className="flex justify-center">
          <button onClick={runSwap}
            disabled={swapping || (!srcFile && !srcB64) || !tgtFile}
            className="w-full sm:w-auto flex items-center justify-center gap-3 bg-teal text-white px-8 sm:px-10 py-3.5 sm:py-4 rounded-xl font-bold text-sm sm:text-base shadow-lg shadow-teal/20 hover:shadow-xl hover:shadow-teal/30 hover:-translate-y-0.5 active:scale-95 transition-all duration-200 disabled:opacity-40 disabled:cursor-not-allowed disabled:transform-none disabled:shadow-none">
            <svg className="w-4 h-4 sm:w-5 sm:h-5 shrink-0" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
              <polyline points="17 1 21 5 17 9"/><path d="M3 11V9a4 4 0 0 1 4-4h14"/>
              <polyline points="7 23 3 19 7 15"/><path d="M21 13v2a4 4 0 0 1-4 4H3"/>
            </svg>
            {swapping ? 'Swapping...' : 'Swap Faces'}
          </button>
        </div>

        {/* PROGRESS */}
        {swapping && <ProgressBar pct={progress} label={pLabel} />}

        {/* RESULTS */}
        <AnimatePresence>
          {result && (
            <motion.section initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
              className="flex flex-col gap-5 sm:gap-6 mt-4 pt-8 border-t border-border">
              <h2 className="text-lg sm:text-xl font-extrabold text-navy text-center">Results</h2>

              {/* 3-panel comparison */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3 sm:gap-4">
                {[
                  { label: 'Source',  src: srcFile ? URL.createObjectURL(srcFile) : srcB64 },
                  { label: 'Target',  src: tgtFile ? URL.createObjectURL(tgtFile) : '' },
                  { label: 'Swapped', src: result.result_image, highlight: true },
                ].map(p => (
                  <TiltedCard key={p.label}
                    className={`bg-white border shadow-sm rounded-2xl overflow-hidden ${p.highlight ? 'border-teal ring-2 ring-teal/20' : 'border-border'}`}>
                    <p className={`px-3 sm:px-4 py-2.5 text-xs font-bold uppercase tracking-widest bg-bg3 border-b border-border ${p.highlight ? 'text-teal' : 'text-slate'}`}>
                      {p.label}{p.highlight ? ' ✓' : ''}
                    </p>
                    <img src={p.src} alt={p.label} className="w-full max-h-56 sm:max-h-80 object-contain bg-bg" />
                  </TiltedCard>
                ))}
              </div>

              {/* metrics */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 sm:gap-3">
                {[
                  { label: 'Alignment',   val: result.quality?.alignment?.toFixed(1) + '/100' },
                  { label: 'Blend',       val: result.quality?.blend?.toFixed(1) + '/100' },
                  { label: 'Colour dE',   val: result.delta_e?.toFixed(2) },
                  { label: 'Naturalness', val: result.quality?.naturalness?.toFixed(1) + '/100' },
                ].map(m => (
                  <div key={m.label} className="bg-white border border-border shadow-sm rounded-xl p-3 sm:p-4 flex flex-col items-center gap-1 text-center">
                    <span className="text-xl sm:text-2xl font-extrabold text-teal">{m.val ?? '-'}</span>
                    <span className="text-[9px] sm:text-[10px] text-slate uppercase tracking-widest font-bold leading-tight">{m.label}</span>
                  </div>
                ))}
              </div>

              {/* download row */}
              <div className="flex flex-col sm:flex-row gap-3 justify-center mt-2">
                <a href={result.download_image || result.result_image}
                  download="face_swap_4k.jpg" target="_blank" rel="noreferrer"
                  className="flex items-center justify-center gap-2 bg-teal text-white px-5 sm:px-6 py-3 rounded-xl font-bold text-sm shadow-md hover:bg-teal-light hover:-translate-y-0.5 transition-all duration-200 w-full sm:w-auto">
                  <svg className="w-4 h-4 shrink-0" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                    <polyline points="7 10 12 15 17 10"/>
                    <line x1="12" y1="15" x2="12" y2="3"/>
                  </svg>
                  Download 4K
                </a>
                <button onClick={reset}
                  className="flex items-center justify-center gap-2 bg-white border border-border shadow-sm text-navy px-5 py-3 rounded-xl text-sm font-semibold hover:border-teal/40 hover:text-teal transition-colors w-full sm:w-auto">
                  <svg className="w-4 h-4 shrink-0" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                    <polyline points="1 4 1 10 7 10"/>
                    <path d="M3.51 15a9 9 0 1 0 .49-3.96"/>
                  </svg>
                  New Swap
                </button>
              </div>
            </motion.section>
          )}
        </AnimatePresence>

      </main>

      <Toast msg={toast} onClose={() => setToast('')} />
    </div>
  );
}
