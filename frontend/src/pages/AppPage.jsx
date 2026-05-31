import { useEffect, useRef, useState } from 'react';
import { useNavigate }    from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { auth, signOut } from '../firebase';
import useAuth from '../hooks/useAuth';
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
          className="fixed bottom-5 left-1/2 -translate-x-1/2 z-50 flex items-center gap-3 bg-red-800 text-white px-4 sm:px-5 py-3 rounded-xl shadow-xl text-sm w-[calc(100%-2rem)] sm:w-auto max-w-sm sm:max-w-md"
        >
          <span className="shrink-0">&#x2715;</span>
          <span className="flex-1">{msg}</span>
          <button onClick={onClose} className="shrink-0 opacity-70 hover:opacity-100">&#x2715;</button>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

/* Progress bar */
function ProgressBar({ pct, label }) {
  return (
    <div className="bg-bg2 border border-border rounded-xl p-4 sm:p-5 flex flex-col gap-3">
      <div className="h-1.5 bg-bg3 rounded-full overflow-hidden">
        <motion.div className="h-full rounded-full bg-gradient-to-r from-olive to-lime"
          animate={{ width: `${pct}%` }} transition={{ duration: 0.35 }} />
      </div>
      <p className="text-xs text-[#6b7345] text-center">{label}</p>
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
        ${dragging ? 'border-lime bg-lime/5' : 'border-border2 hover:border-lime/60 hover:bg-lime/[0.03]'}
        ${disabled ? 'opacity-40 pointer-events-none' : ''}`}
    >
      <svg className="w-9 h-9 sm:w-10 sm:h-10 text-[#6b7345]" fill="none" stroke="currentColor" strokeWidth={1.5} viewBox="0 0 24 24">
        <rect x="3" y="3" width="18" height="18" rx="2"/>
        <circle cx="8.5" cy="8.5" r="1.5"/>
        <polyline points="21 15 16 10 5 21"/>
      </svg>
      <p className="text-sage text-sm font-medium text-center">
        Drag and drop or <span className="text-lime underline">browse</span>
      </p>
      <p className="text-[#6b7345] text-xs">JPG · PNG · WEBP · max {MAX_MB} MB</p>
      <input ref={inputRef} type="file" accept="image/*" hidden onChange={e => handle(e.target.files[0])} />
    </div>
  );
}

/* Image preview */
function ImagePreview({ file, b64, onClear, infoEl }) {
  const src = b64 || (file ? URL.createObjectURL(file) : null);
  if (!src) return null;
  return (
    <div className="relative rounded-xl overflow-hidden border border-border">
      <img src={src} alt="preview" className="w-full max-h-56 sm:max-h-64 object-cover block" />
      {infoEl && (
        <div className="px-3 py-2 bg-bg3 border-t border-border text-xs text-sage min-h-[32px]">
          {infoEl}
        </div>
      )}
      <button onClick={onClear}
        className="absolute top-2 right-2 bg-black/70 text-white px-2 py-1 rounded text-xs hover:bg-red-700 transition-colors">
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
  const navigate = useNavigate();

  const { user, loading: authLoading } = useAuth();
  const authDone = !authLoading;

  useEffect(() => {
    if (authLoading) return;
    if (!user) navigate('/');
  }, [user, authLoading, navigate]);

  const [srcMode, setSrcMode] = useState('upload');
  const [srcFile, setSrcFile] = useState(null);
  const [srcB64,  setSrcB64]  = useState(null);
  const [srcInfo, setSrcInfo] = useState('');

  const [tgtMode, setTgtMode] = useState('upload');
  const [tgtFile, setTgtFile] = useState(null);
  const [tgtB64,  setTgtB64]  = useState(null);
  const [tgtInfo, setTgtInfo] = useState('');

  const [blend, setBlend] = useState(85);
  const [tone,  setTone]  = useState(90);
  const [hair,  setHair]  = useState(80);
  const [neck,  setNeck]  = useState(75);
  const [swapHairOpt, setSwapHairOpt] = useState(false);
  const [keepGlasses, setKeepGlasses] = useState(false);
  const [showSettings, setShowSettings] = useState(false);

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

  const getToken = () => localStorage.getItem('df_token') || sessionStorage.getItem('df_token') || '';

  const detectFaces = async (file, b64) => {
    if (file) {
      const fd = new FormData();
      fd.append('image', file);
      const r = await fetch('/api/detect', { method: 'POST', body: fd, headers: { Authorization: `Bearer ${getToken()}` } });
      return r.json();
    }
    const r = await fetch('/api/detect', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${getToken()}` },
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
      const s = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, facingMode: 'user' } });
      setTgtStream(s); setTgtCamActive(true);
      if (tgtVideoRef.current) tgtVideoRef.current.srcObject = s;
    } catch { setToast('Camera access denied. Use Upload instead.'); }
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

  const startCamera = async () => {
    try {
      const s = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, facingMode: 'user' } });
      setStream(s); setCamActive(true);
      if (videoRef.current) videoRef.current.srcObject = s;
    } catch { setToast('Camera access denied. Use Upload instead.'); }
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
    fd.append('blend_strength', blend);
    fd.append('tone_match',     tone);
    fd.append('hair_preserve',  hair);
    fd.append('neck_blend',     neck);
    fd.append('swap_hair',      swapHairOpt ? '1' : '0');
    fd.append('keep_glasses',   keepGlasses ? '1' : '0');

    try {
      const resp = await fetch('/api/swap', {
        method: 'POST', body: fd,
        headers: { Authorization: `Bearer ${getToken()}` },
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

  if (!authDone) return (
    <div className="min-h-screen bg-bg flex items-center justify-center">
      <div className="w-8 h-8 border-2 border-forest border-t-lime rounded-full animate-spin" />
    </div>
  );

  return (
    <div className="min-h-screen bg-bg font-sans">

      {/* HEADER */}
      <header className="sticky top-0 z-40 bg-bg2 border-b border-border">
        <div className="max-w-5xl mx-auto px-4 sm:px-6 h-13 sm:h-14 flex items-center justify-between gap-3">
          <div className="flex items-center gap-2 shrink-0">
            <span className="text-lime text-base sm:text-lg">&#11043;</span>
            <span className="text-lime font-extrabold tracking-tight text-sm sm:text-base">DeepFace Studio</span>
          </div>
          <div className="flex items-center gap-2 sm:gap-3 min-w-0">

            {/* Avatar + name pill */}
            <div className="flex items-center gap-2 bg-bg3 border border-border2 rounded-full pl-1 pr-3 py-1">
              {user?.photoURL ? (
                <img
                  src={user.photoURL}
                  alt={user.displayName || ''}
                  referrerPolicy="no-referrer"
                  className="w-7 h-7 sm:w-8 sm:h-8 rounded-full ring-2 ring-olive shrink-0 object-cover"
                />
              ) : (
                <span className="w-7 h-7 sm:w-8 sm:h-8 rounded-full bg-olive flex items-center justify-center text-lime font-bold text-sm shrink-0">
                  {(user?.displayName || user?.email || 'U')[0].toUpperCase()}
                </span>
              )}
              <div className="hidden sm:flex flex-col leading-none">
                <span className="text-lime text-xs font-semibold max-w-[110px] truncate">
                  {user?.displayName || user?.email}
                </span>
                <span className="text-[#6b7345] text-[10px]">Signed in</span>
              </div>
            </div>

            <button
              onClick={async () => { await signOut(auth); localStorage.clear(); sessionStorage.clear(); navigate('/'); }}
              className="flex items-center gap-1.5 text-xs text-red-400 border border-red-900/50 bg-red-950/30 rounded-md px-2.5 sm:px-3 py-1.5 hover:text-red-300 hover:border-red-500/60 hover:bg-red-900/40 transition-colors whitespace-nowrap shrink-0"
            >
              <svg className="w-3.5 h-3.5 shrink-0" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/>
                <polyline points="16 17 21 12 16 7"/>
                <line x1="21" y1="12" x2="9" y2="12"/>
              </svg>
              Logout
            </button>
          </div>
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-4 sm:px-6 py-6 sm:py-8 flex flex-col gap-5 sm:gap-7">

        {/* NOTICE */}
        <div className="flex items-start sm:items-center gap-2 bg-yellow-900/10 border border-yellow-700/30 rounded-lg px-3 sm:px-4 py-2.5 text-yellow-300/80 text-xs leading-relaxed">
          <span className="shrink-0 mt-0.5 sm:mt-0">&#9888;</span>
          <span>Use responsibly. Only process images you own or have explicit consent to use.</span>
        </div>

        {/* INPUT PANELS */}
        <div className="grid grid-cols-1 md:grid-cols-[1fr_32px_1fr] items-start gap-4 md:gap-0">

          {/* SOURCE */}
          <SpotlightCard className="bg-bg2 border border-border rounded-2xl p-4 sm:p-6 flex flex-col gap-4">
            <div>
              <h2 className="text-lime font-bold flex items-center gap-2 text-sm sm:text-base">
                <span className="w-5 h-5 sm:w-6 sm:h-6 bg-gradient-to-br from-olive to-lime rounded-full flex items-center justify-center text-xs font-extrabold text-forest shrink-0">1</span>
                Source Face
              </h2>
              <p className="text-[#6b7345] text-xs mt-1">Your face — upload or use camera</p>
            </div>

            {/* mode toggle */}
            <div className="flex bg-bg3 border border-border rounded-lg p-1 gap-1">
              {['upload', 'camera'].map(m => (
                <button key={m} onClick={() => setSrcMode(m)}
                  className={`flex-1 flex items-center justify-center gap-1.5 sm:gap-2 py-2 rounded-md text-xs sm:text-sm font-semibold transition-all duration-200 ${srcMode === m ? 'bg-olive text-lime shadow' : 'text-[#6b7345] hover:text-sage'}`}>
                  {m === 'upload'
                    ? <svg className="w-3.5 h-3.5 sm:w-4 sm:h-4 shrink-0" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
                    : <svg className="w-3.5 h-3.5 sm:w-4 sm:h-4 shrink-0" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24"><path d="M23 7l-7 5 7 5V7z"/><rect x="1" y="5" width="15" height="14" rx="2" ry="2"/></svg>}
                  {m.charAt(0).toUpperCase() + m.slice(1)}
                </button>
              ))}
            </div>

            {/* upload */}
            {srcMode === 'upload' && !srcFile && !srcB64 && (
              <DropZone onFile={onSrcFile} />
            )}

            {/* camera */}
            {srcMode === 'camera' && !srcB64 && (
              <div className="flex flex-col gap-3">
                <div className="relative rounded-xl overflow-hidden bg-black aspect-[4/3]">
                  <video ref={videoRef} autoPlay playsInline muted className="w-full h-full object-cover" />
                  <canvas ref={canvasRef} className="hidden" />
                  {camActive && (
                    <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                      <div className="w-28 h-28 sm:w-32 sm:h-32 border-2 border-lime/60 rounded-full animate-pulse-ring shadow-[0_0_0_9999px_rgba(0,0,0,0.3)]" />
                    </div>
                  )}
                </div>
                <div className="flex gap-2 justify-center flex-wrap">
                  {!camActive && (
                    <button onClick={startCamera}
                      className="flex items-center gap-2 bg-bg3 border border-border2 text-sage px-4 py-2 rounded-lg text-xs sm:text-sm font-semibold hover:border-lime/50 transition-colors">
                      Start Camera
                    </button>
                  )}
                  {camActive && (
                    <button onClick={capture}
                      className="flex items-center gap-2 bg-lime text-forest px-4 sm:px-5 py-2 rounded-lg text-xs sm:text-sm font-bold hover:bg-[#c8d280] transition-colors">
                      Capture
                    </button>
                  )}
                  {camActive && (
                    <button onClick={stopCamera}
                      className="text-xs sm:text-sm text-[#6b7345] border border-border px-3 py-2 rounded-lg hover:text-sage transition-colors">
                      Cancel
                    </button>
                  )}
                </div>
              </div>
            )}

            {(srcFile || srcB64) && (
              <ImagePreview file={srcFile} b64={srcB64} infoEl={srcInfo}
                onClear={() => { setSrcFile(null); setSrcB64(null); setSrcInfo(''); }} />
            )}
          </SpotlightCard>

          {/* arrow — only on md+ */}
          <div className="hidden md:flex items-center justify-center pt-24 text-[#3D4127]">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
              <line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/>
            </svg>
          </div>

          {/* mobile separator */}
          <div className="md:hidden flex items-center gap-3">
            <div className="flex-1 h-px bg-border" />
            <span className="text-[#6b7345] text-xs font-semibold">THEN</span>
            <div className="flex-1 h-px bg-border" />
          </div>

          {/* TARGET */}
          <SpotlightCard className="bg-bg2 border border-border rounded-2xl p-4 sm:p-6 flex flex-col gap-4">
            <div>
              <h2 className="text-lime font-bold flex items-center gap-2 text-sm sm:text-base">
                <span className="w-5 h-5 sm:w-6 sm:h-6 bg-gradient-to-br from-olive to-lime rounded-full flex items-center justify-center text-xs font-extrabold text-forest shrink-0">2</span>
                Target Face
              </h2>
              <p className="text-[#6b7345] text-xs mt-1">The body or background to swap your face onto</p>
            </div>

            {/* mode toggle */}
            <div className="flex bg-bg3 border border-border rounded-lg p-1 gap-1">
              {['upload', 'camera'].map(m => (
                <button key={m} onClick={() => setTgtMode(m)}
                  className={`flex-1 flex items-center justify-center gap-1.5 sm:gap-2 py-2 rounded-md text-xs sm:text-sm font-semibold transition-all duration-200 ${tgtMode === m ? 'bg-olive text-lime shadow' : 'text-[#6b7345] hover:text-sage'}`}>
                  {m === 'upload'
                    ? <svg className="w-3.5 h-3.5 sm:w-4 sm:h-4 shrink-0" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
                    : <svg className="w-3.5 h-3.5 sm:w-4 sm:h-4 shrink-0" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24"><path d="M23 7l-7 5 7 5V7z"/><rect x="1" y="5" width="15" height="14" rx="2" ry="2"/></svg>}
                  {m.charAt(0).toUpperCase() + m.slice(1)}
                </button>
              ))}
            </div>

            {/* upload */}
            {tgtMode === 'upload' && !tgtFile && !tgtB64 && (
              <DropZone onFile={onTgtFile} />
            )}

            {/* camera */}
            {tgtMode === 'camera' && !tgtB64 && (
              <div className="flex flex-col gap-3">
                <div className="relative rounded-xl overflow-hidden bg-black aspect-[4/3]">
                  <video ref={tgtVideoRef} autoPlay playsInline muted className="w-full h-full object-cover" />
                  <canvas ref={tgtCanvasRef} className="hidden" />
                  {tgtCamActive && (
                    <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                      <div className="w-28 h-28 sm:w-32 sm:h-32 border-2 border-lime/60 rounded-full animate-pulse-ring shadow-[0_0_0_9999px_rgba(0,0,0,0.3)]" />
                    </div>
                  )}
                </div>
                <div className="flex gap-2 justify-center flex-wrap">
                  {!tgtCamActive && (
                    <button onClick={startTgtCamera}
                      className="flex items-center gap-2 bg-bg3 border border-border2 text-sage px-4 py-2 rounded-lg text-xs sm:text-sm font-semibold hover:border-lime/50 transition-colors">
                      Start Camera
                    </button>
                  )}
                  {tgtCamActive && (
                    <button onClick={captureTgt}
                      className="flex items-center gap-2 bg-lime text-forest px-4 sm:px-5 py-2 rounded-lg text-xs sm:text-sm font-bold hover:bg-[#c8d280] transition-colors">
                      Capture
                    </button>
                  )}
                  {tgtCamActive && (
                    <button onClick={stopTgtCamera}
                      className="text-xs sm:text-sm text-[#6b7345] border border-border px-3 py-2 rounded-lg hover:text-sage transition-colors">
                      Cancel
                    </button>
                  )}
                </div>
              </div>
            )}

            {(tgtFile || tgtB64) && (
              <ImagePreview file={tgtFile} b64={tgtB64} infoEl={tgtInfo}
                onClear={() => { setTgtFile(null); setTgtB64(null); setTgtInfo(''); }} />
            )}
          </SpotlightCard>
        </div>

        {/* SETTINGS */}
        <div className="bg-bg2 border border-border rounded-2xl overflow-hidden">
          <button onClick={() => setShowSettings(p => !p)}
            className="w-full flex items-center gap-3 px-4 sm:px-5 py-3.5 sm:py-4 text-sage text-xs sm:text-sm font-semibold hover:text-lime transition-colors">
            <svg className="w-4 h-4 shrink-0" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
              <circle cx="12" cy="12" r="3"/>
              <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/>
            </svg>
            Advanced Settings
            <span className={`ml-auto text-xs transition-transform duration-200 ${showSettings ? 'rotate-180' : ''}`}>&#9662;</span>
          </button>

          <AnimatePresence>
            {showSettings && (
              <motion.div
                initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }}
                transition={{ duration: 0.22 }} className="overflow-hidden">
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-5 px-4 sm:px-5 pb-4 sm:pb-5">
                  {[
                    { label: 'Blend Strength',   val: blend, set: setBlend },
                    { label: 'Skin Tone Match',   val: tone,  set: setTone  },
                    { label: 'Hair Preservation', val: hair,  set: setHair  },
                    { label: 'Neck Blend',        val: neck,  set: setNeck  },
                  ].map(s => (
                    <div key={s.label} className="flex flex-col gap-2">
                      <label className="flex justify-between text-xs text-sage font-medium">
                        {s.label}<span className="text-lime font-bold">{s.val}%</span>
                      </label>
                      <input type="range" min="0" max="100" value={s.val}
                        onChange={e => s.set(+e.target.value)}
                        className="w-full accent-lime h-1 bg-border2 rounded-full cursor-pointer" />
                    </div>
                  ))}
                </div>
                <div className="px-4 sm:px-5 pb-4 sm:pb-5 flex flex-col gap-3">
                  <label className="flex items-center gap-3 cursor-pointer select-none">
                    <input type="checkbox" checked={swapHairOpt}
                      onChange={e => setSwapHairOpt(e.target.checked)}
                      className="w-4 h-4 accent-lime cursor-pointer" />
                    <span className="text-xs text-sage font-medium">
                      Swap hair too <span className="text-lime">(experimental)</span>
                      <span className="block text-[10px] text-sage/60">
                        Transplants the source's hair. Works best with a clear, front-facing source; may look artificial.
                      </span>
                    </span>
                  </label>
                  <label className="flex items-center gap-3 cursor-pointer select-none">
                    <input type="checkbox" checked={keepGlasses}
                      onChange={e => setKeepGlasses(e.target.checked)}
                      className="w-4 h-4 accent-lime cursor-pointer" />
                    <span className="text-xs text-sage font-medium">
                      Keep source glasses <span className="text-lime">(experimental)</span>
                      <span className="block text-[10px] text-sage/60">
                        Carries the source's spectacles onto the swap. Best with a front-facing source; may look slightly pasted.
                      </span>
                    </span>
                  </label>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* SWAP BUTTON */}
        <div className="flex justify-center">
          <button onClick={runSwap}
            disabled={swapping || (!srcFile && !srcB64) || !tgtFile}
            className="w-full sm:w-auto flex items-center justify-center gap-3 bg-gradient-to-r from-olive to-lime text-forest px-8 sm:px-10 py-3.5 sm:py-4 rounded-xl font-bold text-sm sm:text-base hover:-translate-y-0.5 transition-all duration-200 disabled:opacity-40 disabled:cursor-not-allowed disabled:transform-none">
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
              className="flex flex-col gap-5 sm:gap-6">
              <h2 className="text-lg sm:text-xl font-extrabold text-lime text-center">Results</h2>

              {/* 3-panel comparison: 1 col on mobile, 3 on md+ */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3 sm:gap-4">
                {[
                  { label: 'Source',  src: srcFile ? URL.createObjectURL(srcFile) : srcB64 },
                  { label: 'Target',  src: tgtFile ? URL.createObjectURL(tgtFile) : '' },
                  { label: 'Swapped', src: result.result_image, highlight: true },
                ].map(p => (
                  <TiltedCard key={p.label}
                    className={`bg-bg2 border rounded-2xl overflow-hidden ${p.highlight ? 'border-olive shadow-[0_0_30px_rgba(99,107,47,0.3)]' : 'border-border'}`}>
                    <p className={`px-3 sm:px-4 py-2.5 text-xs font-bold uppercase tracking-widest bg-bg3 border-b border-border ${p.highlight ? 'text-lime' : 'text-[#6b7345]'}`}>
                      {p.label}{p.highlight ? ' ✓' : ''}
                    </p>
                    <img src={p.src} alt={p.label} className="w-full max-h-48 sm:max-h-72 object-cover" />
                  </TiltedCard>
                ))}
              </div>

              {/* metrics: 2 cols on mobile, 4 on sm+ */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 sm:gap-3">
                {[
                  { label: 'Alignment',   val: result.quality?.alignment?.toFixed(1) + '/100' },
                  { label: 'Blend',       val: result.quality?.blend?.toFixed(1) + '/100' },
                  { label: 'Colour dE',   val: result.delta_e?.toFixed(2) },
                  { label: 'Naturalness', val: result.quality?.naturalness?.toFixed(1) + '/100' },
                ].map(m => (
                  <div key={m.label} className="bg-bg2 border border-border rounded-xl p-3 sm:p-4 flex flex-col items-center gap-1 text-center">
                    <span className="text-xl sm:text-2xl font-extrabold text-lime">{m.val ?? '-'}</span>
                    <span className="text-[9px] sm:text-[10px] text-[#6b7345] uppercase tracking-widest leading-tight">{m.label}</span>
                  </div>
                ))}
              </div>

              {/* tone chips */}
              {(result.src_tone || result.tgt_tone) && (
                <div className="flex flex-wrap gap-2 justify-center text-xs text-sage">
                  <span className="bg-bg3 border border-border rounded-full px-3 py-1">
                    <strong>Source:</strong> {result.src_tone?.category} {result.src_tone?.undertone}
                  </span>
                  <span className="bg-bg3 border border-border rounded-full px-3 py-1">
                    <strong>Target:</strong> {result.tgt_tone?.category} {result.tgt_tone?.undertone}
                  </span>
                  <span className={`bg-bg3 border border-border rounded-full px-3 py-1 ${result.delta_e < 10 ? 'text-green-400' : result.delta_e < 20 ? 'text-yellow-400' : 'text-red-400'}`}>
                    dE {result.delta_e?.toFixed(2)} {result.delta_e < 10 ? 'Excellent' : result.delta_e < 15 ? 'Good' : 'Moderate'}
                  </span>
                </div>
              )}

              {/* download row: full width on mobile, auto on sm+ */}
              <div className="flex flex-col sm:flex-row gap-3 justify-center">
                <a href={result.output_file ? `/api/download/${result.output_file}` : result.result_image}
                  download="face_swap_result.png"
                  className="flex items-center justify-center gap-2 bg-gradient-to-r from-olive to-lime text-forest px-5 sm:px-6 py-3 rounded-xl font-bold text-sm hover:-translate-y-0.5 transition-all duration-200 w-full sm:w-auto">
                  <svg className="w-4 h-4 shrink-0" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                    <polyline points="7 10 12 15 17 10"/>
                    <line x1="12" y1="15" x2="12" y2="3"/>
                  </svg>
                  Download 4K PNG
                </a>
                <button onClick={reset}
                  className="flex items-center justify-center gap-2 border border-border text-sage px-5 py-3 rounded-xl text-sm font-semibold hover:border-border2 hover:text-lime transition-colors w-full sm:w-auto">
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
