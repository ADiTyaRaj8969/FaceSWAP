import { useEffect, useRef, useState, useMemo } from 'react';
import { useNavigate }           from 'react-router-dom';
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
    <div className="bg-white border border-border/60 shadow-sm rounded-2xl p-5 sm:p-6 flex flex-col items-center gap-4">
      <div className="flex items-center gap-3">
        <svg className="w-5 h-5 text-teal animate-spin shrink-0" fill="none" viewBox="0 0 24 24">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
          <path className="opacity-90" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4z" />
        </svg>
        <span className="text-sm font-bold text-navy">{label}</span>
      </div>
      <div className="w-full h-2 bg-bg3 rounded-full overflow-hidden">
        <motion.div className="h-full rounded-full bg-gradient-to-r from-teal to-teal-light"
          animate={{ width: `${pct}%` }} transition={{ duration: 0.35 }} />
      </div>
      <span className="text-[11px] text-slate font-semibold tracking-wide">{Math.round(pct)}%</span>
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
      className={`border-2 border-dashed rounded-xl p-5 sm:p-6 flex flex-col items-center gap-2 cursor-pointer transition-all duration-200
        ${dragging ? 'border-teal bg-teal/5' : 'border-border2 hover:border-teal/50 hover:bg-teal/[0.02]'}
        ${disabled ? 'opacity-40 pointer-events-none' : ''}`}
    >
      <svg className="w-8 h-8 text-slate" fill="none" stroke="currentColor" strokeWidth={1.5} viewBox="0 0 24 24">
        <rect x="3" y="3" width="18" height="18" rx="2"/>
        <circle cx="8.5" cy="8.5" r="1.5"/>
        <polyline points="21 15 16 10 5 21"/>
      </svg>
      <p className="text-navy text-sm font-medium text-center">
        Drag and drop or <span className="text-teal underline">browse</span>
      </p>
      <p className="text-slate text-[11px]">JPG · PNG · WEBP · max {MAX_MB} MB</p>
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
      <img src={src} alt="preview" className="w-full max-h-48 sm:max-h-56 object-contain block mx-auto bg-white" />
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

// Friendly, non-technical loading messages (no internal pipeline jargon).
const STAGES = [
  [12, 'Analysing your photo...'],
  [28, 'Preparing your campus look...'],
  [45, 'Placing you in the scene...'],
  [62, 'Blending everything naturally...'],
  [78, 'Enhancing the details...'],
  [90, 'Adding the final touches...'],
  [97, 'Almost ready...'],
];


// A unique, gently-motivating message for every location (keyed by the clean
// label). Each weaves in the person's name and nudges toward Marwadi University.
const LOCATION_MESSAGES = {
  'Marwadi University':       (n) => `${n}, standing tall at Marwadi University looks good on you! 🎓 This could be your campus every single morning. Your journey starts right here.`,
  'ICT Department':           (n) => `${n}, you fit right into the ICT Department! 💻 Imagine building the future from here. Marwadi University is ready to welcome you.`,
  'Library':                  (n) => `${n}, the Library suits you. 📚 A quiet corner, big dreams, endless books. Picture yourself studying here at Marwadi University.`,
  'Library G Floor':          (n) => `${n}, the Library looks like your kind of place. 📖 Room to think, room to grow. Your study spot at Marwadi University is waiting.`,
  'Electronic Circuit Lab':   (n) => `${n}, you look right at home in the Electronic Circuit Lab! ⚡ Hands-on learning like this is waiting for you at Marwadi University.`,
  'Embedded System Lab':      (n) => `${n}, the Embedded System Lab is calling! 🔧 Design real systems from scratch. Your seat at Marwadi University is ready.`,
  'Data Science And AI Lab':  (n) => `${n}, you belong in the Data Science & AI Lab! 🤖 Train models, shape tomorrow. Start your journey at Marwadi University.`,
  'Ideation Lab':             (n) => `${n}, the Ideation Lab fits your spark! 💡 This is where ideas become startups. Come build yours at Marwadi University.`,
  'IoT Lab':                  (n) => `${n}, you look great in the IoT Lab! 📡 Connect the whole world from here. Marwadi University is ready for you.`,
  'VLSI Lab':                 (n) => `${n}, the VLSI Lab suits you perfectly! 🔬 Design the chips that power the future at Marwadi University.`,
  'Web Development Lab':      (n) => `${n}, you fit right into the Web Development Lab! 🌐 Build the web of tomorrow. Your place at Marwadi University awaits.`,
  'Project Lab':              (n) => `${n}, the Project Lab looks like your kind of place! 🛠️ Turn bold ideas into reality here at Marwadi University.`,
  'Programming Lab':          (n) => `${n}, you belong in the Programming Lab! 👨‍💻 Code your future one line at a time. Marwadi University is ready to welcome you.`,
  'MUIIR':                    (n) => `${n}, MUIIR fits your ambition! 🚀 Innovation and incubation live here. Bring your ideas to Marwadi University.`,
  'Classroom':                (n) => `${n}, you look right at home in the Classroom! 📖 Learn from the very best at Marwadi University.`,
  'Sports Ground':            (n) => `${n}, the Sports Ground suits your energy! 🏅 Play, compete, belong. This could be your campus at Marwadi University.`,
  'Field':                    (n) => `${n}, you look great out on the Field! 🌳 Campus life is so much more than classes. Come live it at Marwadi University.`,
  'Music Room':               (n) => `${n}, the Music Room fits your rhythm! 🎶 Find your beat on campus at Marwadi University.`,
  'Ground':                   (n) => `${n}, you own the Ground! 🌟 From right here, every dream feels reachable. Start yours at Marwadi University.`,
};

// Generic fallback for any location not in the map above.
const FALLBACK_MESSAGES = [
  (n, loc) => `${n}, you look right at home in the ${loc}! 🎓 Imagine making memories like this every day at Marwadi University.`,
  (n, loc) => `${n}, the ${loc} suits you perfectly! ✨ Your future at Marwadi University is just one step away.`,
  (n, loc) => `Welcome to the ${loc}, ${n}! 🌟 Some people just belong on this campus, and you're clearly one of them.`,
];

const motivationFor = (name, loc) => {
  const n = (name || 'Future Student').trim();
  const l = (loc || 'campus').trim();
  if (LOCATION_MESSAGES[l]) return LOCATION_MESSAGES[l](n);
  const idx = (n.length + l.length) % FALLBACK_MESSAGES.length;
  return FALLBACK_MESSAGES[idx](n, l);
};

export default function AppPage() {
  const navigate = useNavigate();

  // user details
  const [name,   setName]   = useState('');
  const [gender, setGender] = useState('Male');

  // source photo
  const [srcMode, setSrcMode] = useState('upload');
  const [srcFile, setSrcFile] = useState(null);
  const [srcB64,  setSrcB64]  = useState(null);
  const [srcInfo, setSrcInfo] = useState('');

  // location
  const [locations,  setLocations]  = useState([]);
  const [location,   setLocation]   = useState('');     // folder name
  const [loadingLocs, setLoadingLocs] = useState(false);

  // camera (source only)
  const videoRef  = useRef(null);
  const canvasRef = useRef(null);
  const [stream,    setStream]    = useState(null);
  const [camActive, setCamActive] = useState(false);

  const [swapping, setSwapping] = useState(false);
  const [progress, setProgress] = useState(0);
  const [pLabel,   setPLabel]   = useState('');
  const [result,   setResult]   = useState(null);
  const [toast,    setToast]    = useState('');

  // ── fetch locations whenever gender changes ──────────────────────────────
  useEffect(() => {
    let cancelled = false;
    setLoadingLocs(true);
    setLocation('');
    // cache:'no-store' + timestamp so newly-added/updated locations always sync
    fetch(`/api/locations?gender=${gender}&t=${Date.now()}`, { cache: 'no-store' })
      .then(r => r.json())
      .then(d => { if (!cancelled) setLocations(d.ok ? d.locations : []); })
      .catch(() => { if (!cancelled) setLocations([]); })
      .finally(() => { if (!cancelled) setLoadingLocs(false); });
    return () => { cancelled = true; };
  }, [gender]);

  // ── source face detection ────────────────────────────────────────────────
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

  // ── source camera ─────────────────────────────────────────────────────────
  useEffect(() => {
    if (camActive && stream && videoRef.current) {
      videoRef.current.srcObject = stream;
      videoRef.current.play?.().catch(() => {});
    }
  }, [camActive, stream]);

  const _camError = (e) => {
    if (!navigator.mediaDevices?.getUserMedia)
      return 'Camera not available here (needs HTTPS or localhost). Use Upload.';
    switch (e?.name) {
      case 'NotAllowedError':  return 'Camera permission blocked - click the camera icon in the address bar, Allow, then retry. Or use Upload.';
      case 'NotFoundError':    return 'No camera found - use Upload instead.';
      case 'NotReadableError': return 'Camera is busy in another app - close it and retry.';
      default:                 return `Camera error (${e?.name || 'unknown'}) - use Upload instead.`;
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
    try { const s = await _getCamStream(); setStream(s); setCamActive(true); }
    catch (e) { setToast(_camError(e)); }
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

  // ── run swap ───────────────────────────────────────────────────────────────
  const runSwap = async () => {
    if (!name.trim())          return setToast('Please enter your name.');
    if (!srcFile && !srcB64)   return setToast('Please provide your photo.');
    if (!location)             return setToast('Please choose a location.');

    setSwapping(true); setResult(null); setProgress(5); setPLabel('Initialising...');

    let idx = 0;
    const timer = setInterval(() => {
      if (idx < STAGES.length) { const [p,l] = STAGES[idx++]; setProgress(p); setPLabel(l); }
    }, 1100);

    const fd = new FormData();
    if (srcFile) fd.append('source_file', srcFile); else fd.append('source_b64', srcB64);
    fd.append('name', name.trim());
    fd.append('gender', gender);
    fd.append('location', location);

    try {
      const resp = await fetch('/api/swap', { method: 'POST', body: fd });
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
    setLocation('');
    setResult(null); setProgress(0);
    stopCamera();
  };

  const selectedLoc  = locations.find(l => l.folder === location);
  const locationLabel = selectedLoc?.label || '';
  const customMessage = selectedLoc?.message || '';

  // Result message: the owner's custom message for this location if set
  // (with {name}/{location} filled in), otherwise the built-in per-location one.
  const motivation = useMemo(() => {
    if (!result) return '';
    const n = (result.name || 'Future Student').trim();
    const l = locationLabel || 'campus';
    if (customMessage) {
      return customMessage.replaceAll('{name}', n).replaceAll('{location}', l);
    }
    return motivationFor(n, l);
  }, [result, locationLabel, customMessage]);

  return (
    <div className="bg-bg font-sans text-navy overflow-x-hidden">

      <div className="min-h-[100dvh] relative flex flex-col">
        {/* Floating Home Button */}
        <button onClick={() => navigate('/')}
          className="absolute top-4 left-4 sm:top-6 sm:left-6 z-50 flex items-center gap-2 text-xs sm:text-sm text-navy border border-border/60 bg-white/90 backdrop-blur-md rounded-lg px-3 sm:px-4 py-1.5 sm:py-2 hover:border-teal/50 hover:bg-white transition-all duration-200 whitespace-nowrap font-bold shadow-sm hover:shadow">
          <svg className="w-4 h-4 shrink-0 text-teal" fill="none" stroke="currentColor" strokeWidth={2.5} viewBox="0 0 24 24">
            <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>
            <polyline points="9 22 9 12 15 12 15 22"/>
          </svg>
          Home
        </button>

        <main className="flex-1 w-full max-w-5xl mx-auto px-3 sm:px-6 pt-14 sm:pt-16 pb-6 flex flex-col gap-3.5 sm:gap-4">

          {/* ── PAGE HEADER ──────────────────────────────────────────────── */}
          <div className="text-center flex flex-col items-center gap-0.5">
            <h1 className="text-xl sm:text-2xl font-extrabold text-navy tracking-tight">
              See yourself on campus
            </h1>
            <p className="text-xs sm:text-sm text-slate font-medium">
              Add your details, a photo, and pick a spot - we'll place you there.
            </p>
          </div>

          {/* ── STEP 1: YOUR DETAILS (name + gender) ───────────────────────── */}
          <SpotlightCard className="bg-white border border-border/60 rounded-2xl p-4 sm:p-5 flex flex-col gap-3.5 shadow-[0_1px_2px_rgba(15,23,42,0.03),0_12px_32px_-20px_rgba(15,23,42,0.12)]">
            <div>
              <h2 className="text-navy font-bold flex items-center gap-2 text-sm sm:text-base">
                <span className="w-5 h-5 sm:w-6 sm:h-6 bg-teal rounded-full flex items-center justify-center text-xs font-extrabold text-white shrink-0 shadow-sm">1</span>
                Your Details
              </h2>
              <p className="text-slate text-xs mt-1 font-medium">Tell us your name and gender</p>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 sm:gap-4">
              {/* Name */}
              <div className="flex flex-col gap-1.5">
                <label className="text-xs font-semibold text-navy-light uppercase tracking-wide">Name</label>
                <input
                  type="text" value={name} onChange={e => setName(e.target.value)}
                  placeholder="Enter your name"
                  className="border border-border rounded-lg px-3 py-2.5 text-sm text-navy bg-bg3 focus:outline-none focus:ring-2 focus:ring-teal/30 focus:border-teal transition-all"
                />
              </div>

              {/* Gender */}
              <div className="flex flex-col gap-1.5">
                <label className="text-xs font-semibold text-navy-light uppercase tracking-wide">Gender</label>
                <div className="flex bg-bg3 border border-border rounded-lg p-1 gap-1">
                  {['Male', 'Female'].map(g => (
                    <button key={g} onClick={() => setGender(g)}
                      className={`flex-1 flex items-center justify-center gap-1.5 py-2 rounded-md text-xs sm:text-sm font-semibold transition-all duration-200 ${gender === g ? 'bg-white text-teal shadow-sm border border-border/50' : 'text-slate hover:text-navy'}`}>
                      {g === 'Male'
                        ? <svg className="w-3.5 h-3.5 shrink-0" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24"><circle cx="10" cy="14" r="5"/><path d="M19 5l-5.4 5.4M19 5h-4M19 5v4"/></svg>
                        : <svg className="w-3.5 h-3.5 shrink-0" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24"><circle cx="12" cy="8" r="5"/><path d="M12 13v8M9 18h6"/></svg>}
                      {g}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </SpotlightCard>

          {/* ── STEP 2 + 3: photo + location, side-by-side from tablet up ────── */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3.5 sm:gap-4 items-start">

            {/* SOURCE PHOTO */}
            <SpotlightCard className="bg-white border border-border/60 rounded-2xl p-4 sm:p-5 flex flex-col gap-3.5 shadow-[0_1px_2px_rgba(15,23,42,0.03),0_12px_32px_-20px_rgba(15,23,42,0.12)]">
              <div>
                <h2 className="text-navy font-bold flex items-center gap-2 text-sm sm:text-base">
                  <span className="w-5 h-5 sm:w-6 sm:h-6 bg-teal rounded-full flex items-center justify-center text-xs font-extrabold text-white shrink-0 shadow-sm">2</span>
                  Your Photo
                </h2>
                <p className="text-slate text-xs mt-1 font-medium">Upload or capture your face</p>
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

            {/* LOCATION */}
            <SpotlightCard className="bg-white border border-border/60 rounded-2xl p-4 sm:p-5 flex flex-col gap-3.5 shadow-[0_1px_2px_rgba(15,23,42,0.03),0_12px_32px_-20px_rgba(15,23,42,0.12)]">
              <div>
                <h2 className="text-navy font-bold flex items-center gap-2 text-sm sm:text-base">
                  <span className="w-5 h-5 sm:w-6 sm:h-6 bg-teal rounded-full flex items-center justify-center text-xs font-extrabold text-white shrink-0 shadow-sm">3</span>
                  Choose Location
                </h2>
                <p className="text-slate text-xs mt-1 font-medium">Pick where you want to appear</p>
              </div>

              {/* location dropdown */}
              <div className="flex flex-col gap-1.5">
                <label className="text-xs font-semibold text-navy-light uppercase tracking-wide">
                  Location {loadingLocs && <span className="text-slate normal-case font-normal">· loading…</span>}
                </label>
                <div className="relative">
                  <select
                    value={location} onChange={e => setLocation(e.target.value)}
                    disabled={loadingLocs || locations.length === 0}
                    className="w-full appearance-none border border-border rounded-lg px-3 py-2.5 pr-9 text-sm text-navy bg-bg3 focus:outline-none focus:ring-2 focus:ring-teal/30 focus:border-teal transition-all disabled:opacity-50">
                    <option value="">{locations.length ? 'Select a location…' : 'No locations available'}</option>
                    {locations.map(l => (
                      <option key={l.folder} value={l.folder} disabled={!l.has_image}>
                        {l.label}{l.has_image ? '' : ' (coming soon)'}
                      </option>
                    ))}
                  </select>
                  <svg className="w-4 h-4 text-slate absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24"><polyline points="6 9 12 15 18 9"/></svg>
                </div>
              </div>

              {/* selected confirmation (no backend image preview) */}
              {location && (
                <div className="flex items-center gap-2.5 bg-teal/5 border border-teal/20 rounded-xl px-4 py-3">
                  <svg className="w-5 h-5 text-teal shrink-0" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                    <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/><circle cx="12" cy="10" r="3"/>
                  </svg>
                  <div className="flex flex-col leading-tight">
                    <span className="text-sm font-bold text-navy">{locationLabel}</span>
                    <span className="text-[11px] text-slate font-medium">Selected · we'll place you here</span>
                  </div>
                </div>
              )}
            </SpotlightCard>
          </div>

          {/* SWAP BUTTON */}
          <div className="flex justify-center mt-1">
            <button onClick={runSwap}
              disabled={swapping || !name.trim() || (!srcFile && !srcB64) || !location}
              className="w-full sm:w-auto flex items-center justify-center gap-2.5 bg-teal text-white px-8 sm:px-12 py-3 rounded-xl font-bold text-sm sm:text-base shadow-lg shadow-teal/20 hover:shadow-xl hover:shadow-teal/30 hover:-translate-y-0.5 active:scale-95 transition-all duration-200 disabled:opacity-40 disabled:cursor-not-allowed disabled:transform-none disabled:shadow-none">
              <svg className="w-4 h-4 sm:w-5 sm:h-5 shrink-0" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                <polyline points="17 1 21 5 17 9"/><path d="M3 11V9a4 4 0 0 1 4-4h14"/>
                <polyline points="7 23 3 19 7 15"/><path d="M21 13v2a4 4 0 0 1-4 4H3"/>
              </svg>
              {swapping ? 'Generating...' : 'Generate'}
            </button>
          </div>

          {/* PROGRESS */}
          {swapping && <ProgressBar pct={progress} label={pLabel} />}

          {/* RESULTS */}
          <AnimatePresence>
            {result && (
              <motion.section initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
                className="flex flex-col gap-5 sm:gap-6 mt-4 pt-8 border-t border-border">
                <h2 className="text-xl sm:text-2xl font-extrabold text-navy text-center">
                  {result.name ? `${result.name}, here you are! ✨` : 'Here you are! ✨'}
                </h2>

                {/* Personalised motivational message */}
                <motion.div
                  initial={{ opacity: 0, scale: 0.97 }} animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: 0.15 }}
                  className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-teal to-teal-dark text-white px-5 sm:px-7 py-5 sm:py-6 shadow-lg shadow-teal/20">
                  <div className="absolute -right-6 -top-6 opacity-10">
                    <svg className="w-28 h-28" fill="currentColor" viewBox="0 0 24 24"><path d="M12 3L1 9l11 6 9-4.91V17h2V9M5 13.18v4L12 21l7-3.82v-4L12 17z"/></svg>
                  </div>
                  <div className="relative flex items-start gap-3">
                    <div className="w-9 h-9 rounded-xl bg-white/15 backdrop-blur flex items-center justify-center shrink-0">
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24"><path d="M22 10v6M2 10l10-5 10 5-10 5z"/><path d="M6 12v5c3 3 9 3 12 0v-5"/></svg>
                    </div>
                    <div className="flex flex-col gap-1">
                      <span className="text-[10px] font-bold uppercase tracking-widest text-white/70">A note for you</span>
                      <p className="text-sm sm:text-base font-semibold leading-relaxed">{motivation}</p>
                    </div>
                  </div>
                </motion.div>

                {/* result only */}
                <div className="flex justify-center">
                  <TiltedCard className="bg-white border border-teal ring-2 ring-teal/20 shadow-lg shadow-teal/10 rounded-2xl overflow-hidden max-w-md w-full">
                    <p className="px-4 py-2.5 text-xs font-bold uppercase tracking-widest bg-teal/5 border-b border-teal/20 text-teal text-center">
                      ✨ Your Campus Look
                    </p>
                    <img src={result.result_image} alt="Your campus look"
                      className="w-full object-contain bg-bg" />
                  </TiltedCard>
                </div>

                {/* download row */}
                <div className="flex flex-col sm:flex-row gap-3 justify-center mt-2">
                  <a href={result.download_image || result.result_image}
                    download={`${(result.name || 'campus_look').replace(/\s+/g,'_')}.jpg`} target="_blank" rel="noreferrer"
                    className="flex items-center justify-center gap-2 bg-teal text-white px-5 sm:px-6 py-3 rounded-xl font-bold text-sm shadow-md hover:bg-teal-light hover:-translate-y-0.5 transition-all duration-200 w-full sm:w-auto">
                    <svg className="w-4 h-4 shrink-0" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                      <polyline points="7 10 12 15 17 10"/>
                      <line x1="12" y1="15" x2="12" y2="3"/>
                    </svg>
                    Download
                  </a>
                  <button onClick={reset}
                    className="flex items-center justify-center gap-2 bg-white border border-border shadow-sm text-navy px-5 py-3 rounded-xl text-sm font-semibold hover:border-teal/40 hover:text-teal transition-colors w-full sm:w-auto">
                    <svg className="w-4 h-4 shrink-0" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                      <polyline points="1 4 1 10 7 10"/>
                      <path d="M3.51 15a9 9 0 1 0 .49-3.96"/>
                    </svg>
                    New
                  </button>
                </div>
              </motion.section>
            )}
          </AnimatePresence>

        </main>
      </div>

      <Toast msg={toast} onClose={() => setToast('')} />

      {/* Footer */}
      <div className="flex flex-col bg-white">
        <footer className="bg-bg3 border-t border-border py-2 px-4">
          <div className="max-w-6xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-3 text-xs sm:text-sm text-slate">
            <div className="flex flex-col sm:flex-row items-center gap-2 sm:gap-4 text-center sm:text-left w-full sm:w-auto">
              <img src="/logos/ict_logo_black.png" alt="ICT Department" className="h-6 sm:h-8 w-auto opacity-70" />
              <div className="w-12 h-px sm:w-px sm:h-8 bg-border" />
              <div className="flex flex-col leading-tight">
                <strong className="text-navy">DeepFace Studio</strong>
                <span>Dept. of ICT, Marwadi University</span>
              </div>
            </div>
            <div className="flex items-center gap-2 sm:gap-3 flex-wrap justify-center font-medium">
              <span>Educational use only</span>
              <span className="text-border2">·</span>
              <span>Local processing</span>
              <span className="text-border2">·</span>
              <span>© Aditya Raj 2026</span>
            </div>
          </div>
        </footer>
      </div>
    </div>
  );
}
