import { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { auth, googleProvider, signInWithPopup, signInWithRedirect, signOut, onAuthStateChanged } from '../firebase';

// ═══ AUTHORISED OWNER EMAILS ═══════════════════════════════════════════════════
// Only these Google accounts can open the Control Panel. Anyone else who signs
// in with Google is rejected (here AND on the backend). Keep this list in sync
// with ALLOWED_ADMIN_EMAILS in the backend .env.
//   ⚠️  Replace the two placeholders below with your real owner Gmail addresses.
const ALLOWED_ADMINS = {
  'adivid198986@gmail.com':       { name: 'Aditya Raj',    role: 'Primary Admin' },
  'nishith.kotak@gmail.com':      { name: 'Nishith Kotak', role: 'Admin' },
  'cdparmar9824416484@gmail.com': { name: 'C D Parmar',    role: 'Admin' },
};
// ═══════════════════════════════════════════════════════════════════════════════

const isAllowed = (email) => !!ALLOWED_ADMINS[(email || '').toLowerCase()];

// Fresh Firebase ID token for backend admin calls (auto-refreshes when needed).
async function getIdToken() {
  const u = auth.currentUser;
  return u ? await u.getIdToken() : '';
}

// ── Google sign-in screen ─────────────────────────────────────────────────────
function GoogleLogin({ denied }) {
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState('');

  const signIn = async () => {
    setLoading(true); setError('');
    try {
      const res   = await signInWithPopup(auth, googleProvider);
      const email = (res.user.email || '').toLowerCase();
      if (!isAllowed(email)) {
        await signOut(auth);   // parent's auth listener also guards this
        setError(`${res.user.email} is not authorised for the Control Panel.`);
      }
      // allowed → onAuthStateChanged in the parent sets the owner + shows dashboard
    } catch (e) {
      const code = e?.code || '';
      if (code === 'auth/popup-closed-by-user' || code === 'auth/cancelled-popup-request') {
        // user closed the popup — ignore
      } else if (code === 'auth/unauthorized-domain') {
        setError('This site\'s domain isn\'t authorised in Firebase. Add it under '
               + 'Firebase Console → Authentication → Settings → Authorized domains.');
      } else if (code === 'auth/popup-blocked' || code === 'auth/operation-not-supported-in-this-environment') {
        // Popup blocked (or COOP) — fall back to full-page redirect sign-in.
        try { await signInWithRedirect(auth, googleProvider); return; }
        catch { setError('Sign-in failed (popup blocked). Allow popups and retry.'); }
      } else {
        setError(`Sign-in failed: ${code || e?.message || 'unknown error'}`);
      }
    } finally {
      setLoading(false);
    }
  };

  const shownError = error || denied;

  return (
    <motion.div initial={{ opacity: 0, y: 24 }} animate={{ opacity: 1, y: 0 }}
      className="w-full max-w-sm mx-auto">
      <div className="bg-white border border-border rounded-2xl shadow-xl p-7 sm:p-8">
        <div className="flex justify-center mb-5">
          <div className="w-14 h-14 bg-teal/10 rounded-2xl flex items-center justify-center">
            <svg className="w-7 h-7 text-teal" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
              <rect x="3" y="11" width="18" height="11" rx="2" ry="2"/>
              <path d="M7 11V7a5 5 0 0 1 10 0v4"/>
            </svg>
          </div>
        </div>

        <h2 className="text-xl font-extrabold text-navy text-center mb-1">Control Panel</h2>
        <p className="text-xs text-slate text-center mb-6">Authorised owners only — sign in with Google</p>

        <button onClick={signIn} disabled={loading}
          className="w-full flex items-center justify-center gap-3 border border-border bg-white rounded-xl py-3 font-semibold text-sm text-navy shadow-sm hover:border-teal/50 hover:shadow active:scale-95 transition-all disabled:opacity-60 disabled:cursor-not-allowed">
          {loading
            ? <svg className="w-5 h-5 animate-spin text-teal" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"/></svg>
            : <svg className="w-5 h-5" viewBox="0 0 48 48"><path fill="#FFC107" d="M43.611 20.083H42V20H24v8h11.303c-1.649 4.657-6.08 8-11.303 8c-6.627 0-12-5.373-12-12s5.373-12 12-12c3.059 0 5.842 1.154 7.961 3.039l5.657-5.657C34.046 6.053 29.268 4 24 4C12.955 4 4 12.955 4 24s8.955 20 20 20s20-8.955 20-20c0-1.341-.138-2.65-.389-3.917z"/><path fill="#FF3D00" d="m6.306 14.691 6.571 4.819C14.655 15.108 18.961 12 24 12c3.059 0 5.842 1.154 7.961 3.039l5.657-5.657C34.046 6.053 29.268 4 24 4C16.318 4 9.656 8.337 6.306 14.691z"/><path fill="#4CAF50" d="M24 44c5.166 0 9.86-1.977 13.409-5.192l-6.19-5.238A11.91 11.91 0 0 1 24 36c-5.202 0-9.619-3.317-11.283-7.946l-6.522 5.025C9.505 39.556 16.227 44 24 44z"/><path fill="#1976D2" d="M43.611 20.083H42V20H24v8h11.303a12.04 12.04 0 0 1-4.087 5.571l.003-.002 6.19 5.238C36.971 39.205 44 34 44 24c0-1.341-.138-2.65-.389-3.917z"/></svg>}
          {loading ? 'Signing in…' : 'Sign in with Google'}
        </button>

        <AnimatePresence>
          {shownError && (
            <motion.p initial={{ opacity: 0, y: -4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
              className="mt-4 text-xs text-red-600 font-medium bg-red-50 border border-red-200 rounded-lg px-3 py-2 flex items-center gap-2">
              <svg className="w-3.5 h-3.5 shrink-0" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
              {shownError}
            </motion.p>
          )}
        </AnimatePresence>

        <p className="mt-5 text-[11px] text-slate text-center leading-relaxed">
          Access is restricted to a fixed list of owner accounts.
        </p>
      </div>
    </motion.div>
  );
}

// ── Manage Locations (complete CRUD) ─────────────────────────────────────────
function ManageLocations({ isPrimary }) {
  const [gender,    setGender]    = useState('Male');
  const [locations, setLocations] = useState([]);
  const [loading,   setLoading]   = useState(false);

  const [name,    setName]    = useState('');
  const [file,    setFile]    = useState(null);
  const [preview, setPreview] = useState(null);
  const [busy,    setBusy]    = useState(false);
  const [msg,     setMsg]     = useState(null);   // { type: 'ok'|'err', text }
  const [cacheBust, setCacheBust] = useState(Date.now());

  const fileRef        = useRef();   // add-form picker
  const replaceRef     = useRef();   // per-card replace picker
  const replaceFolder  = useRef(null);

  const load = (g) => {
    setLoading(true);
    // no-store + timestamp so the panel always reflects the latest photos
    fetch(`/api/locations?gender=${g}&t=${Date.now()}`, { cache: 'no-store' })
      .then(r => r.json())
      .then(d => { setLocations(d.ok ? d.locations : []); setCacheBust(Date.now()); })
      .catch(() => setLocations([]))
      .finally(() => setLoading(false));
  };

  useEffect(() => { load(gender); }, [gender]);

  const flash = (type, text) => setMsg({ type, text });

  const pickFile = (f) => {
    if (!f) return;
    if (!f.type.startsWith('image/')) { flash('err', 'Please choose an image file.'); return; }
    setFile(f); setPreview(URL.createObjectURL(f)); setMsg(null);
  };

  // Upload helper used by both the add-form and per-card replace.
  const upload = async (locName, imgFile) => {
    const fd = new FormData();
    fd.append('gender', gender);
    fd.append('location', locName);
    fd.append('image', imgFile);
    const tok = await getIdToken();
    const r = await fetch('/api/admin/location', {
      method: 'POST', headers: { 'X-Admin-Token': tok }, body: fd,
    });
    return r.json();
  };

  const submit = async () => {
    if (!name.trim()) { flash('err', 'Enter a location name.'); return; }
    if (!file)        { flash('err', 'Choose an image.'); return; }
    setBusy(true); setMsg(null);
    try {
      const d = await upload(name.trim(), file);
      if (d.ok) {
        flash('ok', `Saved "${d.label}" for ${d.gender} — live on the app page.`);
        setName(''); setFile(null); setPreview(null);
        if (fileRef.current) fileRef.current.value = '';
        load(gender);
      } else flash('err', d.error || 'Upload failed.');
    } catch (e) { flash('err', 'Network error: ' + e.message); }
    finally     { setBusy(false); }
  };

  // Per-card "replace photo" → opens the hidden picker for that folder.
  const startReplace = (folder) => { replaceFolder.current = folder; replaceRef.current?.click(); };
  const onReplaceFile = async (f) => {
    const folder = replaceFolder.current;
    if (!f || !folder) return;
    if (!f.type.startsWith('image/')) { flash('err', 'Please choose an image file.'); return; }
    setBusy(true); setMsg(null);
    try {
      const d = await upload(folder, f);          // folder name resolves to existing location
      if (d.ok) { flash('ok', `Updated photo for "${d.label}".`); load(gender); }
      else      flash('err', d.error || 'Update failed.');
    } catch (e) { flash('err', 'Network error: ' + e.message); }
    finally { setBusy(false); if (replaceRef.current) replaceRef.current.value = ''; }
  };

  const rename = async (folder, label) => {
    const next = window.prompt(`Rename "${label}" to:`, label);
    if (next == null || !next.trim() || next.trim() === label) return;
    try {
      const tok = await getIdToken();
      const r = await fetch('/api/admin/location/rename', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-Admin-Token': tok },
        body: JSON.stringify({ gender, location: folder, new_name: next.trim() }),
      });
      const d = await r.json();
      if (d.ok) { flash('ok', `Renamed to "${d.label}".`); load(gender); }
      else      flash('err', d.error || 'Rename failed.');
    } catch (e) { flash('err', 'Network error: ' + e.message); }
  };

  const remove = async (folder, label) => {
    if (!window.confirm(`Delete "${label}" (${gender})? This removes the location and its photo.`)) return;
    try {
      const tok = await getIdToken();
      const r = await fetch('/api/admin/location/delete', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-Admin-Token': tok },
        body: JSON.stringify({ gender, location: folder }),
      });
      const d = await r.json();
      if (d.ok) { flash('ok', `Deleted "${label}".`); load(gender); }
      else      flash('err', d.error || 'Delete failed.');
    } catch (e) { flash('err', 'Network error: ' + e.message); }
  };

  const withPhoto = locations.filter(l => l.has_image).length;
  const pending   = locations.length - withPhoto;

  return (
    <div className="bg-white border border-border rounded-2xl shadow-sm p-5 sm:p-6">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-xs font-bold uppercase tracking-widest text-slate">Manage Locations</h3>
        <button onClick={() => load(gender)} title="Refresh"
          className="text-slate hover:text-teal transition-colors">
          <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24"><polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/></svg>
        </button>
      </div>
      <p className="text-[11px] text-slate mb-4">
        {isPrimary
          ? 'You can add, edit and delete locations.'
          : 'You can add and edit locations. Deleting is restricted to the Primary Admin.'}
      </p>

      {/* Gender toggle */}
      <div className="flex bg-bg3 border border-border rounded-lg p-1 gap-1 mb-3">
        {['Male', 'Female'].map(g => (
          <button key={g} onClick={() => setGender(g)}
            className={`flex-1 py-2 rounded-md text-xs sm:text-sm font-semibold transition-all duration-200 ${gender === g ? 'bg-white text-teal shadow-sm border border-border/50' : 'text-slate hover:text-navy'}`}>
            {g}
          </button>
        ))}
      </div>

      {/* Summary */}
      <div className="grid grid-cols-3 gap-2 mb-5">
        {[
          { k: 'Total',   v: locations.length },
          { k: 'With Photo', v: withPhoto },
          { k: 'Pending', v: pending },
        ].map(s => (
          <div key={s.k} className="bg-bg3 rounded-lg px-3 py-2 text-center">
            <p className="text-lg font-extrabold text-teal leading-none">{loading ? '–' : s.v}</p>
            <p className="text-[9px] font-bold uppercase tracking-widest text-slate mt-1">{s.k}</p>
          </div>
        ))}
      </div>

      {/* Add / update form */}
      <div className="flex flex-col gap-3 bg-bg3 rounded-xl p-4 mb-5">
        <p className="text-[11px] font-bold uppercase tracking-widest text-slate">Add / Update</p>
        <div className="flex flex-col gap-1.5">
          <label className="text-xs font-semibold text-navy-light uppercase tracking-wide">Location Name</label>
          <input
            type="text" value={name} onChange={e => setName(e.target.value)}
            list="loc-suggestions" placeholder="e.g. Auditorium"
            className="border border-border rounded-lg px-3 py-2.5 text-sm text-navy bg-white focus:outline-none focus:ring-2 focus:ring-teal/30 focus:border-teal transition-all"
          />
          <datalist id="loc-suggestions">
            {locations.map(l => <option key={l.folder} value={l.label} />)}
          </datalist>
          <p className="text-[11px] text-slate">Type a new name, or pick an existing one to replace its photo.</p>
        </div>

        {/* Image picker */}
        <div className="flex flex-col gap-1.5">
          <label className="text-xs font-semibold text-navy-light uppercase tracking-wide">Photo for {gender}</label>
          <div
            onClick={() => fileRef.current?.click()}
            onDrop={e => { e.preventDefault(); pickFile(e.dataTransfer.files[0]); }}
            onDragOver={e => e.preventDefault()}
            className="border-2 border-dashed border-border2 rounded-lg p-4 flex items-center gap-3 cursor-pointer hover:border-teal/50 hover:bg-teal/[0.02] transition-all">
            {preview
              ? <img src={preview} alt="preview" className="w-16 h-16 object-cover rounded-md border border-border shrink-0" />
              : <div className="w-16 h-16 rounded-md bg-white border border-border flex items-center justify-center shrink-0">
                  <svg className="w-6 h-6 text-slate" fill="none" stroke="currentColor" strokeWidth={1.5} viewBox="0 0 24 24"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>
                </div>}
            <div className="flex flex-col">
              <span className="text-sm font-semibold text-navy">{file ? file.name : 'Choose / drop image'}</span>
              <span className="text-[11px] text-slate">JPG · PNG · WEBP</span>
            </div>
            <input ref={fileRef} type="file" accept="image/*" hidden onChange={e => pickFile(e.target.files[0])} />
          </div>
        </div>

        <button onClick={submit} disabled={busy}
          className="bg-teal text-white rounded-xl py-2.5 font-bold text-sm shadow hover:bg-teal-light active:scale-95 transition-all disabled:opacity-60 disabled:cursor-not-allowed flex items-center justify-center gap-2">
          {busy
            ? <><svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"/></svg> Saving…</>
            : <>+ Add / Update Location</>}
        </button>

        <AnimatePresence>
          {msg && (
            <motion.p initial={{ opacity: 0, y: -4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
              className={`text-xs font-medium rounded-lg px-3 py-2 ${msg.type === 'ok' ? 'text-teal bg-teal/10 border border-teal/20' : 'text-red-600 bg-red-50 border border-red-200'}`}>
              {msg.text}
            </motion.p>
          )}
        </AnimatePresence>
      </div>

      {/* hidden picker for per-card replace */}
      <input ref={replaceRef} type="file" accept="image/*" hidden onChange={e => onReplaceFile(e.target.files[0])} />

      {/* Existing locations */}
      <p className="text-[11px] font-bold uppercase tracking-widest text-slate mb-2">
        {gender} Locations {loading ? '· loading…' : `· ${locations.length}`}
      </p>
      <div className="grid grid-cols-2 xs:grid-cols-3 gap-2.5">
        {locations.map(l => (
          <div key={l.folder} className="rounded-xl overflow-hidden border border-border bg-white flex flex-col">
            <div className="relative aspect-square bg-bg3 flex items-center justify-center overflow-hidden group">
              {l.has_image
                ? <img src={`/api/location-image?gender=${gender}&location=${encodeURIComponent(l.folder)}&t=${cacheBust}`}
                    alt={l.label} className="w-full h-full object-cover" />
                : <div className="flex flex-col items-center gap-1 text-slate">
                    <svg className="w-6 h-6" fill="none" stroke="currentColor" strokeWidth={1.5} viewBox="0 0 24 24"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>
                    <span className="text-[9px] font-medium">No photo</span>
                  </div>}
              {/* status dot */}
              <span className={`absolute top-1.5 left-1.5 w-2 h-2 rounded-full ${l.has_image ? 'bg-teal' : 'bg-amber-400'}`} title={l.has_image ? 'Has photo' : 'Pending'} />
            </div>

            <div className="px-2 py-1.5 flex-1">
              <p className="text-[11px] font-semibold text-navy leading-tight line-clamp-2" title={l.label}>{l.label}</p>
            </div>

            {/* actions */}
            <div className="flex border-t border-border divide-x divide-border">
              <button onClick={() => startReplace(l.folder)} title="Replace photo"
                className="flex-1 py-1.5 flex items-center justify-center text-slate hover:text-teal hover:bg-teal/[0.04] transition-colors">
                <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
              </button>
              <button onClick={() => rename(l.folder, l.label)} title="Rename"
                className="flex-1 py-1.5 flex items-center justify-center text-slate hover:text-teal hover:bg-teal/[0.04] transition-colors">
                <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.12 2.12 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>
              </button>
              {isPrimary && (
                <button onClick={() => remove(l.folder, l.label)} title="Delete (Primary Admin only)"
                  className="flex-1 py-1.5 flex items-center justify-center text-slate hover:text-red-600 hover:bg-red-50 transition-colors">
                  <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
                </button>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Admin Dashboard ─────────────────────────────────────────────────────────
function Dashboard({ owner, onLogout }) {
  const stats = [
    { label: 'App Version', value: 'v1.0.0' },
    { label: 'Backend',     value: 'Flask + InsightFace' },
    { label: 'Frontend',    value: 'React 19 + Vite 8' },
    { label: 'Models',      value: 'inswapper + GFPGAN + RealESRGAN' },
  ];

  const links = [
    { label: 'GitHub Repository', href: 'https://github.com/ADiTyaRaj8969/FaceSWAP', icon: 'github' },
    { label: 'HuggingFace Space', href: 'https://aditya-raj19-faceswap.hf.space', icon: 'hf' },
  ];

  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
      className="w-full max-w-2xl mx-auto flex flex-col gap-5">

      {/* Header */}
      <div className="bg-white border border-border rounded-2xl shadow-sm p-5 sm:p-6 flex items-center justify-between gap-4">
        <div className="flex items-center gap-3 min-w-0">
          {owner.photo
            ? <img src={owner.photo} alt={owner.name} referrerPolicy="no-referrer"
                className="w-10 h-10 rounded-xl object-cover shrink-0 border border-border" />
            : <div className="w-10 h-10 bg-teal rounded-xl flex items-center justify-center shrink-0">
                <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                  <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>
                </svg>
              </div>}
          <div className="min-w-0">
            <p className="font-extrabold text-navy text-sm truncate">{owner.name}</p>
            <p className="text-xs text-slate truncate">{owner.role} · {owner.email}</p>
          </div>
        </div>
        <button onClick={onLogout}
          className="flex items-center gap-1.5 text-xs font-semibold text-slate border border-border bg-bg3 px-3 py-1.5 rounded-lg hover:text-red-600 hover:border-red-200 hover:bg-red-50 transition-all shrink-0">
          <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24"><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/><polyline points="16 17 21 12 16 7"/><line x1="21" y1="12" x2="9" y2="12"/></svg>
          Sign Out
        </button>
      </div>

      {/* System Info */}
      <div className="bg-white border border-border rounded-2xl shadow-sm p-5 sm:p-6">
        <h3 className="text-xs font-bold uppercase tracking-widest text-slate mb-4">System Info</h3>
        <div className="grid grid-cols-1 xs:grid-cols-2 gap-3">
          {stats.map(s => (
            <div key={s.label} className="bg-bg3 rounded-xl px-4 py-3 flex flex-col gap-0.5">
              <span className="text-[10px] font-bold uppercase tracking-widest text-slate">{s.label}</span>
              <span className="text-sm font-semibold text-navy">{s.value}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Manage Locations */}
      <ManageLocations isPrimary={owner.role === 'Primary Admin'} />

      {/* Quick Links */}
      <div className="bg-white border border-border rounded-2xl shadow-sm p-5 sm:p-6">
        <h3 className="text-xs font-bold uppercase tracking-widest text-slate mb-4">Quick Links</h3>
        <div className="flex flex-col gap-2">
          {links.map(l => (
            <a key={l.label} href={l.href} target="_blank" rel="noreferrer"
              className="flex items-center justify-between gap-3 border border-border rounded-xl px-4 py-3 hover:border-teal/40 hover:bg-teal/[0.02] transition-all group">
              <span className="text-sm font-semibold text-navy group-hover:text-teal transition-colors">{l.label}</span>
              <svg className="w-4 h-4 text-slate group-hover:text-teal transition-colors shrink-0" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg>
            </a>
          ))}
        </div>
      </div>

      {/* Owners */}
      <div className="bg-white border border-border rounded-2xl shadow-sm p-5 sm:p-6">
        <h3 className="text-xs font-bold uppercase tracking-widest text-slate mb-4">Authorised Owners</h3>
        <div className="flex flex-col gap-2">
          {Object.entries(ALLOWED_ADMINS).map(([email, o], i) => (
            <div key={email} className="flex items-center gap-3 bg-bg3 rounded-xl px-4 py-3">
              <div className={`w-8 h-8 rounded-full flex items-center justify-center text-xs font-extrabold text-white shrink-0 ${i === 0 ? 'bg-teal' : 'bg-teal-dark'}`}>
                {o.name.charAt(0)}
              </div>
              <div className="min-w-0">
                <p className="text-sm font-semibold text-navy truncate">{o.name}</p>
                <p className="text-xs text-slate truncate">{o.role} · {email}</p>
              </div>
              {owner.email === email && (
                <span className="ml-auto text-[10px] font-bold text-teal bg-teal/10 px-2 py-0.5 rounded-full shrink-0">You</span>
              )}
            </div>
          ))}
        </div>
      </div>

    </motion.div>
  );
}

// ── Main page ────────────────────────────────────────────────────────────────
export default function AdminPage() {
  const navigate = useNavigate();
  const [owner,  setOwner]  = useState(undefined);  // undefined = auth still loading
  const [denied, setDenied] = useState('');

  // Single source of truth: Firebase auth state. Restores the session on reload
  // and enforces the email allowlist on every sign-in.
  useEffect(() => {
    const unsub = onAuthStateChanged(auth, (u) => {
      if (!u) { setOwner(null); return; }
      const email   = (u.email || '').toLowerCase();
      const profile = ALLOWED_ADMINS[email];
      if (profile) {
        setOwner({ email, name: u.displayName || profile.name,
                   role: profile.role, photo: u.photoURL || '' });
        setDenied('');
      } else {
        // Signed in with Google but not on the allowlist → reject + sign out.
        signOut(auth).catch(() => {});
        setOwner(null);
        setDenied(`${u.email} is not authorised for the Control Panel.`);
      }
    });
    return unsub;
  }, []);

  const handleLogout = () => { signOut(auth).catch(() => {}); setOwner(null); };

  return (
    <div className="min-h-screen bg-bg font-sans text-navy overflow-x-hidden">

      {/* Top bar */}
      <div className="sticky top-0 z-40 bg-white/90 backdrop-blur-md border-b border-border px-4 py-3 flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <button onClick={() => navigate('/')}
            className="flex items-center gap-1.5 text-xs font-semibold text-slate hover:text-navy transition-colors">
            <svg className="w-4 h-4 text-teal" fill="none" stroke="currentColor" strokeWidth={2.5} viewBox="0 0 24 24">
              <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>
              <polyline points="9 22 9 12 15 12 15 22"/>
            </svg>
            Home
          </button>
          <span className="text-border2 text-xs">/</span>
          <span className="text-xs font-bold text-navy">Control Panel</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-2 h-2 rounded-full bg-teal animate-pulse" />
          <span className="text-[10px] font-semibold text-slate uppercase tracking-wide">Admin</span>
        </div>
      </div>

      {/* Content */}
      <div className="max-w-2xl mx-auto px-4 py-10 sm:py-14">
        <AnimatePresence mode="wait">
          {owner === undefined ? (
            <motion.div key="loading" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
              className="flex flex-col items-center justify-center gap-3 py-20 text-slate">
              <svg className="w-7 h-7 animate-spin text-teal" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"/></svg>
              <span className="text-xs font-medium">Checking access…</span>
            </motion.div>
          ) : owner ? (
            <Dashboard key="dashboard" owner={owner} onLogout={handleLogout} />
          ) : (
            <GoogleLogin key="login" denied={denied} />
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
