import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';

// ─── Owner credentials ────────────────────────────────────────────────────────
// Change these to your real usernames / passwords before deploying.
const OWNERS = [
  { username: 'owner1', password: 'deepface@admin1', name: 'Owner 1', role: 'Primary Admin' },
  { username: 'owner2', password: 'deepface@admin2', name: 'Owner 2', role: 'Secondary Admin' },
];
// ─────────────────────────────────────────────────────────────────────────────

const SESSION_KEY = 'df_admin_session';

function getSession() {
  try { return JSON.parse(sessionStorage.getItem(SESSION_KEY)); } catch { return null; }
}
function setSession(owner) {
  sessionStorage.setItem(SESSION_KEY, JSON.stringify(owner));
}
function clearSession() {
  sessionStorage.removeItem(SESSION_KEY);
}

// ── Login form ──────────────────────────────────────────────────────────────
function LoginForm({ onLogin }) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [showPw,   setShowPw]   = useState(false);
  const [error,    setError]    = useState('');
  const [loading,  setLoading]  = useState(false);

  const submit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    await new Promise(r => setTimeout(r, 600)); // subtle delay
    const match = OWNERS.find(o => o.username === username.trim() && o.password === password);
    if (match) {
      setSession(match);
      onLogin(match);
    } else {
      setError('Invalid credentials. Access denied.');
    }
    setLoading(false);
  };

  return (
    <motion.div initial={{ opacity: 0, y: 24 }} animate={{ opacity: 1, y: 0 }}
      className="w-full max-w-sm mx-auto">
      <div className="bg-white border border-border rounded-2xl shadow-xl p-7 sm:p-8">
        {/* Lock icon */}
        <div className="flex justify-center mb-5">
          <div className="w-14 h-14 bg-teal/10 rounded-2xl flex items-center justify-center">
            <svg className="w-7 h-7 text-teal" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
              <rect x="3" y="11" width="18" height="11" rx="2" ry="2"/>
              <path d="M7 11V7a5 5 0 0 1 10 0v4"/>
            </svg>
          </div>
        </div>

        <h2 className="text-xl font-extrabold text-navy text-center mb-1">Control Panel</h2>
        <p className="text-xs text-slate text-center mb-6">Authorised owners only</p>

        <form onSubmit={submit} className="flex flex-col gap-4">
          <div className="flex flex-col gap-1.5">
            <label className="text-xs font-semibold text-navy-light uppercase tracking-wide">Username</label>
            <input
              type="text" autoComplete="username" required
              value={username} onChange={e => setUsername(e.target.value)}
              className="border border-border rounded-lg px-3 py-2.5 text-sm text-navy bg-bg3 focus:outline-none focus:ring-2 focus:ring-teal/30 focus:border-teal transition-all"
              placeholder="Enter username"
            />
          </div>

          <div className="flex flex-col gap-1.5">
            <label className="text-xs font-semibold text-navy-light uppercase tracking-wide">Password</label>
            <div className="relative">
              <input
                type={showPw ? 'text' : 'password'} autoComplete="current-password" required
                value={password} onChange={e => setPassword(e.target.value)}
                className="w-full border border-border rounded-lg px-3 py-2.5 pr-10 text-sm text-navy bg-bg3 focus:outline-none focus:ring-2 focus:ring-teal/30 focus:border-teal transition-all"
                placeholder="Enter password"
              />
              <button type="button" onClick={() => setShowPw(v => !v)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-slate hover:text-navy transition-colors">
                {showPw
                  ? <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24"><path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94"/><path d="M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19"/><line x1="1" y1="1" x2="23" y2="23"/></svg>
                  : <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>}
              </button>
            </div>
          </div>

          <AnimatePresence>
            {error && (
              <motion.p initial={{ opacity: 0, y: -4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
                className="text-xs text-red-600 font-medium bg-red-50 border border-red-200 rounded-lg px-3 py-2 flex items-center gap-2">
                <svg className="w-3.5 h-3.5 shrink-0" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
                {error}
              </motion.p>
            )}
          </AnimatePresence>

          <button type="submit" disabled={loading}
            className="w-full bg-teal text-white rounded-xl py-3 font-bold text-sm shadow hover:bg-teal-light active:scale-95 transition-all duration-200 disabled:opacity-60 disabled:cursor-not-allowed flex items-center justify-center gap-2 mt-1">
            {loading
              ? <><svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"/></svg> Verifying...</>
              : 'Sign In'}
          </button>
        </form>
      </div>
    </motion.div>
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
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-teal rounded-xl flex items-center justify-center shrink-0">
            <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
              <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>
            </svg>
          </div>
          <div>
            <p className="font-extrabold text-navy text-sm">{owner.name}</p>
            <p className="text-xs text-slate">{owner.role}</p>
          </div>
        </div>
        <button onClick={onLogout}
          className="flex items-center gap-1.5 text-xs font-semibold text-slate border border-border bg-bg3 px-3 py-1.5 rounded-lg hover:text-red-600 hover:border-red-200 hover:bg-red-50 transition-all">
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
          {OWNERS.map((o, i) => (
            <div key={i} className="flex items-center gap-3 bg-bg3 rounded-xl px-4 py-3">
              <div className={`w-8 h-8 rounded-full flex items-center justify-center text-xs font-extrabold text-white shrink-0 ${i === 0 ? 'bg-teal' : 'bg-teal-dark'}`}>
                {o.name.charAt(0)}
              </div>
              <div>
                <p className="text-sm font-semibold text-navy">{o.name}</p>
                <p className="text-xs text-slate">{o.role} · @{o.username}</p>
              </div>
              {owner.username === o.username && (
                <span className="ml-auto text-[10px] font-bold text-teal bg-teal/10 px-2 py-0.5 rounded-full">Active</span>
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
  const [owner, setOwner] = useState(getSession);

  const handleLogout = () => {
    clearSession();
    setOwner(null);
  };

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
          {owner
            ? <Dashboard key="dashboard" owner={owner} onLogout={handleLogout} />
            : <LoginForm  key="login"     onLogin={setOwner} />}
        </AnimatePresence>
      </div>
    </div>
  );
}
