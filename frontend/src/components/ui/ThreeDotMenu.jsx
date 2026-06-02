import { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { AnimatePresence, motion } from 'framer-motion';

export default function ThreeDotMenu() {
  const [open, setOpen] = useState(false);
  const ref = useRef(null);
  const navigate = useNavigate();

  // Close on outside click
  useEffect(() => {
    const handler = (e) => { if (ref.current && !ref.current.contains(e.target)) setOpen(false); };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  return (
    <div ref={ref} className="fixed bottom-5 right-5 z-50">
      {/* Dropdown */}
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, scale: 0.92, y: 6 }}
            animate={{ opacity: 1, scale: 1,    y: 0 }}
            exit={{   opacity: 0, scale: 0.92, y: 6 }}
            transition={{ duration: 0.15 }}
            className="absolute bottom-12 right-0 bg-white border border-border rounded-xl shadow-xl overflow-hidden min-w-[170px]">

            <button
              onClick={() => { setOpen(false); navigate('/admin'); }}
              className="w-full flex items-center gap-3 px-4 py-3 text-sm font-semibold text-navy hover:bg-bg3 transition-colors text-left">
              <svg className="w-4 h-4 text-teal shrink-0" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                <rect x="3" y="11" width="18" height="11" rx="2" ry="2"/>
                <path d="M7 11V7a5 5 0 0 1 10 0v4"/>
              </svg>
              Control Panel
            </button>

            <div className="border-t border-border mx-3" />

            <div className="px-4 py-2">
              <p className="text-[10px] text-slate font-medium">Authorised owners only</p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Trigger button */}
      <button
        onClick={() => setOpen(v => !v)}
        className={`w-9 h-9 flex items-center justify-center rounded-full border shadow-md transition-all duration-200
          ${open
            ? 'bg-teal text-white border-teal shadow-teal/30'
            : 'bg-white text-navy border-border hover:border-teal/50 hover:shadow-lg'}`}
        title="Menu"
      >
        <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
          <circle cx="12" cy="5"  r="1.5"/>
          <circle cx="12" cy="12" r="1.5"/>
          <circle cx="12" cy="19" r="1.5"/>
        </svg>
      </button>
    </div>
  );
}
