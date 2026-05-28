import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { auth, googleProvider, signInWithPopup } from '../firebase';
import useAuth, { saveSession, daysLeft } from '../hooks/useAuth';
import Aurora          from '../components/ui/Aurora';
import SplitText       from '../components/ui/SplitText';
import BlurText        from '../components/ui/BlurText';
import SpotlightCard   from '../components/ui/SpotlightCard';
import AnimatedContent from '../components/ui/AnimatedContent';
import Magnet          from '../components/ui/Magnet';
import ShinyText       from '../components/ui/ShinyText';

const FEATURE_ICONS = {
  camera: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} className="w-5 h-5"><path d="M23 7l-7 5 7 5V7z"/><rect x="1" y="5" width="15" height="14" rx="2" ry="2"/></svg>,
  layers: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} className="w-5 h-5"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>,
  palette: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} className="w-5 h-5"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="10" r="3"/><path d="M12 13v2"/><circle cx="9" cy="17" r="1" fill="currentColor" stroke="none"/><circle cx="12" cy="18" r="1" fill="currentColor" stroke="none"/><circle cx="15" cy="17" r="1" fill="currentColor" stroke="none"/></svg>,
  chart:  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} className="w-5 h-5"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>,
  zap:    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} className="w-5 h-5"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>,
  lock:   <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} className="w-5 h-5"><rect x="3" y="11" width="18" height="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg>,
};

const FEATURES = [
  { icon: FEATURE_ICONS.camera,  title: 'Live Camera + Upload',  desc: 'Capture from webcam or upload any image. Instant face detection with bounding box feedback.' },
  { icon: FEATURE_ICONS.layers,  title: 'Hair to Neck Blending', desc: 'Seamless multi-layer mask blending from hairline through face to neck using Poisson cloning.' },
  { icon: FEATURE_ICONS.palette, title: 'Skin Tone Matching',    desc: 'CIE LAB colour transfer adapts the swapped face to match target skin across all skin types.' },
  { icon: FEATURE_ICONS.chart,   title: 'Quality Metrics',       desc: 'Alignment score, blend quality, colour Delta E, and naturalness score after every swap.' },
  { icon: FEATURE_ICONS.zap,     title: 'GPU + CPU Support',     desc: 'CUDA optimised InsightFace on GPU, automatic CPU fallback. Same output quality on both.' },
  { icon: FEATURE_ICONS.lock,    title: 'Fully Private',         desc: 'All processing runs on your own machine. No images are uploaded to any external server.' },
];

const PIPELINE = ['Detect', 'Align', 'Swap', 'Tone Match', 'Blend', 'Enhance', 'Output'];

const GoogleIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" className="flex-shrink-0">
    <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4"/>
    <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853"/>
    <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" fill="#FBBC05"/>
    <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335"/>
  </svg>
);

export default function LandingPage() {
  const navigate          = useNavigate();
  const { user, loading } = useAuth();
  const [signingIn, setSigningIn] = useState(false);
  const [error,     setError]     = useState('');

  const isSignedIn = !!user;
  const days       = isSignedIn ? daysLeft() : 0;

  const doGoogleSignIn = async () => {
    setSigningIn(true);
    setError('');
    try {
      const result = await signInWithPopup(auth, googleProvider);
      // Save session immediately with empty token so isSessionExpired() won't block
      saveSession(result.user, '');
      // Fetch real token in background — do NOT await before navigating
      result.user.getIdToken(false).then(t => localStorage.setItem('df_token', t)).catch(() => {});
      navigate('/app');
    } catch (err) {
      if (err?.code !== 'auth/popup-closed-by-user') {
        setError('Sign in failed. Please try again.');
      }
      setSigningIn(false);
    }
  };

  const handleTry = () => isSignedIn ? navigate('/app') : doGoogleSignIn();

  return (
    <div className="min-h-screen bg-bg font-sans overflow-x-hidden">

      {/* NAVBAR */}
      <nav className="fixed top-0 left-0 right-0 z-50 bg-bg/80 backdrop-blur-xl border-b border-white/5 shadow-[0_1px_0_rgba(99,107,47,0.15)]">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 h-14 sm:h-16 flex items-center justify-between gap-3">

          <button onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
            className="flex items-center gap-2 group shrink-0">
            <span className="text-xl sm:text-2xl leading-none group-hover:scale-110 transition-transform duration-200 text-lime drop-shadow-[0_0_8px_rgba(212,222,149,0.4)]">
              &#11043;
            </span>
            <span className="text-lime font-extrabold text-base sm:text-lg tracking-tight group-hover:text-[#e4f0a0] transition-colors duration-200">
              DeepFace Studio
            </span>
          </button>

          <div className="flex items-center gap-2 sm:gap-3">
            {loading ? (
              <div className="w-5 h-5 border-2 border-forest border-t-lime rounded-full animate-spin" />
            ) : isSignedIn ? (
              <div className="flex items-center gap-2 bg-bg3 border border-border2 rounded-full pl-1.5 pr-2.5 sm:pr-3 py-1">
                {user.photoURL
                  ? <img src={user.photoURL} alt="" referrerPolicy="no-referrer" className="w-6 h-6 sm:w-7 sm:h-7 rounded-full ring-2 ring-olive/50 shrink-0" />
                  : <span className="w-6 h-6 sm:w-7 sm:h-7 rounded-full bg-olive flex items-center justify-center text-lime font-bold text-xs shrink-0">
                      {(user.displayName || user.email || 'U')[0].toUpperCase()}
                    </span>}
                <div className="hidden sm:flex flex-col leading-none">
                  <span className="text-lime text-xs font-semibold max-w-[100px] truncate">
                    {user.displayName || user.email}
                  </span>
                  <span className="text-[#6b7345] text-[10px]">{days}d left</span>
                </div>
                <button onClick={() => navigate('/app')}
                  className="ml-1 bg-olive text-lime text-xs font-bold px-2.5 sm:px-3 py-1.5 rounded-full hover:bg-lime hover:text-forest transition-all duration-200 whitespace-nowrap">
                  Open App
                </button>
              </div>
            ) : (
              <button onClick={doGoogleSignIn} disabled={signingIn}
                className="flex items-center gap-1.5 sm:gap-2 bg-bg3 border border-border2 text-sage text-xs sm:text-sm font-semibold px-3 sm:px-4 py-1.5 sm:py-2 rounded-full hover:border-lime/50 hover:text-lime hover:bg-olive/10 transition-all duration-200 disabled:opacity-50">
                {signingIn
                  ? <span className="w-3.5 h-3.5 border-2 border-sage/30 border-t-sage rounded-full animate-spin" />
                  : <GoogleIcon />}
                <span>{signingIn ? 'Signing in...' : 'Sign In'}</span>
              </button>
            )}
          </div>
        </div>
      </nav>

      {/* HERO */}
      <section className="relative min-h-screen flex items-center justify-center pt-14 sm:pt-16 overflow-hidden">
        <Aurora />
        <div className="absolute inset-0 pointer-events-none" style={{
          backgroundImage: 'linear-gradient(rgba(99,107,47,0.06) 1px,transparent 1px),linear-gradient(90deg,rgba(99,107,47,0.06) 1px,transparent 1px)',
          backgroundSize: '50px 50px',
        }} />

        <div className="relative z-10 w-full max-w-3xl mx-auto text-center px-4 sm:px-6 py-12 sm:py-16 flex flex-col items-center gap-6 sm:gap-8">

          <h1 className="text-[clamp(2.2rem,8vw,5rem)] font-black leading-[1.05] tracking-[-0.04em]">
            <SplitText text="Swap Faces with" className="text-lime block" delay={0.1} />
            <ShinyText text="Studio Precision" className="block mt-1" />
          </h1>

          <BlurText
            text="Professional grade deepfake face swapping with seamless skin tone matching, hair integration, and neck blending."
            className="text-base sm:text-lg text-sage max-w-lg leading-relaxed px-2"
            delay={0.35}
          />

          <AnimatedContent delay={0.55} direction="up" distance={20} className="w-full max-w-xs sm:max-w-none">

            {/* signed in welcome strip */}
            {isSignedIn && (
              <motion.div initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }}
                className="flex items-center justify-center gap-2 bg-olive/15 border border-lime/20 rounded-full px-4 py-2 mb-5 text-xs sm:text-sm text-sage flex-wrap text-center">
                {user.photoURL && <img src={user.photoURL} alt="" referrerPolicy="no-referrer" className="w-5 h-5 rounded-full shrink-0 object-cover" />}
                <span>
                  Welcome back, <strong className="text-lime">{user.displayName?.split(' ')[0] || 'User'}</strong>
                  {' '} Session valid for <strong className="text-lime">{days} day{days !== 1 ? 's' : ''}</strong>
                </span>
              </motion.div>
            )}

            {/* buttons */}
            <div className="flex flex-col sm:flex-row gap-3 sm:gap-4 justify-center">

              {/* Try Now — plain, no glow */}
              <Magnet className="w-full sm:w-auto">
                <button onClick={handleTry} disabled={signingIn || loading}
                  className="w-full sm:w-auto flex items-center justify-center gap-2.5 bg-lime text-forest px-7 sm:px-8 py-3.5 sm:py-4 rounded-xl font-bold text-base hover:bg-[#c8d280] active:scale-95 transition-all duration-200 disabled:opacity-50">
                  {signingIn ? (
                    <><span className="w-4 h-4 border-2 border-forest/40 border-t-forest rounded-full animate-spin" />Signing in...</>
                  ) : (
                    <><svg className="w-4 h-4 sm:w-5 sm:h-5 shrink-0" viewBox="0 0 24 24"><polygon points="5 3 19 12 5 21 5 3" fill="currentColor"/></svg>Try Now</>
                  )}
                </button>
              </Magnet>

              {/* Sign in with Google */}
              <Magnet className="w-full sm:w-auto">
                <button onClick={isSignedIn ? () => navigate('/app') : doGoogleSignIn}
                  disabled={signingIn || loading}
                  className="w-full sm:w-auto flex items-center justify-center gap-2.5 bg-bg3 border border-border2 text-sage px-6 sm:px-7 py-3.5 sm:py-4 rounded-xl font-semibold text-base hover:border-lime/50 hover:text-lime transition-all duration-200 disabled:opacity-50">
                  {isSignedIn ? (
                    <><svg className="w-4 h-4 text-lime shrink-0" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>Go to App</>
                  ) : (
                    <><GoogleIcon />Sign in with Google</>
                  )}
                </button>
              </Magnet>
            </div>

            {error && (
              <p className="text-red-400 text-sm bg-red-900/20 border border-red-800/40 rounded-lg px-4 py-2 mt-3">
                {error}
              </p>
            )}
          </AnimatedContent>
        </div>
      </section>

      {/* FEATURES */}
      <section id="features" className="py-16 sm:py-24 px-4 sm:px-6 bg-bg2">
        <div className="max-w-6xl mx-auto">
          <AnimatedContent direction="up">
            <p className="text-xs font-bold text-olive uppercase tracking-[0.12em] mb-2">What you get</p>
            <h2 className="text-[clamp(1.6rem,4vw,2.6rem)] font-extrabold text-lime tracking-tight mb-8 sm:mb-14">
              Everything for flawless face swaps
            </h2>
          </AnimatedContent>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 sm:gap-5">
            {FEATURES.map((f, i) => (
              <AnimatedContent key={f.title} delay={i * 0.07} direction="up">
                <SpotlightCard className="bg-bg3 border border-forest rounded-2xl p-5 sm:p-7 h-full hover:border-olive hover:-translate-y-1 transition-all duration-200">
                  <div className="w-10 h-10 sm:w-11 sm:h-11 bg-olive/20 border border-olive/40 rounded-xl flex items-center justify-center text-lime mb-3 sm:mb-4">
                    {f.icon}
                  </div>
                  <h3 className="text-lime font-bold mb-1.5 text-sm sm:text-base">{f.title}</h3>
                  <p className="text-sage text-sm leading-relaxed">{f.desc}</p>
                </SpotlightCard>
              </AnimatedContent>
            ))}
          </div>
        </div>
      </section>

      {/* PIPELINE */}
      <section className="py-16 sm:py-24 px-4 sm:px-6 bg-bg">
        <div className="max-w-4xl mx-auto">
          <AnimatedContent direction="up">
            <p className="text-xs font-bold text-olive uppercase tracking-[0.12em] mb-2">Processing pipeline</p>
            <h2 className="text-[clamp(1.6rem,4vw,2.6rem)] font-extrabold text-lime tracking-tight mb-8 sm:mb-14">
              Professional grade workflow
            </h2>
          </AnimatedContent>
          <div className="relative">
            <div className="absolute top-[18px] left-[6%] right-[6%] h-0.5 bg-gradient-to-r from-olive to-lime hidden md:block" />
            <div className="grid grid-cols-4 md:flex md:justify-center gap-y-6 md:gap-0">
              {PIPELINE.map((step, i) => (
                <AnimatedContent key={step} delay={i * 0.06} direction="up"
                  className="flex flex-col items-center gap-2 md:px-4 z-10">
                  <div className="w-9 h-9 sm:w-10 sm:h-10 bg-bg2 border-2 border-olive rounded-full flex items-center justify-center text-xs font-extrabold text-lime hover:bg-olive hover:border-lime transition-all duration-200 cursor-default">
                    {i + 1}
                  </div>
                  <span className="text-[10px] sm:text-xs text-sage font-semibold text-center leading-tight max-w-[60px] sm:max-w-[80px]">{step}</span>
                </AnimatedContent>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* FOOTER */}
      <footer className="bg-bg2 border-t border-forest py-5 px-4 text-center text-xs text-[#6b7345] leading-relaxed">
        <span className="text-sage">DeepFace Studio</span>
        &nbsp;&nbsp;
        Educational use only &nbsp; All processing is local &nbsp; &copy; Aditya Raj 2026
      </footer>

    </div>
  );
}
