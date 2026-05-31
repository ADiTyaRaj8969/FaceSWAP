import { useNavigate, Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import SplitText from '../components/ui/SplitText';
import BlurText from '../components/ui/BlurText';
import AnimatedContent from '../components/ui/AnimatedContent';
import Magnet from '../components/ui/Magnet';
import ShinyText from '../components/ui/ShinyText';
import Typewriter from '../components/ui/Typewriter';

export default function LandingPage() {
  const navigate = useNavigate();

  return (
    // min-h-screen and overflow-x-hidden allow normal scrolling, but we make the Hero 100vh
    <div className="min-h-screen w-full bg-bg font-sans overflow-x-hidden flex flex-col text-navy">

      {/* ═══════════════ TOP LOGOS (No Navbar) ═══════════════ */}
      <div className="absolute top-0 left-0 right-0 z-50 flex items-start justify-between px-3 sm:px-8 pt-3 sm:pt-4 pointer-events-none gap-2">
        {/* Left — Marwadi University */}
        <div className="shrink-0 pointer-events-auto max-w-[48%]">
          <img src="/logos/marwadi_university.jpeg" alt="Marwadi University"
            className="h-10 sm:h-14 md:h-20 w-auto object-contain mix-blend-multiply" />
        </div>

        {/* Right — Faculty of Engineering */}
        <div className="shrink-0 pointer-events-auto mt-1 sm:mt-2 max-w-[48%] flex justify-end">
          <img src="/logos/engineering_faculty.jpeg" alt="Faculty of Engineering"
            className="h-7 sm:h-12 md:h-16 w-auto object-contain mix-blend-multiply" />
        </div>
      </div>


      {/* ═══════════════════════════ HERO (Takes up exactly 100vh) ═══════════════════════════ */}
      <main className="h-screen min-h-[550px] relative flex flex-col items-center justify-center px-4 sm:px-6">
        {/* Subtle grid background */}
        <div className="absolute inset-0 pointer-events-none" style={{
          backgroundImage: 'linear-gradient(rgba(14,116,144,0.02) 1px,transparent 1px),linear-gradient(90deg,rgba(14,116,144,0.02) 1px,transparent 1px)',
          backgroundSize: '60px 60px',
        }} />

        <div className="relative z-10 w-full max-w-4xl mx-auto text-center flex flex-col items-center gap-6 sm:gap-8 -mt-10 sm:-mt-16 md:-mt-20">

          {/* ICT Logo - Fixed */}
          <AnimatedContent delay={0} direction="down" distance={20}>
            <div className="inline-flex px-4">
              <img src="/logos/ict_logo.png" alt="ICT Department"
                className="w-[180px] sm:w-[280px] md:w-[380px] max-w-full h-auto object-contain mix-blend-multiply" />
            </div>
          </AnimatedContent>

          {/* Headline */}
          <div className="w-full">
            <h1 className="text-[clamp(2.2rem,6vw,4.5rem)] font-black leading-[1.05] tracking-[-0.04em] text-navy flex flex-col md:flex-row items-center justify-center md:gap-x-1">
              <span className="inline-flex items-center justify-center md:justify-start h-[1.2em] overflow-visible whitespace-nowrap w-full md:w-[6.4em]">
                <Typewriter words={[
                  "Your Future",       // English
                  "તમારું ભવિષ્ય",       // Gujarati
                  "మీ భవిష్యత్తు",          // Telugu
                  "आपका भविष्य",         // Hindi
                  "Seu Futuro"         // Portuguese
                ]} />
              </span>
              <ShinyText text="Starts Here" className="inline-block mt-2 md:mt-4 text-[clamp(1.4rem,4vw,2.5rem)] opacity-90" />
            </h1>
          </div>

          {/* Subtitle */}
          <BlurText
            text="Visualize your potential. Experience campus life virtually and see yourself wearing our colors in just one click."
            className="text-sm sm:text-base text-slate max-w-2xl leading-relaxed px-2 font-medium"
            delay={0.3}
          />

          {/* CTA Button */}
          <AnimatedContent delay={0.45} direction="up" distance={20} className="w-full max-w-xs sm:max-w-none mt-2">
            <div className="flex justify-center">
              <Magnet className="w-full sm:w-auto">
                <button onClick={() => navigate('/app')}
                  className="w-full sm:w-auto flex items-center justify-center gap-2.5 bg-teal text-white px-10 py-3.5 sm:py-4 rounded-xl font-bold text-sm sm:text-base shadow-lg shadow-teal/20 hover:shadow-xl hover:shadow-teal/30 hover:-translate-y-0.5 active:scale-95 transition-all duration-200">
                  <svg className="w-5 h-5 shrink-0" viewBox="0 0 24 24"><polygon points="5 3 19 12 5 21 5 3" fill="currentColor" /></svg>
                  Enter Studio
                </button>
              </Magnet>
            </div>
          </AnimatedContent>

          {/* Excellence Trust Bar */}
          <AnimatedContent delay={0.6} direction="up" distance={10} className="mt-6 sm:mt-10 w-full">
            <div className="flex items-center justify-center opacity-90 px-4">
              <img src="/logos/qs_ranking.png" alt="QS Ranking" className="h-12 sm:h-16 md:h-20 w-auto object-contain mix-blend-multiply" />
            </div>
          </AnimatedContent>
        </div>
      </main>


      {/* ═══════════════════════════ BOTTOM AREA (Scrollable) ═══════════════════════════ */}
      <div className="flex flex-col bg-white">

        {/* Footer Strip */}
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
