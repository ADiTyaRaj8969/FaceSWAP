import { useNavigate } from 'react-router-dom';
import BlurText from '../components/ui/BlurText';
import AnimatedContent from '../components/ui/AnimatedContent';
import Magnet from '../components/ui/Magnet';
import ShinyText from '../components/ui/ShinyText';
import Typewriter from '../components/ui/Typewriter';

export default function LandingPage() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen w-full bg-bg font-sans overflow-x-hidden flex flex-col text-navy">

      {/* ═══ TOP LOGOS ═══ */}
      <div className="absolute top-0 left-0 right-0 z-50 flex items-start justify-between px-3 sm:px-6 pt-3 sm:pt-4 pointer-events-none gap-2">
        <div className="shrink-0 pointer-events-auto max-w-[45%]">
          <img src="/logos/marwadi_university.jpeg" alt="Marwadi University"
            className="h-8 sm:h-12 md:h-16 w-auto max-w-full object-contain mix-blend-multiply" />
        </div>
        <div className="shrink-0 pointer-events-auto mt-1 sm:mt-2 max-w-[45%] flex justify-end">
          <img src="/logos/engineering_faculty.jpeg" alt="Faculty of Engineering"
            className="h-6 sm:h-10 md:h-14 w-auto max-w-full object-contain mix-blend-multiply" />
        </div>
      </div>

      {/* ═══ HERO ═══ */}
      <main className="flex-1 relative flex flex-col items-center justify-center px-4 sm:px-6 py-24 sm:py-28 min-h-screen">
        {/* Grid background */}
        <div className="absolute inset-0 pointer-events-none" style={{
          backgroundImage: 'linear-gradient(rgba(14,116,144,0.02) 1px,transparent 1px),linear-gradient(90deg,rgba(14,116,144,0.02) 1px,transparent 1px)',
          backgroundSize: '60px 60px',
        }} />

        <div className="relative z-10 w-full max-w-3xl mx-auto text-center flex flex-col items-center gap-5 sm:gap-7">

          {/* ICT Logo */}
          <AnimatedContent delay={0} direction="down" distance={20}>
            <img src="/logos/ict_logo.png" alt="ICT Department"
              className="w-[140px] sm:w-[220px] md:w-[300px] max-w-[70vw] h-auto object-contain mix-blend-multiply mx-auto" />
          </AnimatedContent>

          {/* Headline */}
          <div className="w-full px-2">
            <h1 className="font-black tracking-tight text-navy flex flex-col items-center gap-1 sm:gap-2"
                style={{ fontSize: 'clamp(1.8rem, 6vw, 4rem)', lineHeight: 1.25 }}>
              {/* Typewriter - extra line-height + padding so Gujarati/Telugu/Hindi
                  vowel marks (above & below the baseline) are never clipped */}
              <span className="block w-full text-center leading-[1.5] py-1.5 overflow-visible">
                <Typewriter words={[
                  'Your Future',
                  'તમારું ભવિષ્ય',
                  'మీ భవిష్యత్తు',
                  'आपका भविष्य',
                  'Seu Futuro',
                ]} />
              </span>
              <ShinyText text="Starts Here"
                className="block text-[0.65em] sm:text-[0.7em] opacity-90" />
            </h1>
          </div>

          {/* Subtitle */}
          <BlurText
            text="Visualize your potential. Experience campus life virtually and see yourself wearing our colors in just one click."
            className="text-xs sm:text-sm md:text-base text-slate max-w-xl leading-relaxed px-2 font-medium"
            delay={0.3}
          />

          {/* CTA */}
          <AnimatedContent delay={0.45} direction="up" distance={20} className="w-full mt-1">
            <div className="flex justify-center px-4 sm:px-0">
              <Magnet className="w-full sm:w-auto max-w-xs sm:max-w-none">
                <button onClick={() => navigate('/app')}
                  className="w-full sm:w-auto flex items-center justify-center gap-2.5 bg-teal text-white px-8 sm:px-10 py-3.5 rounded-xl font-bold text-sm sm:text-base shadow-lg shadow-teal/20 hover:shadow-xl hover:shadow-teal/30 hover:-translate-y-0.5 active:scale-95 transition-all duration-200">
                  <svg className="w-4 h-4 sm:w-5 sm:h-5 shrink-0" viewBox="0 0 24 24">
                    <polygon points="5 3 19 12 5 21 5 3" fill="currentColor" />
                  </svg>
                  Enter Studio
                </button>
              </Magnet>
            </div>
          </AnimatedContent>

          {/* QS Ranking badge */}
          <AnimatedContent delay={0.6} direction="up" distance={10} className="mt-2 sm:mt-6 w-full">
            <div className="flex items-center justify-center px-4">
              <img src="/logos/qs_ranking.png" alt="QS Ranking"
                className="h-10 sm:h-14 md:h-18 w-auto max-w-[80vw] object-contain mix-blend-multiply opacity-90" />
            </div>
          </AnimatedContent>
        </div>
      </main>

      {/* ═══ FOOTER ═══ */}
      <footer className="bg-bg3 border-t border-border py-3 px-4">
        <div className="max-w-6xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-2 sm:gap-4 text-xs sm:text-sm text-slate">
          <div className="flex flex-col sm:flex-row items-center gap-2 sm:gap-4 text-center sm:text-left w-full sm:w-auto">
            <img src="/logos/ict_logo_black.png" alt="ICT Department" className="h-6 sm:h-8 w-auto opacity-70" />
            <div className="w-10 h-px sm:w-px sm:h-7 bg-border" />
            <div className="flex flex-col leading-tight">
              <strong className="text-navy">DeepFace Studio</strong>
              <span>Dept. of ICT, Marwadi University</span>
            </div>
          </div>
          <div className="flex items-center gap-2 flex-wrap justify-center font-medium text-[11px] sm:text-xs">
            <span>Educational use only</span>
            <span className="text-border2">·</span>
            <span>Local processing</span>
            <span className="text-border2">·</span>
            <span>© Aditya Raj 2026</span>
          </div>
        </div>
      </footer>

    </div>
  );
}
