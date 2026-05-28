import { useEffect, useRef, useState } from 'react';
import { useInView } from 'framer-motion';

/* Counts from 0 to `end` when scrolled into view */
export default function CountUp({ end, suffix = '', duration = 1.8, className = '' }) {
  const ref    = useRef(null);
  const inView = useInView(ref, { once: true });
  const [val, setVal] = useState(0);

  useEffect(() => {
    if (!inView) return;
    let start = 0;
    const steps    = 60;
    const stepTime = (duration * 1000) / steps;

    const timer = setInterval(() => {
      start++;
      const progress = start / steps;
      const eased    = 1 - Math.pow(1 - progress, 3);
      setVal(Math.round(eased * end));
      if (start >= steps) clearInterval(timer);
    }, stepTime);

    return () => clearInterval(timer);
  }, [inView, end, duration]);

  return (
    <span ref={ref} className={className}>
      {val}{suffix}
    </span>
  );
}
