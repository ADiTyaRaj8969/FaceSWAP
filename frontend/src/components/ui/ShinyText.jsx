/* Shiny shimmer gradient text — Marwadi teal/gold */
export default function ShinyText({ text, className = '' }) {
  return (
    <span
      className={`inline-block bg-clip-text text-transparent ${className}`}
      style={{
        backgroundImage: 'linear-gradient(120deg, #0e7490 0%, #22b8cf 25%, #c5a55a 50%, #e0c97d 65%, #22b8cf 80%, #0e7490 100%)',
        backgroundSize: '200% auto',
        animation: 'shine 3s linear infinite',
      }}
    >
      {text}
      <style>{`
        @keyframes shine {
          from { background-position: 200% center; }
          to   { background-position: -200% center; }
        }
      `}</style>
    </span>
  );
}
