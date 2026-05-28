/* Shiny shimmer gradient text -- React Bits style */
export default function ShinyText({ text, className = '' }) {
  return (
    <span
      className={`inline-block bg-clip-text text-transparent ${className}`}
      style={{
        backgroundImage: 'linear-gradient(120deg, #636B2F 0%, #D4DE95 40%, #BAC095 60%, #636B2F 100%)',
        backgroundSize: '200% auto',
        animation: 'shine 2.5s linear infinite',
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
