import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import LandingPage from './pages/LandingPage';
import AppPage     from './pages/AppPage';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/"    element={<LandingPage />} />
        <Route path="/app" element={<AppPage />} />
        <Route path="*"    element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}
