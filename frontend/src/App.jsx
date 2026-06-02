import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import LandingPage from './pages/LandingPage';
import AppPage     from './pages/AppPage';
import AdminPage   from './pages/AdminPage';
import ThreeDotMenu from './components/ui/ThreeDotMenu';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/"      element={<LandingPage />} />
        <Route path="/app"   element={<AppPage />} />
        <Route path="/admin" element={<AdminPage />} />
        <Route path="*"      element={<Navigate to="/" replace />} />
      </Routes>
      {/* Three-dot menu visible on every page except admin */}
      <Routes>
        <Route path="/admin" element={null} />
        <Route path="*"      element={<ThreeDotMenu />} />
      </Routes>
    </BrowserRouter>
  );
}
