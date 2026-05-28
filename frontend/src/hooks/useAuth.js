import { useEffect, useState } from 'react';
import { auth, onAuthStateChanged, signOut } from '../firebase';

const SESSION_DAYS = 10;
const SESSION_MS   = SESSION_DAYS * 24 * 60 * 60 * 1000;
const LOGIN_KEY    = 'df_login_time';
const TOKEN_KEY    = 'df_token';
const NAME_KEY     = 'df_name';
const PHOTO_KEY    = 'df_photo';

export function saveSession(user, token) {
  localStorage.setItem(LOGIN_KEY, Date.now().toString());
  localStorage.setItem(TOKEN_KEY, token);
  localStorage.setItem(NAME_KEY,  user.displayName || user.email || '');
  localStorage.setItem(PHOTO_KEY, user.photoURL || '');
}

export function clearSession() {
  [LOGIN_KEY, TOKEN_KEY, NAME_KEY, PHOTO_KEY].forEach(k => localStorage.removeItem(k));
  sessionStorage.clear();
}

function isSessionValid() {
  const t = localStorage.getItem(LOGIN_KEY);
  if (!t) return false;
  return Date.now() - parseInt(t, 10) < SESSION_MS;
}

export function daysLeft() {
  const t = localStorage.getItem(LOGIN_KEY);
  if (!t) return 0;
  const remaining = SESSION_MS - (Date.now() - parseInt(t, 10));
  return Math.max(0, Math.ceil(remaining / (24 * 60 * 60 * 1000)));
}

export default function useAuth() {
  const [user,    setUser]    = useState(undefined); // undefined = still loading
  const [expired, setExpired] = useState(false);

  useEffect(() => {
    const unsub = onAuthStateChanged(auth, async (fbUser) => {
      if (!fbUser) {
        clearSession();
        setUser(null);
        return;
      }

      // Session expired: sign out silently
      if (!isSessionValid()) {
        await signOut(auth);
        clearSession();
        setExpired(true);
        setUser(null);
        return;
      }

      // Refresh token
      try {
        const token = await fbUser.getIdToken(true);
        localStorage.setItem(TOKEN_KEY, token);
        sessionStorage.setItem(TOKEN_KEY, token);
      } catch {/* no-op */}

      setUser(fbUser);
    });
    return unsub;
  }, []);

  return {
    user,                              // null = not signed in, undefined = loading
    loading:   user === undefined,
    signedIn:  !!user,
    expired,
    daysLeft:  daysLeft(),
  };
}
