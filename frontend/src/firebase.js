import { initializeApp }  from 'firebase/app';
import { getAuth, GoogleAuthProvider, signInWithPopup, signOut, onAuthStateChanged } from 'firebase/auth';

const firebaseConfig = {
  apiKey:            "AIzaSyCF9t3xPf_Se4qkqAHFRoW-YqG8LfoQIpo",
  authDomain:        "deepface-aditya.firebaseapp.com",
  projectId:         "deepface-aditya",
  storageBucket:     "deepface-aditya.firebasestorage.app",
  messagingSenderId: "596616284131",
  appId:             "1:596616284131:web:8ad02bce5645a03e1b42d6",
  measurementId:     "G-44EVTLV0VL",
};

const fbApp          = initializeApp(firebaseConfig);
const auth           = getAuth(fbApp);
const googleProvider = new GoogleAuthProvider();

export { auth, googleProvider, signInWithPopup, signOut, onAuthStateChanged };
