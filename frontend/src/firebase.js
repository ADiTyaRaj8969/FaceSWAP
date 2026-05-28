import { initializeApp }  from 'firebase/app';
import { getAuth, GoogleAuthProvider, signInWithPopup, signOut, onAuthStateChanged } from 'firebase/auth';

/*
  FILL IN YOUR FIREBASE CONFIG
  Go to: https://console.firebase.google.com
  Create project -> Add Web App -> Copy the config object here
*/
const firebaseConfig = {
  apiKey:            'YOUR_API_KEY',
  authDomain:        'YOUR_PROJECT_ID.firebaseapp.com',
  projectId:         'YOUR_PROJECT_ID',
  storageBucket:     'YOUR_PROJECT_ID.appspot.com',
  messagingSenderId: 'YOUR_SENDER_ID',
  appId:             'YOUR_APP_ID',
};

const fbApp  = initializeApp(firebaseConfig);
const auth   = getAuth(fbApp);
const googleProvider = new GoogleAuthProvider();

export { auth, googleProvider, signInWithPopup, signOut, onAuthStateChanged };
