import React, { useState, useRef, useEffect } from 'react';
import './App.css';
import BodyTracking from './BodyTracking';

function App() {
  const [showWelcome, setShowWelcome] = useState(true);
  const [showCamera, setShowCamera] = useState(false);
  const [showBodyTracking, setShowBodyTracking] = useState(false);
  const [cameraReady, setCameraReady] = useState(false);
  const videoRef = useRef(null);
  const streamRef = useRef(null);

  useEffect(() => {
    // Hide welcome screen after 3 seconds
    const timer = setTimeout(() => {
      setShowWelcome(false);
    }, 3000);

    return () => clearTimeout(timer);
  }, []);

  const startCamera = async () => {
    setShowCamera(true);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { facingMode: 'user' },
        audio: false 
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setCameraReady(true);
      }
    } catch (err) {
      console.error("Error accessing camera:", err);
      alert("Unable to access camera. Please allow camera permissions.");
    }
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    setShowCamera(false);
    setCameraReady(false);
  };

  const startBodyTracking = () => {
    setShowBodyTracking(true);
  };

  const stopBodyTracking = () => {
    setShowBodyTracking(false);
  };

  // If body tracking is active, show only that component
  if (showBodyTracking) {
    return <BodyTracking onClose={stopBodyTracking} />;
  }

  return (
    <div className="App">
      {/* Welcome Screen */}
      {showWelcome && (
        <div className="welcome-screen">
          <div className="welcome-content">
            <div className="ar-icon">📱</div>
            <h1 className="welcome-title">
              Welcome to <span className="gradient-text">AR Try-On</span>
            </h1>
            <p className="welcome-subtitle">Experience virtual fitting like never before</p>
            <div className="loading-dots">
              <span></span>
              <span></span>
              <span></span>
            </div>
          </div>
        </div>
      )}

      {/* Main Content */}
      {!showWelcome && (
        <div className="main-content">
          {!showCamera ? (
            <div className="home-screen">
              <h1 className="app-title">
                <span className="gradient-text">AR Try-On</span>
              </h1>
              <p className="app-subtitle">Virtual Try-On Experience</p>
              
              <div className="phone-container" onClick={startCamera}>
                <div className="phone-frame">
                  <div className="phone-notch"></div>
                  <div className="phone-screen">
                    <div className="camera-icon-wrapper">
                      <svg className="camera-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z" />
                        <circle cx={12} cy={13} r={4} />
                      </svg>
                      <p className="tap-text">Tap to open camera</p>
                    </div>
                  </div>
                  <div className="phone-button"></div>
                </div>
              </div>

              <div className="features">
                <div className="feature-item">
                  <span className="feature-icon">✨</span>
                  <span>Real-time AR</span>
                </div>
                <div className="feature-item">
                  <span className="feature-icon">🎯</span>
                  <span>Accurate Fitting</span>
                </div>
                <div className="feature-item">
                  <span className="feature-icon">📸</span>
                  <span>Instant Preview</span>
                </div>
              </div>

              <button 
                onClick={startBodyTracking}
                style={{
                  marginTop: '40px',
                  padding: '15px 40px',
                  fontSize: '18px',
                  fontWeight: '600',
                  background: 'linear-gradient(135deg, #00ff88 0%, #00cc66 100%)',
                  border: 'none',
                  borderRadius: '15px',
                  color: '#000',
                  cursor: 'pointer',
                  boxShadow: '0 10px 30px rgba(0,255,136,0.3)',
                  transition: 'all 0.3s ease',
                }}
                onMouseOver={(e) => {
                  e.target.style.transform = 'translateY(-3px)';
                  e.target.style.boxShadow = '0 15px 40px rgba(0,255,136,0.4)';
                }}
                onMouseOut={(e) => {
                  e.target.style.transform = 'translateY(0)';
                  e.target.style.boxShadow = '0 10px 30px rgba(0,255,136,0.3)';
                }}
              >
                🏃‍♂️ Launch Body Tracking
              </button>
            </div>
          ) : (
            <div className="camera-screen">
              <div className="camera-header">
                <button className="close-btn" onClick={stopCamera}>✕</button>
                <h2>AR Try-On Camera</h2>
                <div></div>
              </div>
              
              <div className="phone-camera-container">
                <div className="phone-frame active">
                  <div className="phone-notch"></div>
                  <div className="phone-screen">
                    <video 
                      ref={videoRef} 
                      autoPlay 
                      playsInline
                      className="camera-feed"
                    />
                    {!cameraReady && (
                      <div className="camera-loading">
                        <div className="spinner"></div>
                        <p>Starting camera...</p>
                      </div>
                    )}
                  </div>
                  <div className="phone-button"></div>
                </div>
              </div>

              <div className="camera-controls">
                <button className="control-btn">
                  <span className="control-icon">🔄</span>
                  <span>Flip</span>
                </button>
                <button className="control-btn capture">
                  <span className="control-icon">📷</span>
                  <span>Capture</span>
                </button>
                <button className="control-btn">
                  <span className="control-icon">🎨</span>
                  <span>Effects</span>
                </button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default App;
