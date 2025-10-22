import React, { useEffect, useRef, useState } from 'react';
import * as posenet from '@tensorflow-models/posenet';
import '@tensorflow/tfjs-backend-webgl';
import '@tensorflow/tfjs-backend-cpu';

// CSS animations
const styleSheet = document.createElement("style");
styleSheet.textContent = `
  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
  @keyframes slideIn {
    from { opacity: 0; transform: translateY(-10px); }
    to { opacity: 1; transform: translateY(0); }
  }
`;
if (!document.head.querySelector('style[data-spinner]')) {
  styleSheet.setAttribute('data-spinner', 'true');
  document.head.appendChild(styleSheet);
}

const BodyTracking = ({ onClose }) => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const modelRef = useRef(null);
  const animationIdRef = useRef(null);
  const frameCountRef = useRef(0);
  const keypointHistoryRef = useRef([]);
  const SMOOTHING_WINDOW = 5;
  const previousKeypointsRef = useRef(null);
  const MOVEMENT_THRESHOLD = 15;
  const CONFIDENCE_THRESHOLD = 0.5;
  const FRAME_SKIP = 5;
  
  const [modelLoaded, setModelLoaded] = useState(false);
  const [modelError, setModelError] = useState(null);
  const [detecting, setDetecting] = useState(false);
  const [motionLog, setMotionLog] = useState([]);

  const BODY_PARTS = {
    head: [0, 1, 2, 3, 4],
    leftArm: [5, 7, 9],
    rightArm: [6, 8, 10],
    leftLeg: [11, 13, 15],
    rightLeg: [12, 14, 16],
    torso: [5, 6, 11, 12],
  };

  const BODY_COLORS = {
    head: { fill: 'rgba(255, 0, 0, 0.7)', stroke: '#ff0000', name: 'Red' },
    leftArm: { fill: 'rgba(255, 0, 255, 0.7)', stroke: '#ff00ff', name: 'Magenta' },
    rightArm: { fill: 'rgba(0, 255, 255, 0.7)', stroke: '#00ffff', name: 'Cyan' },
    leftLeg: { fill: 'rgba(0, 255, 0, 0.7)', stroke: '#00ff00', name: 'Green' },
    rightLeg: { fill: 'rgba(255, 255, 0, 0.7)', stroke: '#ffff00', name: 'Yellow' },
    torso: { fill: 'rgba(255, 165, 0, 0.7)', stroke: '#ffa500', name: 'Orange' },
  };

  useEffect(() => {
    let stream = null;

    const setupCamera = async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
          audio: false
        });

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          return new Promise((resolve) => {
            videoRef.current.onloadedmetadata = () => {
              videoRef.current.width = 640;
              videoRef.current.height = 480;
              videoRef.current.play();
              resolve();
            };
          });
        }
      } catch (err) {
        console.error('Error accessing camera:', err);
        alert('Unable to access camera. Please grant camera permissions.');
      }
    };

    const loadModel = async () => {
      try {
        console.log('%c🚀 Loading PoseNet Model...', 'color: #00ff88; font-size: 18px; font-weight: bold;');
        
        const net = await posenet.load({
          architecture: 'MobileNetV1',
          outputStride: 16,
          inputResolution: { width: 640, height: 480 },
          multiplier: 0.75,
          quantBytes: 2
        });
        
        if (!net) throw new Error('Model loaded but returned null');
        
        modelRef.current = net;
        setModelLoaded(true);
        setModelError(null);
        console.log('%c✅ PoseNet Model Loaded Successfully!', 'color: #00ff88; font-size: 18px; font-weight: bold;');
      } catch (err) {
        console.error('❌ Error loading PoseNet model:', err);
        let errorMessage = 'Failed to load model. ';
        if (err.message.includes('fetch') || err.message.includes('network')) {
          errorMessage += 'Check your internet connection.';
        } else {
          errorMessage += err.message;
        }
        setModelError(errorMessage);
      }
    };

    const init = async () => {
      await setupCamera();
      await loadModel();
    };

    init();

    return () => {
      if (stream) stream.getTracks().forEach(track => track.stop());
      if (animationIdRef.current) cancelAnimationFrame(animationIdRef.current);
    };
  }, []);

  const applyTemporalSmoothing = (keypoints) => {
    keypointHistoryRef.current.push(keypoints);
    if (keypointHistoryRef.current.length > SMOOTHING_WINDOW) {
      keypointHistoryRef.current.shift();
    }
    if (keypointHistoryRef.current.length < 2) return keypoints;
    
    const smoothedKeypoints = keypoints.map((kp, i) => {
      let sumX = 0, sumY = 0, sumScore = 0, count = 0;
      keypointHistoryRef.current.forEach(frame => {
        if (frame[i] && frame[i].score >= CONFIDENCE_THRESHOLD) {
          sumX += frame[i].position.x;
          sumY += frame[i].position.y;
          sumScore += frame[i].score;
          count++;
        }
      });
      if (count === 0) return kp;
      return {
        ...kp,
        position: { x: sumX / count, y: sumY / count },
        score: sumScore / count
      };
    });
    return smoothedKeypoints;
  };

  const calculatePartCenter = (keypoints, partIndices) => {
    let sumX = 0, sumY = 0, count = 0;
    partIndices.forEach(idx => {
      const kp = keypoints[idx];
      if (kp && kp.score >= CONFIDENCE_THRESHOLD) {
        sumX += kp.position.x;
        sumY += kp.position.y;
        count++;
      }
    });
    if (count === 0) return null;
    return { x: sumX / count, y: sumY / count };
  };

  const analyzeMovement = (smoothedKeypoints) => {
    if (!smoothedKeypoints || smoothedKeypoints.length === 0) return;

    const currentParts = {
      head: calculatePartCenter(smoothedKeypoints, BODY_PARTS.head),
      leftArm: calculatePartCenter(smoothedKeypoints, BODY_PARTS.leftArm),
      rightArm: calculatePartCenter(smoothedKeypoints, BODY_PARTS.rightArm),
      leftLeg: calculatePartCenter(smoothedKeypoints, BODY_PARTS.leftLeg),
      rightLeg: calculatePartCenter(smoothedKeypoints, BODY_PARTS.rightLeg),
      torso: calculatePartCenter(smoothedKeypoints, BODY_PARTS.torso),
    };

    if (previousKeypointsRef.current) {
      const movements = [];

      const headCurrent = currentParts.head;
      const headPrevious = previousKeypointsRef.current.head;
      if (headCurrent && headPrevious) {
        const headDeltaX = headCurrent.x - headPrevious.x;
        const headDeltaY = headCurrent.y - headPrevious.y;
        if (Math.abs(headDeltaX) > MOVEMENT_THRESHOLD) {
          const direction = headDeltaX > 0 ? 'left' : 'right';
          const message = `👤 Head turned ${direction} (${Math.abs(headDeltaX).toFixed(1)}px)`;
          movements.push(`Head turned ${direction}`);
          console.log(`%c${message}`, 'color: #ff0000; font-weight: bold; font-size: 14px;');
        }
        if (Math.abs(headDeltaY) > MOVEMENT_THRESHOLD) {
          const direction = headDeltaY > 0 ? 'down' : 'up';
          const message = `👤 Head tilted ${direction} (${Math.abs(headDeltaY).toFixed(1)}px)`;
          movements.push(`Head tilted ${direction}`);
          console.log(`%c${message}`, 'color: #ff0000; font-weight: bold; font-size: 14px;');
        }
      }

      ['leftArm', 'rightArm', 'leftLeg', 'rightLeg'].forEach(part => {
        const current = currentParts[part];
        const previous = previousKeypointsRef.current[part];
        if (current && previous) {
          const distance = Math.sqrt(
            Math.pow(current.x - previous.x, 2) + Math.pow(current.y - previous.y, 2)
          );
          if (distance > MOVEMENT_THRESHOLD) {
            const partName = part.replace(/([A-Z])/g, ' $1').trim();
            const emoji = part.includes('Arm') ? '💪' : '🦵';
            const message = `${emoji} ${partName} moved (${distance.toFixed(1)}px)`;
            movements.push(`${partName} movement`);
            const color = BODY_COLORS[part].stroke;
            console.log(`%c${message}`, `color: ${color}; font-weight: bold; font-size: 14px;`);
          }
        }
      });

      const torsoCurrent = currentParts.torso;
      const torsoPrevious = previousKeypointsRef.current.torso;
      if (torsoCurrent && torsoPrevious) {
        const distance = Math.sqrt(
          Math.pow(torsoCurrent.x - torsoPrevious.x, 2) +
          Math.pow(torsoCurrent.y - torsoPrevious.y, 2)
        );
        if (distance > MOVEMENT_THRESHOLD * 1.5) {
          const message = `🚶 Body moved (${distance.toFixed(1)}px)`;
          movements.push('Body movement');
          console.log(`%c${message}`, 'color: #ffa500; font-weight: bold; font-size: 14px;');
        }
      }

      if (movements.length > 0) {
        setMotionLog(prev => {
          const timestamp = new Date().toLocaleTimeString();
          const newEntries = movements.map(m => ({ text: m, time: timestamp }));
          return [...newEntries, ...prev].slice(0, 10);
        });
      }
    }

    previousKeypointsRef.current = currentParts;
  };

  // OpenPose-style skeleton connections
  const SKELETON_CONNECTIONS = [
    // Face connections
    { start: 0, end: 1, color: '#FF0000', name: 'nose-leftEye' },
    { start: 0, end: 2, color: '#FF0000', name: 'nose-rightEye' },
    { start: 1, end: 3, color: '#FF0000', name: 'leftEye-leftEar' },
    { start: 2, end: 4, color: '#FF0000', name: 'rightEye-rightEar' },
    
    // Torso connections
    { start: 5, end: 6, color: '#FFA500', name: 'shoulders' },
    { start: 5, end: 11, color: '#FFA500', name: 'leftTorso' },
    { start: 6, end: 12, color: '#FFA500', name: 'rightTorso' },
    { start: 11, end: 12, color: '#FFA500', name: 'hips' },
    
    // Left arm connections
    { start: 5, end: 7, color: '#FF00FF', name: 'leftUpperArm' },
    { start: 7, end: 9, color: '#FF00FF', name: 'leftForearm' },
    
    // Right arm connections
    { start: 6, end: 8, color: '#00FFFF', name: 'rightUpperArm' },
    { start: 8, end: 10, color: '#00FFFF', name: 'rightForearm' },
    
    // Left leg connections
    { start: 11, end: 13, color: '#00FF00', name: 'leftThigh' },
    { start: 13, end: 15, color: '#00FF00', name: 'leftShin' },
    
    // Right leg connections
    { start: 12, end: 14, color: '#FFFF00', name: 'rightThigh' },
    { start: 14, end: 16, color: '#FFFF00', name: 'rightShin' },
  ];

  const drawBodyParts = (keypoints) => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;

    const ctx = canvas.getContext('2d');
    const videoWidth = video.videoWidth || 640;
    const videoHeight = video.videoHeight || 480;
    
    canvas.width = videoWidth;
    canvas.height = videoHeight;
    
    // Clear canvas (transparent, so video shows through)
    ctx.clearRect(0, 0, videoWidth, videoHeight);

    // Draw thick colored body segments covering the whole body area
    ctx.lineWidth = 60; // Very thick lines to cover body area
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    
    SKELETON_CONNECTIONS.forEach(connection => {
      const startKp = keypoints[connection.start];
      const endKp = keypoints[connection.end];
      
      if (startKp && endKp && 
          startKp.score >= CONFIDENCE_THRESHOLD && 
          endKp.score >= CONFIDENCE_THRESHOLD) {
        
        const startX = videoWidth - startKp.position.x;
        const startY = startKp.position.y;
        const endX = videoWidth - endKp.position.x;
        const endY = endKp.position.y;
        
        // Draw thick colored line with transparency
        ctx.strokeStyle = connection.color + 'BB'; // Semi-transparent
        ctx.shadowColor = connection.color;
        ctx.shadowBlur = 15;
        
        ctx.beginPath();
        ctx.moveTo(startX, startY);
        ctx.lineTo(endX, endY);
        ctx.stroke();
      }
    });

    // Draw large filled circles at each keypoint to cover body parts
    Object.entries(BODY_PARTS).forEach(([partName, indices]) => {
      const color = BODY_COLORS[partName];
      const validKeypoints = indices
        .map(idx => keypoints[idx])
        .filter(kp => kp && kp.score >= CONFIDENCE_THRESHOLD);

      if (validKeypoints.length === 0) return;

      // Draw large circles covering each body part
      validKeypoints.forEach(kp => {
        const x = videoWidth - kp.position.x;
        const y = kp.position.y;
        
        // Determine radius based on body part
        let radius = 40;
        if (partName === 'head') radius = 70;
        else if (partName === 'torso') radius = 80;
        else if (partName === 'leftArm' || partName === 'rightArm') radius = 45;
        else if (partName === 'leftLeg' || partName === 'rightLeg') radius = 50;
        
        // Draw large shadow/glow
        ctx.fillStyle = color.fill;
        ctx.shadowColor = color.stroke;
        ctx.shadowBlur = 25;
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, 2 * Math.PI);
        ctx.fill();
        
        // Reset shadow
        ctx.shadowBlur = 0;
      });

      // Draw filled polygon connecting keypoints for solid coverage
      if (validKeypoints.length >= 2) {
        ctx.fillStyle = color.fill;
        ctx.beginPath();
        validKeypoints.forEach((kp, i) => {
          const x = videoWidth - kp.position.x;
          const y = kp.position.y;
          if (i === 0) {
            ctx.moveTo(x, y);
          } else {
            ctx.lineTo(x, y);
          }
        });
        ctx.closePath();
        ctx.fill();
      }
    });
  };

  const detectPose = async () => {
    if (!modelRef.current || !videoRef.current) return;
    const video = videoRef.current;
    if (video.readyState < 2) {
      animationIdRef.current = requestAnimationFrame(detectPose);
      return;
    }

    try {
      const pose = await modelRef.current.estimateSinglePose(video, {
        flipHorizontal: false,
        decodingMethod: 'single-person'
      });

      if (pose && pose.keypoints) {
        const smoothedKeypoints = applyTemporalSmoothing(pose.keypoints);
        drawBodyParts(smoothedKeypoints);
        frameCountRef.current++;
        if (frameCountRef.current % FRAME_SKIP === 0) {
          analyzeMovement(smoothedKeypoints);
        }
      }

      animationIdRef.current = requestAnimationFrame(detectPose);
    } catch (err) {
      console.error('Error during pose detection:', err);
      animationIdRef.current = requestAnimationFrame(detectPose);
    }
  };

  const toggleDetection = () => {
    if (!modelLoaded) {
      alert('Please wait for the model to load first!');
      return;
    }

    setDetecting(prev => {
      const newState = !prev;
      if (newState) {
        console.log('%c🎯 BODY TRACKING STARTED', 'color: #00ff88; font-size: 18px; font-weight: bold;');
        frameCountRef.current = 0;
        detectPose();
      } else {
        console.log('%c⏸ Body tracking paused', 'color: #ff4444; font-size: 16px; font-weight: bold;');
        if (animationIdRef.current) {
          cancelAnimationFrame(animationIdRef.current);
          animationIdRef.current = null;
        }
      }
      return newState;
    });
  };

  const clearLog = () => {
    setMotionLog([]);
    console.clear();
    console.log('%c🧹 Motion log cleared!', 'color: #00ccff; font-size: 16px; font-weight: bold;');
  };

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <h1 style={styles.title}>🎯 Real-Time AR Body Tracking</h1>
        <p style={styles.subtitle}>Powered by TensorFlow.js PoseNet with OpenPose Skeleton</p>
        {onClose && (
          <button onClick={onClose} style={styles.closeButton}>
            ← Back to Home
          </button>
        )}
      </div>

      <div style={styles.statusBar}>
        <div style={styles.statusItem}>
          <span style={styles.statusLabel}>Model Status:</span>
          <span style={{
            ...styles.statusValue,
            color: modelError ? '#ff4444' : (modelLoaded ? '#00ff00' : '#ffaa00')
          }}>
            {modelError ? '❌ Error' : (modelLoaded ? '✅ Loaded' : '⏳ Loading...')}
          </span>
        </div>
        <div style={styles.statusItem}>
          <span style={styles.statusLabel}>Detection:</span>
          <span style={{
            ...styles.statusValue,
            color: detecting ? '#00ff00' : '#888'
          }}>
            {detecting ? '🟢 Active' : '⚫ Paused'}
          </span>
        </div>
        <div style={styles.statusItem}>
          <span style={styles.statusLabel}>Smoothing:</span>
          <span style={styles.statusValue}>
            {SMOOTHING_WINDOW} frames
          </span>
        </div>
      </div>

      {modelError && (
        <div style={styles.errorAlert}>
          <h3 style={styles.errorTitle}>⚠️ Model Loading Failed</h3>
          <p style={styles.errorText}>{modelError}</p>
          <button onClick={() => window.location.reload()} style={styles.reloadButton}>
            🔄 Reload Page
          </button>
        </div>
      )}

      {!modelLoaded && !modelError && (
        <div style={styles.loadingAlert}>
          <div style={styles.loadingSpinner}></div>
          <p style={styles.loadingText}>
            Loading PoseNet AI Model... This may take 10-30 seconds.
          </p>
          <p style={styles.loadingSubtext}>
            Please wait until the model is fully loaded before starting detection.
          </p>
        </div>
      )}

      <div style={styles.videoContainer}>
        <video ref={videoRef} style={styles.video} playsInline muted />
        <canvas ref={canvasRef} style={styles.canvas} />
      </div>

      <div style={styles.legend}>
        <h4 style={styles.legendTitle}>Color Legend:</h4>
        <div style={styles.legendItems}>
          {Object.entries(BODY_COLORS).map(([part, color]) => (
            <div key={part} style={styles.legendItem}>
              <div style={{ ...styles.colorBox, background: color.stroke }}></div>
              <span>{part.replace(/([A-Z])/g, ' $1').trim()}: {color.name}</span>
            </div>
          ))}
        </div>
      </div>

      <div style={styles.controls}>
        <button
          onClick={toggleDetection}
          disabled={!modelLoaded}
          style={{
            ...styles.button,
            ...(detecting ? styles.buttonStop : styles.buttonStart),
            ...(modelLoaded ? {} : styles.buttonDisabled)
          }}
        >
          {detecting ? '⏸ Stop Detection' : '▶️ Start Detection'}
        </button>
        <button onClick={clearLog} style={styles.buttonClear}>
          🧹 Clear Log
        </button>
      </div>

      <div style={styles.logContainer}>
        <h3 style={styles.logTitle}>📊 Motion Log (Real-time)</h3>
        <div style={styles.logContent}>
          {motionLog.length === 0 ? (
            <p style={styles.emptyLog}>
              {detecting 
                ? '👋 Move around to detect body movements...' 
                : '▶️ Start detection to see motion logs'}
            </p>
          ) : (
            motionLog.map((log, index) => (
              <div key={index} style={styles.logItem}>
                <span style={styles.logTimestamp}>{log.time}</span>
                <span style={styles.logText}>{log.text}</span>
              </div>
            ))
          )}
        </div>
      </div>

      <div style={styles.info}>
        <h3 style={styles.sectionTitle}>📋 Instructions</h3>
        <ul style={styles.instructionList}>
          <li>✅ Allow camera access when prompted</li>
          <li>🎯 Wait for PoseNet model to load (~10-30 seconds)</li>
          <li>▶️ Click "Start Detection" to begin</li>
          <li>🚶 Move around - the system tracks 17 body keypoints with OpenPose skeleton</li>
          <li>� Motion logs appear when movements exceed {MOVEMENT_THRESHOLD}px threshold</li>
          <li>� Each body part is color-coded with smooth skeleton lines</li>
          <li>🧹 Use "Clear Log" to reset the motion history</li>
        </ul>
      </div>

      <div style={styles.techSpecs}>
        <h3 style={styles.sectionTitle}>⚙️ Technical Specifications</h3>
        <ul style={styles.specList}>
          <li><strong>AI Model:</strong> PoseNet (MobileNetV1, OutputStride: 16)</li>
          <li><strong>Keypoints Detected:</strong> 17 (head, torso, arms, legs)</li>
          <li><strong>Confidence Threshold:</strong> {(CONFIDENCE_THRESHOLD * 100).toFixed(0)}%</li>
          <li><strong>Movement Threshold:</strong> {MOVEMENT_THRESHOLD}px (torso: {(MOVEMENT_THRESHOLD * 1.5).toFixed(1)}px)</li>
          <li><strong>Temporal Smoothing:</strong> {SMOOTHING_WINDOW}-frame window averaging</li>
          <li><strong>Performance:</strong> Analysis every {FRAME_SKIP} frames</li>
          <li><strong>Visualization:</strong> OpenPose-style skeleton with gradient lines</li>
        </ul>
      </div>

      <div style={styles.extensions}>
        <h3 style={styles.sectionTitle}>🚀 Possible Extensions</h3>
        <ul style={styles.extensionList}>
          <li>👗 <strong>Virtual Try-On:</strong> Overlay 3D clothing models on detected body</li>
          <li>💪 <strong>Fitness Tracking:</strong> Count reps, track form, analyze exercises</li>
          <li>🎮 <strong>Gesture Control:</strong> Map body movements to game controls</li>
          <li>� <strong>Dance Analysis:</strong> Compare poses with reference choreography</li>
          <li>📸 <strong>Photo Booth:</strong> Trigger capture on specific poses</li>
          <li>� <strong>Sports Analytics:</strong> Analyze athletic movements and technique</li>
        </ul>
      </div>
    </div>
  );
};

const styles = {
  container: {
    minHeight: '100vh',
    background: '#000000',
    padding: '20px',
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    color: 'white',
  },
  header: {
    textAlign: 'center',
    marginBottom: '20px',
    position: 'relative',
  },
  closeButton: {
    position: 'absolute',
    top: '0',
    left: '20px',
    padding: '10px 20px',
    fontSize: '14px',
    fontWeight: '600',
    background: 'rgba(255,255,255,0.2)',
    border: 'none',
    borderRadius: '10px',
    color: 'white',
    cursor: 'pointer',
    transition: 'all 0.3s ease',
    backdropFilter: 'blur(10px)',
  },
  title: {
    fontSize: '36px',
    fontWeight: 'bold',
    margin: '0 0 10px 0',
    textShadow: '2px 2px 4px rgba(0,0,0,0.3)',
  },
  subtitle: {
    fontSize: '16px',
    opacity: 0.9,
    margin: 0,
  },
  statusBar: {
    display: 'flex',
    justifyContent: 'center',
    gap: '20px',
    marginBottom: '20px',
    padding: '15px',
    background: 'rgba(0,0,0,0.2)',
    borderRadius: '15px',
    backdropFilter: 'blur(10px)',
    flexWrap: 'wrap',
  },
  statusItem: {
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
  },
  statusLabel: {
    fontWeight: '600',
    fontSize: '14px',
  },
  statusValue: {
    fontSize: '14px',
    fontWeight: 'bold',
  },
  errorAlert: {
    maxWidth: '640px',
    margin: '0 auto 20px',
    padding: '20px',
    background: 'rgba(255, 68, 68, 0.2)',
    border: '2px solid rgba(255, 68, 68, 0.6)',
    borderRadius: '15px',
    textAlign: 'center',
    backdropFilter: 'blur(10px)',
  },
  errorTitle: {
    margin: '0 0 10px 0',
    fontSize: '20px',
    fontWeight: '600',
    color: '#ff4444',
  },
  errorText: {
    margin: '0 0 15px 0',
    fontSize: '14px',
  },
  reloadButton: {
    padding: '10px 20px',
    fontSize: '16px',
    fontWeight: '600',
    background: 'linear-gradient(135deg, #00ff88 0%, #00cc66 100%)',
    border: 'none',
    borderRadius: '10px',
    color: '#000',
    cursor: 'pointer',
    transition: 'all 0.3s ease',
  },
  loadingAlert: {
    maxWidth: '640px',
    margin: '0 auto 20px',
    padding: '20px',
    background: 'rgba(255, 165, 0, 0.2)',
    border: '2px solid rgba(255, 165, 0, 0.6)',
    borderRadius: '15px',
    textAlign: 'center',
    backdropFilter: 'blur(10px)',
  },
  loadingSpinner: {
    width: '40px',
    height: '40px',
    border: '4px solid rgba(255, 255, 255, 0.3)',
    borderTop: '4px solid #ffaa00',
    borderRadius: '50%',
    animation: 'spin 1s linear infinite',
    margin: '0 auto 15px',
  },
  loadingText: {
    fontSize: '16px',
    fontWeight: '600',
    margin: '0 0 10px 0',
    color: '#ffaa00',
  },
  loadingSubtext: {
    fontSize: '14px',
    margin: '0',
    opacity: 0.9,
  },
  videoContainer: {
    position: 'relative',
    maxWidth: '640px',
    margin: '0 auto 20px',
    borderRadius: '15px',
    overflow: 'hidden',
    boxShadow: '0 20px 60px rgba(0,0,0,0.5)',
    background: '#000',
  },
  video: {
    width: '100%',
    display: 'block',
    transform: 'scaleX(-1)',
  },
  canvas: {
    position: 'absolute',
    top: 0,
    left: 0,
    width: '100%',
    height: '100%',
    pointerEvents: 'none',
  },
  legend: {
    maxWidth: '640px',
    margin: '0 auto 20px',
    padding: '15px',
    background: 'rgba(0,0,0,0.2)',
    borderRadius: '15px',
    backdropFilter: 'blur(10px)',
  },
  legendTitle: {
    margin: '0 0 10px 0',
    fontSize: '16px',
    fontWeight: '600',
  },
  legendItems: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
    gap: '10px',
  },
  legendItem: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    fontSize: '13px',
  },
  colorBox: {
    width: '20px',
    height: '20px',
    borderRadius: '4px',
    border: '2px solid white',
  },
  controls: {
    display: 'flex',
    justifyContent: 'center',
    gap: '15px',
    marginBottom: '20px',
    flexWrap: 'wrap',
  },
  button: {
    padding: '12px 30px',
    fontSize: '16px',
    fontWeight: '600',
    border: 'none',
    borderRadius: '10px',
    cursor: 'pointer',
    transition: 'all 0.3s ease',
    boxShadow: '0 4px 15px rgba(0,0,0,0.3)',
  },
  buttonStart: {
    background: 'linear-gradient(135deg, #00ff88 0%, #00cc66 100%)',
    color: '#000',
  },
  buttonStop: {
    background: 'linear-gradient(135deg, #ff4444 0%, #cc0000 100%)',
    color: '#fff',
  },
  buttonClear: {
    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    color: '#fff',
    padding: '12px 30px',
    fontSize: '16px',
    fontWeight: '600',
    border: 'none',
    borderRadius: '10px',
    cursor: 'pointer',
    transition: 'all 0.3s ease',
    boxShadow: '0 4px 15px rgba(0,0,0,0.3)',
  },
  buttonDisabled: {
    background: '#666',
    cursor: 'not-allowed',
    opacity: 0.5,
  },
  logContainer: {
    maxWidth: '640px',
    margin: '0 auto 20px',
    background: 'rgba(0,0,0,0.2)',
    borderRadius: '15px',
    padding: '20px',
    backdropFilter: 'blur(10px)',
  },
  logTitle: {
    margin: '0 0 15px 0',
    fontSize: '18px',
    fontWeight: '600',
  },
  logContent: {
    maxHeight: '200px',
    overflowY: 'auto',
    background: 'rgba(0,0,0,0.3)',
    borderRadius: '10px',
    padding: '15px',
  },
  emptyLog: {
    textAlign: 'center',
    opacity: 0.7,
    fontStyle: 'italic',
    margin: 0,
  },
  logItem: {
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
    padding: '8px',
    marginBottom: '8px',
    background: 'rgba(255,255,255,0.1)',
    borderRadius: '8px',
    animation: 'slideIn 0.3s ease-out',
  },
  logTimestamp: {
    fontSize: '11px',
    opacity: 0.7,
    minWidth: '70px',
  },
  logText: {
    fontSize: '14px',
    fontWeight: '500',
  },
  info: {
    maxWidth: '640px',
    margin: '0 auto 20px',
    padding: '20px',
    background: 'rgba(0,0,0,0.2)',
    borderRadius: '15px',
    backdropFilter: 'blur(10px)',
  },
  infoTitle: {
    margin: '0 0 15px 0',
    fontSize: '18px',
    fontWeight: '600',
  },
  infoList: {
    margin: '0',
    padding: '0 0 0 20px',
    lineHeight: '1.8',
  },
  technicalInfo: {
    maxWidth: '640px',
    margin: '0 auto 20px',
    padding: '20px',
    background: 'rgba(0,0,0,0.2)',
    borderRadius: '15px',
    backdropFilter: 'blur(10px)',
  },
  technicalTitle: {
    margin: '0 0 15px 0',
    fontSize: '16px',
    fontWeight: '600',
  },
  specs: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
    gap: '10px',
  },
  spec: {
    padding: '10px',
    background: 'rgba(255,255,255,0.1)',
    borderRadius: '8px',
    fontSize: '13px',
  },
  extensionsInfo: {
    maxWidth: '640px',
    margin: '0 auto 20px',
    padding: '20px',
    background: 'rgba(0,255,136,0.1)',
    border: '2px solid rgba(0,255,136,0.3)',
    borderRadius: '15px',
    backdropFilter: 'blur(10px)',
  },
  extensionsTitle: {
    margin: '0 0 10px 0',
    fontSize: '16px',
    fontWeight: '600',
    color: '#00ff88',
  },
  extensionsText: {
    margin: '0 0 10px 0',
    fontSize: '14px',
  },
  extensionsList: {
    margin: '0',
    padding: '0 0 0 20px',
    lineHeight: '1.8',
    fontSize: '14px',
  },
};

export default BodyTracking;
