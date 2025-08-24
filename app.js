// Futuristic Gait Authentication System - Complete ML Implementation

// Decision Tree class definition (needs to be outside the main class)
class DecisionTree {
    constructor(maxDepth = 10, minSamples = 2) {
        this.maxDepth = maxDepth;
        this.minSamples = minSamples;
        this.tree = null;
    }

    fit(X, y) {
        this.tree = this.buildTree(X, y, 0);
    }

    buildTree(X, y, depth) {
        const nSamples = X.length;
        const nFeatures = X[0] ? X[0].length : 0;

        // Stopping criteria
        if (depth >= this.maxDepth || nSamples < this.minSamples || this.isPure(y)) {
            return {
                type: 'leaf',
                prediction: this.mostCommon(y)
            };
        }

        // Find best split
        let bestFeature = null;
        let bestThreshold = null;
        let bestGini = 1;

        for (let feature = 0; feature < nFeatures; feature++) {
            const thresholds = [...new Set(X.map(x => x[feature]))].sort((a, b) => a - b);
            
            for (let i = 0; i < thresholds.length - 1; i++) {
                const threshold = (thresholds[i] + thresholds[i + 1]) / 2;
                const gini = this.calculateSplitGini(X, y, feature, threshold);
                
                if (gini < bestGini) {
                    bestGini = gini;
                    bestFeature = feature;
                    bestThreshold = threshold;
                }
            }
        }

        if (bestFeature === null) {
            return {
                type: 'leaf',
                prediction: this.mostCommon(y)
            };
        }

        // Split data
        const leftIndices = [];
        const rightIndices = [];
        
        for (let i = 0; i < X.length; i++) {
            if (X[i][bestFeature] <= bestThreshold) {
                leftIndices.push(i);
            } else {
                rightIndices.push(i);
            }
        }

        const leftX = leftIndices.map(i => X[i]);
        const leftY = leftIndices.map(i => y[i]);
        const rightX = rightIndices.map(i => X[i]);
        const rightY = rightIndices.map(i => y[i]);

        return {
            type: 'node',
            feature: bestFeature,
            threshold: bestThreshold,
            left: this.buildTree(leftX, leftY, depth + 1),
            right: this.buildTree(rightX, rightY, depth + 1)
        };
    }

    predict(X) {
        return X.map(x => this.predictSingle(x, this.tree));
    }

    predictSingle(x, node) {
        if (node.type === 'leaf') {
            return node.prediction;
        }

        if (x[node.feature] <= node.threshold) {
            return this.predictSingle(x, node.left);
        } else {
            return this.predictSingle(x, node.right);
        }
    }

    isPure(y) {
        return new Set(y).size <= 1;
    }

    mostCommon(y) {
        const counts = {};
        for (const label of y) {
            counts[label] = (counts[label] || 0) + 1;
        }
        return Object.keys(counts).reduce((a, b) => counts[a] > counts[b] ? a : b);
    }

    calculateSplitGini(X, y, feature, threshold) {
        const leftY = [];
        const rightY = [];

        for (let i = 0; i < X.length; i++) {
            if (X[i][feature] <= threshold) {
                leftY.push(y[i]);
            } else {
                rightY.push(y[i]);
            }
        }

        const totalSamples = y.length;
        const leftWeight = leftY.length / totalSamples;
        const rightWeight = rightY.length / totalSamples;

        return leftWeight * this.calculateGini(leftY) + rightWeight * this.calculateGini(rightY);
    }

    calculateGini(y) {
        if (y.length === 0) return 0;
        
        const counts = {};
        for (const label of y) {
            counts[label] = (counts[label] || 0) + 1;
        }

        let gini = 1;
        for (const count of Object.values(counts)) {
            const prob = count / y.length;
            gini -= prob * prob;
        }

        return gini;
    }
}

// Random Forest class definition
class RandomForest {
    constructor(nTrees = 10, maxDepth = 10, minSamples = 2) {
        this.nTrees = nTrees;
        this.maxDepth = maxDepth;
        this.minSamples = minSamples;
        this.trees = [];
    }

    fit(X, y) {
        this.trees = [];
        const nSamples = X.length;

        for (let i = 0; i < this.nTrees; i++) {
            // Bootstrap sampling
            const indices = [];
            for (let j = 0; j < nSamples; j++) {
                indices.push(Math.floor(Math.random() * nSamples));
            }

            const bootstrapX = indices.map(idx => X[idx]);
            const bootstrapY = indices.map(idx => y[idx]);

            // Create and train tree
            const tree = new DecisionTree(this.maxDepth, this.minSamples);
            tree.fit(bootstrapX, bootstrapY);
            this.trees.push(tree);
        }
    }

    predict(X) {
        const treePredictions = this.trees.map(tree => tree.predict(X));
        const results = [];

        for (let i = 0; i < X.length; i++) {
            const votes = {};
            for (let j = 0; j < this.trees.length; j++) {
                const prediction = treePredictions[j][i];
                votes[prediction] = (votes[prediction] || 0) + 1;
            }

            const bestPrediction = Object.keys(votes).reduce((a, b) => 
                votes[a] > votes[b] ? a : b
            );

            results.push({
                prediction: bestPrediction,
                confidence: votes[bestPrediction] / this.trees.length
            });
        }

        return results;
    }
}

class GaitAuthSystem {
    constructor() {
        this.isRecording = false;
        this.sensorData = [];
        this.profiles = this.loadProfiles();
        this.currentMode = null;
        this.recordingStartTime = 0;
        this.recordingDuration = 7000; // 7 seconds
        this.sampleRate = 50; // 50 Hz sampling
        this.canvas = null;
        this.ctx = null;
        this.animationFrame = null;
        
        // Initialize the system
        this.init();
    }

    init() {
        this.setupCanvas();
        this.setupEventListeners();
        this.updateProfileDisplay();
        this.checkSensorSupport();
        this.startBackgroundAnimations();
    }

    setupCanvas() {
        this.canvas = document.getElementById('motion-canvas');
        if (this.canvas) {
            this.ctx = this.canvas.getContext('2d');
        }
    }

    setupEventListeners() {
        // Handle device motion events
        if (window.DeviceMotionEvent) {
            window.addEventListener('devicemotion', (event) => {
                if (this.isRecording) {
                    this.handleMotionData(event);
                }
            });
        }

        // Handle screen orientation changes
        window.addEventListener('orientationchange', () => {
            setTimeout(() => this.setupCanvas(), 500);
        });
    }

    checkSensorSupport() {
        if (!window.DeviceMotionEvent) {
            this.showError('Device motion sensors are not supported on this device.');
            return false;
        }
        return true;
    }

    // ML Feature Extraction Functions
    extractFeatures(data) {
        if (!data || data.length === 0) return [];

        const features = [];
        const axes = ['x', 'y', 'z'];
        const sensors = ['acceleration', 'gyroscope'];

        sensors.forEach(sensorType => {
            axes.forEach(axis => {
                const values = data.map(d => d[sensorType][axis]).filter(v => v !== null && v !== undefined);
                
                if (values.length > 0) {
                    // Time-domain features
                    features.push(this.calculateMean(values));
                    features.push(this.calculateStd(values));
                    features.push(this.calculateMin(values));
                    features.push(this.calculateMax(values));
                    features.push(this.calculateRange(values));
                    features.push(this.calculateRMS(values));
                    features.push(this.calculateKurtosis(values));
                    features.push(this.calculateSkewness(values));
                    features.push(this.calculateEnergy(values));
                    features.push(this.calculatePercentile(values, 25));
                    features.push(this.calculatePercentile(values, 75));
                    
                    // Frequency-domain features
                    features.push(this.calculateAutocorrelation(values));
                    features.push(this.calculatePeakFrequency(values));
                } else {
                    // Fill with zeros if no data
                    for (let i = 0; i < 13; i++) features.push(0);
                }
            });
        });

        return features;
    }

    // Statistical calculation functions
    calculateMean(values) {
        return values.reduce((sum, val) => sum + val, 0) / values.length;
    }

    calculateStd(values) {
        const mean = this.calculateMean(values);
        const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
        return Math.sqrt(variance);
    }

    calculateMin(values) {
        return Math.min(...values);
    }

    calculateMax(values) {
        return Math.max(...values);
    }

    calculateRange(values) {
        return this.calculateMax(values) - this.calculateMin(values);
    }

    calculateRMS(values) {
        const sumSquares = values.reduce((sum, val) => sum + val * val, 0);
        return Math.sqrt(sumSquares / values.length);
    }

    calculateKurtosis(values) {
        const mean = this.calculateMean(values);
        const std = this.calculateStd(values);
        if (std === 0) return 0;
        
        const n = values.length;
        const fourthMoment = values.reduce((sum, val) => sum + Math.pow((val - mean) / std, 4), 0) / n;
        return fourthMoment - 3;
    }

    calculateSkewness(values) {
        const mean = this.calculateMean(values);
        const std = this.calculateStd(values);
        if (std === 0) return 0;
        
        const n = values.length;
        return values.reduce((sum, val) => sum + Math.pow((val - mean) / std, 3), 0) / n;
    }

    calculateEnergy(values) {
        return values.reduce((sum, val) => sum + val * val, 0);
    }

    calculatePercentile(values, percentile) {
        const sorted = [...values].sort((a, b) => a - b);
        const index = (percentile / 100) * (sorted.length - 1);
        const lower = Math.floor(index);
        const upper = Math.ceil(index);
        
        if (lower === upper) return sorted[lower];
        return sorted[lower] * (upper - index) + sorted[upper] * (index - lower);
    }

    calculateAutocorrelation(values) {
        if (values.length < 2) return 0;
        
        const mean = this.calculateMean(values);
        let numerator = 0;
        let denominator = 0;
        
        for (let i = 0; i < values.length - 1; i++) {
            numerator += (values[i] - mean) * (values[i + 1] - mean);
            denominator += (values[i] - mean) * (values[i] - mean);
        }
        
        return denominator === 0 ? 0 : numerator / denominator;
    }

    calculatePeakFrequency(values) {
        // Simplified frequency analysis
        const fft = this.simpleFFT(values);
        let maxMagnitude = 0;
        let peakIndex = 0;
        
        for (let i = 1; i < fft.length / 2; i++) {
            const magnitude = Math.sqrt(fft[i].real * fft[i].real + fft[i].imag * fft[i].imag);
            if (magnitude > maxMagnitude) {
                maxMagnitude = magnitude;
                peakIndex = i;
            }
        }
        
        return peakIndex * this.sampleRate / values.length;
    }

    simpleFFT(values) {
        // Simplified FFT implementation for peak frequency detection
        const N = values.length;
        const result = new Array(N);
        
        for (let k = 0; k < N; k++) {
            let real = 0;
            let imag = 0;
            
            for (let n = 0; n < N; n++) {
                const angle = -2 * Math.PI * k * n / N;
                real += values[n] * Math.cos(angle);
                imag += values[n] * Math.sin(angle);
            }
            
            result[k] = { real, imag };
        }
        
        return result;
    }

    // Data preprocessing
    normalizeFeatures(features) {
        if (features.length === 0) return features;
        
        // Z-score normalization
        const mean = this.calculateMean(features);
        const std = this.calculateStd(features);
        
        if (std === 0) return features.map(() => 0);
        
        return features.map(f => (f - mean) / std);
    }

    // Sensor data handling
    handleMotionData(event) {
        if (!this.isRecording) return;

        const acceleration = event.acceleration || event.accelerationIncludingGravity || { x: 0, y: 0, z: 0 };
        const gyroscope = event.rotationRate || { alpha: 0, beta: 0, gamma: 0 };

        const dataPoint = {
            timestamp: Date.now(),
            acceleration: {
                x: acceleration.x || 0,
                y: acceleration.y || 0,
                z: acceleration.z || 0
            },
            gyroscope: {
                x: gyroscope.alpha || 0,
                y: gyroscope.beta || 0,
                z: gyroscope.gamma || 0
            }
        };

        this.sensorData.push(dataPoint);
        this.updateSensorDisplay(dataPoint);
        this.updateVisualization(dataPoint);
    }

    updateSensorDisplay(dataPoint) {
        const accelX = document.getElementById('accel-x');
        const accelY = document.getElementById('accel-y');
        const accelZ = document.getElementById('accel-z');
        const gyroX = document.getElementById('gyro-x');
        const gyroY = document.getElementById('gyro-y');
        const gyroZ = document.getElementById('gyro-z');

        if (accelX) accelX.textContent = dataPoint.acceleration.x.toFixed(2);
        if (accelY) accelY.textContent = dataPoint.acceleration.y.toFixed(2);
        if (accelZ) accelZ.textContent = dataPoint.acceleration.z.toFixed(2);
        if (gyroX) gyroX.textContent = dataPoint.gyroscope.x.toFixed(2);
        if (gyroY) gyroY.textContent = dataPoint.gyroscope.y.toFixed(2);
        if (gyroZ) gyroZ.textContent = dataPoint.gyroscope.z.toFixed(2);
    }

    updateVisualization(dataPoint) {
        if (!this.ctx) return;

        const canvas = this.canvas;
        const ctx = this.ctx;
        
        // Clear canvas with fade effect
        ctx.globalAlpha = 0.1;
        ctx.fillStyle = 'rgba(0, 0, 0, 1)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.globalAlpha = 1;

        // Draw motion trails
        const centerX = canvas.width / 2;
        const centerY = canvas.height / 2;
        const scale = 20;

        // Accelerometer visualization (cyan)
        ctx.strokeStyle = '#00f5ff';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(
            centerX + dataPoint.acceleration.x * scale,
            centerY + dataPoint.acceleration.y * scale,
            Math.abs(dataPoint.acceleration.z * scale / 4) + 2,
            0, 2 * Math.PI
        );
        ctx.stroke();

        // Gyroscope visualization (magenta)
        ctx.strokeStyle = '#ff1493';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(
            centerX + dataPoint.gyroscope.x * scale / 10,
            centerY + dataPoint.gyroscope.y * scale / 10,
            Math.abs(dataPoint.gyroscope.z * scale / 20) + 1,
            0, 2 * Math.PI
        );
        ctx.stroke();
    }

    // Recording functionality
    async startRecording(mode) {
        if (!await this.requestSensorPermission()) return;

        this.currentMode = mode;
        this.sensorData = [];
        this.isRecording = true;
        this.recordingStartTime = Date.now();

        this.showScreen('recording-screen');
        this.updateRecordingUI();
        this.startRecordingTimer();
        this.startRecordingAnimations();

        const title = document.getElementById('recording-title');
        if (title) {
            title.textContent = mode === 'registration' ? 'Recording New Profile' : 'Verifying Identity';
        }
    }

    startRecordingTimer() {
        const startTime = Date.now();
        const updateTimer = () => {
            if (!this.isRecording) return;

            const elapsed = Date.now() - startTime;
            const remaining = Math.max(0, (this.recordingDuration - elapsed) / 1000);
            
            const timer = document.getElementById('countdown-timer');
            if (timer) {
                timer.textContent = remaining.toFixed(1);
            }
            
            // Update progress ring
            const progress = elapsed / this.recordingDuration;
            this.updateProgressRing(Math.min(progress * 100, 100));

            if (elapsed >= this.recordingDuration) {
                this.stopRecording();
            } else {
                requestAnimationFrame(updateTimer);
            }
        };

        requestAnimationFrame(updateTimer);
    }

    updateProgressRing(percentage) {
        const ring = document.querySelector('.progress-ring-fill');
        const circumference = 2 * Math.PI * 90; // radius is 90
        const offset = circumference - (percentage / 100 * circumference);
        
        if (ring) {
            ring.style.strokeDashoffset = offset;
        }
        
        const progressText = document.getElementById('progress-percentage');
        if (progressText) {
            progressText.textContent = Math.round(percentage) + '%';
        }
    }

    startRecordingAnimations() {
        // Animate footsteps
        const footsteps = document.querySelectorAll('.footstep');
        footsteps.forEach((step, index) => {
            step.style.animationDelay = (index * 0.3) + 's';
            step.style.animationPlayState = 'running';
        });

        // Animate ripples
        const ripples = document.querySelectorAll('.ripple');
        ripples.forEach((ripple, index) => {
            ripple.style.animationDelay = (index * 0.7) + 's';
            ripple.style.animationPlayState = 'running';
        });

        // Animate trails
        const trails = document.querySelectorAll('.trail');
        trails.forEach((trail, index) => {
            trail.style.animationDelay = (index * 1) + 's';
            trail.style.animationPlayState = 'running';
        });
    }

    stopRecording() {
        this.isRecording = false;
        
        if (this.currentMode === 'registration') {
            this.processRegistration();
        } else if (this.currentMode === 'authentication') {
            this.processAuthentication();
        }
    }

    // Registration processing
    processRegistration() {
        const usernameField = document.getElementById('username');
        const username = usernameField ? usernameField.value.trim() : '';
        
        if (!username) {
            this.showError('Please enter a username');
            this.showScreen('registration-screen');
            return;
        }

        if (this.profiles.length >= 5) {
            this.showError('Maximum 5 profiles allowed. Please delete a profile first.');
            this.showScreen('registration-screen');
            return;
        }

        if (this.sensorData.length < 50) {
            this.showError('Insufficient motion data. Please try again.');
            this.showScreen('registration-screen');
            return;
        }

        // Extract features
        const features = this.extractFeatures(this.sensorData);
        const normalizedFeatures = this.normalizeFeatures(features);

        // Create new profile
        const profile = {
            id: Date.now().toString(),
            name: username,
            features: normalizedFeatures,
            created: new Date().toLocaleDateString(),
            accuracy: 0.95 // Default accuracy estimate
        };

        this.profiles.push(profile);
        this.saveProfiles();
        this.updateProfileDisplay();

        // Show results
        this.showRegistrationResult(username, features.length);
    }

    // Authentication processing
    processAuthentication() {
        if (this.profiles.length === 0) {
            this.showError('No profiles found. Please register first.');
            this.showScreen('mode-selection');
            return;
        }

        if (this.sensorData.length < 50) {
            this.showError('Insufficient motion data. Please try again.');
            this.showScreen('authentication-screen');
            return;
        }

        // Extract and normalize features
        const features = this.extractFeatures(this.sensorData);
        const normalizedFeatures = this.normalizeFeatures(features);

        // Prepare training data for Random Forest
        const X = this.profiles.map(profile => profile.features);
        const y = this.profiles.map(profile => profile.name);

        if (X.length === 0 || X[0].length === 0) {
            this.showError('Invalid profile data. Please re-register.');
            this.showScreen('mode-selection');
            return;
        }

        try {
            // Train Random Forest classifier
            const rf = new RandomForest(10, 10, 1);
            rf.fit(X, y);

            // Predict
            const predictions = rf.predict([normalizedFeatures]);
            const result = predictions[0];

            // Calculate similarity scores for all profiles
            const scores = this.profiles.map(profile => ({
                name: profile.name,
                similarity: this.calculateSimilarity(normalizedFeatures, profile.features)
            }));

            scores.sort((a, b) => b.similarity - a.similarity);
            const bestMatch = scores[0];
            const threshold = 0.70;

            if (bestMatch.similarity >= threshold && result.confidence >= 0.6) {
                this.showAuthenticationSuccess(result.prediction, bestMatch.similarity, result.confidence);
            } else {
                this.showAuthenticationFailure(bestMatch.name, bestMatch.similarity);
            }
        } catch (error) {
            console.error('Authentication error:', error);
            this.showError('Authentication failed. Please try again.');
            this.showScreen('authentication-screen');
        }
    }

    calculateSimilarity(features1, features2) {
        if (features1.length !== features2.length) return 0;
        
        let dotProduct = 0;
        let norm1 = 0;
        let norm2 = 0;

        for (let i = 0; i < features1.length; i++) {
            dotProduct += features1[i] * features2[i];
            norm1 += features1[i] * features1[i];
            norm2 += features2[i] * features2[i];
        }

        if (norm1 === 0 || norm2 === 0) return 0;
        
        const cosineSimilarity = dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));
        return Math.max(0, Math.min(1, (cosineSimilarity + 1) / 2)); // Normalize to [0, 1]
    }

    // Results display
    showRegistrationResult(username, featureCount) {
        const registeredName = document.getElementById('registered-name');
        const featureCountEl = document.getElementById('feature-count');
        
        if (registeredName) registeredName.textContent = username;
        if (featureCountEl) featureCountEl.textContent = featureCount;
        
        const registrationResult = document.getElementById('registration-result');
        const successResult = document.getElementById('success-result');
        const failureResult = document.getElementById('failure-result');
        
        if (registrationResult) registrationResult.classList.remove('hidden');
        if (successResult) successResult.classList.add('hidden');
        if (failureResult) failureResult.classList.add('hidden');
        
        this.showScreen('results-screen');
    }

    showAuthenticationSuccess(identity, accuracy, confidence) {
        const matchedIdentity = document.getElementById('matched-identity');
        const accuracyScore = document.getElementById('accuracy-score');
        const confidenceLevel = document.getElementById('confidence-level');
        
        if (matchedIdentity) matchedIdentity.textContent = identity;
        if (accuracyScore) accuracyScore.textContent = (accuracy * 100).toFixed(1) + '%';
        if (confidenceLevel) confidenceLevel.textContent = (confidence * 100).toFixed(1) + '%';
        
        const successResult = document.getElementById('success-result');
        const failureResult = document.getElementById('failure-result');
        const registrationResult = document.getElementById('registration-result');
        
        if (successResult) successResult.classList.remove('hidden');
        if (failureResult) failureResult.classList.add('hidden');
        if (registrationResult) registrationResult.classList.add('hidden');
        
        this.showScreen('results-screen');
        this.triggerSuccessAnimation();
    }

    showAuthenticationFailure(bestMatch, maxScore) {
        const bestMatchEl = document.getElementById('best-match');
        const maxScoreEl = document.getElementById('max-score');
        
        if (bestMatchEl) bestMatchEl.textContent = bestMatch || 'None';
        if (maxScoreEl) maxScoreEl.textContent = (maxScore * 100).toFixed(1) + '%';
        
        const failureResult = document.getElementById('failure-result');
        const successResult = document.getElementById('success-result');
        const registrationResult = document.getElementById('registration-result');
        
        if (failureResult) failureResult.classList.remove('hidden');
        if (successResult) successResult.classList.add('hidden');
        if (registrationResult) registrationResult.classList.add('hidden');
        
        this.showScreen('results-screen');
    }

    triggerSuccessAnimation() {
        // Add confetti effect
        this.createConfetti();
    }

    createConfetti() {
        const colors = ['#00f5ff', '#ff1493', '#8a2be2', '#00ffff'];
        const confettiContainer = document.createElement('div');
        confettiContainer.style.position = 'fixed';
        confettiContainer.style.top = '0';
        confettiContainer.style.left = '0';
        confettiContainer.style.width = '100%';
        confettiContainer.style.height = '100%';
        confettiContainer.style.pointerEvents = 'none';
        confettiContainer.style.zIndex = '9999';
        
        document.body.appendChild(confettiContainer);

        for (let i = 0; i < 50; i++) {
            const confetti = document.createElement('div');
            confetti.style.position = 'absolute';
            confetti.style.width = '6px';
            confetti.style.height = '6px';
            confetti.style.backgroundColor = colors[Math.floor(Math.random() * colors.length)];
            confetti.style.left = Math.random() * 100 + '%';
            confetti.style.top = '-10px';
            confetti.style.borderRadius = '2px';
            confetti.style.animation = `confettiFall ${2 + Math.random() * 3}s linear forwards`;
            
            confettiContainer.appendChild(confetti);
        }

        // Clean up after animation
        setTimeout(() => {
            if (document.body.contains(confettiContainer)) {
                document.body.removeChild(confettiContainer);
            }
        }, 5000);
    }

    // Profile management
    loadProfiles() {
        try {
            const saved = localStorage.getItem('gaitauth_profiles');
            return saved ? JSON.parse(saved) : [];
        } catch (error) {
            console.error('Failed to load profiles:', error);
            return [];
        }
    }

    saveProfiles() {
        try {
            localStorage.setItem('gaitauth_profiles', JSON.stringify(this.profiles));
        } catch (error) {
            console.error('Failed to save profiles:', error);
            this.showError('Failed to save profile. Storage may be full.');
        }
    }

    updateProfileDisplay() {
        const container = document.getElementById('profile-list');
        if (!container) return;

        if (this.profiles.length === 0) {
            container.innerHTML = '<p style="text-align: center; color: rgba(255, 255, 255, 0.5); grid-column: 1 / -1;">No profiles stored</p>';
            return;
        }

        container.innerHTML = this.profiles.map(profile => `
            <div class="profile-card">
                <button class="delete-profile" onclick="gaitAuth.deleteProfile('${profile.id}')" title="Delete Profile">
                    ×
                </button>
                <div class="profile-avatar">
                    ${profile.name.charAt(0).toUpperCase()}
                </div>
                <div class="profile-name">${profile.name}</div>
                <div class="profile-accuracy">Accuracy: ${(profile.accuracy * 100).toFixed(0)}%</div>
            </div>
        `).join('');
    }

    deleteProfile(profileId) {
        if (confirm('Are you sure you want to delete this profile?')) {
            this.profiles = this.profiles.filter(p => p.id !== profileId);
            this.saveProfiles();
            this.updateProfileDisplay();
        }
    }

    // Permission handling
    async requestSensorPermission() {
        if (typeof DeviceMotionEvent !== 'undefined' && typeof DeviceMotionEvent.requestPermission === 'function') {
            try {
                const permission = await DeviceMotionEvent.requestPermission();
                if (permission !== 'granted') {
                    this.showError('Motion sensor permission denied. Please grant permission to use this feature.');
                    return false;
                }
            } catch (error) {
                console.error('Permission request failed:', error);
                this.showError('Failed to request sensor permission.');
                return false;
            }
        }
        return true;
    }

    // UI Management
    showScreen(screenId) {
        document.querySelectorAll('.screen').forEach(screen => {
            screen.classList.remove('active');
        });
        
        const targetScreen = document.getElementById(screenId);
        if (targetScreen) {
            targetScreen.classList.add('active');
        }
    }

    showError(message) {
        const errorMessage = document.getElementById('error-message');
        const errorModal = document.getElementById('error-modal');
        
        if (errorMessage) errorMessage.textContent = message;
        if (errorModal) errorModal.classList.remove('hidden');
    }

    closeErrorModal() {
        const errorModal = document.getElementById('error-modal');
        if (errorModal) errorModal.classList.add('hidden');
    }

    showPermissionModal() {
        const permissionModal = document.getElementById('permission-modal');
        if (permissionModal) permissionModal.classList.remove('hidden');
    }

    closePermissionModal() {
        const permissionModal = document.getElementById('permission-modal');
        if (permissionModal) permissionModal.classList.add('hidden');
    }

    // Background animations
    startBackgroundAnimations() {
        // Add CSS for confetti animation
        if (!document.getElementById('confetti-style')) {
            const style = document.createElement('style');
            style.id = 'confetti-style';
            style.textContent = `
                @keyframes confettiFall {
                    0% { transform: translateY(-100vh) rotate(0deg); opacity: 1; }
                    100% { transform: translateY(100vh) rotate(720deg); opacity: 0; }
                }
            `;
            document.head.appendChild(style);
        }

        // Add SVG gradient for progress ring
        const progressRing = document.querySelector('.progress-ring');
        if (progressRing && !progressRing.querySelector('defs')) {
            const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
            const gradient = document.createElementNS('http://www.w3.org/2000/svg', 'linearGradient');
            gradient.id = 'progressGradient';
            gradient.setAttribute('x1', '0%');
            gradient.setAttribute('y1', '0%');
            gradient.setAttribute('x2', '100%');
            gradient.setAttribute('y2', '0%');
            
            const stop1 = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
            stop1.setAttribute('offset', '0%');
            stop1.setAttribute('stop-color', '#00f5ff');
            
            const stop2 = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
            stop2.setAttribute('offset', '50%');
            stop2.setAttribute('stop-color', '#8a2be2');
            
            const stop3 = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
            stop3.setAttribute('offset', '100%');
            stop3.setAttribute('stop-color', '#ff1493');
            
            gradient.appendChild(stop1);
            gradient.appendChild(stop2);
            gradient.appendChild(stop3);
            defs.appendChild(gradient);
            progressRing.appendChild(defs);
        }
    }

    updateRecordingUI() {
        // Reset progress
        this.updateProgressRing(0);
        
        // Reset sensor displays
        const sensorIds = ['accel-x', 'accel-y', 'accel-z', 'gyro-x', 'gyro-y', 'gyro-z'];
        sensorIds.forEach(id => {
            const element = document.getElementById(id);
            if (element) element.textContent = '0.00';
        });
    }
}

// Global instance
let gaitAuth = null;

// Global functions for HTML event handlers
function showWelcome() {
    if (gaitAuth) gaitAuth.showScreen('welcome-screen');
}

function showModeSelection() {
    if (gaitAuth) gaitAuth.showScreen('mode-selection');
}

function startRegistration() {
    if (gaitAuth) {
        gaitAuth.showScreen('registration-screen');
        const usernameField = document.getElementById('username');
        if (usernameField) usernameField.value = '';
    }
}

function startAuthentication() {
    if (gaitAuth) gaitAuth.showScreen('authentication-screen');
}

function beginRegistration() {
    if (!gaitAuth) return;
    
    const usernameField = document.getElementById('username');
    const username = usernameField ? usernameField.value.trim() : '';
    
    if (!username) {
        gaitAuth.showError('Please enter a username');
        return;
    }
    
    // Check if username already exists
    if (gaitAuth.profiles.some(p => p.name.toLowerCase() === username.toLowerCase())) {
        gaitAuth.showError('Username already exists. Please choose a different name.');
        return;
    }
    
    gaitAuth.startRecording('registration');
}

function beginAuthentication() {
    if (!gaitAuth) return;
    
    if (gaitAuth.profiles.length === 0) {
        gaitAuth.showError('No profiles found. Please register first.');
        return;
    }
    gaitAuth.startRecording('authentication');
}

function closeErrorModal() {
    if (gaitAuth) gaitAuth.closeErrorModal();
}

function requestSensorPermission() {
    if (gaitAuth) {
        gaitAuth.closePermissionModal();
        gaitAuth.requestSensorPermission();
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    try {
        // Initialize the system
        gaitAuth = new GaitAuthSystem();
        console.log('Gait Authentication System initialized successfully');
        
        // Add keyboard navigation
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                closeErrorModal();
            }
        });
        
        // Add touch feedback for mobile
        document.addEventListener('touchstart', (e) => {
            if (e.target.classList.contains('btn') || 
                e.target.classList.contains('mode-button') ||
                e.target.classList.contains('cta-button') ||
                e.target.classList.contains('action-button')) {
                e.target.style.transform = 'scale(0.95)';
            }
        });
        
        document.addEventListener('touchend', (e) => {
            if (e.target.classList.contains('btn') || 
                e.target.classList.contains('mode-button') ||
                e.target.classList.contains('cta-button') ||
                e.target.classList.contains('action-button')) {
                setTimeout(() => {
                    e.target.style.transform = '';
                }, 100);
            }
        });
        
    } catch (error) {
        console.error('Failed to initialize Gait Authentication System:', error);
        alert('System initialization failed. Please refresh the page.');
    }
});
