const predictBtn = document.getElementById('predictBtn');
const messageInput = document.getElementById('messageInput');
const result = document.getElementById('result');
const predictionLabel = document.getElementById('predictionLabel');
const confidenceText = document.getElementById('confidenceText');
const loading = document.getElementById('loading');

const API_URL = 'http://127.0.0.1:8000/predict';

predictBtn.addEventListener('click', async () => {
    const message = messageInput.value.trim();

    if (!message) {
        alert('Please enter a message');
        return;
    }

    loading.classList.remove('hidden');
    result.classList.add('hidden');
    predictBtn.disabled = true;

    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message: message })
        });

        const data = await response.json();

        loading.classList.add('hidden');
        result.classList.remove('hidden');

        if (data.prediction === 'spam') {
            predictionLabel.textContent = '🚨 SPAM';
            result.className = 'result spam';
        } else {
            predictionLabel.textContent = '✅ HAM (Not Spam)';
            result.className = 'result ham';
        }

        confidenceText.textContent = `Confidence: ${(data.confidence * 100).toFixed(1)}%`;

    } catch (error) {
        loading.classList.add('hidden');
        alert('Error: Could not connect to the server. Make sure the backend is running.');
        console.error('Error:', error);
    } finally {
        predictBtn.disabled = false;
    }
});