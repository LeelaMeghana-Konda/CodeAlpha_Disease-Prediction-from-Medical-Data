let myChart = null;

// Navigation
document.querySelectorAll('.nav-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const target = btn.dataset.target;
        switchTab(target);
    });
});

function switchTab(tabId) {
    document.querySelectorAll('.page-section').forEach(s => s.classList.remove('active'));
    document.querySelector(`#${tabId}`).classList.add('active');
    
    document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
    document.querySelector(`.nav-btn[data-target="${tabId}"]`)?.classList.add('active');

    if (tabId !== 'home') {
        const mapping = {
            'heart': 'heart_disease',
            'diabetes': 'diabetes',
            'cancer': 'breast_cancer'
        };
        loadFeatures(mapping[tabId], `${tabId}-features`);
    }

    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Feature Metadata for better UI
const CATEGORICAL_FEATURES = {
    "Sex": [
        { val: 1, label: "Male" },
        { val: 0, label: "Female" }
    ],
    "Chest Pain Type": [
        { val: 1, label: "Typical Angina" },
        { val: 2, label: "Atypical Angina" },
        { val: 3, label: "Non-anginal Pain" },
        { val: 4, label: "Asymptomatic" }
    ],
    "Fasting Blood Sugar": [
        { val: 1, label: "> 120 mg/dl" },
        { val: 0, label: "< 120 mg/dl" }
    ],
    "Resting ECG": [
        { val: 0, label: "Normal" },
        { val: 1, label: "ST-T Wave Abnormality" },
        { val: 2, label: "Left Ventricular Hypertrophy" }
    ],
    "Exercise Angina": [
        { val: 1, label: "Yes" },
        { val: 0, label: "No" }
    ],
    "Thalassemia": [
        { val: 3, label: "Normal" },
        { val: 6, label: "Fixed Defect" },
        { val: 7, label: "Reversable Defect" }
    ]
};

// Load Features from API
async function loadFeatures(disease, containerId) {
    const container = document.getElementById(containerId);
    if (container.children.length > 0) container.innerHTML = ''; // Clear and reload for reduced features

    try {
        const response = await fetch(`/api/features/${disease}`);
        const data = await response.json();
        
        data.features.forEach(feat => {
            const div = document.createElement('div');
            div.className = 'input-group';
            
            if (CATEGORICAL_FEATURES[feat]) {
                const options = CATEGORICAL_FEATURES[feat].map(opt => 
                    `<option value="${opt.val}">${opt.label}</option>`
                ).join('');
                div.innerHTML = `
                    <label for="${disease}-${feat}">${feat}</label>
                    <select id="${disease}-${feat}" name="${feat}" required>
                        ${options}
                    </select>
                `;
            } else {
                div.innerHTML = `
                    <label for="${disease}-${feat}">${feat}</label>
                    <input type="number" step="any" id="${disease}-${feat}" name="${feat}" placeholder="Enter value" required>
                `;
            }
            container.appendChild(div);
        });
    } catch (err) {
        console.error("Failed to load features:", err);
    }
}

// Handle Forms
['heart', 'diabetes', 'cancer'].forEach(type => {
    const form = document.getElementById(`${type}-form`);
    form?.addEventListener('submit', async (e) => {
        e.preventDefault();
        showLoader(true);

        const diseaseMap = { 'heart': 'heart_disease', 'diabetes': 'diabetes', 'cancer': 'breast_cancer' };
        const formData = new FormData(form);
        const jsonData = {};
        formData.forEach((value, key) => jsonData[key] = value);

        try {
            const response = await fetch(`/api/predict/${diseaseMap[type]}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(jsonData)
            });
            const result = await response.json();
            showResult(result);
        } catch (err) {
            alert("Prediction failed. Make sure all fields are filled correctly.");
        } finally {
            showLoader(false);
        }
    });
});

function showLoader(show) {
    document.getElementById('loading').style.display = show ? 'flex' : 'none';
}

function showResult(data) {
    const modal = document.getElementById('result-modal');
    const badge = document.getElementById('res-badge');
    const probFill = document.getElementById('res-prob-fill');
    const probText = document.getElementById('res-prob-text');
    const icon = document.getElementById('res-icon');

    badge.innerText = data.prediction_label;
    badge.className = 'prediction-badge ' + (data.prediction === 1 ? 'badge-positive' : 'badge-negative');
    
    icon.innerText = data.prediction === 1 ? '⚠️' : '✓';
    icon.style.background = data.prediction === 1 ? 'var(--danger)' : 'var(--secondary)';

    const percentage = Math.round(data.probability * 100);
    probFill.style.width = percentage + '%';
    probText.innerText = percentage + '% Risk';

    modal.style.display = 'flex';
    updateChart(data);
}

function updateChart(data) {
    const ctx = document.getElementById('predictionChart').getContext('2d');
    if (myChart) myChart.destroy();

    myChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Negative Class', 'Positive Class'],
            datasets: [{
                data: data.all_probabilities,
                backgroundColor: ['#10b981', '#ef4444'],
                borderWidth: 0,
                spacing: 5
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'bottom', labels: { color: '#94a3b8' } }
            }
        }
    });
}

// Close Modal
document.querySelector('.close-btn').addEventListener('click', () => {
    document.getElementById('result-modal').style.display = 'none';
});

window.addEventListener('click', (e) => {
    if (e.target.classList.contains('modal')) {
        e.target.style.display = 'none';
    }
});
