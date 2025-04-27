// Dashboard JavaScript for Predictive Maintenance System

// Initialize charts and gauges when the document is ready
document.addEventListener('DOMContentLoaded', function() {
    // Initialize gauges
    initializeGauges();
    
    // Load visualizations
    loadVisualizations();
    
    // Load sensor data and initialize charts
    loadSensorData();
    
    // Set up button event listeners
    document.getElementById('predictBtn').addEventListener('click', runPrediction);
    document.getElementById('refreshBtn').addEventListener('click', refreshData);
});

// Initialize health and RUL gauges
function initializeGauges() {
    // Health gauge
    const healthGaugeCtx = document.getElementById('healthGauge').getContext('2d');
    window.healthGauge = new Chart(healthGaugeCtx, {
        type: 'doughnut',
        data: {
            datasets: [{
                data: [75, 25], // Initial value: 75% health
                backgroundColor: [
                    getHealthColor(75),
                    '#f5f5f5'
                ],
                borderWidth: 0
            }]
        },
        options: {
            cutout: '70%',
            circumference: 180,
            rotation: 270,
            plugins: {
                tooltip: {
                    enabled: false
                },
                legend: {
                    display: false
                }
            },
            animation: {
                animateRotate: true,
                animateScale: true
            }
        }
    });
    
    // RUL gauge
    const rulGaugeCtx = document.getElementById('rulGauge').getContext('2d');
    window.rulGauge = new Chart(rulGaugeCtx, {
        type: 'doughnut',
        data: {
            datasets: [{
                data: [65, 35], // Initial value: 65 RUL
                backgroundColor: [
                    getRulColor(65),
                    '#f5f5f5'
                ],
                borderWidth: 0
            }]
        },
        options: {
            cutout: '70%',
            circumference: 180,
            rotation: 270,
            plugins: {
                tooltip: {
                    enabled: false
                },
                legend: {
                    display: false
                }
            },
            animation: {
                animateRotate: true,
                animateScale: true
            }
        }
    });
    
    // Set initial values
    updateHealthGauge(75);
    updateRulGauge(65);
}

// Update health gauge with new value
function updateHealthGauge(value) {
    const color = getHealthColor(value);
    window.healthGauge.data.datasets[0].data = [value, 100 - value];
    window.healthGauge.data.datasets[0].backgroundColor[0] = color;
    window.healthGauge.update();
    
    document.getElementById('healthValue').textContent = value.toFixed(1) + '%';
    updateStatusAlert(value);
}

// Update RUL gauge with new value
function updateRulGauge(value) {
    const color = getRulColor(value);
    // Scale RUL to 0-100 for gauge display
    const scaledValue = Math.min(100, Math.max(0, value));
    window.rulGauge.data.datasets[0].data = [scaledValue, 100 - scaledValue];
    window.rulGauge.data.datasets[0].backgroundColor[0] = color;
    window.rulGauge.update();
    
    document.getElementById('rulValue').textContent = value.toFixed(1);
}

// Get color based on health value
function getHealthColor(value) {
    if (value >= 70) {
        return '#28a745'; // Green - Healthy
    } else if (value >= 40) {
        return '#ffc107'; // Yellow - Warning
    } else {
        return '#dc3545'; // Red - Danger
    }
}

// Get color based on RUL value
function getRulColor(value) {
    if (value >= 70) {
        return '#28a745'; // Green - Long RUL
    } else if (value >= 30) {
        return '#ffc107'; // Yellow - Medium RUL
    } else {
        return '#dc3545'; // Red - Short RUL
    }
}

// Update status alert based on health value
function updateStatusAlert(healthValue) {
    const statusAlert = document.getElementById('statusAlert');
    
    if (healthValue >= 70) {
        statusAlert.className = 'alert alert-healthy';
        statusAlert.textContent = 'System is healthy. No maintenance required.';
    } else if (healthValue >= 40) {
        statusAlert.className = 'alert alert-warning';
        statusAlert.textContent = 'Warning: System showing signs of wear. Schedule maintenance soon.';
    } else {
        statusAlert.className = 'alert alert-danger';
        statusAlert.textContent = 'Critical: System at high risk of failure. Immediate maintenance required!';
    }
}

// Load visualizations from the API
function loadVisualizations() {
    fetch('/api/visualizations')
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                // Update visualization images
                document.getElementById('trainingHistoryImg').src = 'data:image/png;base64,' + data.visualizations.training_history;
                document.getElementById('actualVsPredictedImg').src = 'data:image/png;base64,' + data.visualizations.actual_vs_predicted;
                document.getElementById('modelComparisonImg').src = 'data:image/png;base64,' + data.visualizations.model_comparison;
                document.getElementById('featureImportanceImg').src = 'data:image/png;base64,' + data.visualizations.feature_importance;
                document.getElementById('rulDistributionImg').src = 'data:image/png;base64,' + data.visualizations.rul_distribution;
                document.getElementById('healthIndexImg').src = 'data:image/png;base64,' + data.visualizations.health_index;
            } else {
                console.error('Error loading visualizations:', data.message);
            }
        })
        .catch(error => {
            console.error('Error fetching visualizations:', error);
        });
}

// Load sensor data and initialize vibration chart
function loadSensorData() {
    fetch('/api/sensor_data')
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                initializeVibrationChart(data.data);
            } else {
                console.error('Error loading sensor data:', data.message);
            }
        })
        .catch(error => {
            console.error('Error fetching sensor data:', error);
        });
}

// Initialize vibration chart with sensor data
function initializeVibrationChart(data) {
    const vibrationChartCtx = document.getElementById('vibrationChart').getContext('2d');
    window.vibrationChart = new Chart(vibrationChartCtx, {
        type: 'line',
        data: {
            labels: data.timestamps,
            datasets: [
                {
                    label: 'Vibration X',
                    data: data.vibration_x,
                    borderColor: '#007bff',
                    backgroundColor: 'rgba(0, 123, 255, 0.1)',
                    borderWidth: 2,
                    tension: 0.1
                },
                {
                    label: 'Vibration Y',
                    data: data.vibration_y,
                    borderColor: '#28a745',
                    backgroundColor: 'rgba(40, 167, 69, 0.1)',
                    borderWidth: 2,
                    tension: 0.1
                },
                {
                    label: 'Vibration Z',
                    data: data.vibration_z,
                    borderColor: '#dc3545',
                    backgroundColor: 'rgba(220, 53, 69, 0.1)',
                    borderWidth: 2,
                    tension: 0.1
                }
            ]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Bearing Vibration Data'
                },
                tooltip: {
                    mode: 'index',
                    intersect: false
                },
                legend: {
                    position: 'top'
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Time'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Amplitude'
                    }
                }
            }
        }
    });
}

// Run prediction when button is clicked
function runPrediction() {
    fetch('/api/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({})  // Empty body for sample prediction
    })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                // Update gauges with prediction results
                updateHealthGauge(data.prediction.health_index);
                updateRulGauge(data.prediction.rul);
                
                // Update last updated timestamp
                document.getElementById('lastUpdated').textContent = data.prediction.timestamp;
            } else {
                console.error('Error making prediction:', data.message);
                alert('Error making prediction: ' + data.message);
            }
        })
        .catch(error => {
            console.error('Error fetching prediction:', error);
            alert('Error fetching prediction: ' + error);
        });
}

// Refresh data when button is clicked
function refreshData() {
    // Reload sensor data
    loadSensorData();
    
    // Update last updated timestamp
    document.getElementById('lastUpdated').textContent = new Date().toLocaleString();
}
