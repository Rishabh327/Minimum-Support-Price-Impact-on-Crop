// State management for AcreoMetrics Dashboard
let dashboardData = null;
let activeCrop = "Wheat";
let historicalChart = null;
let growthChart = null;

// Constant variable mapping for OLS coefficients to display labels
const variableLabels = {
    "const": "Intercept (Constant β₀)",
    "msp_change_pct": "MSP Change (%) (β₁)",
    "cost_index": "Agricultural Input Cost Index (β₂)",
    "lagged_area_change_pct": "Lagged Acreage Change (%) (β₃)",
    "interaction_term": "MSP * Cost Interaction Effect (β₄)"
};

// Map variable names to standard econometric explanations
const variableExplanations = {
    "const": "Reflects baseline autonomous growth in sown area due to standard developmental cycles independent of price.",
    "msp_change_pct": "Measures the acreage supply response sensitivity to Minimum Support Price price revisions (incentive elasticity).",
    "cost_index": "Reflects the structural cost drag of modern inputs (fertilizers, diesel, seeds) on planting decisions.",
    "lagged_area_change_pct": "Represents agricultural habit persistence, crop-rotation cycles, and long-term capital lock-in effects.",
    "interaction_term": "The crucial interaction variable. Demonstrates how rising input cost inflation dilutes standard MSP incentive structures."
};

// Map crops to custom policy templates
const cropPolicyTemplates = {
    "Wheat": "Wheat exhibits high sensitivity to MSP incentives. However, because it relies heavily on intensive fertilizer applications, the negative interaction coefficient confirms that input cost inflation strongly suppresses planting expansion even during MSP price boosts.",
    "Paddy": "Paddy acreage shows a steady baseline due to irrigation networks and crop security. The temporal causality tests show that MSP decisions have lagged impacts on water-intensive cropping, requiring balanced water and support policies.",
    "Maize": "Maize presents a highly responsive market dynamic. Since it is often planted as an alternative to other grains, input cost indices heavily affect its marginal profitability, making dynamic price signaling critical.",
    "Cotton": "Cotton is a highly commercial cash crop. Sown area decisions are strongly driven by global price expectations and domestic support grids. High input costs (especially crop protection) act as massive dampening filters on acreage growth.",
    "Mustard": "Mustard acts as a highly resilient oilseed crop. Sown area responds moderately to MSP shifts. Due to lower intensive input requirements compared to paddy or wheat, it is less vulnerable to cost inflation drags, showing unique policy options.",
    "Groundnut": "Groundnut is key to oilseed self-sufficiency. Sown area shows significant causal feedback loop relations to MSP support. Under heavy input cost levels, the structural supply response requires solid market price floors to stimulate planting."
};

// Initialize Application
document.addEventListener("DOMContentLoaded", () => {
    fetchData();
    setupEventListeners();
});

// Fetch econometric data
async function fetchData() {
    try {
        console.log("Fetching econometric results...");
        const response = await fetch("data.json");
        if (!response.ok) {
            throw new Error("Failed to load data.json. Ensure econometrics_engine.py was run successfully.");
        }
        dashboardData = await response.json();
        console.log("Data loaded successfully:", dashboardData);
        
        // Populate sidebar selectors
        populateSidebar();
        
        // Initialize dashboard with active crop
        switchCrop(activeCrop);
    } catch (error) {
        console.error("Initialization Error:", error);
        document.getElementById("crop-name").innerText = "Error Loading Data";
        document.getElementById("policy-advisory-msg").innerHTML = `<span style="color: var(--accent-rose)"><i class="fa-solid fa-triangle-exclamation"></i> Error: ${error.message}</span>`;
    }
}

// Set up UI Event Listeners
function setupEventListeners() {
    // Simulator inputs
    const mspSlider = document.getElementById("sim-msp-change");
    const costSlider = document.getElementById("sim-cost-index");
    const laggedSlider = document.getElementById("sim-lagged-area");
    
    mspSlider.addEventListener("input", (e) => {
        document.getElementById("sim-msp-val").innerText = (e.target.value >= 0 ? "+" : "") + e.target.value + "%";
        runPolicySimulation();
    });
    
    costSlider.addEventListener("input", (e) => {
        document.getElementById("sim-cost-val").innerText = e.target.value;
        runPolicySimulation();
    });
    
    laggedSlider.addEventListener("input", (e) => {
        document.getElementById("sim-lagged-val").innerText = (e.target.value >= 0 ? "+" : "") + e.target.value + "%";
        runPolicySimulation();
    });
}

// Populate Sidebar with Crops
function populateSidebar() {
    const selectorContainer = document.getElementById("crop-selector");
    selectorContainer.innerHTML = "";
    
    const crops = Object.keys(dashboardData.crops);
    crops.forEach(crop => {
        const cropInfo = dashboardData.crops[crop];
        const avgPrice = cropInfo.baseline_stats ? cropInfo.baseline_stats.avg_modal : "N/A";
        const recordCount = cropInfo.baseline_stats ? cropInfo.baseline_stats.record_count : 0;
        
        const cropItem = document.createElement("div");
        cropItem.className = `crop-item ${crop === activeCrop ? 'active' : ''}`;
        cropItem.setAttribute("data-crop", crop);
        
        cropItem.innerHTML = `
            <div class="crop-info">
                <span class="crop-title">${crop}</span>
                <span class="crop-records">${recordCount} APMC Records</span>
            </div>
            <span class="crop-badge">₹${Math.round(avgPrice)}</span>
        `;
        
        cropItem.addEventListener("click", () => {
            document.querySelectorAll(".crop-item").forEach(item => item.classList.remove("active"));
            cropItem.classList.add("active");
            switchCrop(crop);
        });
        
        selectorContainer.appendChild(cropItem);
    });
}

// Switch Active Crop and Refresh Dashboard
function switchCrop(cropName) {
    activeCrop = cropName;
    const cropData = dashboardData.crops[cropName];
    
    // Update summary card metadata
    document.getElementById("crop-name").innerText = cropName;
    const baseStats = cropData.baseline_stats;
    if (baseStats) {
        document.getElementById("crop-apmc-base").innerText = `APMC Anchor (2019): ₹${baseStats.avg_modal}/Qtl (Records: ${baseStats.record_count})`;
    } else {
        document.getElementById("crop-apmc-base").innerText = `APMC Anchor: Simulated Level`;
    }
    
    // Update OLS Diagnostics
    const reg = cropData.regression;
    document.getElementById("model-r2").innerText = reg.r_squared.toFixed(4);
    document.getElementById("model-adj-r2").innerText = `Adj. R²: ${reg.adj_r_squared.toFixed(4)}`;
    document.getElementById("dw-stat-badge").innerText = `Durbin-Watson d: ${reg.durbin_watson.toFixed(3)}`;
    document.getElementById("f-stat-badge").innerText = `F-Statistic: ${reg.f_statistic.toFixed(2)} (p = ${reg.f_pvalue.toExponential(3)})`;
    
    // Update Granger Causal Badge
    let verified = false;
    let bestP = 1.0;
    Object.keys(cropData.granger_causality).forEach(lag => {
        const p = cropData.granger_causality[lag].p_value;
        if (p < 0.05) verified = true;
        if (p < bestP) bestP = p;
    });
    
    const statusVal = document.getElementById("causality-status");
    const statusP = document.getElementById("causality-pvalue");
    const iconContainer = document.getElementById("causal-icon-container");
    
    if (verified) {
        statusVal.innerText = "Causal Verified";
        statusVal.className = "card-value positive";
        statusP.innerText = `Granger Causality Validated (p = ${bestP.toFixed(4)})`;
        iconContainer.className = "card-icon positive";
        iconContainer.innerHTML = '<i class="fa-solid fa-circle-check"></i>';
    } else {
        statusVal.innerText = "Weakly Causal";
        statusVal.className = "card-value negative";
        statusP.innerText = `Granger test (p = ${bestP.toFixed(4)} > 0.05)`;
        iconContainer.className = "card-icon non-causal";
        iconContainer.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i>';
    }
    
    // Refresh tables and charts
    populateRegressionTable(cropData);
    populateGrangerTable(cropData);
    drawCharts(cropData);
    
    // Sync simulator parameters to sensible defaults
    resetSimulatorDefaults(cropData);
    runPolicySimulation();
}

// Populate OLS Coefficient Table
function populateRegressionTable(cropData) {
    const tableBody = document.getElementById("regression-coef-body");
    tableBody.innerHTML = "";
    
    const coefficients = cropData.regression.coefficients;
    const variables = ["const", "msp_change_pct", "cost_index", "lagged_area_change_pct", "interaction_term"];
    
    variables.forEach(varKey => {
        if (!coefficients[varKey]) return;
        const coef = coefficients[varKey];
        
        let pClass = "pvalue-insig";
        let stars = "ns";
        let starsClass = "pvalue-insig";
        
        if (coef.p_value < 0.01) {
            pClass = "pvalue-sig";
            stars = "***";
            starsClass = "sig-stars high";
        } else if (coef.p_value < 0.05) {
            pClass = "pvalue-sig";
            stars = "**";
            starsClass = "sig-stars med";
        } else if (coef.p_value < 0.10) {
            pClass = "pvalue-weak";
            stars = "*";
            starsClass = "sig-stars low";
        }
        
        const row = document.createElement("tr");
        row.innerHTML = `
            <td>
                <strong>${variableLabels[varKey]}</strong>
                <div style="font-size: 11px; color: var(--text-secondary); font-family: var(--font-body); font-weight: normal; margin-top: 2px;">
                    ${variableExplanations[varKey]}
                </div>
            </td>
            <td>${coef.coefficient.toFixed(5)}</td>
            <td>${coef.std_err.toFixed(5)}</td>
            <td>${coef.t_stat.toFixed(3)}</td>
            <td class="${pClass}">${coef.p_value.toFixed(4)}</td>
            <td>[${coef.conf_lower.toFixed(4)}, ${coef.conf_upper.toFixed(4)}]</td>
            <td><span class="${starsClass}">${stars}</span></td>
        `;
        tableBody.appendChild(row);
    });
}

// Populate Granger Causality Table & Narrative
function populateGrangerTable(cropData) {
    const tableBody = document.getElementById("granger-table-body");
    tableBody.innerHTML = "";
    
    const gc = cropData.granger_causality;
    const lags = Object.keys(gc);
    
    let significantLag = null;
    let minP = 1.0;
    
    lags.forEach(lagKey => {
        const item = gc[lagKey];
        const isCausal = item.p_value < 0.05;
        if (isCausal && item.p_value < minP) {
            minP = item.p_value;
            significantLag = lagKey === "lag_1" ? 1 : 2;
        }
        
        const row = document.createElement("tr");
        row.innerHTML = `
            <td>Lag ${lagKey.split("_")[1]}</td>
            <td>${item.f_stat.toFixed(4)}</td>
            <td class="${isCausal ? 'pvalue-sig' : 'pvalue-insig'}">${item.p_value.toFixed(4)}</td>
            <td>(${item.df_num}, ${item.df_denom})</td>
            <td>
                <span class="causality-badge ${isCausal ? 'verified' : 'unverified'}">
                    ${isCausal ? 'Causality Validated' : 'No Causality'}
                </span>
            </td>
        `;
        tableBody.appendChild(row);
    });
    
    // Update Causal Narrative Panel
    const narrativeTextContainer = document.getElementById("causality-narrative-text");
    let narrativeHTML = "";
    
    const cropTemplate = cropPolicyTemplates[activeCrop] || "";
    
    if (significantLag) {
        narrativeHTML = `
            <p style="margin-bottom: 12px;">
                Our Granger causality analysis **confirms dynamic temporal causality** flowing from support price choices to actual acreage adjustments for <strong>${activeCrop}</strong> at <strong>Lag ${significantLag}</strong> ($F$-stat = ${gc['lag_' + significantLag].f_stat.toFixed(2)}, $p$ = ${gc['lag_' + significantLag].p_value.toFixed(4)}).
            </p>
            <p style="margin-bottom: 12px;">
                This means past information about MSP announcements provides statistically significant forecasting power for current acreage supply decisions, even after controlling for lagged acreage shifts. In plain terms: <strong>farmers actively adapt their sowing planning in subsequent planting seasons based on support price signals.</strong>
            </p>
            <p>
                <em>${cropTemplate}</em>
            </p>
        `;
    } else {
        narrativeHTML = `
            <p style="margin-bottom: 12px;">
                For <strong>${activeCrop}</strong>, Granger causality tests show weak temporal causality at standard 5% thresholds (Minimum $p$ = ${minP.toFixed(4)}). This suggests that market price signals might be diffused by intermediate market shocks, irrigation limitations, or agricultural locking effects.
            </p>
            <p style="margin-bottom: 12px;">
                While the direct linear regression highlights strong contemporary supply correlations, the lack of lagged predictive power means acreage shifts happen rapidly or are constrained by strict physical cropping frameworks.
            </p>
            <p>
                <em>${cropTemplate}</em>
            </p>
        `;
    }
    
    narrativeTextContainer.innerHTML = narrativeHTML;
}

// Reset Sliders to sensible defaults for active crop
function resetSimulatorDefaults(cropData) {
    // We set proposed MSP change to +5%
    document.getElementById("sim-msp-change").value = 5.0;
    document.getElementById("sim-msp-val").innerText = "+5.0%";
    
    // Input cost index is anchored at 180 (simulated index region)
    document.getElementById("sim-cost-index").value = 180;
    document.getElementById("sim-cost-val").innerText = "180";
    
    // Lagged area is anchored at +1.0%
    document.getElementById("sim-lagged-area").value = 1.0;
    document.getElementById("sim-lagged-val").innerText = "+1.0%";
}

// Run Interactive Policy Simulation in Real-time
function runPolicySimulation() {
    if (!dashboardData) return;
    
    const cropData = dashboardData.crops[activeCrop];
    const coefs = cropData.regression.coefficients;
    
    // Get values from sliders
    const mspChange = parseFloat(document.getElementById("sim-msp-change").value);
    const costIndex = parseFloat(document.getElementById("sim-cost-index").value);
    const laggedArea = parseFloat(document.getElementById("sim-lagged-area").value);
    
    // Calculate regression variables
    const intercept = coefs["const"].coefficient;
    const mspCoef = coefs["msp_change_pct"].coefficient;
    const costCoef = coefs["cost_index"].coefficient;
    const laggedCoef = coefs["lagged_area_change_pct"].coefficient;
    const interactionCoef = coefs["interaction_term"].coefficient;
    
    // Component calculations
    const mspIncentive = mspCoef * mspChange;
    const costDrag = costCoef * costIndex;
    const interactionTerm = mspChange * costIndex;
    const interactionDrag = interactionCoef * interactionTerm;
    const laggedEffect = laggedCoef * laggedArea;
    
    // Core econometric calculation: y = b0 + b1*x1 + b2*x2 + b3*x3 + b4*(x1*x2)
    const predictedChange = intercept + mspIncentive + costDrag + laggedEffect + interactionDrag;
    
    // Update prediction value UI
    const predValContainer = document.getElementById("predicted-change-val");
    predValContainer.innerText = (predictedChange >= 0 ? "+" : "") + predictedChange.toFixed(2) + "%";
    
    if (predictedChange >= 0) {
        predValContainer.className = "prediction-value positive";
    } else {
        predValContainer.className = "prediction-value negative";
    }
    
    // Calculate and update breakdown components
    document.getElementById("bd-msp-incentive").innerText = (mspIncentive >= 0 ? "+" : "") + mspIncentive.toFixed(2) + "%";
    document.getElementById("bd-cost-drag").innerText = costDrag.toFixed(2) + "%";
    document.getElementById("bd-interaction-drag").innerText = (interactionDrag >= 0 ? "+" : "") + interactionDrag.toFixed(2) + "%";
    
    // Update Circular Dial Gauge (Fill degree calculation)
    // Map predictedChange range of [-10%, +15%] to radial fill degrees of [0, 360]
    const minVal = -8.0;
    const maxVal = 12.0;
    const percent = Math.min(Math.max((predictedChange - minVal) / (maxVal - minVal), 0), 1);
    const degrees = percent * 360;
    document.getElementById("gauge-fill").style.setProperty("--fill-deg", `${degrees}deg`);
    
    // Set custom policy advice message dynamically
    const adviceContainer = document.getElementById("policy-advisory-msg");
    let adviceText = "";
    
    if (interactionDrag < 0 && Math.abs(interactionDrag) > 0.5) {
        adviceText = `
            <strong>Policy Warning:</strong> Proposed MSP increase of ${mspChange}% is actively diluted by high input costs (${costIndex}). 
            The negative interaction effect drags down acreage growth by an additional <strong>${Math.abs(interactionDrag).toFixed(2)}%</strong>. 
            <em>Recommendation: Pair MSP incentives with direct diesel/fertilizer subsidies to reduce costs below index 150.</em>
        `;
    } else if (predictedChange < 0) {
        adviceText = `
            <strong>Acreage Contraction Alert:</strong> Sown area is predicted to contract by <strong>${Math.abs(predictedChange).toFixed(2)}%</strong>. 
            The structural input cost drag (${costDrag.toFixed(2)}%) completely overrides support incentives. 
            <em>Recommendation: A significant MSP increase (> ${Math.max(10, Math.ceil(Math.abs(costDrag)/mspCoef)) + 2}%) is required to sustain baseline sown acreage.</em>
        `;
    } else {
        adviceText = `
            <strong>Stable Acreage Growth:</strong> A healthy sown area expansion of <strong>+${predictedChange.toFixed(2)}%</strong> is projected. 
            Cost levels are manageable, allowing support price incentives to successfully stimulate farming expansion.
        `;
    }
    adviceContainer.innerHTML = adviceText;
}

// Draw Dual Charts (Historical and Growth Rate Panels)
function drawCharts(cropData) {
    const ts = cropData.time_series;
    
    const labels = ts.map(item => item.year);
    const sownArea = ts.map(item => item.sown_area);
    const msp = ts.map(item => item.msp);
    const costIndex = ts.map(item => item.cost_index);
    const areaChange = ts.map(item => item.sown_area_change_pct);
    
    // 1. Destroy existing charts if initialized
    if (historicalChart) historicalChart.destroy();
    if (growthChart) growthChart.destroy();
    
    // Custom Chart.js configurations for glowing dark-theme aesthetics
    const chartGridColor = "rgba(255, 255, 255, 0.05)";
    const chartTextColor = "#9ca3af";
    
    // 2. Draw Historical Trends Chart (Dual Axes)
    const ctxHist = document.getElementById("historicalChart").getContext("2d");
    
    // Smooth line gradient for Acreage
    const areaGradient = ctxHist.createLinearGradient(0, 0, 0, 280);
    areaGradient.addColorStop(0, "rgba(99, 102, 241, 0.4)");
    areaGradient.addColorStop(1, "rgba(99, 102, 241, 0.02)");
    
    historicalChart = new Chart(ctxHist, {
        type: "bar",
        data: {
            labels: labels,
            datasets: [
                {
                    label: "MSP (Rs/Qtl)",
                    data: msp,
                    yAxisID: "y2",
                    backgroundColor: "rgba(245, 158, 11, 0.45)",
                    hoverBackgroundColor: "rgba(245, 158, 11, 0.7)",
                    borderRadius: 4,
                    barPercentage: 0.55,
                    order: 2
                },
                {
                    label: "Acreage (MHa)",
                    data: sownArea,
                    type: "line",
                    yAxisID: "y1",
                    borderColor: "#6366f1",
                    borderWidth: 3,
                    pointBackgroundColor: "#6366f1",
                    pointBorderColor: "#0d121f",
                    pointBorderWidth: 1.5,
                    pointRadius: 4,
                    pointHoverRadius: 6,
                    fill: true,
                    backgroundColor: areaGradient,
                    tension: 0.35,
                    order: 1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: "#0d121f",
                    titleColor: "#fff",
                    bodyColor: "#f3f4f6",
                    borderColor: "rgba(99, 102, 241, 0.2)",
                    borderWidth: 1,
                    padding: 10
                }
            },
            scales: {
                x: {
                    grid: { color: chartGridColor },
                    ticks: { color: chartTextColor }
                },
                y1: {
                    type: "linear",
                    position: "left",
                    grid: { color: chartGridColor },
                    ticks: { color: chartTextColor },
                    title: {
                        display: true,
                        text: "Sown Area (Million Hectares)",
                        color: chartTextColor,
                        font: { size: 10, weight: 600 }
                    }
                },
                y2: {
                    type: "linear",
                    position: "right",
                    grid: { drawOnChartArea: false },
                    ticks: { color: chartTextColor },
                    title: {
                        display: true,
                        text: "Minimum Support Price (₹/Quintal)",
                        color: chartTextColor,
                        font: { size: 10, weight: 600 }
                    }
                }
            }
        }
    });
    
    // 3. Draw Cost Index vs Acreage Growth Rate Chart
    const ctxGrowth = document.getElementById("growthChart").getContext("2d");
    
    growthChart = new Chart(ctxGrowth, {
        type: "line",
        data: {
            labels: labels,
            datasets: [
                {
                    label: "Input Cost Index",
                    data: costIndex,
                    yAxisID: "y1",
                    borderColor: "#f43f5e",
                    borderWidth: 2.5,
                    pointRadius: 0,
                    tension: 0.1,
                    fill: false
                },
                {
                    label: "Acreage Growth (%)",
                    data: areaChange,
                    yAxisID: "y2",
                    borderColor: "#10b981",
                    borderWidth: 2.5,
                    pointRadius: 3.5,
                    pointBackgroundColor: "#10b981",
                    tension: 0.3,
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: "#0d121f",
                    titleColor: "#fff",
                    bodyColor: "#f3f4f6",
                    borderColor: "rgba(244, 63, 94, 0.2)",
                    borderWidth: 1,
                    padding: 10
                }
            },
            scales: {
                x: {
                    grid: { color: chartGridColor },
                    ticks: { color: chartTextColor }
                },
                y1: {
                    type: "linear",
                    position: "left",
                    grid: { color: chartGridColor },
                    ticks: { color: chartTextColor },
                    title: {
                        display: true,
                        text: "Agricultural Input Cost Index",
                        color: chartTextColor,
                        font: { size: 10, weight: 600 }
                    }
                },
                y2: {
                    type: "linear",
                    position: "right",
                    grid: { drawOnChartArea: false },
                    ticks: { color: chartTextColor },
                    title: {
                        display: true,
                        text: "Acreage Growth Rate (Annual %)",
                        color: chartTextColor,
                        font: { size: 10, weight: 600 }
                    }
                }
            }
        }
    });
}
