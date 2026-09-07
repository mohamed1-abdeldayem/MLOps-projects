// ==========================================================================
// Configuration
// ==========================================================================
const API_BASE_URL = "http://localhost:8000";

// ==========================================================================
// Element references
// ==========================================================================
const form = document.getElementById("churnForm");
const predictBtn = document.getElementById("predictBtn");
const btnSpinner = document.getElementById("btnSpinner");
const resetBtn = document.getElementById("resetBtn");

const statusIndicator = document.getElementById("statusIndicator");
const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");
const refreshStatusBtn = document.getElementById("refreshStatusBtn");

const errorBox = document.getElementById("errorBox");

const resultCard = document.getElementById("resultCard");
const resultBadge = document.getElementById("resultBadge");
const probabilityValue = document.getElementById("probabilityValue");
const probabilityBar = document.getElementById("probabilityBar");
const probabilityFill = document.getElementById("probabilityFill");
const resultNote = document.getElementById("resultNote");

// Fields that must be sent as numbers rather than strings.
const INTEGER_FIELDS = ["SeniorCitizen", "tenure"];
const FLOAT_FIELDS = ["MonthlyCharges", "TotalCharges"];

// ==========================================================================
// Health check
// ==========================================================================
async function checkHealth() {
  setStatus("checking");

  try {
    const response = await fetch(`${API_BASE_URL}/health`);

    if (!response.ok) {
      setStatus("offline");
      return;
    }

    const data = await response.json();

    if (data && data.model_loaded === true) {
      setStatus("online");
    } else {
      setStatus("not-ready");
    }
  } catch (err) {
    setStatus("offline");
  }
}

function setStatus(state) {
  statusIndicator.classList.remove("status--online", "status--offline");

  switch (state) {
    case "checking":
      statusText.textContent = "Checking API…";
      break;
    case "online":
      statusIndicator.classList.add("status--online");
      statusText.textContent = "API Online — Model Ready";
      break;
    case "not-ready":
      statusIndicator.classList.add("status--offline");
      statusText.textContent = "API Online — Model Not Ready";
      break;
    case "offline":
      statusIndicator.classList.add("status--offline");
      statusText.textContent = "API Offline";
      break;
    default:
      statusText.textContent = "Unknown status";
  }
}

// ==========================================================================
// Building the request payload
// ==========================================================================
function buildPredictionPayload(formData) {
  const payload = {};

  for (const [key, rawValue] of formData.entries()) {
    if (INTEGER_FIELDS.includes(key)) {
      payload[key] = parseInt(rawValue, 10);
    } else if (FLOAT_FIELDS.includes(key)) {
      payload[key] = parseFloat(rawValue);
    } else {
      payload[key] = rawValue;
    }
  }

  return payload;
}

// ==========================================================================
// Form validation
// ==========================================================================
function validateForm() {
  if (!form.checkValidity()) {
    form.reportValidity();
    return false;
  }
  return true;
}

// ==========================================================================
// Loading state
// ==========================================================================
function setLoadingState(isLoading) {
  predictBtn.disabled = isLoading;
  btnSpinner.hidden = !isLoading;
  predictBtn.querySelector(".btn__label").textContent = isLoading
    ? "Predicting…"
    : "Predict Churn";
}

// ==========================================================================
// Prediction request
// ==========================================================================
async function handlePrediction(event) {
  event.preventDefault();

  hideError();

  if (!validateForm()) {
    return;
  }

  const payload = buildPredictionPayload(new FormData(form));

  setLoadingState(true);
  resultCard.hidden = true;

  try {
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    if (response.status === 422) {
      let detailMessage = "The submitted customer data is invalid. Please check the form fields and try again.";
      try {
        const errorData = await response.json();
        if (errorData && errorData.detail) {
          detailMessage = formatValidationError(errorData.detail);
        }
      } catch (_) {
        // fall back to default message
      }
      displayError(detailMessage);
      return;
    }

    if (!response.ok) {
      displayError(`The prediction service returned an error (HTTP ${response.status}). Please try again.`);
      return;
    }

    const data = await response.json();
    displayPrediction(data);
  } catch (err) {
    displayError("Unable to connect to the prediction API. Please make sure the FastAPI server is running.");
  } finally {
    setLoadingState(false);
  }
}

function formatValidationError(detail) {
  if (Array.isArray(detail)) {
    const fields = detail
      .map((item) => (Array.isArray(item.loc) ? item.loc[item.loc.length - 1] : null))
      .filter(Boolean);
    if (fields.length > 0) {
      return `The submitted customer data is invalid. Please check: ${fields.join(", ")}.`;
    }
  }
  return "The submitted customer data is invalid. Please check the form fields and try again.";
}

// ==========================================================================
// Displaying results
// ==========================================================================
function displayPrediction(data) {
  if (!data || typeof data.prediction === "undefined" || typeof data.probability === "undefined") {
    displayError("The prediction service returned an unexpected response format.");
    return;
  }

  const isChurn = normalizeIsChurn(data.prediction);
  const probabilityPercent = normalizeProbability(data.probability);

  resultBadge.textContent = isChurn ? "Churn" : "No Churn";
  resultBadge.classList.remove("result-badge--churn", "result-badge--no-churn");
  resultBadge.classList.add(isChurn ? "result-badge--churn" : "result-badge--no-churn");

  probabilityValue.textContent = `${probabilityPercent}%`;
  probabilityBar.setAttribute("aria-valuenow", String(probabilityPercent));

  probabilityFill.classList.remove("probability-bar__fill--churn", "probability-bar__fill--no-churn");
  probabilityFill.classList.add(isChurn ? "probability-bar__fill--churn" : "probability-bar__fill--no-churn");
  // Trigger transition on next frame.
  requestAnimationFrame(() => {
    probabilityFill.style.width = `${probabilityPercent}%`;
  });

  resultNote.textContent = isChurn
    ? "This customer shows a high likelihood of churning. Consider proactive retention outreach."
    : "This customer shows a low likelihood of churning based on the submitted profile.";

  resultCard.hidden = false;
  resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function normalizeIsChurn(prediction) {
  if (typeof prediction === "number") {
    return prediction === 1;
  }
  if (typeof prediction === "string") {
    const value = prediction.trim().toLowerCase();
    return value === "yes" || value === "churn" || value === "true" || value === "1";
  }
  if (typeof prediction === "boolean") {
    return prediction;
  }
  return Boolean(prediction);
}

function normalizeProbability(probability) {
  let value = Number(probability);
  if (Number.isNaN(value)) {
    value = 0;
  }
  // Accept either 0-1 or 0-100 scales.
  if (value <= 1) {
    value = value * 100;
  }
  return Math.max(0, Math.min(100, Math.round(value * 10) / 10));
}

// ==========================================================================
// Error display
// ==========================================================================
function displayError(message) {
  errorBox.textContent = message;
  errorBox.hidden = false;
}

function hideError() {
  errorBox.hidden = true;
  errorBox.textContent = "";
}

// ==========================================================================
// Reset
// ==========================================================================
function handleReset() {
  form.reset();
  hideError();
  resultCard.hidden = true;
  probabilityFill.style.width = "0%";
}

// ==========================================================================
// Event bindings
// ==========================================================================
form.addEventListener("submit", handlePrediction);
resetBtn.addEventListener("click", handleReset);
refreshStatusBtn.addEventListener("click", checkHealth);

// Run health check on page load.
document.addEventListener("DOMContentLoaded", checkHealth);
