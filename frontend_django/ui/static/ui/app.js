const form = document.getElementById("predictForm");
const apiUrlInput = document.getElementById("apiUrl");
const imageInput = document.getElementById("imageFile");
const responseBox = document.getElementById("responseBox");
const statusBadge = document.getElementById("statusBadge");
const latencyBadge = document.getElementById("latencyBadge");
const previewImage = document.getElementById("previewImage");
const previewPlaceholder = document.getElementById("previewPlaceholder");
const healthBtn = document.getElementById("healthBtn");

const setStatus = (text, tone = "idle") => {
  statusBadge.textContent = text;
  if (tone === "ok") {
    statusBadge.style.background = "rgba(29, 116, 179, 0.18)";
    statusBadge.style.color = "#1b5f99";
  } else if (tone === "error") {
    statusBadge.style.background = "rgba(217, 30, 24, 0.18)";
    statusBadge.style.color = "#9b2a1f";
  } else {
    statusBadge.style.background = "rgba(217, 30, 24, 0.12)";
    statusBadge.style.color = "#d91e18";
  }
};

const updateResponse = (payload) => {
  responseBox.textContent = JSON.stringify(payload, null, 2);
};

const showLatency = (durationMs) => {
  latencyBadge.textContent = `${durationMs} ms`;
};

const updatePreview = () => {
  const file = imageInput.files[0];
  if (!file) {
    previewImage.style.display = "none";
    previewPlaceholder.style.display = "block";
    return;
  }

  const reader = new FileReader();
  reader.onload = () => {
    previewImage.src = reader.result;
    previewImage.style.display = "block";
    previewPlaceholder.style.display = "none";
  };
  reader.readAsDataURL(file);
};

const fetchHealth = async () => {
  const baseUrl = apiUrlInput.value.trim().replace(/\/$/, "");
  if (!baseUrl) {
    setStatus("Missing URL", "error");
    return;
  }

  setStatus("Checking", "idle");
  latencyBadge.textContent = "-- ms";
  const started = performance.now();

  try {
    const response = await fetch(`${baseUrl}/health`);
    const duration = Math.round(performance.now() - started);
    showLatency(duration);

    if (!response.ok) {
      setStatus("Health failed", "error");
      updateResponse({ status: response.status, message: "Health check failed" });
      return;
    }

    const data = await response.json();
    setStatus("Healthy", "ok");
    updateResponse(data);
  } catch (error) {
    setStatus("Unavailable", "error");
    updateResponse({ error: error.message });
  }
};

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const baseUrl = apiUrlInput.value.trim().replace(/\/$/, "");
  const file = imageInput.files[0];

  if (!baseUrl || !file) {
    setStatus("Missing input", "error");
    return;
  }

  setStatus("Sending", "idle");
  latencyBadge.textContent = "-- ms";
  const formData = new FormData();
  formData.append("file", file);
  const started = performance.now();

  try {
    const response = await fetch(`${baseUrl}/predict`, {
      method: "POST",
      body: formData,
    });
    const duration = Math.round(performance.now() - started);
    showLatency(duration);

    if (!response.ok) {
      setStatus("Error", "error");
      const errorText = await response.text();
      updateResponse({ status: response.status, detail: errorText });
      return;
    }

    const data = await response.json();
    setStatus("Success", "ok");
    updateResponse(data);
  } catch (error) {
    setStatus("Network error", "error");
    updateResponse({ error: error.message });
  }
});

imageInput.addEventListener("change", updatePreview);
healthBtn.addEventListener("click", fetchHealth);
updatePreview();
