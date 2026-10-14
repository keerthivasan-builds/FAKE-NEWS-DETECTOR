const form = document.getElementById("detectorForm");
const newsInput = document.getElementById("newsInput");
const checkBtn = document.getElementById("checkBtn");
const resultCard = document.getElementById("resultCard");
const resultBadge = document.getElementById("resultBadge");
const resultIcon = document.getElementById("resultIcon");
const labelValue = document.getElementById("labelValue");
const confidenceValue = document.getElementById("confidenceValue");
const progressValue = document.getElementById("progressValue");
const progressBar = document.getElementById("progressBar");
const themeToggle = document.getElementById("themeToggle");
const themeText = document.getElementById("themeText");
const locateBtn = document.getElementById("locateBtn");
const locationStatus = document.getElementById("locationStatus");
const locationCoords = document.getElementById("locationCoords");
const mapsLink = document.getElementById("mapsLink");

const THEME_KEY = "fnd-theme";

function setTheme(mode) {
  document.body.classList.remove("light", "dark");
  document.body.classList.add(mode);
  themeText.textContent = mode === "light" ? "LIGHT" : "DARK";
  localStorage.setItem(THEME_KEY, mode);
}

(function initTheme() {
  const saved = localStorage.getItem(THEME_KEY);
  if (saved === "light" || saved === "dark") {
    setTheme(saved);
    return;
  }

  const prefersLight = window.matchMedia("(prefers-color-scheme: light)").matches;
  setTheme(prefersLight ? "light" : "dark");
})();

themeToggle.addEventListener("click", () => {
  const next = document.body.classList.contains("light") ? "dark" : "light";
  setTheme(next);
});

function updateLocationStatus(message) {
  locationStatus.textContent = message;
}

function showCoordinates(latitude, longitude) {
  locationCoords.textContent = `Latitude: ${latitude.toFixed(6)} | Longitude: ${longitude.toFixed(6)}`;
}

locateBtn.addEventListener("click", () => {
  if (!navigator.geolocation) {
    updateLocationStatus("Geolocation is not supported in this browser.");
    return;
  }

  locateBtn.disabled = true;
  updateLocationStatus("Requesting live location access...");

  navigator.geolocation.getCurrentPosition(
    (position) => {
      const { latitude, longitude } = position.coords;
      showCoordinates(latitude, longitude);
      updateLocationStatus("Live location captured successfully.");
      mapsLink.href = `https://www.google.com/maps?q=${latitude},${longitude}`;
      mapsLink.hidden = false;
      locateBtn.disabled = false;
    },
    (error) => {
      const messageByCode = {
        [error.PERMISSION_DENIED]: "Location permission denied. Please allow access and try again.",
        [error.POSITION_UNAVAILABLE]: "Location data is unavailable right now.",
        [error.TIMEOUT]: "Location request timed out. Please retry.",
      };

      updateLocationStatus(messageByCode[error.code] || "Unable to fetch live location.");
      locateBtn.disabled = false;
    },
    {
      enableHighAccuracy: true,
      timeout: 12000,
      maximumAge: 0,
    }
  );
});

function analyzeNews(text) {
  const fakeWords = ["shocking", "viral", "click", "secret", "hoax", "exposed", "miracle"];
  const realWords = ["report", "official", "statement", "analysis", "confirmed", "agency"];

  const content = text.toLowerCase();
  let fakeScore = 0;
  let realScore = 0;

  fakeWords.forEach((word) => {
    if (content.includes(word)) fakeScore += 1;
  });

  realWords.forEach((word) => {
    if (content.includes(word)) realScore += 1;
  });

  if (content.length < 50) fakeScore += 1;

  const isReal = realScore >= fakeScore;
  const base = isReal ? 62 : 58;
  const spread = Math.min(Math.abs(realScore - fakeScore) * 8, 35);
  const confidence = Math.min(base + spread + Math.floor(Math.random() * 7), 96);

  return {
    label: isReal ? "Real" : "Fake",
    confidence,
  };
}

function showResult({ label, confidence }) {
  const isReal = label === "Real";

  labelValue.textContent = label;
  confidenceValue.textContent = `${confidence}%`;
  progressValue.textContent = `${confidence}%`;
  resultBadge.textContent = isReal ? "Likely Reliable" : "Potentially Misleading";

  if (isReal) {
    labelValue.style.color = "#18d26e";
    resultIcon.className = "fa-solid fa-circle-check";
    resultIcon.style.color = "#18d26e";
    progressBar.style.background = "linear-gradient(90deg, #00d68f, #a9ff68)";
  } else {
    labelValue.style.color = "#ff3232";
    resultIcon.className = "fa-solid fa-circle-exclamation";
    resultIcon.style.color = "#ff3232";
    progressBar.style.background = "linear-gradient(90deg, #ff2727, #ff5d5d)";
  }

  progressBar.style.width = "0%";

  requestAnimationFrame(() => {
    progressBar.style.width = `${confidence}%`;
  });
}

form.addEventListener("submit", (event) => {
  event.preventDefault();

  const text = newsInput.value.trim();
  if (!text) {
    newsInput.focus();
    return;
  }

  checkBtn.classList.add("loading");
  checkBtn.disabled = true;

  setTimeout(() => {
    const result = analyzeNews(text);
    showResult(result);
    checkBtn.classList.remove("loading");
    checkBtn.disabled = false;
  }, 1300);
});
