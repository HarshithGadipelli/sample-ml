
document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".metric-value, .score-box strong").forEach((el) => {
    const text = el.textContent.trim();
    const num = parseFloat(text);
    if (!Number.isNaN(num)) {
      el.style.transition = "transform 0.35s ease, opacity 0.35s ease";
      el.style.opacity = "0";
      el.style.transform = "translateY(8px)";
      setTimeout(() => {
        el.style.opacity = "1";
        el.style.transform = "translateY(0)";
      }, 100);
    }
  });
});
