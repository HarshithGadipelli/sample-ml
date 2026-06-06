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

  // Adding subtle hover scale effect for cards
  document.querySelectorAll(".card").forEach((el) => {
    el.style.transition = "transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out";
    el.addEventListener("mouseenter", () => {
      el.style.transform = "translateY(-2px)";
      el.style.boxShadow = "0 18px 48px rgba(27, 60, 35, .15)";
    });
    el.addEventListener("mouseleave", () => {
      el.style.transform = "none";
      el.style.boxShadow = "var(--shadow)";
    });
  });
});
