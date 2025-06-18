//  static/claim_strength.js
// Throttle: ensures fn runs at most once every `delay` ms
function throttle(fn, delay) {
  let lastCall = 0;
  let timer;
  return function(...args) {
    const now = Date.now();
    const remaining = delay - (now - lastCall);
    if (remaining <= 0) {
      clearTimeout(timer);
      lastCall = now;
      fn.apply(this, args);
    } else {
      clearTimeout(timer);
      timer = setTimeout(() => {
        lastCall = Date.now();
        fn.apply(this, args);
      }, remaining);
    }
  };
}
window.throttledEvaluateClaimStrength = throttle(evaluateClaimStrength, 300);

function evaluateClaimStrength(textarea) {
  

  const text = textarea.value.trim().toLowerCase();

  /* ───── heuristic scoring ───── */
  const lengthScore = Math.min(text.length / 40, 1);
  const hasColor    = /(black|white|red|blue|green|brown|grey)/.test(text) ? 0.2 : 0;
  const hasBrand    = /(casio|hp|nike|titan|sony|apple)/.test(text)       ? 0.2 : 0;
  const hasLocation = /(library|canteen|lab|class|hall|ground)/.test(text)? 0.2 : 0;
  const hasNumber   = /\d/.test(text) ? 0.1 : 0;

  const score = Math.min(lengthScore + hasColor + hasBrand + hasLocation + hasNumber, 1);

  let msg, colour;
  if (score > 0.7)      { msg = "🟢 Strong claim";  colour = "green";  }
  else if (score > 0.4) { msg = "🟡 Add more detail"; colour = "orange"; }
  else                  { msg = "🔴 Too vague";     colour = "red";    }

  /* update the sibling feedback div */
  const meter = textarea.parentElement.querySelector('.claim-strength');
  if (meter) {
    meter.textContent = msg;
    meter.style.color = colour;
  }
}


// gate the Submit button by strength score
function evaluateReportStrength(textarea) {
  const text = textarea.value.trim().toLowerCase();   // <-- missing line

  /* ── revised heuristic ── */
  const words        = text.split(/\s+/).filter(Boolean);
  const lengthScore  = Math.min(words.length / 12, 0.4);     // 0-0.4
  const hasColor     = /(black|white|red|blue|green|brown|grey)/.test(text)      ? 0.3 : 0;
  const hasBrand     = /(casio|hp|nike|titan|sony|apple|samsung|lenovo)/.test(text) ? 0.3 : 0;
  const hasNumber    = /\b\d{3,}\b/.test(text)                                           ? 0.2 : 0;
  const hasKeyword   = /(wallet|phone|book|id|card|pen|earphones|calculator|bag)/.test(text) ? 0.2 : 0;
  const hasLocation  = /(library|canteen|lab|class|hall|ground)/.test(text)             ? 0.2 : 0;

  const score = Math.min(
      lengthScore + hasColor + hasBrand + hasNumber + hasKeyword + hasLocation,
      1);

  /* colour & message */
  let msg, clr;
  if (score >= 0.8)      { msg = "🟢 Strong";   clr = "green";  }
  else if (score >= 0.5) { msg = "🟡 Add more"; clr = "orange"; }
  else                   { msg = "🔴 Too vague";clr = "red";    }

  const meter = document.getElementById("report-strength");
  meter.textContent = msg;
  meter.style.color = clr;

  /* enable submit only for score ≥ 0.8 */
  document.getElementById("reportSubmitBtn").disabled = !(score >= 0.8);

  // Create a throttled wrapper around evaluateClaimStrength (300ms)
  


}
