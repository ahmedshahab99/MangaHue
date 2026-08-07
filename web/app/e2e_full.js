// Full-journey verification: token -> demo -> start -> WAIT FOR COMPLETION -> result renders.
const puppeteer = require("puppeteer-core");
const TOKEN = "mh_J6PFAj-GmSQ2CIJu5dI-mTJatP3sCv08FCG_r6xxM9g";

(async () => {
  const browser = await puppeteer.launch({
    executablePath: "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
    headless: "new",
    args: ["--no-sandbox"],
  });
  const page = await browser.newPage();
  const errors = [];
  page.on("pageerror", (e) => errors.push("[pageerror] " + e.message.slice(0, 150)));
  page.on("console", (m) => {
    if (m.type() === "error") errors.push("[console] " + m.text().slice(0, 150));
  });
  await page.setViewport({ width: 1440, height: 900 });

  await page.goto("http://127.0.0.1:7890/app", { waitUntil: "networkidle2" });
  // Token via dialog
  await page.evaluate((token) => {
    const pw = [...document.querySelectorAll("input")].find((i) => i.type === "password");
    if (!pw) return;
    const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value").set;
    setter.call(pw, token);
    pw.dispatchEvent(new Event("input", { bubbles: true }));
  }, TOKEN);
  await new Promise((r) => setTimeout(r, 300));
  await page.evaluate(() => {
    const save = [...document.querySelectorAll("button")].find((b) => b.textContent.includes("Save"));
    if (save) save.click();
  });
  await new Promise((r) => setTimeout(r, 3500));
  console.log("1) connected:", await page.evaluate(() => document.body.innerText.includes("Engine ready")));

  // Demo page + start
  await page.evaluate(() => {
    const demo = [...document.querySelectorAll("button")].find((b) => /Try a demo page/.test(b.textContent));
    if (demo) demo.click();
  });
  await new Promise((r) => setTimeout(r, 2000));
  await page.evaluate(() => {
    const start = [...document.querySelectorAll("button")].find((b) => /Start colorization/.test(b.textContent));
    if (start) start.click();
  });
  console.log("2) job started, waiting for completion (up to 3 min)...");

  // Poll for completion: look for the before/after images or "Colorization complete" toast
  let done = false;
  for (let i = 0; i < 36; i++) {
    await new Promise((r) => setTimeout(r, 5000));
    const state = await page.evaluate(() => {
      const imgs = [...document.querySelectorAll("img")].map((img) => ({
        alt: img.alt,
        loaded: img.complete && img.naturalWidth > 0,
        w: img.naturalWidth,
      }));
      const text = document.body.innerText;
      return {
        imgs,
        completeToast: text.includes("Colorization complete"),
        progressPercent: /(\d+)%/.test(text) ? text.match(/(\d+)%/)[1] : null,
      };
    });
    const hasResult = state.imgs.some((i) => i.alt === "Colorized manga page" && i.loaded && i.w > 100);
    if (hasResult || state.completeToast) {
      console.log(`3) completed at poll ${i + 1}:`, JSON.stringify(state));
      done = true;
      break;
    }
    if (i % 6 === 0) console.log(`   ...still running (${state.progressPercent}%)`);
  }
  if (!done) {
    const finalState = await page.evaluate(() => ({
      imgs: [...document.querySelectorAll("img")].map((i) => ({ alt: i.alt, loaded: i.complete && i.naturalWidth > 0 })),
      text: document.body.innerText.slice(0, 400),
    }));
    console.log("3) NOT completed in time:", JSON.stringify(finalState));
  }

  console.log("4) errors:", errors.length ? errors.join("\n   ") : "none");
  await browser.close();
})().catch((e) => { console.error("FAILED:", e.message); process.exit(1); });
