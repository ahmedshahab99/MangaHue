// Verify completed job renders in the Library + viewer (no waiting for new jobs).
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

  // Quota chip sanity
  const quota = await page.evaluate(() => {
    const badge = [...document.querySelectorAll("span,div")].find((el) => /pages today/.test(el.textContent || ""));
    return badge ? badge.textContent.trim() : null;
  });
  console.log("2) quota chip:", quota);

  // Library view
  await page.evaluate(() => {
    const lib = [...document.querySelectorAll("button")].find((b) => b.textContent.includes("Library"));
    if (lib) lib.click();
  });
  await new Promise((r) => setTimeout(r, 2000));
  const libRows = await page.evaluate(() => {
    const text = document.body.innerText;
    return { hasRows: /5F8EFF96|C527F925|LIBRARY/i.test(text), text: text.slice(0, 200) };
  });
  console.log("3) library:", JSON.stringify(libRows));

  // Open the completed demo job (5F8EFF96)
  await page.evaluate(() => {
    const rows = [...document.querySelectorAll("button")];
    const row = rows.find((b) => b.textContent.includes("5F8EFF96"));
    if (row) row.click();
  });
  console.log("4) opened completed job, waiting for images...");
  let result = null;
  for (let i = 0; i < 12; i++) {
    await new Promise((r) => setTimeout(r, 3000));
    result = await page.evaluate(() => {
      const imgs = [...document.querySelectorAll("img")].map((img) => ({
        alt: img.alt,
        loaded: img.complete && img.naturalWidth > 0,
        w: img.naturalWidth,
      }));
      return {
        colored: imgs.find((i) => i.alt === "Colorized manga page") || null,
        original: imgs.find((i) => i.alt === "Original manga page") || null,
      };
    });
    if (result.colored?.loaded && result.colored.w > 100) break;
  }
  console.log("5) viewer:", JSON.stringify(result));

  console.log("6) errors:", errors.length ? errors.join("\n   ") : "none");
  await browser.close();
})().catch((e) => { console.error("FAILED:", e.message); process.exit(1); });
