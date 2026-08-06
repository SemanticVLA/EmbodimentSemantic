"use strict";

const demos = {
  libero: {title: "LIBERO Demo", url: "libero/"},
  so101: {title: "SO101 Demo", url: "so101/"},
  "libero-tour": {title: "LIBERO Reviewer Tour", url: "libero/"},
  "so101-tour": {title: "SO101 Reviewer Tour", url: "so101/?tour=reviewer"},
};

const frame = document.getElementById("demoFrame");
const modeBadge = document.getElementById("modeBadge");
const tabs = {
  libero: document.getElementById("liberoTab"),
  so101: document.getElementById("so101Tab"),
};

function selectDemo(name, updateHash = true) {
  if (!(name in demos)) name = "libero";
  const demo = demos[name];
  const selectedTab = name.startsWith("so101") ? "so101" : "libero";
  document.title = demo.title;
  frame.title = demo.title;
  if (frame.getAttribute("src") !== demo.url) {
    frame.contentWindow?.postMessage({type: "demo:stop"}, window.location.origin);
    frame.src = demo.url;
  }
  for (const [key, tab] of Object.entries(tabs)) {
    tab.setAttribute("aria-selected", String(key === selectedTab));
    tab.tabIndex = key === selectedTab ? 0 : -1;
  }
  if (updateHash) history.replaceState(null, "", `#${name}`);
}

for (const [name, tab] of Object.entries(tabs)) {
  tab.addEventListener("click", () => selectDemo(name));
}

window.addEventListener("hashchange", () => selectDemo(location.hash.slice(1), false));
selectDemo(location.hash.slice(1) || "libero", false);

async function loadModeBadge() {
  if (!modeBadge) return;
  try {
    const response = await fetch("api/health", {cache: "no-store"});
    const health = await response.json();
    if (!response.ok || health.error || !health.demo_mode) return;
    modeBadge.textContent = health.demo_mode === "online"
      ? "Online cached showcase | episode_0 | read-only"
      : "Offline localhost tool | writable annotations";
    modeBadge.title = health.demo_dataset_scope || "";
    modeBadge.hidden = false;
  } catch {
    modeBadge.hidden = true;
  }
}

loadModeBadge();
