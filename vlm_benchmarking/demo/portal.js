"use strict";

const demos = {
  libero: {title: "LIBERO Demo", url: "libero/"},
  so101: {title: "SO101 Demo", url: "so101/"},
};

const frame = document.getElementById("demoFrame");
const tabs = {
  libero: document.getElementById("liberoTab"),
  so101: document.getElementById("so101Tab"),
};

function selectDemo(name, updateHash = true) {
  if (!(name in demos)) name = "libero";
  const demo = demos[name];
  document.title = demo.title;
  frame.title = demo.title;
  if (frame.getAttribute("src") !== demo.url) {
    frame.contentWindow?.postMessage({type: "demo:stop"}, window.location.origin);
    frame.src = demo.url;
  }
  for (const [key, tab] of Object.entries(tabs)) {
    tab.setAttribute("aria-selected", String(key === name));
    tab.tabIndex = key === name ? 0 : -1;
  }
  if (updateHash) history.replaceState(null, "", `#${name}`);
}

for (const [name, tab] of Object.entries(tabs)) {
  tab.addEventListener("click", () => selectDemo(name));
}

window.addEventListener("hashchange", () => selectDemo(location.hash.slice(1), false));
selectDemo(location.hash.slice(1) || "libero", false);
