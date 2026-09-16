const $ = (id) => document.getElementById(id);
const escapeHTML = (text) =>
  String(text ?? "").replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
let busy = false,
  result = null,
  selectedProtein = null,
  isReady = false;
async function api(path, body) {
  const r = await fetch("/api/" + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const d = await r.json();
  if (!r.ok)
    throw new Error(
      typeof d.detail === "string"
        ? d.detail
        : "Request failed; check the server log.",
    );
  return d;
}
function setBusy(value) {
  busy = value;
  $("columns").setAttribute("aria-busy", String(value));
  $("columns").classList.toggle("searching", value);
  $("submit").disabled = value || !isReady;
  $("dice").disabled = value || !isReady || !hasExamples[$("mode").value];
  $("mode").disabled = value;
}
let hasExamples = {};
async function search(text = $("query").value, mode = $("mode").value) {
  if (busy) throw new Error("Wait for the current search to finish");
  if (!text.trim()) throw new Error("Enter a function or reaction description");
  const expected = result?.query === text ? result.expected?.target.id : null;
  $("query").value = text;
  $("mode").value = mode;
  updateModeNote();
  setBusy(true);
  $("error").hidden = true;
  try {
    const d = await api("search", { text, mode, expected });
    render(d);
    return d;
  } catch (e) {
    $("error").textContent = e.message;
    $("error").hidden = false;
    throw e;
  } finally {
    setBusy(false);
  }
}
function render(d) {
  result = d;
  $("columns").classList.toggle("protein-first", d.mode === "protein");
  $("query").value = d.query;
  $("result-label").textContent = "Matches for “" + d.query + "”";
  $("timing").textContent =
    d.seconds.toFixed(2) +
    "s · " +
    ({catalogue: "catalogue-assisted", protein: "Swiss-Prot adapter", adapter: "reaction adapter"}[d.mode]);
  $("reactions").innerHTML = d.reactions
    .map(
      (r, i) =>
        `<article class="reaction-card"><div class="card-top"><span class="rank">${i + 1}</span><a href="${escapeHTML(r.url)}" target="_blank" rel="noreferrer">${escapeHTML(r.id.replace("Rh_", "RHEA "))} ↗</a><span class="score">${r.score.toFixed(3)}</span></div><p class="equation">${escapeHTML(r.equation)}</p><div class="card-footer"><span>${escapeHTML(r.ec || "Reaction library")}</span><span>cosine similarity</span></div></article>`,
    )
    .join("");
  $("proteins").innerHTML = d.proteins
    .map(
      (p, i) =>
        `<article class="protein-card"><div class="card-top"><span class="rank">${i + 1}</span><a href="${escapeHTML(p.url)}" target="_blank" rel="noreferrer">${escapeHTML(p.id)} ↗</a><span class="score">${p.score.toFixed(3)}</span></div><div class="protein-body"><div class="protein-thumbnail" data-protein="${escapeHTML(p.id)}"><span>Loading structure…</span></div><div><h3>${escapeHTML(p.name || p.id)}</h3><p class="organism">${escapeHTML(p.organism)}</p><button class="compare-link" data-compare="${escapeHTML(p.id)}">Compare descriptions →</button></div></div><div class="card-footer"><span>${escapeHTML(p.ec || "Protein library")}</span><span>cosine similarity</span></div></article>`,
    )
    .join("");
  if (!d.proteins.length)
    $("proteins").textContent =
      "No protein library loaded. Reaction search is available.";
  $("answer").hidden = !d.expected;
  if (d.expected) {
    const e = d.expected;
    const protein = e.kind === "protein";
    $("answer").innerHTML =
      `<div class="test-answer-heading"><span>Diagnostic example</span><strong>Target rank #${e.rank.toLocaleString()} <small>/ ${e.groups.toLocaleString()} ${protein ? "proteins" : "groups"}</small></strong></div><p>${escapeHTML(protein ? e.target.name : e.target.equation)}</p><a href="${escapeHTML(e.target.url)}" target="_blank" rel="noreferrer">${escapeHTML(e.target.id)}</a><div class="test-answer-note">Exact-ID rank: ${e.exact_rank.toLocaleString()}. ${protein ? "Swiss-Prot function text, held out from adapter training. Similar proteins may share this function." : "Synthetic description; demo examples are separate from formal evaluation."}</div>`;
  }
  document
    .querySelectorAll("[data-compare]")
    .forEach((b) => (b.onclick = () => openComparison(b.dataset.compare)));
  document.querySelectorAll("[data-protein]").forEach((el) => {
    renderQueue = renderQueue
      .then(() => thumbnail(el))
      .catch(() => {
        el.textContent = "No structure";
      });
  });
}
$("search").onsubmit = (e) => {
  e.preventDefault();
  search().catch(() => {});
};
$("query").onkeydown = (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    search().catch(() => {});
  }
};
function updateModeNote() {
  $("method-note").textContent =
    $("mode").value === "catalogue"
      ? "Text finds named equations; their Horizyn vectors retrieve proteins."
      : $("mode").value === "protein"
        ? "Learned from Swiss-Prot protein descriptions; searches proteins and reactions."
        : "Learned from reaction descriptions; searches reactions and proteins.";
}
$("mode").onchange = () => {
  search().catch(() => {});
};
$("dice").onclick = async () => {
  if (busy) return;
  setBusy(true);
  $("error").hidden = true;
  try {
    render(await api("example", { mode: $("mode").value }));
  } catch (e) {
    $("error").textContent = e.message;
    $("error").hidden = false;
  } finally {
    setBusy(false);
  }
};
document
  .querySelectorAll("[data-query]")
  .forEach((b) => (b.onclick = () => search(b.dataset.query).catch(() => {})));
function addDescription(text = "") {
  const label = document.createElement("label");
  label.textContent =
    "Description " + ($("compare-inputs").children.length + 1);
  const t = document.createElement("textarea");
  t.required = true;
  t.maxLength = 3000;
  t.value = text;
  label.appendChild(t);
  $("compare-inputs").appendChild(label);
  $("add-description").disabled = $("compare-inputs").children.length >= 8;
}
function openComparison(id) {
  selectedProtein = id;
  $("compare-name").textContent = "Compare descriptions · " + id;
  $("compare-inputs").replaceChildren();
  $("compare-results").replaceChildren();
  addDescription(result.query);
  addDescription();
  $("comparison").showModal();
}
$("close-comparison").onclick = () => $("comparison").close();
$("add-description").onclick = () => addDescription();
$("compare-form").onsubmit = async (e) => {
  e.preventDefault();
  const button = e.submitter;
  button.disabled = true;
  try {
    const d = await api("compare", {
      protein_id: selectedProtein,
      mode: $("mode").value,
      descriptions: [...$("compare-inputs").querySelectorAll("textarea")].map(
        (t) => t.value,
      ),
    });
    $("compare-results").innerHTML = d
      .map(
        (r) =>
          `<article><strong>${r.score.toFixed(3)}</strong><p>${escapeHTML(r.text)}</p></article>`,
      )
      .join("");
  } catch (e) {
    $("compare-results").textContent = e.message;
  } finally {
    button.disabled = false;
  }
};
let viewer = null,
  renderQueue = Promise.resolve();
const thumbnails = new Map();
async function thumbnail(el) {
  if (!el.isConnected) return;
  const id = el.dataset.protein;
  if (!window.$3Dmol) {
    el.textContent = "Structure viewer unavailable";
    return;
  }
  if (!thumbnails.has(id)) {
    const r = await fetch("/api/structure/" + encodeURIComponent(id));
    if (!r.ok) throw new Error("No structure");
    const pdb = await r.text();
    if (!viewer) {
      const box = document.createElement("div");
      box.style.cssText =
        "position:fixed;left:-1000px;top:0;width:250px;height:216px";
      document.body.appendChild(box);
      viewer = $3Dmol.createViewer(box, {
        backgroundColor: "#f6faf9",
        antialias: true,
      });
    }
    viewer.clear();
    viewer.addModel(pdb, "pdb");
    viewer.setStyle({}, { cartoon: { color: "spectrum" } });
    viewer.zoomTo();
    viewer.rotate(25, "y");
    viewer.rotate(15, "x");
    viewer.render();
    thumbnails.set(id, viewer.pngURI());
  }
  const img = document.createElement("img");
  img.src = thumbnails.get(id);
  img.alt = "AlphaFold predicted structure of " + id;
  el.replaceChildren(img);
}
async function health() {
  try {
    const d = await (await fetch("/api/health")).json();
    isReady = d.ready;
    hasExamples = {adapter: d.examples, catalogue: d.examples, protein: d.protein_examples};
    $("status").textContent = d.ready
      ? `${d.reactions.toLocaleString()} reactions · ${d.proteins.toLocaleString()} proteins`
      : d.error || d.stage;
    $("mode").querySelector("[value=catalogue]").disabled = !d.catalogue;
    $("mode").querySelector("[value=protein]").disabled = !d.protein_adapter;
    setBusy(busy);
    if (d.ready) {
      clearInterval(poll);
      search().catch(() => {});
    }
  } catch (e) {
    $("status").textContent = "Server unavailable";
  }
}
const poll = setInterval(health, 3000);
health();
// Optional browser-agent interface; ordinary browsers use the same visible controls.
if (document.modelContext) {
  document.modelContext.registerTool({
    name: "search_biochemistry",
    description:
      "Search reaction and protein libraries and update the visible results.",
    inputSchema: {
      type: "object",
      properties: {
        text: { type: "string", minLength: 1, maxLength: 3000 },
        mode: { type: "string", enum: ["adapter", "catalogue", "protein"] },
      },
      required: ["text"],
      additionalProperties: false,
    },
    execute: async ({ text, mode = "adapter" }) => {
      if (typeof text !== "string" || !text.trim() || text.length > 3000)
        throw new Error("Enter 1–3000 characters");
      if (!["adapter", "catalogue", "protein"].includes(mode))
        throw new Error("Unknown mode");
      return search(text, mode);
    },
  });
}
