/* Apuli — Local-First Obsidian Companion Frontend */

const state = {
  status: null,
  conversations: [],
  activeConvId: null,
  messages: [],
  mode: "companion",
  streaming: false,
  memories: [],
  profile: null,
};

const $ = (id) => document.getElementById(id);

function showToast(msg, ms = 3200) {
  const el = $("toast");
  el.textContent = msg;
  el.classList.add("show");
  setTimeout(() => el.classList.remove("show"), ms);
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function renderMarkdown(md) {
  if (!md) return "";
  let text = escapeHtml(md);

  text = text.replace(/```([\s\S]*?)```/g, (_, code) => `<pre><code>${code.trim()}</code></pre>`);
  text = text.replace(/`([^`]+)`/g, "<code>$1</code>");
  text = text.replace(/\[\[([^\]]+)\]\]/g, '<span class="wikilink-chip">[[$1]]</span>');
  text = text.replace(/^### (.+)$/gm, "<h3>$1</h3>");
  text = text.replace(/^## (.+)$/gm, "<h2>$1</h2>");
  text = text.replace(/^# (.+)$/gm, "<h1>$1</h1>");
  text = text.replace(/^&gt; (.+)$/gm, "<blockquote>$1</blockquote>");
  text = text.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  text = text.replace(/\*([^*]+)\*/g, "<em>$1</em>");

  const lines = text.split("\n");
  const out = [];
  let inList = false;
  for (const line of lines) {
    const m = line.match(/^\s*[-*]\s+(.+)$/);
    if (m) {
      if (!inList) {
        out.push("<ul>");
        inList = true;
      }
      out.push(`<li>${m[1]}</li>`);
    } else {
      if (inList) {
        out.push("</ul>");
        inList = false;
      }
      out.push(line);
    }
  }
  if (inList) out.push("</ul>");

  return out
    .join("\n")
    .split(/\n{2,}/)
    .map((block) => {
      const t = block.trim();
      if (!t) return "";
      if (/^<(h[1-3]|ul|pre|blockquote)/.test(t)) return t;
      return `<p>${t.replace(/\n/g, "<br />")}</p>`;
    })
    .join("\n");
}

async function api(url, options = {}) {
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(detail || `HTTP ${res.status}`);
  }
  return res.json();
}

/* --- Status, Onboarding, Hardware Advisor & Settings --- */

function updateModelPrivacyHint() {
  const sel = $("modelSelect");
  const hint = $("composerHint");
  if (!sel || !hint || state.streaming) return;
  const val = sel.value || "";
  if (val.startsWith("ollama::") || val.includes("127.0.0.1") || !val.includes("::")) {
    hint.textContent = "🔒 Running locally on your hardware — nothing leaves your PC.";
  } else {
    hint.textContent = "☁️ Using an external cloud API for this model (vault indexing stays local).";
  }
}

async function loadStatus() {
  const data = await api("/api/status");
  state.status = data;

  const cfg = data.config;
  $("brandTitle").textContent = cfg.companion_name || "Apuli";
  $("welcomeHeading").textContent = `Hey ${cfg.user_name && cfg.user_name !== "Friend" ? cfg.user_name : "there"}, I’m here.`;

  // Onboarding card visibility
  if (data.needs_onboarding) {
    $("onboardingCard").classList.remove("hidden");
    $("welcomeCard").classList.add("hidden");
    $("onboardName").value = cfg.user_name && cfg.user_name !== "Friend" ? cfg.user_name : "";
    const rec = data.model_recommendations;
    if (rec) {
      $("onboardHardwareSummary").innerHTML = `<strong>Detected Hardware:</strong> ${escapeHtml(rec.summary)}`;
    }
    const candidates = data.candidate_vaults || [];
    $("candidateVaultsBox").innerHTML = candidates.length
      ? candidates
          .map(
            (p) =>
              `<button type="button" class="mini-btn candidate-vault-btn" data-path="${escapeHtml(p)}">Use ${escapeHtml(p)}</button>`
          )
          .join("")
      : "";
    $("candidateVaultsBox").querySelectorAll(".candidate-vault-btn").forEach((b) => {
      b.onclick = () => {
        $("onboardVault").value = b.dataset.path;
      };
    });
  } else {
    $("onboardingCard").classList.add("hidden");
  }

  // Ollama pill
  const oPill = $("ollamaPill");
  const oText = $("ollamaPillText");
  oPill.classList.remove("ok", "warn", "err");
  if (data.ollama.online) {
    oPill.classList.add("ok");
    oText.textContent = `Ollama online · ${data.ollama.chat_models.length} local models`;
  } else {
    oPill.classList.add("err");
    oText.textContent = "Ollama offline (start Ollama)";
  }

  // Index pill
  const iPill = $("indexPill");
  const iText = $("indexPillText");
  const idx = data.index_status;
  iPill.classList.remove("ok", "warn", "err");
  if (data.needs_onboarding) {
    iPill.classList.add("warn");
    iText.textContent = "Connect your vault to begin";
  } else if (idx.state === "error") {
    iPill.classList.add("err");
    iText.textContent = "Vault index error";
  } else if (idx.state !== "idle") {
    iPill.classList.add("warn");
    iText.textContent = idx.message || `Indexing (${idx.processed_notes}/${idx.total_notes})…`;
  } else {
    iPill.classList.add("ok");
    const modeTag = idx.embeddings_available ? "semantic+keyword" : "keyword";
    iText.textContent = `${idx.indexed_notes} notes · ${modeTag}`;
  }

  // Memory badge
  const memCnt = data.memory_count ?? 0;
  const chatCnt = data.chat_chunk_count ?? 0;
  $("memoryBadge").textContent = `🔒 Local-first · ${memCnt} memories · ${chatCnt} chat turns`;

  // Unified Provider Model Selector
  const pModels = data.provider_models || [];
  const prevModel = $("modelSelect").value;
  const defaultSpec = `${cfg.provider || "ollama"}::${
    cfg.provider === "openai"
      ? cfg.openai_model
      : cfg.provider === "anthropic"
      ? cfg.anthropic_model
      : cfg.provider === "gemini"
      ? cfg.gemini_model
      : cfg.provider === "openai_compatible"
      ? cfg.custom_model
      : cfg.chat_model
  }`;

  const localModels = pModels.filter((m) => m.is_local);
  const cloudModels = pModels.filter((m) => !m.is_local);

  let modelHtml = "";
  if (localModels.length) {
    modelHtml += `<optgroup label="Local-First (Private)">${localModels
      .map((m) => `<option value="${escapeHtml(m.id)}">${escapeHtml(m.label)}</option>`)
      .join("")}</optgroup>`;
  }
  if (cloudModels.length) {
    modelHtml += `<optgroup label="Frontier Cloud APIs">${cloudModels
      .map((m) => `<option value="${escapeHtml(m.id)}">${escapeHtml(m.label)}</option>`)
      .join("")}</optgroup>`;
  }
  $("modelSelect").innerHTML = modelHtml || `<option value="ollama::${escapeHtml(cfg.chat_model)}">Local · ${escapeHtml(cfg.chat_model)}</option>`;

  if (prevModel && $("modelSelect").querySelector(`option[value="${CSS.escape(prevModel)}"]`)) {
    $("modelSelect").value = prevModel;
  } else if ($("modelSelect").querySelector(`option[value="${CSS.escape(defaultSpec)}"]`)) {
    $("modelSelect").value = defaultSpec;
  }
  updateModelPrivacyHint();

  // Default Local Chat Model select in settings
  const ollamaChatModels = data.ollama.chat_models.length ? data.ollama.chat_models : [cfg.chat_model];
  $("setChatModel").innerHTML = ollamaChatModels
    .map((m) => `<option value="${escapeHtml(m)}">${escapeHtml(m)}</option>`)
    .join("");
  $("setChatModel").value = cfg.chat_model;

  // Background Memory Model select (optional split between unfiltered voice & structured memory)
  const memChoices = [
    `<option value="">Same as active chat model</option>`,
    ...pModels.map((m) => `<option value="${escapeHtml(m.id)}">${escapeHtml(m.label)}</option>`),
  ];
  $("setMemoryModel").innerHTML = memChoices.join("");
  $("setMemoryModel").value = cfg.memory_model || "";

  // Folder scope selector
  const scopeSelect = $("folderScopeSelect");
  const prevScope = scopeSelect.value || "all";
  const folderOptions = [
    `<option value="all">All Indexed Folders</option>`,
    ...data.indexed_folders.map(
      (f) => `<option value="${escapeHtml(f.folder)}">${escapeHtml(f.folder)} (${f.count})</option>`
    ),
  ];
  scopeSelect.innerHTML = folderOptions.join("");
  if (scopeSelect.querySelector(`option[value="${CSS.escape(prevScope)}"]`)) {
    scopeSelect.value = prevScope;
  }

  renderHardwareAdvisor(data);
  renderSettingsPanel(data);
}

function renderHardwareAdvisor(data) {
  const rec = data.model_recommendations;
  if (!rec) return;
  const hw = rec.hardware || {};
  $("hwSpecsLine").textContent = `${hw.os || ""} (${hw.arch || ""}) · ${hw.cpu_cores || "?"} CPU threads · ${hw.ram_gb || "?"} GB RAM · ${hw.gpu_name || "CPU"}${hw.vram_gb ? ` (${hw.vram_gb} GB VRAM)` : ""}`;
  $("hwSummaryBox").innerHTML = `<strong>Recommendation (${escapeHtml(rec.tier)} tier):</strong> ${escapeHtml(rec.summary)}`;

  const cards = (rec.chat_recommendations || []).map((c) => {
    const statusBadge = c.installed
      ? `<span class="hw-badge ok">Installed ✓</span>`
      : `<button type="button" class="mini-btn pull-model-btn" data-model="${escapeHtml(c.name)}">Pull (${escapeHtml(c.size)})</button>`;
    const recBadge = c.recommended ? `<span class="hw-badge">Best Match</span>` : "";
    return `
      <div class="hw-rec-card ${c.recommended ? "recommended" : ""}">
        <div>
          <div class="hw-rec-top">
            <strong>${escapeHtml(c.name)}</strong>
            ${recBadge}
          </div>
          <div class="muted small" style="margin-top:3px;">${escapeHtml(c.role)} · ${escapeHtml(c.size)}</div>
          <div class="muted small" style="margin-top:6px;">${escapeHtml(c.reason)}</div>
        </div>
        <div style="display:flex; justify-content:space-between; align-items:center; gap:6px;">
          ${statusBadge}
          ${
            c.installed
              ? `<button type="button" class="mini-btn use-model-btn" data-model="ollama::${escapeHtml(c.name)}">Use</button>`
              : ""
          }
        </div>
      </div>
    `;
  });

  $("hwRecGrid").innerHTML = cards.join("");
  $("hwRecGrid").querySelectorAll(".pull-model-btn").forEach((btn) => {
    btn.onclick = () => pullOllamaModel(btn.dataset.model, $("pullAnyProgress"), btn);
  });
  $("hwRecGrid").querySelectorAll(".use-model-btn").forEach((btn) => {
    btn.onclick = () => {
      if ($("modelSelect").querySelector(`option[value="${CSS.escape(btn.dataset.model)}"]`)) {
        $("modelSelect").value = btn.dataset.model;
        updateModelPrivacyHint();
        showToast(`Switched chat model to ${btn.dataset.model.replace("ollama::", "")}`);
      }
    };
  });
}

function renderSettingsPanel(data) {
  const cfg = data.config;
  const idx = data.index_status;

  $("setUserName").value = cfg.user_name || "";
  $("setCompanionName").value = cfg.companion_name || "";
  $("setVaultPath").value = cfg.vault_path || "";
  $("setOllamaUrl").value = cfg.ollama_url || "";
  $("setEmbedModel").value = cfg.embedding_model || "";
  $("setTemp").value = cfg.temperature ?? 0.65;
  $("setTopK").value = cfg.top_k_chunks ?? 8;
  $("setGraphExpand").checked = !!cfg.enable_graph_expansion;
  $("setAutoLearn").checked = cfg.auto_learn !== false;
  $("setCrossChat").checked = cfg.enable_cross_chat_recall !== false;
  $("setAutoTurns").value = cfg.auto_distill_turns ?? 2;
  $("setProvider").value = cfg.provider || "ollama";

  $("setOpenaiKey").value = cfg.openai_api_key || "";
  $("setOpenaiModel").value = cfg.openai_model || "gpt-4o-mini";
  $("setAnthropicKey").value = cfg.anthropic_api_key || "";
  $("setAnthropicModel").value = cfg.anthropic_model || "claude-sonnet-4-20250514";
  $("setGeminiKey").value = cfg.gemini_api_key || "";
  $("setGeminiModel").value = cfg.gemini_model || "gemini-2.5-flash";
  $("setCustomUrl").value = cfg.custom_base_url || "";
  $("setCustomKey").value = cfg.custom_api_key || "";
  $("setCustomModel").value = cfg.custom_model || "";

  $("indexStatsBox").innerHTML = `
    <div><strong>Indexed Notes:</strong> ${idx.indexed_notes} · <strong>Chunks:</strong> ${idx.total_chunks} · <strong>Embedded:</strong> ${idx.embedded_chunks}</div>
    <div class="muted small" style="margin-top:4px;">${escapeHtml(idx.message || "")}</div>
  `;

  const hasEmbed = (data.ollama.embed_models || []).some((m) =>
    m.toLowerCase().includes(cfg.embedding_model.split(":")[0].toLowerCase())
  );
  const embedBox = $("embedBox");
  if (!data.ollama.online) {
    embedBox.innerHTML = `Ollama is offline. Start Ollama so local embeddings and chat can run.`;
  } else if (hasEmbed && idx.embedded_chunks === idx.total_chunks && idx.total_chunks > 0) {
    embedBox.innerHTML = `✅ Semantic memory active using <code>${escapeHtml(cfg.embedding_model)}</code> (${idx.embedded_chunks}/${idx.total_chunks} chunks embedded).`;
  } else if (hasEmbed) {
    embedBox.innerHTML = `Model <code>${escapeHtml(cfg.embedding_model)}</code> is installed (${idx.embedded_chunks}/${idx.total_chunks} chunks embedded). Click <strong>Rebuild Embeddings</strong> if needed.`;
  } else {
    embedBox.innerHTML = `
      <div>Semantic model <code>${escapeHtml(cfg.embedding_model)}</code> is not pulled in Ollama yet (keyword search is already working).</div>
      <button type="button" class="btn-primary" id="pullEmbedBtn" style="margin-top:8px;">
        Pull ${escapeHtml(cfg.embedding_model)} &amp; Embed Notes
      </button>
      <div id="pullEmbedProgress" class="muted small" style="margin-top:6px;"></div>
    `;
    $("pullEmbedBtn").onclick = () =>
      pullOllamaModel(cfg.embedding_model, $("pullEmbedProgress"), $("pullEmbedBtn"));
  }

  const excludedLower = new Set((cfg.excluded_folders || []).map((f) => f.toLowerCase()));
  const folders = data.vault_folders || [];
  $("folderChecks").innerHTML = folders.length
    ? folders
        .map((f) => {
          const checked = !excludedLower.has(f.name.toLowerCase());
          return `
            <div class="folder-check-item">
              <label>
                <input type="checkbox" class="folder-toggle" data-folder="${escapeHtml(f.name)}" ${checked ? "checked" : ""} />
                <span>${escapeHtml(f.name)}</span>
              </label>
              <span class="muted small">${f.note_count} notes</span>
            </div>
          `;
        })
        .join("")
    : `<div class="muted small">Connect a vault path first to configure folder privacy.</div>`;
}

async function pullOllamaModel(modelName, progressEl, btn) {
  if (btn) btn.disabled = true;
  if (progressEl) progressEl.textContent = `Pulling ${modelName} from Ollama…`;

  try {
    const res = await fetch("/api/ollama/pull", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: modelName }),
    });
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const events = buffer.split("\n\n");
      buffer = events.pop();

      for (const ev of events) {
        const lines = ev.split("\n");
        let type = "message";
        let payload = "";
        for (const l of lines) {
          if (l.startsWith("event:")) type = l.slice(6).trim();
          if (l.startsWith("data:")) payload += l.slice(5).trim();
        }
        if (!payload) continue;
        const parsed = JSON.parse(payload);
        if (type === "progress" && progressEl) {
          const pct = parsed.total ? ` (${Math.round((parsed.completed / parsed.total) * 100)}%)` : "";
          progressEl.textContent = `${modelName}: ${parsed.status || "Downloading"}${pct}`;
        } else if (type === "embedding" && progressEl) {
          progressEl.textContent = parsed.status;
        } else if (type === "done") {
          if (progressEl) progressEl.textContent = `${modelName} is ready!`;
          showToast(`${modelName} installed locally!`);
          await loadStatus();
        } else if (type === "error" && progressEl) {
          progressEl.textContent = `Error: ${parsed.error}`;
        }
      }
    }
  } catch (err) {
    if (progressEl) progressEl.textContent = `Failed: ${err.message}`;
  } finally {
    if (btn) btn.disabled = false;
  }
}

/* --- Conversations & Chat --- */

async function loadConversations() {
  const data = await api("/api/conversations");
  state.conversations = data.conversations || [];
  renderConvList();
}

function renderConvList() {
  const list = $("convList");
  if (!state.conversations.length) {
    list.innerHTML = `<div class="muted small" style="padding:8px;">No chats yet. Start one above.</div>`;
    return;
  }
  list.innerHTML = state.conversations
    .map(
      (c) => `
      <div class="conv-item ${c.id === state.activeConvId ? "active" : ""}" data-id="${c.id}">
        <span class="conv-item-title">${escapeHtml(c.title)}</span>
        <button class="conv-del" data-del="${c.id}" title="Delete conversation">✕</button>
      </div>
    `
    )
    .join("");

  list.querySelectorAll(".conv-item").forEach((el) => {
    el.onclick = (e) => {
      if (e.target.dataset.del) return;
      openConversation(el.dataset.id);
      switchView("chat");
    };
  });

  list.querySelectorAll(".conv-del").forEach((btn) => {
    btn.onclick = async (e) => {
      e.stopPropagation();
      const id = btn.dataset.del;
      await api(`/api/conversations/${id}`, { method: "DELETE" });
      if (state.activeConvId === id) {
        state.activeConvId = null;
        state.messages = [];
        renderMessages();
      }
      await loadConversations();
    };
  });
}

async function createConversation() {
  const model = $("modelSelect").value || undefined;
  const folder_scope = $("folderScopeSelect").value || "all";
  const data = await api("/api/conversations", {
    method: "POST",
    body: JSON.stringify({ mode: state.mode, model, folder_scope }),
  });
  state.activeConvId = data.conversation.id;
  state.messages = [];
  await loadConversations();
  renderMessages();
  switchView("chat");
  $("chatInput").focus();
}

async function openConversation(convId) {
  const data = await api(`/api/conversations/${convId}`);
  state.activeConvId = convId;
  state.messages = data.messages || [];
  setMode(data.conversation.mode || "companion");
  if (data.conversation.model && $("modelSelect").querySelector(`option[value="${CSS.escape(data.conversation.model)}"]`)) {
    $("modelSelect").value = data.conversation.model;
  }
  if (data.conversation.folder_scope && $("folderScopeSelect").querySelector(`option[value="${CSS.escape(data.conversation.folder_scope)}"]`)) {
    $("folderScopeSelect").value = data.conversation.folder_scope;
  }
  updateModelPrivacyHint();
  renderConvList();
  renderMessages();
}

function setMode(mode) {
  state.mode = mode;
  document.querySelectorAll(".mode-pill").forEach((b) => {
    b.classList.toggle("active", b.dataset.mode === mode);
  });
}

function renderSourcesHtml(sources) {
  if (!sources || !sources.length) return "";
  const cards = sources
    .map((s) => {
      if (s.source_type === "chat") {
        return `
          <div class="source-card">
            <div class="source-title-row">
              <span class="source-title" title="Past conversation">${escapeHtml(s.title)}</span>
              <button type="button" class="obsidian-link jump-chat-btn" data-conv="${escapeHtml(s.conversation_id)}" style="background:none;border:none;cursor:pointer;padding:0;">Jump ↗</button>
            </div>
            <div class="source-meta">💬 Past Chat${s.note_date ? " · " + escapeHtml(s.note_date) : ""} · ${escapeHtml(s.match_reason)}</div>
            <div class="source-excerpt">${escapeHtml(s.excerpt)}</div>
          </div>
        `;
      }
      return `
        <div class="source-card">
          <div class="source-title-row">
            <span class="source-title" title="${escapeHtml(s.note_path)}">${escapeHtml(s.title)}</span>
            <a class="obsidian-link" href="${escapeHtml(s.obsidian_uri)}" title="Open in Obsidian">Open ↗</a>
          </div>
          <div class="source-meta">📄 ${escapeHtml(s.folder)}${s.note_date ? " · " + escapeHtml(s.note_date) : ""} · ${escapeHtml(s.match_reason)}</div>
          <div class="source-excerpt">${escapeHtml(s.excerpt)}</div>
        </div>
      `;
    })
    .join("");
  return `
    <details class="sources-drawer">
      <summary>📚 ${sources.length} memory source${sources.length > 1 ? "s" : ""} referenced</summary>
      <div class="sources-list">${cards}</div>
    </details>
  `;
}

function renderMessages() {
  const container = $("messagesContainer");
  const welcome = $("welcomeCard");
  const needsOnboarding = state.status?.needs_onboarding;

  if (!state.messages.length) {
    if (!needsOnboarding) welcome.classList.remove("hidden");
    container.innerHTML = "";
    return;
  }
  welcome.classList.add("hidden");

  const companionName = state.status?.config?.companion_name || "Apuli";
  container.innerHTML = state.messages
    .map((m, idx) => {
      if (m.role === "user") {
        return `<div class="msg msg-user"><div class="msg-bubble">${escapeHtml(m.content)}</div></div>`;
      }
      const thinkBlock = m.thinking
        ? `<details class="think-box"><summary>Thought process</summary><div class="think-content">${escapeHtml(m.thinking)}</div></details>`
        : "";
      const bodyHtml = m.content ? renderMarkdown(m.content) : `<span class="muted">Thinking…</span>`;
      return `
        <div class="msg msg-assistant" data-msg-idx="${idx}">
          <div class="msg-assistant-head">◈ ${escapeHtml(companionName)}</div>
          <div class="msg-bubble">
            ${thinkBlock}
            <div class="markdown-body">${bodyHtml}</div>
          </div>
          ${renderSourcesHtml(m.sources)}
          ${
            m.content && !state.streaming
              ? `<div class="msg-actions">
                  <button class="mini-btn save-vault-btn" data-idx="${idx}">Save Reflection to Vault</button>
                 </div>`
              : ""
          }
        </div>
      `;
    })
    .join("");

  container.querySelectorAll(".save-vault-btn").forEach((btn) => {
    btn.onclick = () => exportMessageToVault(Number(btn.dataset.idx));
  });

  container.querySelectorAll(".jump-chat-btn").forEach((btn) => {
    btn.onclick = () => openConversation(btn.dataset.conv);
  });

  const scroller = $("chatScroll");
  scroller.scrollTop = scroller.scrollHeight;
}

async function exportMessageToVault(idx) {
  const msg = state.messages[idx];
  if (!msg) return;
  const title = prompt("Title for this Obsidian note in Personal/:", "Reflection with Apuli");
  if (!title) return;
  try {
    const res = await api("/api/vault/export-reflection", {
      method: "POST",
      body: JSON.stringify({ title, content: msg.content, subfolder: "Personal" }),
    });
    showToast(`Saved to Obsidian: ${res.saved_path}`);
  } catch (err) {
    showToast(`Save failed: ${err.message}`);
  }
}

async function sendMessage(text) {
  const message = text.trim();
  if (!message || state.streaming) return;

  if (!state.activeConvId) {
    await createConversation();
  }

  state.streaming = true;
  $("sendBtn").disabled = true;
  $("composerHint").textContent = "Searching your vault & thinking…";

  state.messages.push({ role: "user", content: message });
  const assistantMsg = { role: "assistant", content: "", thinking: "", sources: [] };
  state.messages.push(assistantMsg);
  renderMessages();

  try {
    const res = await fetch(`/api/conversations/${state.activeConvId}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message,
        mode: state.mode,
        model: $("modelSelect").value,
        folder_scope: $("folderScopeSelect").value,
      }),
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const blocks = buffer.split("\n\n");
      buffer = blocks.pop();

      for (const block of blocks) {
        const lines = block.split("\n");
        let evType = "message";
        let dataStr = "";
        for (const l of lines) {
          if (l.startsWith("event:")) evType = l.slice(6).trim();
          if (l.startsWith("data:")) dataStr += l.slice(5).trim();
        }
        if (!dataStr) continue;
        const payload = JSON.parse(dataStr);

        if (evType === "sources") {
          assistantMsg.sources = payload.sources || [];
          renderMessages();
        } else if (evType === "thinking") {
          assistantMsg.thinking += payload.text || "";
          renderMessages();
        } else if (evType === "token") {
          assistantMsg.content += payload.text || "";
          renderMessages();
        } else if (evType === "error") {
          assistantMsg.content += `\n\n> ⚠️ ${payload.error}`;
          renderMessages();
        }
      }
    }
  } catch (err) {
    assistantMsg.content = `⚠️ Error talking to backend: ${err.message}`;
  } finally {
    state.streaming = false;
    $("sendBtn").disabled = false;
    updateModelPrivacyHint();
    renderMessages();
    await loadConversations();
    setTimeout(loadStatus, 2000);
  }
}

/* --- Mirror & Memories --- */

async function loadMirrorAndMemories() {
  const [profRes, memRes] = await Promise.all([
    api("/api/self-profile"),
    api("/api/memories"),
  ]);
  state.profile = profRes.profile;
  state.memories = memRes.memories || [];

  if (state.profile && state.profile.summary_md) {
    $("profileDisplay").innerHTML = renderMarkdown(state.profile.summary_md);
    $("profileUpdatedAt").textContent = `Last updated: ${new Date(state.profile.updated_at).toLocaleString()}`;
  }

  renderMemories();
}

function renderMemories() {
  const list = $("memoriesList");
  if (!state.memories.length) {
    list.innerHTML = `<p class="muted small">No saved memories yet. Use "✨ Remember Chat" during a conversation or add one above.</p>`;
    return;
  }
  list.innerHTML = state.memories
    .map(
      (m) => `
      <div class="memory-card">
        <div>
          <span class="memory-badge">${escapeHtml(m.category)}</span>
          <span>${escapeHtml(m.content)}</span>
        </div>
        <div class="panel-actions">
          <button class="mini-btn pin-mem" data-id="${m.id}" data-pinned="${m.pinned ? 1 : 0}">
            ${m.pinned ? "★ Pinned" : "☆ Pin"}
          </button>
          <button class="mini-btn del-mem" data-id="${m.id}">Delete</button>
        </div>
      </div>
    `
    )
    .join("");

  list.querySelectorAll(".pin-mem").forEach((btn) => {
    btn.onclick = async () => {
      const id = btn.dataset.id;
      const pinned = btn.dataset.pinned !== "1";
      const res = await api(`/api/memories/${id}`, {
        method: "PATCH",
        body: JSON.stringify({ pinned }),
      });
      state.memories = res.memories;
      renderMemories();
    };
  });

  list.querySelectorAll(".del-mem").forEach((btn) => {
    btn.onclick = async () => {
      await api(`/api/memories/${btn.dataset.id}`, { method: "DELETE" });
      state.memories = state.memories.filter((m) => String(m.id) !== btn.dataset.id);
      renderMemories();
    };
  });
}

/* --- Navigation & Events --- */

function switchView(name) {
  document.querySelectorAll(".nav-tab").forEach((t) => {
    t.classList.toggle("active", t.dataset.view === name);
  });
  document.querySelectorAll(".view").forEach((v) => {
    v.classList.toggle("active", v.id === `view-${name}`);
  });
  if (name === "mirror") loadMirrorAndMemories();
  if (name === "settings") loadStatus();
}

function bindEvents() {
  document.querySelectorAll(".nav-tab").forEach((tab) => {
    tab.onclick = () => switchView(tab.dataset.view);
  });

  $("newChatBtn").onclick = createConversation;
  $("modelSelect").onchange = updateModelPrivacyHint;

  document.querySelectorAll(".mode-pill").forEach((btn) => {
    btn.onclick = () => setMode(btn.dataset.mode);
  });

  document.querySelectorAll(".starter-btn").forEach((btn) => {
    btn.onclick = () => sendMessage(btn.dataset.prompt);
  });

  $("onboardingForm").onsubmit = async (e) => {
    e.preventDefault();
    const user_name = $("onboardName").value.trim() || "Friend";
    const vault_path = $("onboardVault").value.trim();
    if (!vault_path) return;
    await api("/api/settings", {
      method: "POST",
      body: JSON.stringify({ user_name, vault_path }),
    });
    showToast("Vault connected! Scanning your notes now…");
    await loadStatus();
    renderMessages();
  };

  const input = $("chatInput");
  input.addEventListener("input", () => {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 160) + "px";
  });

  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      $("chatForm").requestSubmit();
    }
  });

  $("chatForm").onsubmit = (e) => {
    e.preventDefault();
    const val = input.value;
    if (!val.trim() || state.streaming) return;
    input.value = "";
    input.style.height = "auto";
    sendMessage(val);
  };

  $("distillBtn").onclick = async () => {
    if (!state.activeConvId) return showToast("Start a conversation first.");
    $("distillBtn").disabled = true;
    showToast("Distilling lasting memories from this chat…");
    try {
      const res = await api(`/api/conversations/${state.activeConvId}/distill`, {
        method: "POST",
      });
      const count = (res.saved || []).length;
      showToast(
        count ? `Saved ${count} new memor${count > 1 ? "ies" : "y"} to your Memory Bank.` : "No new enduring memories detected."
      );
    } catch (err) {
      showToast(`Memory extraction failed: ${err.message}`);
    } finally {
      $("distillBtn").disabled = false;
    }
  };

  $("synthProfileBtn").onclick = async () => {
    $("synthProfileBtn").disabled = true;
    $("synthProfileBtn").textContent = "Synthesizing…";
    try {
      const res = await api("/api/self-profile/synthesize", { method: "POST" });
      state.profile = res.profile;
      $("profileDisplay").innerHTML = renderMarkdown(res.profile.summary_md);
      $("profileUpdatedAt").textContent = `Last updated: ${new Date(res.profile.updated_at).toLocaleString()}`;
      showToast("Self-Portrait updated from your vault.");
    } catch (err) {
      showToast(`Failed: ${err.message}`);
    } finally {
      $("synthProfileBtn").disabled = false;
      $("synthProfileBtn").textContent = "Generate from Vault";
    }
  };

  $("editProfileBtn").onclick = () => {
    $("profileEditor").value = state.profile?.summary_md || "";
    $("profileDisplay").classList.add("hidden");
    $("profileEditorWrap").classList.remove("hidden");
  };

  $("cancelProfileBtn").onclick = () => {
    $("profileEditorWrap").classList.add("hidden");
    $("profileDisplay").classList.remove("hidden");
  };

  $("saveProfileBtn").onclick = async () => {
    const summary_md = $("profileEditor").value;
    const res = await api("/api/self-profile", {
      method: "PUT",
      body: JSON.stringify({ summary_md }),
    });
    state.profile = res.profile;
    $("profileDisplay").innerHTML = renderMarkdown(summary_md);
    $("profileEditorWrap").classList.add("hidden");
    $("profileDisplay").classList.remove("hidden");
    showToast("Self-Portrait saved.");
  };

  $("addMemoryForm").onsubmit = async (e) => {
    e.preventDefault();
    const content = $("memoryInput").value.trim();
    if (!content) return;
    await api("/api/memories", {
      method: "POST",
      body: JSON.stringify({ category: $("memoryCategory").value, content, pinned: true }),
    });
    $("memoryInput").value = "";
    await loadMirrorAndMemories();
    showToast("Memory saved.");
  };

  $("reindexBtn").onclick = async () => {
    await api("/api/index?force_reembed=false", { method: "POST" });
    showToast("Incremental vault scan started.");
    setTimeout(loadStatus, 1000);
  };

  $("reembedBtn").onclick = async () => {
    await api("/api/index?force_reembed=true", { method: "POST" });
    showToast("Full re-index & embedding started.");
    setTimeout(loadStatus, 1000);
  };

  $("settingsForm").onsubmit = async (e) => {
    e.preventDefault();
    const excluded = [];
    document.querySelectorAll(".folder-toggle").forEach((cb) => {
      if (!cb.checked) excluded.push(cb.dataset.folder);
    });
    await api("/api/settings", {
      method: "POST",
      body: JSON.stringify({
        user_name: $("setUserName").value.trim(),
        companion_name: $("setCompanionName").value.trim(),
        vault_path: $("setVaultPath").value.trim(),
        ollama_url: $("setOllamaUrl").value.trim(),
        provider: $("setProvider").value,
        chat_model: $("setChatModel").value,
        memory_model: $("setMemoryModel").value,
        embedding_model: $("setEmbedModel").value.trim(),
        temperature: parseFloat($("setTemp").value),
        top_k_chunks: parseInt($("setTopK").value, 10),
        enable_graph_expansion: $("setGraphExpand").checked,
        auto_learn: $("setAutoLearn").checked,
        enable_cross_chat_recall: $("setCrossChat").checked,
        auto_distill_turns: parseInt($("setAutoTurns").value, 10),
        openai_api_key: $("setOpenaiKey").value.trim(),
        openai_model: $("setOpenaiModel").value.trim(),
        anthropic_api_key: $("setAnthropicKey").value.trim(),
        anthropic_model: $("setAnthropicModel").value.trim(),
        gemini_api_key: $("setGeminiKey").value.trim(),
        gemini_model: $("setGeminiModel").value.trim(),
        custom_base_url: $("setCustomUrl").value.trim(),
        custom_api_key: $("setCustomKey").value.trim(),
        custom_model: $("setCustomModel").value.trim(),
        excluded_folders: excluded,
      }),
    });
    $("settingsSavedMsg").textContent = "Saved!";
    showToast("Settings saved.");
    await loadStatus();
  };
}

async function init() {
  bindEvents();
  await Promise.all([loadStatus(), loadConversations()]);
  setInterval(() => {
    if (state.status && state.status.index_status.state !== "idle") {
      loadStatus();
    }
  }, 2500);
}

init();
