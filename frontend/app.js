"use strict";

const el = (id) => document.getElementById(id);
const number = new Intl.NumberFormat("ru-RU", { maximumFractionDigits: 2 });
const money = new Intl.NumberFormat("ru-RU", { style: "currency", currency: "RUB", maximumFractionDigits: 0 });
const messages = {
  pending: "В очереди", running: "Идёт расчёт", waiting_for_human: "Нужна проверка",
  completed: "Готово", failed: "Ошибка",
};
const agentLabels = {
  Supervisor: "Координатор", DataAnalyst: "Анализ данных", ModelSelector: "Выбор модели",
  HybridRetriever: "Поиск кейсов", Forecaster: "Расчёт", Validator: "Проверка",
  Critic: "Оценка качества", Iteration: "Итерация", HumanReview: "Решение эксперта",
  ForecastCache: "Готовый расчёт из кэша",
};
const fields = {
  area_total: "Площадь", floor: "Этаж", floors_total: "Этажность", build_year: "Год постройки",
  district: "Район", house_type: "Тип дома", condition: "Состояние",
};
const state = {
  token: sessionStorage.getItem("repf-token"), kind: "tabular", example: null,
  task: null, forecast: null, evaluation: null, taskWatch: 0, evaluationWatch: 0,
};
const forecastChat = { turns: [], request: null };
const modeColors = { baseline: "#656055", "+rag": "#893d2d", "+agents": "#364b3e" };
const modeLabels = { baseline: "Базовый", "+rag": "Подбор по аналогам", "+agents": "Агенты" };

function node(tag, value, style) {
  const item = document.createElement(tag);
  if (value !== undefined) item.textContent = value;
  if (style) item.className = style;
  return item;
}

function appendInlineMarkdown(container, source) {
  const syntax = /(\\[\\\x60*])|(\x60+)([^\n]+?)\2(?!\x60)|\*\*(?=\S)(.+?)\*\*|\*(?=\S)([^*]+?)\*/g;
  let start = 0;
  for (const match of source.matchAll(syntax)) {
    const [token, escaped, , code, bold, italic] = match;
    container.append(source.slice(start, match.index));
    if (escaped) container.append(escaped.slice(1));
    else if (code) container.append(node("code", code));
    else {
      const emphasis = node(bold ? "strong" : "em");
      appendInlineMarkdown(emphasis, bold ?? italic);
      container.append(emphasis);
    }
    start = match.index + token.length;
  }
  container.append(source.slice(start));
}

function renderMarkdown(container, source) {
  container.replaceChildren();
  let block = null;
  let code = null;
  for (const line of source.split(/\r?\n/)) {
    if (code) {
      if (/^\s*\x60{3,}\s*$/.test(line)) code = null;
      else code.append(line, "\n");
      continue;
    }
    if (/^\s*\x60{3,}/.test(line)) {
      code = node("code");
      const pre = node("pre");
      pre.append(code);
      container.append(pre);
      block = null;
    } else if (!line.trim()) {
      block = null;
    } else if (/^\s*#{1,6}\s+/.test(line)) {
      const heading = node("h3");
      appendInlineMarkdown(heading, line.replace(/^\s*#{1,6}\s+/, ""));
      container.append(heading);
      block = null;
    } else {
      const list = line.match(/^\s*([-+*]|\d+\.)\s+(.+)/);
      const tag = list ? (list[1].endsWith(".") ? "ol" : "ul") : "p";
      if (block?.localName !== tag) {
        block = node(tag);
        if (tag === "ol") block.start = Number.parseInt(list[1], 10);
        container.append(block);
      } else if (!list) block.append(node("br"));
      if (list) {
        const item = node("li");
        appendInlineMarkdown(item, list[2]);
        block.append(item);
      } else appendInlineMarkdown(block, line);
    }
  }
}

function status(id, value) {
  el(id).textContent = messages[value] || value;
  el(id).dataset.state = value;
}

async function api(path, options = {}) {
  const headers = new Headers(options.headers);
  if (state.token) headers.set("Authorization", "Bearer " + state.token);
  if (options.body && typeof options.body === "string") headers.set("Content-Type", "application/json");
  const response = await fetch(path, { ...options, headers });
  if (!response.ok) {
    if (response.status === 401 && !path.startsWith("/auth/")) signOut();
    const body = await response.json().catch(() => ({}));
    const detail = Array.isArray(body.detail) ? body.detail.map((issue) => issue.msg).join(". ") : body.detail;
    throw new Error(detail || "Ошибка запроса " + response.status);
  }
  return response;
}

const json = async (path, options) => (await api(path, options)).json();
const post = (path, value) => json(path, { method: "POST", body: JSON.stringify(value) });
const delay = (ms) => new Promise((done) => setTimeout(done, ms));
const forecastPath = (suffix) => "/api/v1/forecast/" + state.task + "/" + suffix;
const formatNumber = (value) => value === null || value === undefined ? "—" : number.format(value);

function signOut() {
  state.token = null;
  state.taskWatch++;
  state.evaluationWatch++;
  sessionStorage.removeItem("repf-token");
  el("account").textContent = "Войти";
  resetForecastChat();
  clearSettings();
}

function requireSession() {
  if (state.token) return true;
  el("auth-dialog").showModal();
  return false;
}

el("account").textContent = state.token ? "Выйти" : "Войти";
el("account").addEventListener("click", () => {
  if (state.token) {
    signOut();
    resetForecast();
    el("history-table").replaceChildren();
    el("evaluation-table").replaceChildren();
    el("evaluation-chart-panel").hidden = true;
    el("comparisons-panel").hidden = true;
    state.evaluation = null;
    status("evaluation-status", "Ожидает запуска");
  } else {
    el("auth-dialog").showModal();
  }
});
el("close-auth").addEventListener("click", () => el("auth-dialog").close());
el("auth-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = event.submitter;
  button.disabled = true;
  el("auth-error").textContent = "";
  try {
    const values = Object.fromEntries(new FormData(event.currentTarget));
    const session = await post("/auth/" + button.value, values);
    state.token = session.access_token;
    sessionStorage.setItem("repf-token", state.token);
    el("account").textContent = "Выйти";
    el("auth-dialog").close();
    event.target.reset();
    if (!el("history").hidden) await history();
    if (!el("settings").hidden) await loadSettings();
  } catch (error) {
    el("auth-error").textContent = error.message;
  } finally {
    button.disabled = false;
  }
});

function page(id) {
  document.querySelectorAll(".page").forEach((section) => { section.hidden = section.id !== id; });
  document.querySelectorAll("nav button").forEach((button) => {
    if (button.dataset.page === id) button.setAttribute("aria-current", "page");
    else button.removeAttribute("aria-current");
  });
  if (id === "history") history();
  if (id === "experiments") drawEvaluation();
  if (id === "forecast") redraw();
  if (id === "settings") loadSettings();
}

document.querySelectorAll("nav button").forEach((button) =>
  button.addEventListener("click", () => page(button.dataset.page)));

document.querySelectorAll("[data-type]").forEach((button) =>
  button.addEventListener("click", () => {
    state.kind = button.dataset.type;
    const property = state.kind === "tabular";
    for (const [id, hidden] of [["property-fields", !property], ["series-fields", property]]) {
      el(id).hidden = hidden;
      el(id).disabled = hidden;
    }
    document.querySelectorAll("[data-type]").forEach((item) =>
      item.setAttribute("aria-pressed", String(item === button)));
    const label = el("horizon-label");
    const horizon = label.querySelector("input");
    label.firstChild.textContent = property ? "Горизонт, месяцев " : "Горизонт, периодов ";
    horizon.max = property ? 12 : 366;
    if (property && Number(horizon.value) > 12) horizon.value = 12;
  }));

el("load-example").addEventListener("click", async () => {
  try {
    state.example = await json("/api/v1/forecast/demo/series");
    const form = el("forecast-form").elements;
    form.series.value = "";
    form.frequency.value = "QS";
    form.horizon.value = 3;
    el("series-label").textContent = "Татарстанстат · вторичный рынок · 16 кварталов, 2021–2024";
    el("forecast-error").textContent = "";
  } catch (error) {
    el("forecast-error").textContent = error.message;
  }
});
el("forecast-form").elements.series.addEventListener("change", (event) => {
  if (event.target.files.length) {
    state.example = null;
    el("series-label").textContent = event.target.files[0].name;
  }
});

async function readMarket(file) {
  if (!file?.name) return {};
  const source = JSON.parse(await file.text());
  const dataset = Array.isArray(source) ? { market: source } : source;
  if (dataset?.market == null && dataset?.market_index) return dataset;
  if (!Array.isArray(dataset?.market)) throw new Error("Укажите массив записей или поле market.");
  return dataset;
}

const marketForm = el("forecast-form").elements;
const demoDistricts = Array.from(marketForm.district.options, (option) => option.value);
marketForm.market.addEventListener("change", async () => {
  const file = marketForm.market.files[0];
  let districts = demoDistricts;
  let label = "Без файла используется синтетическая выборка по Казани.";
  if (file) {
    try {
      const { market } = await readMarket(file);
      label = file.name;
      if (market) {
        districts = [...new Set(market.map((row) => row?.district)
          .filter((district) => typeof district === "string" && district.trim()))].sort((a, b) => a.localeCompare(b, "ru"));
        if (!districts.length) throw new Error("В истории отсутствуют районы или города.");
        label += " · " + formatNumber(market.length) + " записей";
      }
    } catch (error) {
      districts = demoDistricts;
      label = "Ошибка чтения истории. " + error.message;
    }
  }
  if (marketForm.market.files[0] !== file) return;
  const selected = marketForm.district.value;
  marketForm.district.replaceChildren(...districts.map((district) => new Option(district, district)));
  if (districts.includes(selected)) marketForm.district.value = selected;
  el("market-label").textContent = label;
});

function resetForecast(placeholder = "Ожидает запуска") {
  state.taskWatch++;
  state.forecast = null;
  resetForecastChat();
  el("result").hidden = true;
  el("empty").hidden = false;
  el("trace-panel").hidden = true;
  el("trace").replaceChildren();
  el("review-form").hidden = true;
  el("forecast-error").textContent = "";
  status("status", placeholder);
  redraw();
}

async function submitForecast(form) {
  const values = new FormData(form);
  const horizon = Number(values.get("horizon"));
  const mode = values.get("mode");
  if (state.kind === "tabular") {
    const apartment = {};
    for (const field of Object.keys(fields)) {
      const value = values.get(field);
      apartment[field] = ["area_total", "floor", "floors_total", "build_year"].includes(field) ?
        (value ? Number(value) : null) : value || null;
    }
    const dataset = await readMarket(values.get("market"));
    return post("/api/v1/forecast/", { data: { ...dataset, task_type: "tabular", apartment }, horizon, mode });
  }
  const file = values.get("series");
  const frequency = values.get("frequency") || null;
  const seasonal = values.get("seasonal_period");
  if (file.size) {
    const payload = new FormData();
    payload.set("file", file);
    payload.set("horizon", String(horizon));
    payload.set("mode", mode);
    if (frequency) payload.set("frequency", frequency);
    if (seasonal) payload.set("seasonal_period", seasonal);
    return json("/api/v1/forecast/", { method: "POST", body: payload });
  }
  if (!state.example) throw new Error("Выберите CSV или загрузите пример.");
  return post("/api/v1/forecast/", {
    data: { ...state.example, frequency, seasonal_period: seasonal ? Number(seasonal) : null }, horizon, mode,
  });
}

el("forecast-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!requireSession()) return;
  const button = event.submitter;
  button.disabled = true;
  resetForecast("Отправляется");
  try {
    const task = await submitForecast(event.currentTarget);
    watchTask(task.task_id);
  } catch (error) {
    status("status", "failed");
    el("forecast-error").textContent = error.message;
  } finally {
    button.disabled = false;
  }
});

function traceNode(item) {
  const details = node("details");
  const iteration = item.details.input?.iteration;
  const count = item.llm_calls.length ? " · LLM " + item.llm_calls.length : "";
  details.append(node("summary", (agentLabels[item.agent] || item.agent) +
    (iteration ? " " + iteration : "") + " · " + (item.status === "ok" ? "готово" : messages[item.status] || item.status) + count));
  for (const child of item.children) details.append(traceNode(child));
  details.append(node("pre", JSON.stringify({ ...item.details, llm_calls: item.llm_calls }, null, 2)));
  return details;
}

function showTrace(tree) {
  const root = el("trace");
  const openBefore = new Set(Array.from(root.querySelectorAll("details[open]"), (child) => child.dataset.id));
  const enhance = (item, details) => {
    details.dataset.id = item.id;
    details.open = openBefore.has(item.id);
    item.children.forEach((child, index) => enhance(child, details.children[index + 1]));
  };
  const nodes = tree.spans.map((span) => {
    const details = traceNode(span);
    enhance(span, details);
    return details;
  });
  root.replaceChildren(...nodes);
  el("trace-panel").hidden = !nodes.length;
}

function showResult(result, current) {
  state.forecast = result;
  el("result").hidden = false;
  el("forecast-chat").hidden = !result.chat_enabled;
  el("empty").hidden = true;
  const isApartment = result.predicted_price !== undefined;
  el("price-label").textContent = isApartment ? "Цена на конец горизонта" : "Последняя точка прогноза";
  el("price").textContent = isApartment ? money.format(result.predicted_price) : formatNumber(result.predictions.at(-1));
  el("model").textContent = result.model.name;
  el("result-source").textContent = result.data_origin === "synthetic_kazan_demo" ?
    "Синтетические данные · " + money.format(result.price_per_m2) + "/м²" :
    isApartment ? "Загруженная история · " + money.format(result.price_per_m2) + "/м²" :
      result.dates[0] + " — " + result.dates.at(-1);
  const stats = {
    MAE: result.metrics.mae, WAPE: result.metrics.wape === null ? null : formatNumber(result.metrics.wape) + "%",
    MASE: result.metrics.mase, "Оценка Critic": result.confidence,
  };
  el("stats").replaceChildren(...Object.entries(stats).map(([label, value]) => {
    const item = node("div", undefined, "stat");
    item.append(node("span", label, "note"), node("strong", typeof value === "string" ? value : formatNumber(value)));
    return item;
  }));
  const explanation = el("explanation");
  if (result.explanation) renderMarkdown(explanation, result.explanation);
  else explanation.textContent = (result.validation.reasons || []).join(". ");
  el("factors").replaceChildren(...(result.factors || []).map((factor) =>
    node("li", (fields[factor.feature] || factor.feature) + " · " + money.format(factor.impact_rub))));
  el("review-form").hidden = current.status !== "waiting_for_human";
  el("review-form").elements.model.replaceChildren(new Option("Текущая логика выбора", ""),
    ...current.models.map((name) => new Option(name, name)));
  redraw();
}

function resetForecastChat() {
  forecastChat.request?.abort();
  forecastChat.request = null;
  forecastChat.turns = [];
  el("forecast-chat").hidden = true;
  el("chat-form").reset();
  el("chat-form").querySelector("button").disabled = false;
  el("chat-messages").replaceChildren();
  el("chat-status").textContent = "";
  el("chat-error").textContent = "";
  el("chat-error").hidden = true;
}

function addChatMessage(role, text, speaker) {
  const messages = el("chat-messages");
  const line = node("li");
  line.dataset.role = role;
  const message = node("div", undefined, role === "assistant" ? "markdown" : "chat-question");
  if (role === "assistant") renderMarkdown(message, text);
  else message.textContent = text;
  line.append(node("strong", speaker, "note"), message);
  messages.append(line);
  messages.scrollTop = messages.scrollHeight;
  return line;
}

el("chat-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = event.currentTarget;
  const question = form.elements.question.value.trim();
  if (!question || forecastChat.request || !state.forecast?.chat_enabled) return;
  const button = form.querySelector("button");
  const request = new AbortController();
  forecastChat.request = request;
  button.disabled = true;
  el("chat-error").textContent = "";
  el("chat-error").hidden = true;
  el("chat-status").textContent = "Модель отвечает…";
  form.elements.question.value = "";
  const pending = addChatMessage("user", question, "Вы");
  try {
    const result = await json("/api/v1/forecast/" + state.forecast.task_id + "/chat", {
      method: "POST", signal: request.signal,
      body: JSON.stringify({ question, history: forecastChat.turns.slice(-6) }),
    });
    if (forecastChat.request !== request) return;
    forecastChat.turns.push({ question, answer: result.answer });
    addChatMessage("assistant", result.answer, result.model || "Модель");
  } catch (error) {
    if (forecastChat.request !== request) return;
    pending.remove();
    if (!form.elements.question.value) form.elements.question.value = question;
    el("chat-error").textContent = error.message;
    el("chat-error").hidden = false;
  } finally {
    if (forecastChat.request === request) {
      forecastChat.request = null;
      button.disabled = false;
      el("chat-status").textContent = "";
    }
  }
});

async function watchTask(id) {
  state.task = id;
  const watch = ++state.taskWatch;
  try {
    while (watch === state.taskWatch) {
      const [current, trace] = await Promise.all([json(forecastPath("status")), json(forecastPath("reasoning"))]);
      if (watch !== state.taskWatch) return;
      status("status", current.status);
      showTrace(trace);
      if (current.status === "waiting_for_human" || current.status === "completed") {
        const result = await json(forecastPath("result"));
        if (watch === state.taskWatch) showResult(result, current);
        return;
      }
      if (current.status === "failed") {
        el("forecast-error").textContent = current.error || "Расчёт не выполнен";
        return;
      }
      await delay(700);
    }
  } catch (error) {
    if (watch === state.taskWatch) el("forecast-error").textContent = error.message;
  }
}

el("review-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = event.submitter;
  const form = event.currentTarget;
  const values = new FormData(form);
  const action = button.value;
  const payload = { action, comment: values.get("comment") };
  if (action === "retry") {
    if (values.get("model")) payload.model = values.get("model");
    if (values.get("horizon")) payload.horizon = Number(values.get("horizon"));
  }
  button.disabled = true;
  try {
    await post(forecastPath("hitl"), payload);
    form.reset();
    resetForecast("Решение отправлено");
    watchTask(state.task);
  } catch (error) {
    el("forecast-error").textContent = error.message;
  } finally {
    button.disabled = false;
  }
});
el("download").addEventListener("click", async () => {
  try {
    const response = await api(forecastPath("download"));
    const url = URL.createObjectURL(await response.blob());
    const anchor = node("a");
    anchor.href = url;
    anchor.download = "repf-" + state.task + ".csv";
    anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  } catch (error) {
    el("forecast-error").textContent = error.message;
  }
});

function table(container, columns, rows) {
  const result = node("table");
  const head = result.createTHead().insertRow();
  head.append(...columns.map((name) => node("th", name)));
  const body = result.createTBody();
  for (const row of rows) {
    const line = body.insertRow();
    for (const cell of row) {
      if (cell instanceof Node) line.insertCell().append(cell);
      else line.insertCell().textContent = cell;
    }
  }
  container.replaceChildren(result);
}

async function history() {
  const target = el("history-table");
  if (!state.token) {
    target.replaceChildren(node("p", "Войдите в аккаунт, чтобы открыть свои задачи.", "note"));
    return;
  }
  try {
    const jobs = await json("/api/v1/forecast/?review_queue=" + el("review-only").checked);
    if (!jobs.length) {
      target.replaceChildren(node("p", "Задач пока нет.", "note"));
      return;
    }
    table(target, ["Создано", "Тип", "Горизонт", "Статус", ""], jobs.map((job) => {
      const open = node("button", "Открыть", "secondary");
      open.addEventListener("click", () => {
        page("forecast");
        resetForecast("Загружается");
        watchTask(job.task_id);
      });
      return [new Date(job.created_at).toLocaleString("ru-RU"), job.task_type === "tabular" ? "Квартира" : "Ряд",
        job.horizon, messages[job.status], open];
    }));
  } catch (error) {
    target.replaceChildren(node("p", error.message, "error"));
  }
}
el("review-only").addEventListener("change", history);

el("evaluation-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!requireSession()) return;
  const button = event.submitter;
  button.disabled = true;
  const form = new FormData(event.currentTarget);
  const numbers = (field) => String(form.get(field)).split(",").map((value) => Number(value.trim()));
  const payload = {
    dataset: form.get("dataset"), num_series: Number(form.get("num_series")),
    horizons: numbers("horizons"), data_fractions: numbers("data_fractions"), modes: form.getAll("mode"),
  };
  el("evaluation-error").textContent = "";
  try {
    const run = await post("/api/v1/evaluate/batch", payload);
    status("evaluation-status", run.status);
    watchEvaluation(run.evaluation_id);
  } catch (error) {
    el("evaluation-error").textContent = error.message;
  } finally {
    button.disabled = false;
  }
});

async function watchEvaluation(id) {
  const watch = ++state.evaluationWatch;
  try {
    while (watch === state.evaluationWatch) {
      const run = await json("/api/v1/evaluate/compare?evaluation_id=" + id);
      if (watch !== state.evaluationWatch) return;
      status("evaluation-status", run.status);
      if (run.status === "completed") {
        showEvaluation(run);
        return;
      }
      if (run.status === "failed") throw new Error(run.error);
      await delay(1000);
    }
  } catch (error) {
    if (watch === state.evaluationWatch) el("evaluation-error").textContent = error.message;
  }
}

function showEvaluation(result) {
  state.evaluation = result;
  el("evaluation-horizon").replaceChildren(...[...new Set(result.summary.map((row) => row.horizon))]
    .map((value) => new Option(String(value), String(value))));
  const panel = el("evaluation-chart-panel");
  panel.hidden = !result.summary.length;
  if (panel.hidden) {
    el("evaluation-table").replaceChildren(node("p", "Для выбранных параметров недостаточно наблюдений.", "note"));
  } else {
    table(el("evaluation-table"), ["Режим", "Горизонт", "Обучение", "Рядов", "MASE", "WAPE %", "Итерации", "HITL %", "мс"],
      result.summary.map((row) => [modeLabels[row.mode], row.horizon, Math.round(row.fraction * 100) + "%", row.series,
        formatNumber(row.metrics.mase), formatNumber(row.metrics.wape), formatNumber(row.iterations),
        formatNumber(row.hitl_rate * 100), formatNumber(row.latency_ms)]));
  }
  el("comparisons-panel").hidden = !result.comparisons.length;
  table(el("comparisons"), ["Пара", "Горизонт", "Обучение", "Δ MASE", "Bootstrap 95%", "Wilcoxon p"],
    result.comparisons.map((row) => [
      modeLabels[row.left] + " / " + modeLabels[row.right], row.horizon, Math.round(row.fraction * 100) + "%",
      formatNumber(row.mean_mase_delta), row.bootstrap_ci95?.map(formatNumber).join(" … ") || "—",
      formatNumber(row.wilcoxon_pvalue),
    ]));
  drawEvaluation();
}

function plot(canvas, series, labels = [], band = null) {
  const w = canvas.clientWidth;
  const h = Number(canvas.dataset.plotHeight ||= canvas.getAttribute("height"));
  if (!w) return;
  const ratio = Math.min(2, window.devicePixelRatio || 1);
  canvas.style.height = h + "px";
  canvas.width = w * ratio;
  canvas.height = h * ratio;
  const ctx = canvas.getContext("2d");
  ctx.scale(ratio, ratio);
  const all = series.flatMap((line) => line.values).filter(Number.isFinite);
  if (band) all.push(...band.lower, ...band.upper);
  const floor = Math.min(...all), top = Math.max(...all);
  const range = Math.max(top - floor, Math.abs(top) * .06, .001);
  const min = floor - range * .12, max = top + range * .12;
  const n = series[0].values.length;
  const xMin = 60, xMax = w - 16, yMin = 10, yMax = h - 32;
  const x = (i) => xMin + (n === 1 ? .5 : i / (n - 1)) * (xMax - xMin);
  const y = (v) => yMax - (v - min) / (max - min) * (yMax - yMin);
  ctx.font = "11px system-ui";
  ctx.fillStyle = "#635f55";
  ctx.strokeStyle = "#dad3c5";
  for (let i = 0; i <= 4; i++) {
    const p = yMin + i * (yMax - yMin) / 4;
    ctx.beginPath();
    ctx.moveTo(xMin, p);
    ctx.lineTo(xMax, p);
    ctx.stroke();
    ctx.fillText(formatNumber(max - i * (max - min) / 4), 0, p + 3, 56);
  }
  if (band) {
    ctx.fillStyle = "#eadbd0";
    ctx.beginPath();
    if (n === 1) {
      ctx.fillRect(x(0) - 8, y(band.upper[0]), 16, y(band.lower[0]) - y(band.upper[0]));
    } else {
      band.upper.forEach((value, i) => i ? ctx.lineTo(x(i), y(value)) : ctx.moveTo(x(i), y(value)));
      for (let i = n - 1; i >= 0; i--) ctx.lineTo(x(i), y(band.lower[i]));
      ctx.closePath();
      ctx.fill();
    }
  }
  for (const line of series) {
    ctx.strokeStyle = line.color;
    ctx.fillStyle = line.color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    let started = false;
    line.values.forEach((value, i) => {
      if (!Number.isFinite(value)) return;
      if (started) ctx.lineTo(x(i), y(value));
      else ctx.moveTo(x(i), y(value));
      started = true;
    });
    ctx.stroke();
    if (n < 18) line.values.forEach((value, i) => {
      if (!Number.isFinite(value)) return;
      ctx.beginPath();
      ctx.arc(x(i), y(value), 3, 0, 2 * Math.PI);
      ctx.fill();
    });
  }
  ctx.fillStyle = "#635f55";
  const step = Math.max(1, Math.ceil(labels.length / (w < 420 ? 3 : 6)));
  labels.forEach((label, i) => {
    if (labels.length > 1 && i % step === 0 && i < n - 1) ctx.fillText(label, Math.max(0, x(i) - 24), h - 9);
  });
  if (labels.length === 1) ctx.fillText(labels[0], Math.max(0, x(0) - 26), h - 9);
  if (labels.length > 1) ctx.fillText(labels.at(-1), w - ctx.measureText(labels.at(-1)).width, h - 9);
}

function redraw() {
  if (state.forecast) {
    plot(el("forecast-chart"), [{ values: state.forecast.predictions, color: "#893d2d" }],
      state.forecast.dates, state.forecast.interval);
  }
}

function drawEvaluation() {
  if (!state.evaluation?.summary.length) return;
  const h = Number(el("evaluation-horizon").value);
  const rows = state.evaluation.summary.filter((row) => row.horizon === h);
  const fractions = [...new Set(rows.map((row) => row.fraction))].sort((a, b) => a - b);
  const series = [...new Set(rows.map((row) => row.mode))].map((mode) => ({
    color: modeColors[mode], values: fractions.map((fraction) =>
      rows.find((row) => row.mode === mode && row.fraction === fraction)?.metrics.mase ?? null),
  }));
  if (series.length && series.some((line) => line.values.some(Number.isFinite))) {
    plot(el("evaluation-chart"), series, fractions.map((f) => Math.round(100 * f) + "%"));
  }
}
el("evaluation-horizon").addEventListener("change", drawEvaluation);
window.addEventListener("resize", () => { redraw(); drawEvaluation(); });
redraw();

const llmForm = el("llm-form");
const providerLabels = {
  deepseek: "DeepSeek", ollama: "Ollama", compatible: "Совместимый API", local: "Локальные правила",
};
let llmState = null;
let settingsRequest = 0;
let settingsBusy = false;

function settingsMessage(value = "", kind = "") {
  el("settings-message").textContent = value;
  el("settings-message").dataset.kind = kind;
}

function clearSettings() {
  settingsRequest++;
  llmState = null;
  llmForm.reset();
  llmForm.hidden = true;
  el("settings-gate").hidden = Boolean(state.token);
  el("reset-model").hidden = true;
  el("saved-models").hidden = true;
  el("model-checkpoints").replaceChildren();
  el("current-model").textContent = "Настройки сервера";
  el("current-model-detail").textContent = "";
  settingsMessage();
}

function selectedLLMProvider(provider, endpoint) {
  return provider === "compatible" && endpoint === "deepseek" ? "deepseek" : provider;
}

function updateSettingsFields() {
  if (!llmState) return;
  const { provider, model, endpoint, api_key: key } = llmForm.elements;
  const selected = selectedLLMProvider(provider.value, endpoint.value);
  const compatible = selected === "compatible";
  const cloud = provider.value === "compatible";
  el("endpoint-field").hidden = !cloud;
  endpoint.disabled = !cloud || settingsBusy;
  el("model-field").hidden = selected === "local";
  model.disabled = selected === "local" || settingsBusy;
  model.required = selected !== "local";
  model.placeholder = compatible ? "Например, provider/model" : llmState.models[selected] || "";
  el("key-field").hidden = !cloud;
  key.disabled = !cloud || settingsBusy;
  key.required = selected === "deepseek";
  el("key-note").textContent = cloud ? "Проверка отправляет один пробный запрос провайдеру." : "";
  el("settings-note").textContent = selected === "ollama" ?
    "Используется адрес Ollama, настроенный на сервере." :
    compatible ? "Укажите имя модели в каталоге выбранного провайдера." : "";
  provider.disabled = settingsBusy;
  llmForm.querySelector('button[type="submit"]').disabled = settingsBusy;
  el("reset-model").disabled = settingsBusy;
  llmForm.elements.namedItem("name").disabled = settingsBusy;
  for (const button of el("model-checkpoints").querySelectorAll("button")) {
    button.disabled = settingsBusy || button.dataset.current === "true";
  }
}

function showSettings(payload) {
  llmState = payload;
  const { profile, default: defaults, endpoints, checkpoints, active_checkpoint_id: activeId } = payload;
  const { provider, model, endpoint, api_key: key } = llmForm.elements;
  provider.value = "compatible";
  endpoint.replaceChildren(new Option("DeepSeek", "deepseek"),
    ...endpoints.map((item) => new Option(item.label, item.url)));
  llmForm.elements.namedItem("name").value = "";
  model.value = "";
  key.value = "";

  const active = checkpoints.find((checkpoint) => checkpoint.id === activeId);
  el("current-model").textContent = active?.name ||
    (defaults ? providerLabels[defaults.provider] : "Локальные правила");
  const selected = profile || defaults;
  el("current-model-detail").textContent = selected?.model || payload.models[selected?.provider] || "";
  el("reset-model").hidden = !activeId;
  el("saved-models").hidden = false;
  el("empty-models").hidden = checkpoints.length !== 0;
  el("model-checkpoints").replaceChildren(...checkpoints.map(checkpointRow));
  llmForm.hidden = false;
  el("settings-gate").hidden = true;
  updateSettingsFields();
}

function checkpointRow(checkpoint) {
  const selected = checkpoint.id === llmState.active_checkpoint_id;
  const row = node("li");
  row.append(node("strong", checkpoint.name),
    node("p", [providerLabels[checkpoint.provider], checkpoint.model].filter(Boolean).join(" · "), "note"));
  const actions = node("div", undefined, "actions");
  const use = node("button", selected ? "Активен" : "Использовать", "secondary");
  use.type = "button";
  use.dataset.current = String(selected);
  use.addEventListener("click", () => switchCheckpoint(checkpoint.id));
  actions.append(use);
  if (checkpoint.provider !== "local") {
    const probe = node("button", "Проверить", "secondary");
    probe.type = "button";
    probe.addEventListener("click", () => probeCheckpoint(checkpoint));
    actions.append(probe);
  }
  const remove = node("button", "Удалить", "text-button");
  remove.type = "button";
  remove.addEventListener("click", () => deleteCheckpoint(checkpoint));
  actions.append(remove);
  row.append(actions);
  return row;
}

async function loadSettings() {
  if (!state.token) {
    clearSettings();
    return;
  }
  const request = ++settingsRequest;
  el("settings-gate").hidden = true;
  try {
    const response = await json("/api/v1/llm/settings");
    if (request === settingsRequest) showSettings(response);
  } catch (error) {
    if (request === settingsRequest) {
      el("current-model").textContent = "Настройки недоступны";
      el("current-model-detail").textContent = error.message;
    }
  }
}

el("settings-login").addEventListener("click", requireSession);
llmForm.addEventListener("input", (event) => {
  if (event.target.name === "provider" || event.target.name === "endpoint") {
    llmForm.elements.model.value = "";
    llmForm.elements.api_key.value = "";
  }
  settingsMessage();
  updateSettingsFields();
});

async function settingsAction(callback) {
  if (settingsBusy || !state.token) return;
  const token = state.token;
  settingsBusy = true;
  settingsMessage();
  updateSettingsFields();
  try {
    await callback(() => token === state.token);
  } catch (error) {
    if (token === state.token) settingsMessage(error.message, "error");
  } finally {
    settingsBusy = false;
    updateSettingsFields();
  }
}

llmForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const form = new FormData(llmForm);
  const provider = selectedLLMProvider(form.get("provider"), form.get("endpoint"));
  const payload = { name: form.get("name"), provider, model: form.get("model") || null };
  if (provider === "compatible") payload.endpoint = form.get("endpoint");
  if (form.get("api_key")) payload.api_key = form.get("api_key");
  settingsAction(async (active) => {
    settingsMessage(provider === "local" ? "Сохраняю пресет…" : "Проверяю модель…");
    await post("/api/v1/llm/checkpoints", payload);
    const response = await json("/api/v1/llm/settings");
    if (active()) {
      showSettings(response);
      settingsMessage("Пресет сохранён. Нажмите «Использовать», чтобы переключиться на него.", "ok");
    }
  });
});

function switchCheckpoint(id) {
  settingsAction(async (active) => {
    settingsMessage(id ? "Проверяю и переключаю модель…" : "Возвращаю настройки сервера…");
    const response = await json("/api/v1/llm/selection", {
      method: "PUT", body: JSON.stringify({checkpoint_id: id}),
    });
    if (active()) {
      showSettings(response);
      settingsMessage(id ? "Модель переключена." : "Восстановлены настройки сервера.", "ok");
    }
  });
}

function probeCheckpoint(checkpoint) {
  settingsAction(async (active) => {
    settingsMessage("Проверяю доступность модели…");
    await json("/api/v1/llm/checkpoints/" + encodeURIComponent(checkpoint.id) + "/probe", {method: "POST"});
    if (active()) settingsMessage("Модель из пресета «" + checkpoint.name + "» доступна. Вызов инструмента прошёл.", "ok");
  });
}

function deleteCheckpoint(checkpoint) {
  if (!confirm("Удалить пресет «" + checkpoint.name + "»?")) return;
  settingsAction(async (active) => {
    await api("/api/v1/llm/checkpoints/" + encodeURIComponent(checkpoint.id), {method: "DELETE"});
    const response = await json("/api/v1/llm/settings");
    if (active()) {
      showSettings(response);
      settingsMessage("Пресет удалён.", "ok");
    }
  });
}

el("reset-model").addEventListener("click", () => switchCheckpoint(null));
