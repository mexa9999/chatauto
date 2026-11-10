import { lazy, Suspense, useCallback, useEffect, useRef, useState } from "react";
import type { FormEvent, KeyboardEvent as ReactKeyboardEvent, ReactNode } from "react";
import { FeedbackProvider, useFeedback } from "./Feedback";
import { useStoredState } from "./useStoredState";
import { api } from "./api";

const AnalyticsPage = lazy(() => import("./AnalyticsPage"));

type Page = "overview" | "launch" | "analytics" | "history" | "data" | "live";
type Run = {
  run_id: string;
  kind: string;
  state: string;
  created_at: string;
  request: Record<string, unknown>;
};
type Dataset = {
  dataset_id: string;
  name: string;
  mode: "light" | "full";
  created_at: string;
  active: boolean;
  run: Record<string, unknown>;
};
type Branch = {
  name: string;
  head_commit: string | null;
  generation: number | null;
  archived_at?: string | null;
};
type Genome = {
  genome_id: string;
  source_kind: string;
  member_kind: string;
  member_index: number | null;
  generation: number | null;
  branch_name: string | null;
  metadata: Record<string, unknown>;
};
type TrashItem = {
  entity_type: "run" | "dataset" | "branch" | "genome";
  id?: string;
  name?: string;
  genome_id?: string;
  deleted_at: string;
};
type ImportPreview = {
  token: string;
  name: string;
  manifest: {
    mode: "light" | "full";
    run: { kind?: string; run_id?: string };
  };
  conflicts: Record<string, { name: string; mode: string }>;
};
type JobStatus = {
  job_id: string | null;
  kind?: string;
  state: string;
  progress?: {
    phase?: string;
    generation?: number;
    generations?: number;
    completed_generations?: number;
    best_fitness?: number;
    eta_seconds?: number;
  } | null;
  result?: Record<string, unknown> | null;
  log_url?: string;
};
type DemoCheckpoint = {
  ref: string;
  branch: string;
  generation: number;
  best_fitness: number | null;
  complete: boolean;
};
type LiveFrame = {
  seq: number;
  kind: string;
  round?: number;
  height?: number;
  width?: number;
  type_grid?: number[][];
  quality_grid?: number[][];
  floors_grid?: number[][];
  roof_quality_grid?: number[][];
  owner_grid?: number[][];
  agent_id?: number | null;
  action?: { cell: number | null; type_id: number; quality: number; floors?: number; roof_quality?: number; demolish: boolean } | null;
  accepted?: boolean | null;
};

type ArchiveUpload<T> = {
  promise: Promise<T>;
  abort: () => void;
};

function uploadArchive<T>(
  path: string,
  file: File,
  onProgress: (progress: number | null) => void,
): ArchiveUpload<T> {
  const request = new XMLHttpRequest();
  const promise = new Promise<T>((resolve, reject) => {
    request.open("POST", path);
    request.setRequestHeader("Content-Type", "application/zip");
    request.upload.onprogress = (event) => {
      onProgress(event.lengthComputable ? Math.round((event.loaded / event.total) * 100) : null);
    };
    request.onerror = () => reject(new Error("Не удалось отправить архив: проверьте соединение и повторите попытку."));
    request.onabort = () => {
      const error = new Error("Загрузка архива отменена.");
      error.name = "AbortError";
      reject(error);
    };
    request.onload = () => {
      let payload: unknown;
      try {
        payload = JSON.parse(request.responseText);
      } catch {
        payload = request.responseText;
      }
      if (request.status < 200 || request.status >= 300) {
        const detail = typeof payload === "object" && payload !== null && "detail" in payload
          ? (payload as { detail: unknown }).detail
          : payload;
        reject(new Error(typeof detail === "string" ? detail : `HTTP ${request.status}: ${request.statusText}`));
        return;
      }
      resolve(payload as T);
    };
    request.send(file);
  });
  return {
    promise,
    abort: () => {
      if (request.readyState !== XMLHttpRequest.DONE) request.abort();
    },
  };
}

function isIntegerBetween(value: unknown, min: number, max: number): value is number {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= min && value <= max;
}

function gridValidationError(value: string) {
  const normalized = value.trim().toLowerCase().replace(/[х×]/g, "x").replace(/\s/g, "");
  const parts = normalized.split("x");
  if (parts.length !== 2 || parts.some((part) => !/^\d+$/.test(part))) {
    return "Введите размеры как высота×ширина, например 12x12.";
  }
  const [height, width] = parts.map(Number);
  if (!Number.isSafeInteger(height) || !Number.isSafeInteger(width) || height < 3 || width < 3) {
    return "Каждая сторона сетки должна быть целым числом не меньше 3.";
  }
  return "";
}

function App() {
  return (
    <FeedbackProvider>
      <DashboardApp />
    </FeedbackProvider>
  );
}

function DashboardApp() {
  const { notify } = useFeedback();
  const [page, setPage] = useState<Page>("overview");
  const [runs, setRuns] = useState<Run[]>([]);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [branches, setBranches] = useState<Branch[]>([]);
  const [activeBranch, setActiveBranch] = useState("main");
  const [job, setJob] = useState<JobStatus>({ job_id: null, state: "idle" });
  const [error, setError] = useState("");
  const [connectionError, setConnectionError] = useState("");
  const [connected, setConnected] = useState(true);
  const [stopPending, setStopPending] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const refreshInFlight = useRef(false);

  const refresh = useCallback(async () => {
    if (refreshInFlight.current) return;
    refreshInFlight.current = true;
    try {
      const [catalog, branchData, status] = await Promise.all([
        api<{ runs: Run[]; datasets: Dataset[] }>("/api/data"),
        api<{ active: string; branches: Branch[] }>("/api/branches"),
        api<JobStatus>("/api/jobs"),
      ]);
      setRuns(catalog.runs);
      setDatasets(catalog.datasets);
      setBranches(branchData.branches);
      setActiveBranch(branchData.active);
      setJob(status);
      setConnected(true);
      setConnectionError("");
    } catch (reason) {
      setConnected(false);
      setConnectionError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      refreshInFlight.current = false;
    }
  }, []);

  useEffect(() => {
    const poll = () => {
      if (document.visibilityState === "visible") void refresh();
    };
    poll();
    const timer = window.setInterval(poll, 2500);
    document.addEventListener("visibilitychange", poll);
    return () => {
      window.clearInterval(timer);
      document.removeEventListener("visibilitychange", poll);
    };
  }, [refresh]);

  const start = async (request: Record<string, unknown>) => {
    try {
      const result = await api<{ job_id: string }>("/api/jobs", {
        method: "POST",
        body: JSON.stringify(request),
      });
      setJob({
        job_id: result.job_id,
        kind: typeof request.kind === "string" ? request.kind : "train",
        state: "running",
      });
      setError("");
      notify({ kind: "success", message: "Задача запущена. Статус и управление доступны на любой вкладке." });
      setPage(request.kind === "demo" ? "live" : "overview");
      await refresh();
      return result.job_id;
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
      return null;
    }
  };

  const stop = async () => {
    if (!job.job_id || stopPending) return;
    setStopPending(true);
    try {
      await api(`/api/jobs/${job.job_id}/stop`, { method: "POST" });
      setJob((current) => ({ ...current, state: "stopping" }));
      setError("");
      notify({ kind: "info", message: "Остановка запрошена. Обучение завершит поколение и сохранит checkpoint." });
      await refresh();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setStopPending(false);
    }
  };
  const refreshNow = async () => {
    setRefreshing(true);
    try {
      await refresh();
    } finally {
      setRefreshing(false);
    }
  };

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a className="brand" href="#" onClick={() => setPage("overview")}>
          <span className="brand-mark">C</span>
          <span>
            <strong>CitySim</strong>
            <small>Experiment lab</small>
          </span>
        </a>
        <div className="side-label">РАБОЧЕЕ ПРОСТРАНСТВО</div>
        <nav className="nav-list" aria-label="Основная навигация">
          <NavButton active={page === "overview"} onClick={() => setPage("overview")} icon="overview">
            Обзор
          </NavButton>
          <NavButton active={page === "launch"} onClick={() => setPage("launch")} icon="launch">
            Запуски
          </NavButton>
          <NavButton active={page === "analytics"} onClick={() => setPage("analytics")} icon="analytics">
            Аналитика
          </NavButton>
          <NavButton active={page === "history"} onClick={() => setPage("history")} icon="history">
            История
          </NavButton>
          <NavButton active={page === "data"} onClick={() => setPage("data")} icon="data">
            Данные
          </NavButton>
        </nav>
        <div className="sidebar-bottom">
          <span className={`connection-dot ${!connected ? "offline" : job.state === "running" ? "busy" : ""}`} />
          <span>{!connected ? "Нет соединения" : job.state === "running" ? "Выполняется задача" : "Подключено"}</span>
          <small>127.0.0.1</small>
        </div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <div className="breadcrumb">CitySim <span>/</span> {pageTitle(page)}</div>
          <div className="topbar-actions">
            <span className="branch-pill">Ветка <b>{activeBranch}</b></span>
            <button
              className="icon-button"
              title="Обновить данные"
              aria-label="Обновить данные"
              disabled={refreshing}
              onClick={() => void refreshNow()}
            >
              {refreshing ? "…" : "↻"}
            </button>
          </div>
        </header>
        {!connected && (
          <div className="alert connection-alert" role="alert">
            <span>Не удалось обновить данные: {connectionError}</span>
            <button className="button small-button secondary" disabled={refreshing} onClick={() => void refreshNow()}>
              {refreshing ? "Повтор…" : "Повторить"}
            </button>
          </div>
        )}
        {error && (
          <div className="alert" role="alert">
            <span>{error}</span>
            <button className="alert-close" onClick={() => setError("")} aria-label="Закрыть">
              ×
            </button>
          </div>
        )}
        <TaskCenter
          job={job}
          currentPage={page}
          stopPending={stopPending}
          onNavigate={setPage}
          onStop={() => void stop()}
          onError={setError}
        />
        {page === "overview" && (
          <Overview
            runs={runs}
            datasets={datasets}
            job={job}
            onNavigate={setPage}
          />
        )}
        {page === "launch" && (
          <LaunchPage
            branches={branches}
            activeBranch={activeBranch}
            jobActive={job.state === "running" || job.state === "stopping"}
            onStart={start}
            onError={setError}
          />
        )}
        {page === "analytics" && (
          <Suspense fallback={<div className="analytics-loading panel"><span className="analytics-loader" />Загружаем инструменты аналитики…</div>}>
            <AnalyticsPage
              running={job.state === "running" || job.state === "stopping"}
              progress={job.progress}
              onNavigate={setPage}
            />
          </Suspense>
        )}
        {page === "history" && (
          <HistoryPage branches={branches} job={job} onRefresh={refresh} />
        )}
        {page === "data" && (
          <DataPage
            runs={runs}
            datasets={datasets}
            onRefresh={refresh}
            onError={setError}
          />
        )}
        {page === "live" && (
          <LiveDemoPage job={job} onError={setError} />
        )}
      </main>
    </div>
  );
}

function pageTitle(page: Page) {
  return {
    overview: "Обзор",
    launch: "Запуски",
    analytics: "Аналитика",
    history: "История",
    data: "Наборы данных",
    live: "Живая симуляция",
  }[page];
}

function NavButton({
  active,
  icon,
  children,
  onClick,
}: {
  active: boolean;
  icon: "overview" | "launch" | "analytics" | "history" | "data";
  children: React.ReactNode;
  onClick: () => void;
}) {
  return (
    <button
      className={`nav-item ${active ? "active" : ""}`}
      aria-current={active ? "page" : undefined}
      onClick={onClick}
    >
      <span className="nav-icon" aria-hidden="true">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
          {icon === "overview" && <><path d="m3.5 10 8.5-7 8.5 7" /><path d="M5.5 9v11h13V9M9 20v-6h6v6" /></>}
          {icon === "launch" && <><path d="m8 5 11 7-11 7z" /><path d="M4 5v14" /></>}
          {icon === "analytics" && <><path d="M4 19V5" /><path d="M4 19h17" /><path d="m7 15 4-4 3 2 5-7" /><path d="M19 6h-4" /></>}
          {icon === "history" && <><path d="M3.5 12a8.5 8.5 0 1 0 2.2-5.7L3.5 8.5" /><path d="M3.5 4.5v4h4" /><path d="M12 7v5l3.2 2" /></>}
          {icon === "data" && <><ellipse cx="12" cy="5" rx="8" ry="3" /><path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3" /></>}
        </svg>
      </span>
      {children}
    </button>
  );
}

function TaskCenter({
  job,
  currentPage,
  stopPending,
  onNavigate,
  onStop,
  onError,
}: {
  job: JobStatus;
  currentPage: Page;
  stopPending: boolean;
  onNavigate: (page: Page) => void;
  onStop: () => void;
  onError: (message: string) => void;
}) {
  const [logs, setLogs] = useState("");
  const [logsOpen, setLogsOpen] = useState(false);
  const [logsBusy, setLogsBusy] = useState(false);
  const active = job.state === "running" || job.state === "stopping";
  const progress = job.progress;
  const generation = progress?.completed_generations ?? progress?.generation ?? 0;
  const generations = progress?.generations ?? 0;
  const ratio = generations > 0 ? Math.min(generation / generations, 1) : 0;

  useEffect(() => {
    setLogs("");
    setLogsOpen(false);
  }, [job.job_id]);

  if (!job.job_id) return null;

  const showLogs = async () => {
    if (!job.log_url) return;
    if (logsOpen) {
      setLogsOpen(false);
      return;
    }
    setLogsBusy(true);
    try {
      const result = await api<{ text: string }>(job.log_url);
      setLogs(result.text || "В журнале пока нет записей.");
      setLogsOpen(true);
    } catch (reason) {
      onError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setLogsBusy(false);
    }
  };

  const stateMessage = active
    ? job.state === "stopping"
      ? "Остановка запрошена"
      : job.kind === "demo"
        ? "Симуляция выполняется"
        : job.kind === "experiment"
          ? "Выполняется эксперимент"
          : "Выполняется обучение"
    : job.state === "failed"
      ? "Задача завершилась с ошибкой"
      : job.state === "stopped"
        ? "Задача остановлена"
        : job.state === "completed"
          ? "Задача завершена"
          : "Состояние задачи";

  return (
    <section className={`task-center task-${job.state}`} aria-live="polite">
      <div className="task-center-main">
        <div className="task-center-heading">
          <div>
            <p className="eyebrow">ТЕКУЩАЯ ЗАДАЧА</p>
            <h2>{kindLabel(job.kind ?? "Задача")}</h2>
          </div>
          <span className={`badge task-badge task-badge-${job.state}`}>
            {stateLabel(job.state)}
          </span>
        </div>
        <p className="task-summary">{stateMessage}</p>
        {generations > 0 && (
          <>
            <div className="progress-meta">
              <span>{progress?.phase ?? "Поколения"}</span>
              <b>{generation} / {generations}</b>
            </div>
            <div
              className="progress-track"
              role="progressbar"
              aria-label="Прогресс обучения"
              aria-valuemin={0}
              aria-valuemax={generations}
              aria-valuenow={generation}
            >
              <span style={{ width: `${ratio * 100}%` }} />
            </div>
            <p className="task-hint">
              {progress?.eta_seconds == null
                ? "Прогресс обновляется автоматически."
                : `Осталось примерно ${Math.ceil(progress.eta_seconds)} сек.`}
              {progress?.best_fitness != null &&
                ` Лучший fitness: ${progress.best_fitness.toFixed(3)}.`}
            </p>
          </>
        )}
        {job.state === "stopping" && (
          <p className="task-hint">Обучение остановится после сохранения текущего поколения.</p>
        )}
        {job.state === "failed" && (
          <p className="task-hint">Откройте журнал, чтобы увидеть причину и последние строки вывода.</p>
        )}
      </div>
      <div className="task-center-actions">
        {job.kind === "demo" && currentPage !== "live" && (
          <button className="button secondary" onClick={() => onNavigate("live")}>
            Открыть симуляцию
          </button>
        )}
        {active && (job.kind !== "demo" || currentPage !== "live") && (
          <button
            className="button danger"
            disabled={stopPending || job.state === "stopping"}
            onClick={onStop}
          >
            {stopPending || job.state === "stopping"
              ? "Остановка…"
              : job.kind === "demo"
                ? "Остановить симуляцию"
                : "Остановить после поколения"}
          </button>
        )}
        {job.state === "failed" && (
          <button className="button secondary" disabled={logsBusy} onClick={() => void showLogs()}>
            {logsBusy ? "Загрузка журнала…" : logsOpen ? "Скрыть журнал" : "Показать журнал ошибки"}
          </button>
        )}
      </div>
      {logsOpen && <pre className="task-log">{logs}</pre>}
    </section>
  );
}

function Overview({
  runs,
  datasets,
  job,
  onNavigate,
}: {
  runs: Run[];
  datasets: Dataset[];
  job: JobStatus;
  onNavigate: (page: Page) => void;
}) {
  const completed = runs.filter((run) => run.state === "completed").length;
  const running = job.state === "running" || job.state === "stopping";
  return (
    <div className="page-content">
      <section className="overview-hero">
        <div className="overview-hero-copy">
          <p className="eyebrow">ЛАБОРАТОРИЯ ЭКСПЕРИМЕНТОВ</p>
          <h1>Добро пожаловать в CitySim</h1>
          <p className="muted">Запускайте симуляции, сравнивайте результаты и собирайте данные с разных компьютеров.</p>
          <div className="hero-meta" aria-label="Возможности CitySim">
            <span><i className="hero-meta-dot" /> Симуляция</span>
            <span><i className="hero-meta-dot hero-meta-dot-blue" /> Обучение</span>
            <span><i className="hero-meta-dot hero-meta-dot-purple" /> Аналитика</span>
          </div>
          <button className="button primary" onClick={() => onNavigate("launch")}>
            <span>＋</span> Новый запуск
          </button>
        </div>
        <CityIllustration />
      </section>

      <div className="stat-grid">
        <StatCard label="Всего запусков" value={runs.length} caption="В этом рабочем пространстве" icon="◷" />
        <StatCard label="Успешные" value={completed} caption="Завершились без ошибок" icon="✓" tone="green" />
        <StatCard label="Импортированные наборы" value={datasets.length} caption="Собраны с других устройств" icon="⇣" tone="purple" />
        <StatCard label="Активная задача" value={running ? "В работе" : "Нет"} caption={running ? (job.kind ?? "Задача") : "Можно запускать эксперимент"} icon="●" tone={running ? "orange" : "blue"} />
      </div>

      <section className="panel">
        <div className="section-head">
          <div><p className="eyebrow">АКТИВНОСТЬ</p><h2>Последние запуски</h2></div>
          <button className="text-button" onClick={() => onNavigate("data")}>Все данные <span>→</span></button>
        </div>
        <RunTable runs={runs.slice(0, 5)} />
      </section>
    </div>
  );
}

function CityIllustration() {
  return (
    <figure className="city-illustration" aria-label="Схема изометрического города CitySim">
      <svg viewBox="0 0 520 310" role="img" aria-labelledby="city-illustration-title">
        <title id="city-illustration-title">Изометрический город с жилыми кварталами и парком</title>
        <defs>
          <linearGradient id="city-ground" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0" stopColor="#34234f" stopOpacity=".72" />
            <stop offset="1" stopColor="#17212c" stopOpacity=".84" />
          </linearGradient>
          <linearGradient id="tower-front" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0" stopColor="#593b83" />
            <stop offset="1" stopColor="#302342" />
          </linearGradient>
          <linearGradient id="tower-side" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0" stopColor="#382650" />
            <stop offset="1" stopColor="#202333" />
          </linearGradient>
          <linearGradient id="tower-top" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0" stopColor="#d2bdff" />
            <stop offset="1" stopColor="#8b55e8" />
          </linearGradient>
          <radialGradient id="city-glow">
            <stop offset="0" stopColor="#873cff" stopOpacity=".3" />
            <stop offset="1" stopColor="#873cff" stopOpacity="0" />
          </radialGradient>
          <filter id="city-shadow" x="-.3" y="-.3" width="1.7" height="1.8">
            <feGaussianBlur in="SourceAlpha" stdDeviation="9" />
            <feOffset dy="12" />
            <feComponentTransfer><feFuncA type="linear" slope=".28" /></feComponentTransfer>
            <feMerge><feMergeNode /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
        </defs>
        <ellipse cx="288" cy="158" rx="211" ry="138" fill="url(#city-glow)" />
        <g className="city-orbit city-orbit-back" fill="none" stroke="#b69aff" strokeOpacity=".24">
          <ellipse cx="285" cy="157" rx="207" ry="81" transform="rotate(-18 285 157)" />
          <ellipse cx="285" cy="157" rx="178" ry="102" transform="rotate(25 285 157)" />
        </g>
        <g fill="none" stroke="#d3c4ff" strokeOpacity=".12" strokeWidth="1">
          <path d="m75 208 188-109 184 106-188 109z" />
          <path d="m112 229 188-109 111 64M151 251l187-109 74 42M190 273l187-109" />
          <path d="m151 165 185 107M190 142l185 107M229 120l185 107M267 98l185 107" />
        </g>
        <g className="city-ground" filter="url(#city-shadow)">
          <path d="m93 190 190-110 187 108-190 110z" fill="url(#city-ground)" stroke="#a77cff" strokeOpacity=".46" />
          <path d="m93 190 187 108v10L93 201z" fill="#1d2331" />
          <path d="m280 298 190-110v10L280 308z" fill="#141c27" />
          <path d="m122 190 161-93 158 91-162 94z" fill="none" stroke="#d3c4ff" strokeOpacity=".19" />
        </g>
        <g className="city-road" fill="none" strokeLinecap="round" strokeLinejoin="round">
          <path d="m128 211 155-90 153 89" stroke="#0f2527" strokeWidth="16" />
          <path d="m128 211 155-90 153 89" stroke="#b69aff" strokeOpacity=".42" strokeWidth="1.4" strokeDasharray="5 7" />
          <path d="m230 252 96-55" stroke="#0f2527" strokeWidth="12" />
          <path d="m230 252 96-55" stroke="#ff744f" strokeOpacity=".35" strokeWidth="1" strokeDasharray="4 6" />
        </g>
        <g className="city-trees" stroke="#ff744f" strokeOpacity=".58">
          <path d="m139 186v17m19-29v18m151-37v18m95 52v17m-218-7v13m166-35v14" />
          <circle cx="139" cy="183" r="7" fill="#e55137" fillOpacity=".58" />
          <circle cx="158" cy="169" r="6" fill="#ff9a76" fillOpacity=".54" />
          <circle cx="309" cy="152" r="7" fill="#ff4f00" fillOpacity=".48" />
          <circle cx="404" cy="204" r="7" fill="#e55137" fillOpacity=".54" />
          <circle cx="186" cy="220" r="5" fill="#ff9a76" fillOpacity=".52" />
          <circle cx="352" cy="187" r="5" fill="#ff9a76" fillOpacity=".52" />
        </g>
        <g className="city-buildings" filter="url(#city-shadow)">
          <g>
            <path d="m171 174 31-18 31 18-31 18z" fill="url(#tower-top)" />
            <path d="M171 174v37l31 18v-37z" fill="url(#tower-front)" />
            <path d="m202 192 31-18v37l-31 18z" fill="url(#tower-side)" />
            <path d="m180 178 22-13 22 13-22 13z" fill="#e1d4ff" fillOpacity=".28" />
            <path d="m180 188 5 3v18l-5-3zm12 7 5 3v18l-5-3zm21 4 5-3v18l-5 3zm11-7 5-3v18l-5 3z" fill="#d3c4ff" fillOpacity=".62" />
          </g>
          <g>
            <path d="m254 126 36-21 36 21-36 21z" fill="url(#tower-top)" />
            <path d="M254 126v64l36 21v-64z" fill="url(#tower-front)" />
            <path d="m290 147 36-21v64l-36 21z" fill="url(#tower-side)" />
            <path d="m264 126 26-15 26 15-26 15z" fill="#e1d4ff" fillOpacity=".24" />
            <path d="m264 141 6 4v29l-6-4zm13 8 6 4v29l-6-4zm22 6 6-4v29l-6 4zm13-8 6-4v29l-6 4z" fill="#d3c4ff" fillOpacity=".64" />
            <path d="m281 118 9-5 9 5-9 5z" fill="#fff1e9" fillOpacity=".86" />
          </g>
          <g>
            <path d="m337 184 26-15 26 15-26 15z" fill="#a6c8ef" fillOpacity=".82" />
            <path d="M337 184v30l26 15v-30z" fill="#42617a" />
            <path d="m363 199 26-15v30l-26 15z" fill="#2b4359" />
            <path d="m345 188 5 3v19l-5-3zm12 7 5 3v19l-5-3zm19 4 5-3v19l-5 3zm10-6 5-3v19l-5 3z" fill="#c5dcf4" fillOpacity=".64" />
          </g>
          <g>
            <path d="m211 225 18-11 19 11-19 11z" fill="#a58fd5" fillOpacity=".85" />
            <path d="M211 225v21l19 11v-21z" fill="#574b79" />
            <path d="m230 236 18-11v21l-18 11z" fill="#393553" />
            <path d="m220 228 5 3v11l-5-3zm22 4 4-2v11l-4 2z" fill="#d1c5f6" fillOpacity=".6" />
          </g>
        </g>
        <g className="city-signals">
          <circle cx="290" cy="102" r="3" fill="#ff4f00" />
          <circle cx="290" cy="102" r="8" fill="none" stroke="#ff744f" strokeOpacity=".58" />
          <path d="M290 92v-16" stroke="#ff744f" strokeOpacity=".42" strokeDasharray="2 3" />
          <circle cx="390" cy="123" r="2.5" fill="#b4a5ff" />
          <circle cx="147" cy="142" r="2.5" fill="#9dc7ff" />
        </g>
      </svg>
      <figcaption className="city-illustration-caption">
        <span><i className="hero-meta-dot" /> MODEL PREVIEW</span>
        <b>Изометрический мир CitySim</b>
        <small>Город · агенты · экономика</small>
      </figcaption>
    </figure>
  );
}

function StatCard({
  label,
  value,
  caption,
  icon,
  tone,
}: {
  label: string;
  value: string | number;
  caption: string;
  icon: string;
  tone?: string;
}) {
  return (
    <div className="stat-card">
      <div className="stat-top"><span>{label}</span><span className={`stat-icon ${tone ?? ""}`}>{icon}</span></div>
      <strong>{value}</strong>
      <small>{caption}</small>
    </div>
  );
}

function LaunchPage({
  branches,
  activeBranch,
  jobActive,
  onStart,
  onError,
}: {
  branches: Branch[];
  activeBranch: string;
  jobActive: boolean;
  onStart: (request: Record<string, unknown>) => Promise<string | null>;
  onError: (message: string) => void;
}) {
  const [kind, setKind] = useStoredState(
    "citysim.launch.v1.kind",
    "experiment" as const,
    (value): value is "demo" | "train" | "experiment" => value === "demo" || value === "train" || value === "experiment",
    onError,
  );
  const [branch, setBranch] = useStoredState("citysim.launch.v1.branch", activeBranch, (value): value is string => typeof value === "string", onError);
  const [experiment, setExperiment] = useStoredState("citysim.launch.v1.experiment", "e1", (value): value is string => typeof value === "string" && (/^e([1-9]|1[0-2])$/.test(value) || value === "all"), onError);
  const [policy, setPolicy] = useStoredState(
    "citysim.launch.v1.policy",
    "manual",
    (value): value is string => value === "manual" || value === "random" || value === "greedy" || value === "neural",
    onError,
  );
  const [shock, setShock] = useStoredState(
    "citysim.launch.v1.shock",
    "",
    (value): value is string => value === "" || value === "industry_removal" || value === "population_boom" || value === "earthquake",
    onError,
  );
  const [quick, setQuick] = useStoredState("citysim.launch.v1.quick", true, (value): value is boolean => typeof value === "boolean", onError);
  const [generations, setGenerations] = useStoredState("citysim.launch.v1.generations", 100, (value): value is number => isIntegerBetween(value, 1, 100_000), onError);
  const [episodes, setEpisodes] = useStoredState("citysim.launch.v1.episodes", 10, (value): value is number => isIntegerBetween(value, 1, 100_000), onError);
  const [population, setPopulation] = useStoredState("citysim.launch.v1.population", 32, (value): value is number => isIntegerBetween(value, 8, 10_000), onError);
  const [rounds, setRounds] = useStoredState<number | null>("citysim.launch.v1.rounds", null, (value): value is number | null => value === null || isIntegerBetween(value, 1, 10_000), onError);
  const [agents, setAgents] = useStoredState("citysim.launch.v1.agents", 10, (value): value is number => isIntegerBetween(value, 1, 1000), onError);
  const [workers, setWorkers] = useStoredState("citysim.launch.v1.workers", "", (value): value is string => typeof value === "string" && (value === "" || /^\d+$/.test(value)), onError);
  const [seed, setSeed] = useStoredState("citysim.launch.v1.seed", 0, (value): value is number => isIntegerBetween(value, 0, Number.MAX_SAFE_INTEGER), onError);
  const [seeds, setSeeds] = useStoredState("citysim.launch.v1.seeds", "0-2", (value): value is string => typeof value === "string", onError);
  const [grid, setGrid] = useStoredState("citysim.launch.v1.grid", "12x12", (value): value is string => typeof value === "string", onError);
  const [genomeRef, setGenomeRef] = useStoredState("citysim.launch.v1.genomeRef", "", (value): value is string => typeof value === "string", onError);
  const [checkpoints, setCheckpoints] = useState<DemoCheckpoint[]>([]);
  const [resume, setResume] = useStoredState("citysim.launch.v1.resume", false, (value): value is boolean => typeof value === "boolean", onError);
  const [busy, setBusy] = useState(false);
  const populationValid =
    population >= agents &&
    population % 4 === 0 &&
    episodes * agents >= population;
  const seedsValid = /^\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*$/.test(seeds.trim());
  const gridError = gridValidationError(grid);
  const workersValid = workers === "" || isIntegerBetween(Number(workers), 1, 1024);
  const numericSettingsValid =
    isIntegerBetween(generations, 1, 100_000) &&
    isIntegerBetween(episodes, 1, 100_000) &&
    isIntegerBetween(population, 8, 10_000) &&
    isIntegerBetween(agents, 1, 1000) &&
    isIntegerBetween(seed, 0, Number.MAX_SAFE_INTEGER) &&
    (rounds === null || isIntegerBetween(rounds, 1, 10_000)) &&
    workersValid;
  const selectedBranch = branches.find((item) => item.name === branch);
  const branchModeValid =
    kind !== "train" ||
    Boolean(
      selectedBranch &&
        (selectedBranch.head_commit ? resume : !resume),
    );
  const canStart =
    !jobActive &&
    !gridError &&
    numericSettingsValid &&
    (kind === "demo"
      ? policy !== "neural" || checkpoints.some((checkpoint) => checkpoint.ref === genomeRef)
      : populationValid && branchModeValid && (kind !== "experiment" || seedsValid));

  const resetSettings = () => {
    setKind("experiment");
    setBranch(activeBranch);
    setExperiment("e1");
    setPolicy("manual");
    setShock("");
    setQuick(true);
    setGenerations(100);
    setEpisodes(10);
    setPopulation(32);
    setRounds(null);
    setAgents(10);
    setWorkers("");
    setSeed(0);
    setSeeds("0-2");
    setGrid("12x12");
    setGenomeRef("");
    setResume(false);
  };

  useEffect(() => {
    if (!branch || !branches.some((item) => item.name === branch)) setBranch(activeBranch);
  }, [activeBranch, branch, branches]);
  useEffect(() => {
    if (kind !== "demo" || policy !== "neural") return;
    let active = true;
    api<{ checkpoints: DemoCheckpoint[] }>("/api/demo/checkpoints")
      .then((result) => {
        if (!active) return;
        setCheckpoints(result.checkpoints);
        setGenomeRef((current) =>
          result.checkpoints.some((checkpoint) => checkpoint.ref === current)
            ? current
            : result.checkpoints[0]?.ref ?? "",
        );
      })
      .catch((reason: unknown) => {
        if (active) onError(reason instanceof Error ? reason.message : String(reason));
      });
    return () => { active = false; };
  }, [kind, policy, onError]);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (!canStart || busy) return;
    setBusy(true);
    let result: string | null;
    try {
      result = await onStart({
        kind,
        branch,
        resume,
        experiment,
        policy,
        genome_ref: policy === "neural" ? genomeRef : null,
        shock: shock || null,
        quick,
        generations,
        episodes,
        population,
        rounds: rounds ?? undefined,
        n_agents: agents,
        workers: workers ? Number(workers) : null,
        seed,
        seeds,
        grid,
        speed_ms: 80,
        eval_seeds: 5,
        checkpoint_every: 10,
      });
    } catch (reason) {
      onError(reason instanceof Error ? reason.message : String(reason));
      result = null;
    } finally {
      setBusy(false);
    }
    if (result) window.scrollTo({ top: 0, behavior: "smooth" });
  };

  return (
    <div className="page-content">
      <div className="page-heading">
        <div><p className="eyebrow">КОМАНДНЫЙ ЦЕНТР</p><h1>Новый запуск</h1><p className="muted">Настройте задачу и следите за прогрессом в реальном времени.</p></div>
      </div>
      <form className="panel form-panel" onSubmit={(event) => void submit(event)}>
        <div className="form-section">
          <p className="eyebrow">ТИП ЗАДАЧИ</p>
          <div className="choice-grid">
            {([
              ["demo", "Демо", "Один эпизод симуляции"],
              ["train", "Обучение", "Эволюция стратегий по ветке"],
              ["experiment", "Эксперимент", "Эксперименты E1–E12 и повторные сиды"],
            ] as const).map(([value, title, description]) => (
              <button
                type="button"
                key={value}
                className={`choice-card ${kind === value ? "selected" : ""}`}
                onClick={() => setKind(value)}
              >
                <span className="choice-radio" />
                <b>{title}</b><small>{description}</small>
              </button>
            ))}
          </div>
        </div>

        {kind === "demo" ? (
          <div className="form-section">
            <p className="eyebrow">ПАРАМЕТРЫ СИМУЛЯЦИИ</p>
            <div className="field-grid">
              <Field label="Стратегия">
                <select value={policy} onChange={(event) => setPolicy(event.target.value)}>
                  <option value="manual">Ручная</option><option value="random">Случайная</option>
                  <option value="greedy">Жадная</option><option value="neural">Нейросеть</option>
                </select>
              </Field>
              {policy === "neural" && (
                <Field label="Снимок обученной сети">
                  <select value={genomeRef} onChange={(event) => setGenomeRef(event.target.value)}>
                    {!checkpoints.length && <option value="">Нет доступных снимков</option>}
                    {checkpoints.map((checkpoint) => (
                      <option key={checkpoint.ref} value={checkpoint.ref}>
                        {checkpoint.ref} · fitness {checkpoint.best_fitness?.toFixed(3) ?? "—"}
                      </option>
                    ))}
                  </select>
                </Field>
              )}
              <Field label="Размер сетки">
                <input value={grid} onChange={(event) => setGrid(event.target.value)} placeholder="12x12" aria-invalid={Boolean(gridError)} aria-describedby={gridError ? "grid-format-hint grid-error" : "grid-format-hint"} />
              </Field>
              <Field label="Сид"><input type="number" min="0" value={seed} onChange={(event) => setSeed(Number(event.target.value))} /></Field>
              <Field label="Раундов на эпизод">
                <input
                  type="number"
                  min="1"
                  max="10000"
                  required
                  value={rounds ?? (quick ? 6 : 20)}
                  onChange={(event) => setRounds(Number(event.target.value))}
                />
              </Field>
              <Field label="Сценарий">
                <select value={shock} onChange={(event) => setShock(event.target.value)}>
                  <option value="">Без сценария</option><option value="industry_removal">Удаление промышленности</option>
                  <option value="population_boom">Рост населения</option><option value="earthquake">Землетрясение</option>
                </select>
              </Field>
            </div>
            <label className="check-row"><input type="checkbox" checked={quick} onChange={(event) => setQuick(event.target.checked)} /> Быстрая конфигурация</label>
            <p className="hint" id="grid-format-hint">Размер сетки задаётся как высота×ширина, например 12x12. Без изменения поля быстрый режим использует 6 раундов, обычный — 20; введённое число применяется независимо от режима.</p>
            {gridError && <p className="field-error" id="grid-error" role="alert">{gridError}</p>}
          </div>
        ) : (
          <>
            <div className="form-section">
              <p className="eyebrow">ПАРАМЕТРЫ ЭКСПЕРИМЕНТА</p>
              <div className="field-grid">
                {kind === "train" ? (
                  <>
                    <Field label="Ветка истории">
                      <select value={branch} onChange={(event) => setBranch(event.target.value)}>
                        {branches.map((item) => <option key={item.name} value={item.name}>{item.name}</option>)}
                      </select>
                    </Field>
                    <Field label="Начальный сид"><input type="number" min="0" value={seed} onChange={(event) => setSeed(Number(event.target.value))} /></Field>
                    <label className="check-row span-2"><input type="checkbox" checked={resume} onChange={(event) => setResume(event.target.checked)} /> Продолжить обучение в выбранной ветке</label>
                    <p className="hint span-2">
                      {selectedBranch?.head_commit
                        ? "Ветка уже содержит обучение. Продолжите её или выберите пустую ветку."
                        : "Пустая ветка начнёт новое обучение. Для существующей ветки включите продолжение."}
                    </p>
                  </>
                ) : (
                  <>
                    <Field label="Эксперимент">
                      <select value={experiment} onChange={(event) => setExperiment(event.target.value)}>
                        {Array.from({ length: 12 }, (_, i) => <option key={i + 1} value={`e${i + 1}`}>E{i + 1}</option>)}
                        <option value="all">Все эксперименты</option>
                      </select>
                    </Field>
                    <Field label="Сиды (например, 0-5 или 0,2,4)">
                      <input
                        value={seeds}
                        onChange={(event) => setSeeds(event.target.value)}
                        pattern="[0-9]+(-[0-9]+)?(,[0-9]+(-[0-9]+)?)*"
                        required
                      />
                    </Field>
                  </>
                )}
              </div>
            </div>
            <div className="form-section">
              <div className="section-head compact">
                <div><p className="eyebrow">КОНФИГУРАЦИЯ</p><h2>Параметры обучения</h2></div>
                <label className="check-row"><input type="checkbox" checked={quick} onChange={(event) => setQuick(event.target.checked)} /> Быстрый режим</label>
              </div>
              <div className="field-grid">
                <Field label="Поколений"><input type="number" min="1" max="100000" required value={generations} onChange={(event) => setGenerations(Number(event.target.value))} /></Field>
                <Field label="Эпизодов на поколение"><input type="number" min="1" max="100000" required value={episodes} onChange={(event) => setEpisodes(Number(event.target.value))} /></Field>
                <Field label="Раундов на эпизод">
                  <input
                    type="number"
                    min="1"
                    max="10000"
                    required
                    value={rounds ?? (quick ? 6 : 20)}
                    onChange={(event) => setRounds(Number(event.target.value))}
                  />
                </Field>
                <Field label="Размер популяции"><input type="number" min="8" max="10000" step="4" required value={population} onChange={(event) => setPopulation(Number(event.target.value))} /></Field>
                <Field label="Агентов в городе"><input type="number" min="1" max="1000" required value={agents} onChange={(event) => setAgents(Number(event.target.value))} /></Field>
                <Field label="CPU workers (пусто — авто)"><input type="number" min="1" max="1024" value={workers} onChange={(event) => setWorkers(event.target.value)} placeholder="Авто" /></Field>
              </div>
              <p className={`hint ${!populationValid || (kind === "experiment" && !seedsValid) ? "hint-error" : ""}`}>
                Быстрый режим использует 6 раундов, обычный — 20, если число не задано.
                Популяция должна делиться на 4, быть не меньше числа агентов, а эпизодов × агентов — покрывать популяцию.
                {kind === "experiment" && " Сиды задаются числами, диапазонами и их списками, например 0-5,8."}
              </p>
            </div>
          </>
        )}
        <div className="form-footer">
          <span className="muted small">Параметры сохраняются в этом браузере. Данные задач остаются в выбранном хранилище.</span>
          <button type="button" className="button secondary" disabled={busy} onClick={resetSettings}>Сбросить параметры</button>
          <button className="button primary" aria-disabled={busy || !canStart} disabled={busy || !canStart} title={jobActive ? "Дождитесь завершения активной задачи" : !canStart ? "Проверьте значения и подсказки формы" : undefined}>
            {busy ? "Запуск…" : "▶  Запустить задачу"}
          </button>
          {!canStart && <span className="field-error" role="status">{jobActive ? "Сначала остановите или дождитесь завершения текущей задачи." : "Исправьте параметры выше, чтобы запустить задачу."}</span>}
        </div>
      </form>
    </div>
  );
}

function LiveDemoPage({ job, onError }: { job: JobStatus; onError: (message: string) => void }) {
  const [frame, setFrame] = useState<LiveFrame | null>(null);
  const [paused, setPaused] = useState(false);
  const [speed, setSpeed] = useState(80);
  const [busy, setBusy] = useState(false);
  const [stopBusy, setStopBusy] = useState(false);
  const [imageBusy, setImageBusy] = useState(false);
  const [events, setEvents] = useState<LiveFrame[]>([]);
  const [lastSequence, setLastSequence] = useState(0);
  const running = job.state === "running" || job.state === "stopping";

  useEffect(() => {
    if (!job.job_id || job.kind !== "demo") return;
    setFrame(null);
    setEvents([]);
    setLastSequence(0);
    setPaused(false);
    setSpeed(80);
    let active = true;
    let offset = 0;
    let timer = 0;
    const restoreControl = async () => {
      try {
        const control = await api<{ paused: boolean; speed_ms: number }>(
          `/api/jobs/${job.job_id}/control`,
        );
        if (active) {
          setPaused(control.paused);
          setSpeed(control.speed_ms);
        }
      } catch (reason) {
        onError(reason instanceof Error ? reason.message : String(reason));
      }
    };
    const poll = async () => {
      try {
        const batch = await api<{ frames: LiveFrame[]; next_offset: number }>(
          `/api/jobs/${job.job_id}/frames?offset=${offset}`,
        );
        offset = batch.next_offset;
        const states = batch.frames.filter((item) => item.type_grid);
        if (states.length) {
          setFrame(states[states.length - 1]);
          setLastSequence(states[states.length - 1].seq);
          setEvents((current) => [...states].reverse().concat(current).slice(0, 8));
        }
      } catch (reason) {
        onError(reason instanceof Error ? reason.message : String(reason));
        return;
      }
      if (active) timer = window.setTimeout(() => void poll(), 120);
    };
    void restoreControl();
    void poll();
    return () => {
      active = false;
      window.clearTimeout(timer);
    };
  }, [job.job_id, job.kind, onError]);

  const updateControl = async (value: { paused?: boolean; speed_ms?: number }) => {
    if (!job.job_id) return;
    setBusy(true);
    try {
      await api(`/api/jobs/${job.job_id}/control`, {
        method: "POST",
        body: JSON.stringify(value),
      });
      if (value.paused !== undefined) setPaused(value.paused);
      if (value.speed_ms !== undefined) setSpeed(value.speed_ms);
    } catch (reason) {
      onError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setBusy(false);
    }
  };
  const stop = async () => {
    if (!job.job_id || stopBusy || job.state === "stopping") return;
    setStopBusy(true);
    try {
      await api(`/api/jobs/${job.job_id}/stop`, { method: "POST" });
      setPaused(false);
    } catch (reason) {
      onError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setStopBusy(false);
    }
  };
  const downloadPng = async () => {
    if (!job.job_id || !frame || imageBusy) return;
    setImageBusy(true);
    try {
      const response = await fetch(
        `/api/jobs/${job.job_id}/frames/${frame.seq}/image?projection=isometric`,
      );
      if (!response.ok) throw new Error(`Не удалось создать PNG (${response.status})`);
      const objectUrl = URL.createObjectURL(await response.blob());
      const link = document.createElement("a");
      link.href = objectUrl;
      link.download = `citysim-frame-${frame.seq}.png`;
      link.click();
      window.setTimeout(() => URL.revokeObjectURL(objectUrl), 1000);
    } catch (reason) {
      onError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setImageBusy(false);
    }
  };
  const selectFrame = async (sequence: number) => {
    if (!job.job_id) return;
    try {
      setFrame(await api<LiveFrame>(`/api/jobs/${job.job_id}/frames/${sequence}`));
    } catch (reason) {
      onError(reason instanceof Error ? reason.message : String(reason));
    }
  };

  return (
    <div className="page-content live-page">
      <div className="page-heading">
        <div>
          <p className="eyebrow">ДЕМО В РЕАЛЬНОМ ВРЕМЕНИ</p>
          <h1>Город и агенты</h1>
          <p className="muted">Состояния сохраняются как сжатые данные карты; PNG создаётся только по запросу.</p>
        </div>
        <span className={`badge ${running ? "running" : ""}`}>
          {job.state === "stopping"
            ? "Останавливается"
            : running
              ? paused ? "На паузе" : "Идёт прогон"
              : stateLabel(job.state)}
        </span>
      </div>
      <section className="live-layout">
        <div className="panel live-map-panel">
          <div className="section-head">
            <div>
              <p className="eyebrow">ИЗОМЕТРИЧЕСКАЯ КАРТА</p>
              <h2>{frame?.round ? `Раунд ${frame.round}` : "Ожидание первого состояния"}</h2>
            </div>
            <button className="button secondary" disabled={!frame || imageBusy} onClick={() => void downloadPng()}>
              {imageBusy ? "Создание PNG…" : "Скачать PNG"}
            </button>
          </div>
          <IsometricMap frame={frame} />
          <p className="muted small">Высота показывает этажность, свечение крыши — её качество. Наведите или нажмите на клетку, чтобы открыть детали.</p>
          {!running && lastSequence > 0 && (
            <div className="frame-history">
              <label htmlFor="frame-history">Кадр {frame?.seq ?? 0} из {lastSequence}</label>
              <input
                id="frame-history"
                type="range"
                min="1"
                max={lastSequence}
                value={frame?.seq ?? lastSequence}
                onChange={(event) => void selectFrame(Number(event.target.value))}
              />
            </div>
          )}
          <div className="live-controls">
            <div className="speed-control">
              <label htmlFor="demo-speed">Скорость</label>
              <input
                id="demo-speed"
                type="range"
                min="0"
                max="1000"
                step="25"
                value={speed}
                disabled={!running || busy}
                onChange={(event) => void updateControl({ speed_ms: Number(event.target.value) })}
              />
              <span>{speed === 0 ? "Максимум" : `${speed} мс/ход`}</span>
            </div>
            <div className="button-row">
              <button
                className="button secondary"
                disabled={!running || busy}
                onClick={() => void updateControl({ paused: !paused })}
              >
                {paused ? "Продолжить" : "Пауза"}
              </button>
              <button
                className="button danger"
                disabled={!running || stopBusy || job.state === "stopping"}
                onClick={() => void stop()}
              >
                {stopBusy || job.state === "stopping" ? "Остановка…" : "Остановить симуляцию"}
              </button>
            </div>
          </div>
        </div>
        <aside className="panel live-side-panel">
          <p className="eyebrow">ПОСЛЕДНИЕ ДЕЙСТВИЯ</p>
          <h2>Ход симуляции</h2>
          {!events.length && <p className="muted small">Состояния появятся после начала прогона.</p>}
          <div className="live-event-list">
            {events.map((event) => (
              <div className="live-event" key={event.seq}>
                <span className="event-agent" style={{ background: agentColor(event.agent_id ?? 0) }} />
                <div>
                  <b>{event.agent_id == null ? "Состояние города" : `Агент ${event.agent_id + 1}`}</b>
                  <small>
                    {event.kind === "action"
                      ? event.action?.cell == null
                        ? "Пропуск"
                        : `${event.action.demolish ? "Снос" : `Строительство · ${event.action.floors ?? 1} эт. · крыша ${event.action.roof_quality ?? 0}/3`} · клетка ${event.action.cell + 1}`
                      : event.kind === "round_end" ? "Конец раунда" : "Начало раунда"}
                    {event.accepted === false ? " · не выполнено" : ""}
                  </small>
                </div>
              </div>
            ))}
          </div>
          {frame && <p className="muted small">Кадр {frame.seq} · данные сохраняются в каталоге запуска.</p>}
        </aside>
      </section>
    </div>
  );
}

type IsoPoint = [number, number];
type IsoGeometry = {
  height: number;
  width: number;
  tileWidth: number;
  tileHeight: number;
  originX: number;
  originY: number;
  scale: number;
  offsetX: number;
  offsetY: number;
  viewWidth: number;
  viewHeight: number;
  maxBuildingHeight: number;
};

const BUILDING_MATERIALS = [
  { roof: "#30394a", front: "#273040", side: "#1d2533" },
  { roof: "#b69aff", front: "#634d86", side: "#352b49" },
  { roof: "#ffb08d", front: "#8c594f", side: "#49333b" },
  { roof: "#a9c8f1", front: "#526987", side: "#303d55" },
  { roof: "#d68fa0", front: "#714858", side: "#3d2c3b" },
  { roof: "#93b7d0", front: "#526d83", side: "#303f50" },
];

const BUILDING_LABELS = [
  "Свободная клетка",
  "Жильё",
  "Коммерция",
  "Смешанное здание",
  "Индустрия",
  "Зелёная зона",
];

function isometricHeight(type: number, quality: number, floors: number, tileHeight: number) {
  if (type === 5) return tileHeight * 0.25 * Math.max(0, floors - 1);
  if (type <= 0) return 0;
  const typeScale = [0.9, 1.2, 1.05, 1.45][type - 1] ?? 1;
  const qualityScale = 0.75 + 0.15 * (Math.max(0, Math.min(2, quality)) + 1);
  return tileHeight * typeScale * qualityScale * Math.sqrt(Math.max(1, floors));
}

function mapGeometry(frame: LiveFrame, rect: DOMRect): IsoGeometry | null {
  const typeGrid = frame.type_grid;
  const qualityGrid = frame.quality_grid;
  if (!typeGrid || !qualityGrid || !rect.width || !rect.height) return null;
  const height = frame.height ?? typeGrid.length;
  const width = frame.width ?? typeGrid[0]?.length ?? 0;
  if (!height || !width) return null;
  const tileWidth = 34;
  const tileHeight = 17;
  let maxBuildingHeight = 0;
  for (let row = 0; row < height; row++) {
    for (let col = 0; col < width; col++) {
      const type = typeGrid[row]?.[col] ?? 0;
      maxBuildingHeight = Math.max(
        maxBuildingHeight,
        isometricHeight(
          type,
          qualityGrid[row]?.[col] ?? 0,
          frame.floors_grid?.[row]?.[col] ?? (type > 0 ? 1 : 0),
          tileHeight,
        ),
      );
    }
  }
  const gridWidth = (height + width) * tileWidth / 2;
  const gridHeight = (height + width) * tileHeight / 2 + maxBuildingHeight;
  const scale = Math.max(
    0.1,
    Math.min((rect.width - 32) / gridWidth, (rect.height - 32) / gridHeight),
  );
  return {
    height,
    width,
    tileWidth,
    tileHeight,
    originX: (height + width) * tileWidth / 4,
    originY: maxBuildingHeight + 8,
    scale,
    offsetX: (rect.width - gridWidth * scale) / 2,
    offsetY: (rect.height - gridHeight * scale) / 2,
    viewWidth: rect.width,
    viewHeight: rect.height,
    maxBuildingHeight,
  };
}

function isoDiamond(cx: number, cy: number, tileWidth: number, tileHeight: number): IsoPoint[] {
  return [
    [cx, cy - tileHeight / 2],
    [cx + tileWidth / 2, cy],
    [cx, cy + tileHeight / 2],
    [cx - tileWidth / 2, cy],
  ];
}

function traceIsoPolygon(context: CanvasRenderingContext2D, points: IsoPoint[]) {
  if (!points.length) return;
  context.beginPath();
  context.moveTo(points[0][0], points[0][1]);
  for (const [x, y] of points.slice(1)) context.lineTo(x, y);
  context.closePath();
}

function fillIsoPolygon(
  context: CanvasRenderingContext2D,
  points: IsoPoint[],
  fill: string | CanvasGradient,
  stroke?: string,
  lineWidth = 0.7,
) {
  traceIsoPolygon(context, points);
  context.fillStyle = fill;
  context.fill();
  if (stroke) {
    context.strokeStyle = stroke;
    context.lineWidth = lineWidth;
    context.stroke();
  }
}

function isoCellFaces(frame: LiveFrame, geometry: IsoGeometry, row: number, col: number) {
  const { tileWidth, tileHeight, originX, originY } = geometry;
  const type = frame.type_grid?.[row]?.[col] ?? 0;
  const quality = frame.quality_grid?.[row]?.[col] ?? 0;
  const floors = frame.floors_grid?.[row]?.[col] ?? (type > 0 ? 1 : 0);
  const cx = originX + (col - row) * tileWidth / 2;
  const cy = originY + (row + col) * tileHeight / 2;
  const ground = isoDiamond(cx, cy, tileWidth, tileHeight);
  const elevation = isometricHeight(type, quality, floors, tileHeight);
  const roof = isoDiamond(cx, cy - elevation, tileWidth, tileHeight);
  return {
    type,
    floors,
    cx,
    cy,
    elevation,
    ground,
    roof,
    left: [ground[3], ground[2], roof[2], roof[3]] as IsoPoint[],
    right: [ground[2], ground[1], roof[1], roof[2]] as IsoPoint[],
  };
}

function syncCanvas(canvas: HTMLCanvasElement, width: number, height: number) {
  const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
  const pixelWidth = Math.max(1, Math.round(width * pixelRatio));
  const pixelHeight = Math.max(1, Math.round(height * pixelRatio));
  if (canvas.width !== pixelWidth) canvas.width = pixelWidth;
  if (canvas.height !== pixelHeight) canvas.height = pixelHeight;
  const context = canvas.getContext("2d");
  if (!context) return null;
  context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
  return { context, pixelRatio };
}

function pointInIsoPolygon(point: IsoPoint, polygon: IsoPoint[]) {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const [xi, yi] = polygon[i];
    const [xj, yj] = polygon[j];
    const intersects = (yi > point[1]) !== (yj > point[1])
      && point[0] < ((xj - xi) * (point[1] - yi)) / (yj - yi) + xi;
    if (intersects) inside = !inside;
  }
  return inside;
}

function hitIsoCell(point: IsoPoint, frame: LiveFrame, geometry: IsoGeometry) {
  const worldPoint: IsoPoint = [
    (point[0] - geometry.offsetX) / geometry.scale,
    (point[1] - geometry.offsetY) / geometry.scale,
  ];
  for (let diagonal = geometry.height + geometry.width - 2; diagonal >= 0; diagonal--) {
    for (
      let row = Math.min(geometry.height - 1, diagonal);
      row >= Math.max(0, diagonal - geometry.width + 1);
      row--
    ) {
      const col = diagonal - row;
      const faces = isoCellFaces(frame, geometry, row, col);
      if ([faces.roof, faces.left, faces.right, faces.ground].some(
        (polygon) => pointInIsoPolygon(worldPoint, polygon),
      )) return row * geometry.width + col;
    }
  }
  return null;
}

function IsometricMap({ frame }: { frame: LiveFrame | null }) {
  const backgroundRef = useRef<HTMLCanvasElement>(null);
  const cityRef = useRef<HTMLCanvasElement>(null);
  const interactionRef = useRef<HTMLCanvasElement>(null);
  const geometryRef = useRef<IsoGeometry | null>(null);
  const geometryCacheRef = useRef<{
    frame: LiveFrame;
    width: number;
    height: number;
    geometry: IsoGeometry | null;
  } | null>(null);
  const backgroundCacheRef = useRef<{ key: string; canvas: HTMLCanvasElement } | null>(null);
  const [hoverCell, setHoverCell] = useState<number | null>(null);
  const [selectedCell, setSelectedCell] = useState<number | null>(null);
  const getGeometry = (rect: DOMRect) => {
    if (
      geometryCacheRef.current?.frame === frame
      && geometryCacheRef.current.width === rect.width
      && geometryCacheRef.current.height === rect.height
    ) return geometryCacheRef.current.geometry;
    if (!frame) return null;
    const geometry = mapGeometry(frame, rect);
    geometryCacheRef.current = { frame, width: rect.width, height: rect.height, geometry };
    return geometry;
  };
  const typeGrid = frame?.type_grid;
  const width = frame?.width ?? typeGrid?.[0]?.length ?? 0;
  const selectedRow = selectedCell == null || !width ? -1 : Math.floor(selectedCell / width);
  const selectedCol = selectedCell == null || !width ? -1 : selectedCell % width;
  const selectedType = selectedRow < 0 ? 0 : typeGrid?.[selectedRow]?.[selectedCol] ?? 0;
  const selectedOwner = selectedRow < 0 ? -1 : frame?.owner_grid?.[selectedRow]?.[selectedCol] ?? -1;
  const selectedFloors = selectedRow < 0 ? 0 : frame?.floors_grid?.[selectedRow]?.[selectedCol] ?? 0;
  const selectedQuality = selectedRow < 0 ? 0 : frame?.quality_grid?.[selectedRow]?.[selectedCol] ?? 0;
  const selectedRoof = selectedRow < 0 ? 0 : frame?.roof_quality_grid?.[selectedRow]?.[selectedCol] ?? 0;

  useEffect(() => {
    if (!frame) {
      setHoverCell(null);
      setSelectedCell(null);
    }
  }, [frame]);

  useEffect(() => {
    const canvas = backgroundRef.current;
    if (!canvas || !frame?.type_grid?.length || !frame.type_grid[0]?.length) return;
    const render = () => {
      const rect = canvas.getBoundingClientRect();
      const geometry = getGeometry(rect);
      if (!geometry) return;
      geometryRef.current = geometry;
      const synced = syncCanvas(canvas, rect.width, rect.height);
      if (!synced) return;
      const { pixelRatio } = synced;
      const cacheKey = [
        rect.width, rect.height, pixelRatio, geometry.height, geometry.width,
        geometry.scale, geometry.offsetX, geometry.offsetY, geometry.maxBuildingHeight,
      ].join(":");
      if (backgroundCacheRef.current?.key === cacheKey) {
        synced.context.clearRect(0, 0, rect.width, rect.height);
        synced.context.drawImage(backgroundCacheRef.current.canvas, 0, 0, rect.width, rect.height);
        return;
      }

      const background = document.createElement("canvas");
      background.width = canvas.width;
      background.height = canvas.height;
      const context = background.getContext("2d");
      if (!context) return;
      context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
      context.clearRect(0, 0, rect.width, rect.height);
      const violetGlow = context.createRadialGradient(
        rect.width * 0.7, rect.height * 0.28, 0,
        rect.width * 0.7, rect.height * 0.28, Math.max(rect.width, rect.height) * 0.72,
      );
      violetGlow.addColorStop(0, "rgba(111, 63, 168, .18)");
      violetGlow.addColorStop(1, "rgba(19, 20, 32, 0)");
      context.fillStyle = violetGlow;
      context.fillRect(0, 0, rect.width, rect.height);
      const warmGlow = context.createRadialGradient(
        rect.width * 0.22, rect.height * 0.78, 0,
        rect.width * 0.22, rect.height * 0.78, rect.width * 0.42,
      );
      warmGlow.addColorStop(0, "rgba(229, 81, 55, .08)");
      warmGlow.addColorStop(1, "rgba(229, 81, 55, 0)");
      context.fillStyle = warmGlow;
      context.fillRect(0, 0, rect.width, rect.height);
      context.save();
      context.translate(geometry.offsetX, geometry.offsetY);
      context.scale(geometry.scale, geometry.scale);

      const { height, width: gridWidth, tileWidth, tileHeight, originX, originY } = geometry;
      const platform = [
        [originX, originY - tileHeight / 2],
        [originX + gridWidth * tileWidth / 2, originY + gridWidth * tileHeight / 2 - tileHeight / 2],
        [originX + (gridWidth - height) * tileWidth / 2, originY + (height + gridWidth) * tileHeight / 2 - tileHeight / 2],
        [originX - height * tileWidth / 2, originY + height * tileHeight / 2 - tileHeight / 2],
      ] as IsoPoint[];
      const base = platform.map(([x, y]) => [x, y + 8] as IsoPoint);
      context.save();
      context.shadowColor = "rgba(4, 6, 14, .72)";
      context.shadowBlur = 16;
      context.shadowOffsetY = 7;
      fillIsoPolygon(context, base, "#171b29", "rgba(167, 124, 255, .18)", 1);
      context.restore();
      fillIsoPolygon(context, platform, "#211e30", "rgba(167, 124, 255, .32)", 1.3);

      const groundGradient = context.createLinearGradient(0, originY, 0, originY + (height + gridWidth) * tileHeight / 2);
      groundGradient.addColorStop(0, "#343047");
      groundGradient.addColorStop(1, "#222738");
      for (let diagonal = 0; diagonal < height + gridWidth - 1; diagonal++) {
        for (let row = Math.max(0, diagonal - gridWidth + 1); row <= Math.min(height - 1, diagonal); row++) {
          const col = diagonal - row;
          const { ground } = isoCellFaces(frame, geometry, row, col);
          fillIsoPolygon(context, ground, groundGradient, "rgba(195, 176, 230, .13)", 0.55);
          const [top] = ground;
          context.fillStyle = "rgba(219, 205, 255, .19)";
          context.beginPath();
          context.arc(top[0], top[1] + tileHeight * 0.36, 0.65, 0, Math.PI * 2);
          context.fill();
        }
      }
      context.restore();
      backgroundCacheRef.current = { key: cacheKey, canvas: background };
      synced.context.clearRect(0, 0, rect.width, rect.height);
      synced.context.drawImage(background, 0, 0, rect.width, rect.height);
    };
    render();
    const observer = new ResizeObserver(render);
    observer.observe(canvas);
    return () => observer.disconnect();
  }, [frame]);

  useEffect(() => {
    const canvas = cityRef.current;
    if (!canvas) return;
    if (!frame?.type_grid || !frame.quality_grid || !frame.owner_grid) {
      const context = canvas.getContext("2d");
      context?.clearRect(0, 0, canvas.width, canvas.height);
      return;
    }
    const render = () => {
      const rect = canvas.getBoundingClientRect();
      const geometry = getGeometry(rect);
      if (!geometry) return;
      geometryRef.current = geometry;
      const synced = syncCanvas(canvas, rect.width, rect.height);
      if (!synced) return;
      const { context } = synced;
      context.clearRect(0, 0, rect.width, rect.height);
      context.save();
      context.translate(geometry.offsetX, geometry.offsetY);
      context.scale(geometry.scale, geometry.scale);
      const { height, width: gridWidth, tileWidth, tileHeight } = geometry;
      for (let diagonal = 0; diagonal < height + gridWidth - 1; diagonal++) {
        for (let row = Math.max(0, diagonal - gridWidth + 1); row <= Math.min(height - 1, diagonal); row++) {
          const col = diagonal - row;
          const faces = isoCellFaces(frame, geometry, row, col);
          const { type, floors, cx, cy, elevation, ground, roof, left, right } = faces;
          if (type <= 0) continue;
          const material = BUILDING_MATERIALS[type] ?? BUILDING_MATERIALS[1];
          const shadow = ground.map(([x, y]) => [x, y + 2.2] as IsoPoint);
          fillIsoPolygon(context, shadow, "rgba(6, 8, 17, .35)");
          const sideGradient = context.createLinearGradient(cx - tileWidth / 2, cy, cx + tileWidth / 2, cy - elevation);
          sideGradient.addColorStop(0, material.front);
          sideGradient.addColorStop(1, "#242337");
          fillIsoPolygon(context, left, sideGradient, "rgba(211, 194, 255, .13)", 0.7);
          const frontGradient = context.createLinearGradient(cx, cy, cx + tileWidth / 2, cy - elevation);
          frontGradient.addColorStop(0, material.side);
          frontGradient.addColorStop(1, "#1c2230");
          fillIsoPolygon(context, right, frontGradient, "rgba(211, 194, 255, .1)", 0.7);

          if (elevation > tileHeight * 0.55) {
            const rows = Math.min(4, Math.max(1, Math.floor(elevation / (tileHeight * 0.72))));
            for (const facade of [left, right]) {
              const windowCount = Math.min(3, Math.max(1, Math.floor(floors / 3)));
              for (let level = 0; level < rows; level++) {
                const vertical = 0.2 + (level + 0.35) * 0.58 / rows;
                for (let window = 0; window < windowCount; window++) {
                  const center = (window + 0.5) / windowCount;
                  const halfWidth = 0.14 / windowCount;
                  const pointAt = (horizontal: number, verticalPosition: number): IsoPoint => {
                    const bottomX = facade[0][0] + (facade[1][0] - facade[0][0]) * horizontal;
                    const bottomY = facade[0][1] + (facade[1][1] - facade[0][1]) * horizontal;
                    const topX = facade[3][0] + (facade[2][0] - facade[3][0]) * horizontal;
                    const topY = facade[3][1] + (facade[2][1] - facade[3][1]) * horizontal;
                    return [
                      bottomX + (topX - bottomX) * verticalPosition,
                      bottomY + (topY - bottomY) * verticalPosition,
                    ];
                  };
                  const windowShape = [
                    pointAt(center - halfWidth, vertical),
                    pointAt(center + halfWidth, vertical),
                    pointAt(center + halfWidth, vertical + 0.19 / rows),
                    pointAt(center - halfWidth, vertical + 0.19 / rows),
                  ];
                  fillIsoPolygon(context, windowShape, "rgba(223, 210, 255, .56)");
                }
              }
            }
          }

          const roofGradient = context.createLinearGradient(cx - tileWidth / 2, cy - elevation, cx + tileWidth / 2, cy - elevation + tileHeight);
          roofGradient.addColorStop(0, material.roof);
          roofGradient.addColorStop(1, material.front);
          fillIsoPolygon(context, roof, roofGradient, "rgba(235, 222, 255, .58)", 0.9);
          const roofQuality = frame.roof_quality_grid?.[row]?.[col] ?? 0;
          if (roofQuality > 0) {
            const inset = roof.map(([x, y]) => [
              cx + (x - cx) * 0.66,
              cy - elevation + (y - (cy - elevation)) * 0.66,
            ] as IsoPoint);
            const garden = context.createLinearGradient(cx, cy - elevation - tileHeight / 2, cx, cy - elevation + tileHeight / 2);
            garden.addColorStop(0, `rgba(255, 180, 145, ${0.22 + roofQuality * 0.12})`);
            garden.addColorStop(1, `rgba(167, 124, 255, ${0.18 + roofQuality * 0.1})`);
            fillIsoPolygon(context, inset, garden, "rgba(255, 205, 177, .46)", 0.55);
            for (let detail = 0; detail < roofQuality; detail++) {
              const offset = (detail - (roofQuality - 1) / 2) * tileWidth * 0.13;
              context.fillStyle = "rgba(255, 210, 187, .82)";
              context.beginPath();
              context.arc(cx + offset, cy - elevation - tileHeight * 0.04, 0.9, 0, Math.PI * 2);
              context.fill();
            }
          }

          const owner = frame.owner_grid?.[row]?.[col] ?? -1;
          if (owner >= 0) {
            const beaconY = cy - elevation - 2.4;
            const beacon = context.createRadialGradient(cx, beaconY, 0, cx, beaconY, 8);
            beacon.addColorStop(0, "rgba(255, 226, 205, .95)");
            beacon.addColorStop(0.22, agentColorAlpha(owner, 0.8));
            beacon.addColorStop(1, agentColorAlpha(owner, 0));
            context.fillStyle = beacon;
            context.beginPath();
            context.arc(cx, beaconY, 8, 0, Math.PI * 2);
            context.fill();
            context.beginPath();
            context.arc(cx, beaconY, 2.15, 0, Math.PI * 2);
            context.fillStyle = agentColor(owner);
            context.fill();
          }
        }
      }
      context.restore();
    };
    render();
    const observer = new ResizeObserver(render);
    observer.observe(canvas);
    return () => observer.disconnect();
  }, [frame]);

  useEffect(() => {
    const canvas = interactionRef.current;
    if (!canvas) return;
    const render = () => {
      const rect = canvas.getBoundingClientRect();
      const synced = syncCanvas(canvas, rect.width, rect.height);
      if (!synced) return;
      const { context } = synced;
      context.clearRect(0, 0, rect.width, rect.height);
      const geometry = geometryRef.current;
      if (!frame || !geometry) return;
      context.save();
      context.translate(geometry.offsetX, geometry.offsetY);
      context.scale(geometry.scale, geometry.scale);
      const actionCell = frame.action?.cell;
      const highlighted = new Set(
        [actionCell, hoverCell, selectedCell].filter(
          (cell): cell is number => cell != null && cell >= 0 && cell < geometry.height * geometry.width,
        ),
      );
      for (const cell of highlighted) {
        const row = Math.floor(cell / geometry.width);
        const col = cell % geometry.width;
        const { type, cx, cy, elevation, ground, roof } = isoCellFaces(frame, geometry, row, col);
        if (cell === actionCell) {
          fillIsoPolygon(context, ground, "rgba(255, 116, 79, .13)", "rgba(255, 142, 106, .9)", 1.1);
        }
        if (cell === hoverCell || cell === selectedCell) {
          const selected = cell === selectedCell;
          fillIsoPolygon(
            context,
            ground,
            selected ? "rgba(167, 124, 255, .18)" : "rgba(190, 166, 255, .09)",
            selected ? "rgba(211, 190, 255, .98)" : "rgba(190, 166, 255, .72)",
            selected ? 1.6 : 1,
          );
          if (type > 0) {
            context.shadowColor = "rgba(167, 124, 255, .5)";
            context.shadowBlur = selected ? 9 : 5;
            traceIsoPolygon(context, roof);
            context.strokeStyle = "rgba(219, 202, 255, .88)";
            context.lineWidth = selected ? 1.25 : 0.8;
            context.stroke();
            context.shadowBlur = 0;
          }
        }
        if (cell === actionCell) {
          const beacon = context.createRadialGradient(cx, cy - elevation, 0, cx, cy - elevation, 15);
          beacon.addColorStop(0, "rgba(255, 196, 160, .65)");
          beacon.addColorStop(1, "rgba(255, 116, 79, 0)");
          context.fillStyle = beacon;
          context.beginPath();
          context.arc(cx, cy - elevation, 15, 0, Math.PI * 2);
          context.fill();
        }
      }
      context.restore();
    };
    render();
    const observer = new ResizeObserver(render);
    observer.observe(canvas);
    return () => observer.disconnect();
  }, [frame, hoverCell, selectedCell]);

  const cellAtPointer = (clientX: number, clientY: number, canvas: HTMLCanvasElement) => {
    if (!frame || !geometryRef.current) return null;
    const rect = canvas.getBoundingClientRect();
    return hitIsoCell(
      [clientX - rect.left, clientY - rect.top],
      frame,
      geometryRef.current,
    );
  };
  const onMapKeyDown = (event: ReactKeyboardEvent<HTMLCanvasElement>) => {
    const geometry = geometryRef.current;
    if (!geometry) return;
    if (event.key === "Escape") {
      setSelectedCell(null);
      setHoverCell(null);
      return;
    }
    const current = hoverCell ?? selectedCell ?? 0;
    const row = Math.floor(current / geometry.width);
    const col = current % geometry.width;
    const next = {
      ArrowUp: [row - 1, col],
      ArrowDown: [row + 1, col],
      ArrowLeft: [row, col - 1],
      ArrowRight: [row, col + 1],
    }[event.key];
    if (next) {
      event.preventDefault();
      const [nextRow, nextCol] = next;
      if (nextRow >= 0 && nextRow < geometry.height && nextCol >= 0 && nextCol < geometry.width) {
        const cell = nextRow * geometry.width + nextCol;
        setHoverCell(cell);
      }
    } else if (event.key === "Enter" && hoverCell != null) {
      setSelectedCell(hoverCell);
    }
  };

  return (
    <>
      <div className="live-map">
        <canvas ref={backgroundRef} className="iso-background-layer" aria-hidden="true" />
        <canvas ref={cityRef} className="iso-city-layer" aria-hidden="true" />
        <canvas
          ref={interactionRef}
          className="iso-interaction-layer"
          role="application"
          tabIndex={0}
          aria-label="Изометрическая карта города. Наведите или нажмите на клетку. Используйте стрелки для выбора, Enter для просмотра, Escape для закрытия."
          onPointerMove={(event) => setHoverCell(cellAtPointer(event.clientX, event.clientY, event.currentTarget))}
          onPointerLeave={() => setHoverCell(null)}
          onClick={(event) => {
            const cell = cellAtPointer(event.clientX, event.clientY, event.currentTarget);
            if (cell != null) setSelectedCell(cell);
          }}
          onKeyDown={onMapKeyDown}
        />
        {hoverCell != null && frame && (
          <div className="iso-hover-chip" aria-live="polite">
            Клетка {hoverCell + 1} · {BUILDING_LABELS[frame.type_grid?.[Math.floor(hoverCell / width)]?.[hoverCell % width] ?? 0]}
          </div>
        )}
      </div>
      {selectedCell != null && frame && (
        <section className="iso-inspector" aria-live="polite">
          <div className="iso-inspector-heading">
            <div>
              <span className="eyebrow">ВЫБРАННАЯ КЛЕТКА</span>
              <h3>{BUILDING_LABELS[selectedType] ?? "Здание"} · #{selectedCell + 1}</h3>
            </div>
            <button className="icon-button" aria-label="Закрыть сведения о клетке" onClick={() => setSelectedCell(null)}>×</button>
          </div>
          <div className="iso-inspector-stats">
            <span><small>Позиция</small><b>{selectedRow + 1}, {selectedCol + 1}</b></span>
            <span><small>Этажи</small><b>{selectedFloors || "—"}</b></span>
            <span><small>Качество</small><b>{selectedType > 0 ? `${selectedQuality + 1}/3` : "—"}</b></span>
            <span><small>Крыша</small><b>{selectedType > 0 ? `${selectedRoof}/3` : "—"}</b></span>
            <span><small>Агент</small><b>{selectedOwner >= 0 ? `#${selectedOwner + 1}` : "—"}</b></span>
          </div>
        </section>
      )}
    </>
  );
}

function agentColor(agentId: number) {
  return `hsl(${(agentId * 137.5 + 6) % 360} 72% 60%)`;
}

function agentColorAlpha(agentId: number, alpha: number) {
  return `hsla(${(agentId * 137.5 + 6) % 360}, 72%, 60%, ${alpha})`;
}

function Field({ label, children }: { label: string; children: ReactNode }) {
  return <label className="field"><span>{label}</span>{children}</label>;
}

function HistoryPage({
  branches,
  job,
  onRefresh,
}: {
  branches: Branch[];
  job: JobStatus;
  onRefresh: () => Promise<void>;
}) {
  const { notify, confirm, prompt } = useFeedback();
  const [branch, setBranch] = useState("main");
  const [commits, setCommits] = useState<Record<string, unknown>[]>([]);
  const [refs, setRefs] = useState<string[]>([]);
  const [leftRef, setLeftRef] = useState("");
  const [rightRef, setRightRef] = useState("");
  const [referenceRef, setReferenceRef] = useState("");
  const [seedBase, setSeedBase] = useState(9_000_000);
  const [seedCount, setSeedCount] = useState(5);
  const [name, setName] = useState("");
  const [fromRef, setFromRef] = useState("");
  const [resetRef, setResetRef] = useState("");
  const [fastForwardRef, setFastForwardRef] = useState("");
  const [maintenance, setMaintenance] = useState("");
  const [gcToken, setGcToken] = useState("");
  const [error, setError] = useState("");
  const [busyAction, setBusyAction] = useState("");
  const refBranches = branches.filter((item) => item.head_commit);
  const branchKey = refBranches
    .map((item) => `${item.name}:${item.head_commit}`)
    .join("|");
  useEffect(() => {
    if (!branches.some((item) => item.name === branch) && branches.length) setBranch(branches[0].name);
  }, [branches, branch]);
  const load = useCallback(async () => {
    if (!branch) return;
    try {
      const result = await api<{ commits: Record<string, unknown>[] }>(`/api/history?branch=${encodeURIComponent(branch)}`);
      setCommits(result.commits);
      setError("");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    }
  }, [branch]);
  useEffect(() => { void load(); }, [load]);
  useEffect(() => {
    let active = true;
    const loadRefs = async () => {
      try {
        const lists = await Promise.all(
          refBranches
            .map((item) =>
              api<{ commits: { generation: number }[] }>(
                `/api/history?branch=${encodeURIComponent(item.name)}`,
              ).then((result) =>
                result.commits.map((commit) => `${item.name}@${commit.generation}`),
              ),
            ),
        );
        if (!active) return;
        const nextRefs = [...new Set(lists.flat())];
        setRefs(nextRefs);
        setLeftRef((current) => nextRefs.includes(current) ? current : nextRefs[0] ?? "");
        setRightRef((current) => nextRefs.includes(current) ? current : nextRefs[1] ?? nextRefs[0] ?? "");
        setReferenceRef((current) => nextRefs.includes(current) ? current : nextRefs[0] ?? "");
      } catch (reason) {
        if (active) setError(reason instanceof Error ? reason.message : String(reason));
      }
    };
    void loadRefs();
    return () => { active = false; };
  }, [branchKey]);
  const mutate = async (
    path: string,
    body: unknown,
    successMessage: string,
    method = "POST",
    undo?: { label: string; onSelect: () => void },
  ) => {
    if (busyAction) return;
    setBusyAction(path);
    try {
      const deletingBranch = method === "DELETE" && path.startsWith("/api/branches/");
      await api(path, {
        method,
        ...(body === undefined ? {} : { body: JSON.stringify(body) }),
      });
      setError("");
      notify({ kind: "success", message: successMessage, action: undo });
      if (path === "/api/gc") setGcToken("");
      if (deletingBranch) {
        await onRefresh();
      } else {
        await Promise.all([load(), onRefresh()]);
      }
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setBusyAction("");
    }
  };
  const verify = async () => {
    if (busyAction) return;
    setBusyAction("verify");
    try {
      const report = await api("/api/verify");
      setMaintenance(JSON.stringify(report, null, 2));
      setError("");
      notify({ kind: "success", message: "Проверка хранилища завершена." });
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setBusyAction("");
    }
  };
  const previewGc = async () => {
    if (busyAction) return;
    setBusyAction("preview-gc");
    try {
      const result = await api<{ confirmation_token: string }>("/api/gc/preview");
      setGcToken(result.confirmation_token);
      setMaintenance(JSON.stringify(result, null, 2));
      setError("");
      notify({ kind: "info", message: "Предпросмотр готов. Проверьте список перед подтверждением очистки." });
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setBusyAction("");
    }
  };
  const compare = async () => {
    if (busyAction || !leftRef || !rightRef || !referenceRef) return;
    setBusyAction("compare");
    try {
      await api("/api/compare", {
        method: "POST",
        body: JSON.stringify({
          left_ref: leftRef,
          right_ref: rightRef,
          reference_ref: referenceRef,
          seed_base: seedBase,
          seed_count: seedCount,
        }),
      });
      setError("");
      notify({ kind: "info", message: "Сравнение запущено. Его прогресс и журнал доступны в панели задачи." });
      await onRefresh();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setBusyAction("");
    }
  };
  return (
    <div className="page-content">
      <div className="page-heading"><div><p className="eyebrow">ВЕТКИ И СНИМКИ</p><h1>История обучения</h1><p className="muted">Управляйте ветками, просматривайте поколения и проверяйте целостность.</p></div><button className="button secondary" onClick={() => void onRefresh()}>↻ Обновить</button></div>
      {error && (
        <div className="alert" role="alert">
          <span>{error}</span>
          <button className="alert-close" aria-label="Закрыть сообщение" onClick={() => setError("")}>×</button>
        </div>
      )}
      <div className="two-column">
        <section className="panel">
          <div className="section-head"><div><p className="eyebrow">ИСТОРИЯ</p><h2>Снимки ветки</h2></div></div>
          <Field label="Выбранная ветка">
            <select value={branch} onChange={(event) => setBranch(event.target.value)}>
              {branches.map((item) => <option key={item.name} value={item.name}>{item.name} {item.generation == null ? "· пустая" : `· gen ${item.generation}`}</option>)}
            </select>
          </Field>
          <button
            className="button secondary"
            disabled={Boolean(busyAction) || job.state === "running" || job.state === "stopping"}
            onClick={() => void mutate("/api/checkout", { branch }, "Ветка стала активной.")}
          >
            {busyAction === "/api/checkout" ? "Переключение…" : "Сделать активной"}
          </button>
          <div className="button-row branch-actions">
            <button className="button secondary" disabled={Boolean(busyAction)} onClick={async () => {
              const nextName = (await prompt({
                title: "Переименовать ветку",
                message: "Укажите новое имя. История ветки сохранится.",
                label: "Новое имя",
                initialValue: branch,
              }))?.trim();
              if (nextName && nextName !== branch) {
                void mutate(`/api/branches/${encodeURIComponent(branch)}`, { new_name: nextName }, "Ветка переименована.", "PUT");
              }
            }}>Переименовать</button>
            <button className="button secondary" disabled={Boolean(busyAction) || job.state === "running" || job.state === "stopping"} onClick={() => {
              const archived = !branches.find((item) => item.name === branch)?.archived_at;
              void mutate(`/api/branches/${encodeURIComponent(branch)}/archive`, { archived }, archived ? "Ветка архивирована." : "Ветка восстановлена из архива.");
            }}>{branches.find((item) => item.name === branch)?.archived_at ? "Снять архив" : "Архивировать"}</button>
            <button className="button danger" disabled={Boolean(busyAction) || job.state === "running" || job.state === "stopping"} onClick={async () => {
              if (await confirm({
                title: "Переместить ветку в корзину?",
                message: `Ветка «${branch}» исчезнет из списка, но её снимки сохранятся до очистки.`,
                confirmLabel: "В корзину",
                danger: true,
              })) {
                void mutate(
                  `/api/branches/${encodeURIComponent(branch)}`,
                  undefined,
                  "Ветка перемещена в корзину.",
                  "DELETE",
                  {
                    label: "Восстановить",
                    onSelect: () => void mutate(
                      `/api/branches/${encodeURIComponent(branch)}/restore`,
                      undefined,
                      "Ветка восстановлена.",
                    ),
                  },
                );
              }
            }}>В корзину</button>
          </div>
          <div className="inline-form reset-form">
            <Field label="Переместить указатель вперёд до">
              <input value={fastForwardRef} onChange={(event) => setFastForwardRef(event.target.value)} placeholder="main@40" />
            </Field>
            <button className="button secondary" disabled={!fastForwardRef.trim() || Boolean(busyAction) || job.state === "running" || job.state === "stopping"} onClick={async () => {
              if (await confirm({
                title: "Продвинуть указатель ветки?",
                message: `Ветка «${branch}» будет перемещена до ${fastForwardRef}. Обратное перемещение этим действием не выполняется.`,
                confirmLabel: "Продвинуть",
              })) {
                void mutate(`/api/branches/${encodeURIComponent(branch)}/fast-forward`, { to_ref: fastForwardRef }, "Указатель ветки продвинут.");
              }
            }}>Fast-forward</button>
          </div>
          <div className="inline-form">
            <Field label="Новая ветка"><input value={name} onChange={(event) => setName(event.target.value)} placeholder="experiment" required /></Field>
            <Field label="Создать от (необязательно)"><input value={fromRef} onChange={(event) => setFromRef(event.target.value)} placeholder="main@40" /></Field>
            <button
              className="button primary"
              disabled={!name.trim() || Boolean(busyAction)}
              onClick={() => void mutate("/api/branches", { name, from_ref: fromRef || null }, "Новая ветка создана.")}
            >
              {busyAction === "/api/branches" ? "Создание…" : "Создать ветку"}
            </button>
          </div>
          <div className="inline-form reset-form">
            <Field label="Переместить указатель к"><input value={resetRef} onChange={(event) => setResetRef(event.target.value)} placeholder="main@20" required /></Field>
            <button
              className="button secondary"
              disabled={!resetRef.trim() || Boolean(busyAction) || job.state === "running" || job.state === "stopping"}
              onClick={async () => {
                if (await confirm({
                  title: "Переместить указатель ветки?",
                  message: `Ветка «${branch}» будет указывать на ${resetRef}.`,
                  confirmLabel: "Переместить",
                })) {
                  void mutate("/api/reset", { branch, to_ref: resetRef }, "Указатель ветки перемещён.");
                }
              }}
            >
              {busyAction === "/api/reset" ? "Перемещение…" : "Переместить указатель"}
            </button>
          </div>
          <p className="hint">Перемещение меняет текущую точку ветки; недостижимые снимки удаляются отдельно через безопасную очистку.</p>
          <div className="table-wrap">
            <table><thead><tr><th>Поколение</th><th>Fitness</th><th>Статус</th><th>Снимок</th></tr></thead>
              <tbody>{commits.map((row) => <tr key={String(row.commit_id)}><td>{String(row.generation)}</td><td>{row.best_fitness == null ? "—" : Number(row.best_fitness).toFixed(4)}</td><td><span className="badge">{row.complete ? "Готов" : "Незавершён"}</span></td><td className="mono">{String(row.commit_id).slice(0, 10)}</td></tr>)}
                {!commits.length && <tr><td colSpan={4} className="empty-cell">В этой ветке пока нет снимков.</td></tr>}
              </tbody>
            </table>
          </div>
        </section>
        <section className="panel">
          <div className="section-head"><div><p className="eyebrow">ОБСЛУЖИВАНИЕ</p><h2>Целостность данных</h2></div></div>
          <p className="muted">Проверяйте базу и контрольные точки. Очистка предварительно показывает, что будет удалено.</p>
          <div className="button-row">
            <button className="button secondary" disabled={Boolean(busyAction)} onClick={() => void verify()}>
              {busyAction === "verify" ? "Проверка…" : "Проверить хранилище"}
            </button>
            <button className="button secondary" disabled={Boolean(busyAction) || job.state === "running" || job.state === "stopping"} onClick={() => void previewGc()}>
              {busyAction === "preview-gc" ? "Подготовка…" : "Предпросмотр очистки"}
            </button>
          </div>
          {gcToken && <button className="button danger" onClick={async () => {
            if (await confirm({
              title: "Очистить недостижимые снимки?",
              message: "Будут удалены только снимки из предпросмотра. Это действие нельзя отменить.",
              confirmLabel: "Удалить снимки",
              danger: true,
            })) {
              void mutate(
                "/api/gc",
                { confirmation_token: gcToken },
                "Очистка завершена. Удалены только недостижимые снимки из подтверждённого предпросмотра.",
              );
            }
          }} disabled={Boolean(busyAction)}>{busyAction === "/api/gc" ? "Очистка…" : "Подтвердить очистку"}</button>}
          <pre className="code-panel">{maintenance || "Проверка или предпросмотр ещё не запускались."}</pre>
        </section>
      </div>
      <section className="panel">
        <div className="section-head"><div><p className="eyebrow">СРАВНЕНИЕ СТРАТЕГИЙ</p><h2>Сравнить снимки</h2></div></div>
        <p className="muted small">Обе оценки используют одинаковые тестовые сиды и город из выбранного эталонного снимка.</p>
        <div className="field-grid compare-grid">
          <Field label="Первый снимок"><select value={leftRef} onChange={(event) => setLeftRef(event.target.value)}>{refs.map((ref) => <option key={ref}>{ref}</option>)}</select></Field>
          <Field label="Второй снимок"><select value={rightRef} onChange={(event) => setRightRef(event.target.value)}>{refs.map((ref) => <option key={ref}>{ref}</option>)}</select></Field>
          <Field label="Эталонный снимок"><select value={referenceRef} onChange={(event) => setReferenceRef(event.target.value)}>{refs.map((ref) => <option key={ref}>{ref}</option>)}</select></Field>
          <div className="field-grid nested-fields">
            <Field label="Первый тестовый сид"><input type="number" min="0" value={seedBase} onChange={(event) => setSeedBase(Number(event.target.value))} /></Field>
            <Field label="Число общих сидов"><input type="number" min="1" value={seedCount} onChange={(event) => setSeedCount(Number(event.target.value))} /></Field>
          </div>
        </div>
        <button
          className="button primary"
          disabled={!refs.length || !seedCount || busyAction !== "" || job.state === "running" || job.state === "stopping"}
          onClick={() => void compare()}
        >
          {busyAction === "compare" ? "Запуск сравнения…" : "Сравнить снимки"}
        </button>
        {job.kind === "compare" && job.result && (
          <pre className="code-panel">{JSON.stringify({
            refs: job.result.refs,
            reference_ref: job.result.reference_ref,
            seeds: job.result.seeds,
            metric_differences: job.result.metric_differences,
            config_differences: job.result.config_differences,
          }, null, 2)}</pre>
        )}
      </section>
      <TrashPanel onError={setError} />
    </div>
  );
}

function DataPage({
  runs,
  datasets,
  onRefresh,
  onError,
}: {
  runs: Run[];
  datasets: Dataset[];
  onRefresh: () => Promise<void>;
  onError: (message: string) => void;
}) {
  const { notify, confirm, prompt } = useFeedback();
  const [file, setFile] = useState<File | null>(null);
  const [name, setName] = useState("");
  const [preview, setPreview] = useState<ImportPreview | null>(null);
  const [busy, setBusy] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [busyAction, setBusyAction] = useState("");
  const [uploadProgress, setUploadProgress] = useState<number | null>(null);
  const [dragging, setDragging] = useState(false);
  const [action, setAction] = useState<"skip" | "rename" | "replace">("skip");
  const [datasetQuery, setDatasetQuery] = useState("");
  const [datasetSort, setDatasetSort] = useState<"newest" | "oldest" | "name">("newest");
  const [selectedDataset, setSelectedDataset] = useState("");
  const [datasetDetails, setDatasetDetails] = useState<{
    name: string;
    mode: string;
    run: Record<string, unknown>;
    summaries: { file: string; data: unknown }[];
  } | null>(null);
  const activeUpload = useRef<(() => void) | null>(null);
  const mounted = useRef(true);
  const visibleDatasets = datasets
    .filter((dataset) => `${dataset.name} ${dataset.mode} ${dataset.dataset_id}`.toLocaleLowerCase("ru-RU").includes(datasetQuery.trim().toLocaleLowerCase("ru-RU")))
    .sort((left, right) => datasetSort === "name"
      ? left.name.localeCompare(right.name, "ru")
      : datasetSort === "newest"
        ? right.created_at.localeCompare(left.created_at)
        : left.created_at.localeCompare(right.created_at));

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      activeUpload.current?.();
    };
  }, []);

  const trashItem = async (entityType: "run" | "dataset", id: string, label: string) => {
    if (!await confirm({
      title: "Переместить объект в корзину?",
      message: `«${label}» можно будет восстановить из корзины.`,
      confirmLabel: "В корзину",
      danger: true,
    })) return;
    setBusyAction(`trash:${id}`);
    try {
      await api("/api/data/trash", {
        method: "POST",
        body: JSON.stringify({ entity_type: entityType, ids: [id] }),
      });
      onError("");
      await onRefresh();
      notify({
        kind: "success",
        message: "Объект перемещён в корзину.",
        action: {
          label: "Восстановить",
          onSelect: () => void (async () => {
            setBusyAction(`restore:${id}`);
            try {
              await api("/api/data/restore", {
                method: "POST",
                body: JSON.stringify({ entity_type: entityType, ids: [id] }),
              });
              await onRefresh();
              notify({ kind: "success", message: "Объект восстановлен." });
            } catch (reason) {
              onError(reason instanceof Error ? reason.message : String(reason));
            } finally {
              setBusyAction("");
            }
          })(),
        },
      });
    } catch (reason) {
      onError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setBusyAction("");
    }
  };

  const previewImport = async () => {
    if (!file || busy) return;
    setBusy(true);
    setUploading(true);
    setUploadProgress(0);
    const upload = uploadArchive<ImportPreview>(
      `/api/data/import/preview?name=${encodeURIComponent(name || file.name.replace(/\.zip$/i, ""))}`,
      file,
      setUploadProgress,
    );
    activeUpload.current = upload.abort;
    try {
      const result = await upload.promise;
      if (!mounted.current) return;
      setPreview(result);
      setAction("skip");
      setUploadProgress(null);
      notify({ kind: "success", message: "Архив проверен. Просмотрите сводку и подтвердите импорт." });
      onError("");
    } catch (reason) {
      if (reason instanceof Error && reason.name === "AbortError") {
        if (mounted.current) {
          setUploadProgress(null);
          notify({ kind: "info", message: "Загрузка отменена. Набор не был импортирован." });
        }
      } else if (mounted.current) {
        onError(reason instanceof Error ? reason.message : String(reason));
        setUploadProgress(null);
      }
    } finally {
      if (activeUpload.current === upload.abort) activeUpload.current = null;
      if (mounted.current) {
        setUploading(false);
        setBusy(false);
      }
    }
  };

  const commitImport = async () => {
    if (!preview) return;
    let newName: string | undefined;
    if (action === "rename") {
      newName = (await prompt({
        title: "Импортировать копию",
        message: "Укажите новое имя набора.",
        label: "Название набора",
        initialValue: `${preview.name}-copy`,
        confirmLabel: "Продолжить",
      })) ?? undefined;
      if (!newName) return;
    }
    if (action === "replace" && !await confirm({
      title: "Заменить существующий набор?",
      message: `Данные набора «${preview.name}» будут заменены импортируемым архивом.`,
      confirmLabel: "Заменить набор",
      danger: true,
    })) return;
    setBusy(true);
    try {
      await api("/api/data/import/commit", {
        method: "POST",
        body: JSON.stringify({ token: preview.token, action, name: newName }),
      });
      setPreview(null);
      setFile(null);
      setName("");
      notify({ kind: "success", message: "Набор импортирован и добавлен в коллекцию." });
      await onRefresh();
    } catch (reason) {
      onError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setBusy(false);
    }
  };

  const activate = async (dataset: Dataset) => {
    if (dataset.mode !== "full" || busyAction) return;
    setBusyAction(`activate:${dataset.dataset_id}`);
    try {
      await api(`/api/data/datasets/${dataset.dataset_id}/activate`, { method: "POST" });
      notify({ kind: "success", message: `История «${dataset.name}» открыта.` });
      await onRefresh();
    } catch (reason) {
      onError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setBusyAction("");
    }
  };

  const inspectDataset = async (dataset: Dataset) => {
    if (busyAction) return;
    setBusyAction(`inspect:${dataset.dataset_id}`);
    try {
      const details = await api<NonNullable<typeof datasetDetails>>(
        `/api/data/datasets/${dataset.dataset_id}`,
      );
      setSelectedDataset(dataset.dataset_id);
      setDatasetDetails(details);
      notify({ kind: "success", message: `Сводка набора «${dataset.name}» загружена.` });
    } catch (reason) {
      onError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setBusyAction("");
    }
  };

  return (
    <div className="page-content">
      <div className="page-heading">
        <div><p className="eyebrow">ОБМЕН РЕЗУЛЬТАТАМИ</p><h1>Наборы данных</h1><p className="muted">Переносите эксперименты между компьютерами с помощью ZIP-архивов.</p></div>
        <button className="button secondary" disabled={Boolean(busyAction) || busy} onClick={async () => {
          setBusyAction("default-history");
          try {
            await api("/api/data/history/default", { method: "POST" });
            notify({ kind: "success", message: "Вы вернулись к локальной истории." });
            await onRefresh();
          } catch (reason) {
            onError(reason instanceof Error ? reason.message : String(reason));
          } finally {
            setBusyAction("");
          }
        }}>{busyAction === "default-history" ? "Переключение…" : "Вернуться к локальной истории"}</button>
      </div>
      <div className="two-column data-columns">
        <section className="panel">
          <div className="section-head"><div><p className="eyebrow">ИМПОРТ С КОМПЬЮТЕРА</p><h2>Добавить архив</h2></div><span className="stat-icon purple">⇣</span></div>
          <p className="muted small">Выберите ZIP, который скачали на другом компьютере. Архив проверяется до добавления в хранилище.</p>
          <Field label="Название набора"><input value={name} disabled={busy} onChange={(event) => setName(event.target.value)} placeholder="Например, experiment-gpu-02" /></Field>
          <label
            className={`upload-drop-zone ${dragging ? "dragging" : ""}`}
            onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={(event) => {
              event.preventDefault();
              setDragging(false);
              if (busy) return;
              const selected = event.dataTransfer.files[0] ?? null;
              if (!selected) return;
              if (!/\.zip$/i.test(selected.name)) {
                onError("Выберите ZIP-архив. Другие форматы не поддерживаются.");
                return;
              }
              setFile(selected);
              if (!name) setName(selected.name.replace(/\.zip$/i, ""));
              setPreview(null);
            }}
          >
            <span className="upload-drop-icon" aria-hidden="true">⇧</span>
            <span><b>{file?.name ?? "Перетащите ZIP сюда"}</b><small>или выберите файл · до 20 ГиБ</small></span>
            <input type="file" accept=".zip,application/zip" disabled={busy} onChange={(event) => {
              const selected = event.target.files?.[0] ?? null;
              if (selected && !/\.zip$/i.test(selected.name)) {
                onError("Выберите ZIP-архив. Другие форматы не поддерживаются.");
                return;
              }
              setFile(selected);
              if (selected && !name) setName(selected.name.replace(/\.zip$/i, ""));
              setPreview(null);
            }} />
          </label>
          {file && <div className="upload-file-meta"><span>{(file.size / (1024 * 1024)).toLocaleString("ru-RU", { maximumFractionDigits: 1 })} МиБ</span><button className="text-button" disabled={busy} onClick={() => { setFile(null); setPreview(null); }}>Убрать файл</button></div>}
          {uploading && (
            <div className="upload-progress" role="status" aria-live="polite">
              <div className="progress-meta"><span>{uploadProgress === 100 ? "Архив отправлен · проверяем содержимое" : "Загрузка архива"}</span><b>{uploadProgress == null ? "…" : `${uploadProgress}%`}</b></div>
              <div className={`progress-track ${uploadProgress == null ? "indeterminate" : ""}`} role={uploadProgress == null ? "status" : "progressbar"} aria-label="Загрузка архива" aria-valuemin={uploadProgress == null ? undefined : 0} aria-valuemax={uploadProgress == null ? undefined : 100} aria-valuenow={uploadProgress ?? undefined}><span style={{ width: uploadProgress == null ? undefined : `${uploadProgress}%` }} /></div>
            </div>
          )}
          <div className="button-row upload-actions">
            <button className="button primary" disabled={!file || busy} onClick={() => void previewImport()}>
              {uploading ? "Проверка…" : busy ? "Сохранение…" : "Проверить архив"}
            </button>
            {uploading && <button className="button secondary" onClick={() => activeUpload.current?.()}>Отменить загрузку</button>}
          </div>
          {preview && (
            <div className="preview-box">
              <b>Архив проверен · {preview.manifest.mode === "full" ? "полный" : "легкий"}</b>
              <p className="muted small">Задача: {String(preview.manifest.run.kind ?? "эксперимент")}</p>
              {Object.keys(preview.conflicts).length > 0 ? (
                <>
                  <p className="warning-text">Найдено совпадение. Выберите действие:</p>
                  <div className="choice-row">
                    <label><input type="radio" disabled={busy} checked={action === "skip"} onChange={() => setAction("skip")} /> Пропустить совпадающий набор</label>
                    <label><input type="radio" disabled={busy} checked={action === "rename"} onChange={() => setAction("rename")} /> Импортировать под новым именем</label>
                    <label><input type="radio" disabled={busy} checked={action === "replace"} onChange={() => setAction("replace")} /> Заменить существующий набор</label>
                  </div>
                </>
              ) : <p className="success-text">Дубликатов не найдено. Набор будет добавлен как «{preview.name}».</p>}
              <div className="button-row">
                <button className="button primary" disabled={busy} onClick={() => void commitImport()}>{busy ? "Импорт…" : action === "skip" && Object.keys(preview.conflicts).length ? "Пропустить набор" : "Импортировать набор"}</button>
                <button className="button secondary" disabled={busy} onClick={() => setPreview(null)}>Отмена</button>
              </div>
            </div>
          )}
          <div className="notice"><b>Легкий архив</b><span>JSON, CSV, изображения и параметры эксперимента.</span></div>
          <div className="notice"><b>Полный архив</b><span>Результаты плюс история обучения и контрольные точки для продолжения.</span></div>
        </section>
        <section className="panel">
          <div className="section-head"><div><p className="eyebrow">ИМПОРТИРОВАННЫЕ ДАННЫЕ</p><h2>Коллекция компьютеров</h2></div><span className="count-pill">{visibleDatasets.length} / {datasets.length}</span></div>
          <div className="dataset-toolbar">
            <Field label="Поиск наборов"><input value={datasetQuery} onChange={(event) => setDatasetQuery(event.target.value)} placeholder="Название или ID…" /></Field>
            <Field label="Сортировка"><select value={datasetSort} onChange={(event) => setDatasetSort(event.target.value as typeof datasetSort)}><option value="newest">Сначала новые</option><option value="oldest">Сначала старые</option><option value="name">По названию</option></select></Field>
          </div>
          <div className="dataset-list">
            {visibleDatasets.map((dataset) => (
              <article className="dataset-card" key={dataset.dataset_id}>
                <div className="dataset-avatar">{dataset.name.slice(0, 1).toUpperCase()}</div>
                <div className="dataset-info"><b>{dataset.name}</b><span>{dataset.mode === "full" ? "Полный архив · можно продолжить" : "Легкий архив · результаты"}</span><small>{new Date(dataset.created_at).toLocaleString()}</small></div>
                <button className="button small-button secondary" disabled={Boolean(busyAction) || busy} onClick={() => void inspectDataset(dataset)}>
                  {busyAction === `inspect:${dataset.dataset_id}` ? "Загрузка…" : "Показать сводку"}
                </button>
                {dataset.active ? <span className="badge green">Активная история</span> : dataset.mode === "full" ? (
                  <button className="button small-button secondary" disabled={Boolean(busyAction) || busy} onClick={() => void activate(dataset)}>
                    {busyAction === `activate:${dataset.dataset_id}` ? "Открытие…" : "Открыть историю"}
                  </button>
                ) : <span className="badge">Архив результатов</span>}
                <button className="button small-button danger" title={dataset.active ? "Сначала переключитесь на другую историю" : "Переместить набор в корзину"} disabled={Boolean(busyAction) || busy || dataset.active} onClick={() => void trashItem("dataset", dataset.dataset_id, dataset.name)}>В корзину</button>
              </article>
            ))}
            {!visibleDatasets.length && (
              <div className="empty-state">
                <span className="empty-icon">⇣</span>
                <b>{datasets.length ? "Ничего не найдено" : "Пока нет импортированных наборов"}</b>
                <p>{datasets.length ? "Измените поисковый запрос или сбросьте фильтр." : "Загрузите ZIP-архив, чтобы собрать результаты с другого компьютера."}</p>
                {datasets.length > 0 && <button className="button small-button secondary" onClick={() => setDatasetQuery("")}>Сбросить поиск</button>}
              </div>
            )}
          </div>
          {datasetDetails && (
            <div className="dataset-details">
              <div className="section-head compact">
                <div><p className="eyebrow">СВОДКА НАБОРА</p><h2>{datasetDetails.name}</h2></div>
                <button className="icon-button" onClick={() => { setSelectedDataset(""); setDatasetDetails(null); }} aria-label="Закрыть сводку">×</button>
              </div>
              {datasetDetails.summaries.length ? datasetDetails.summaries.map((summary) => (
                <details key={`${selectedDataset}-${summary.file}`} open>
                  <summary>{summary.file}</summary>
                  <pre className="code-panel">{JSON.stringify(summary.data, null, 2)}</pre>
                </details>
              )) : <p className="muted small">В архиве нет файлов сводных результатов.</p>}
            </div>
          )}
        </section>
      </div>
      <section className="panel">
        <GenomeCatalog onError={onError} />
      </section>
      <TrashPanel onError={onError} />
      <section className="panel">
        <div className="section-head"><div><p className="eyebrow">ЛОКАЛЬНЫЕ ЗАПУСКИ</p><h2>Архивировать результаты</h2></div></div>
        <p className="muted small">Каждый ZIP можно перенести и импортировать на основном компьютере. Полный архив включает активную историю.</p>
        <RunTable runs={runs} exportButtons searchable onTrash={(run) => void trashItem("run", run.run_id, `${kindLabel(run.kind)} ${run.run_id.slice(0, 8)}`)} />
      </section>
    </div>
  );
}

function GenomeCatalog({ onError }: { onError: (message: string) => void }) {
  const { notify, confirm, prompt } = useFeedback();
  const [genomes, setGenomes] = useState<Genome[]>([]);
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState<string[]>([]);
  const [detail, setDetail] = useState<Record<string, unknown> | null>(null);
  const [files, setFiles] = useState<File[]>([]);
  const [busy, setBusy] = useState(false);

  const load = useCallback(async () => {
    try {
      const result = await api<{ genomes: Genome[] }>(
        `/api/genomes?query=${encodeURIComponent(query)}`,
      );
      setGenomes(result.genomes);
      setSelected((current) => current.filter((id) => result.genomes.some((item) => item.genome_id === id)));
    } catch (reason) {
      onError(reason instanceof Error ? reason.message : String(reason));
    }
  }, [onError, query]);
  useEffect(() => {
    const timer = window.setTimeout(() => void load(), 250);
    return () => window.clearTimeout(timer);
  }, [load]);

  const act = async (
    label: string,
    operation: () => Promise<unknown>,
    action?: { label: string; onSelect: () => void },
  ) => {
    if (busy) return;
    setBusy(true);
    try {
      await operation();
      notify({ kind: "success", message: label, action });
      onError("");
      await load();
    } catch (reason) {
      onError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setBusy(false);
    }
  };
  const importFiles = async () => {
    if (!files.length || busy) return;
    setBusy(true);
    const failures: string[] = [];
    let imported = 0;
    try {
      for (const file of files) {
        try {
          const name = file.name.replace(/\.npz$/i, "") || "Imported genome";
          await api("/api/genomes/import?name=" + encodeURIComponent(name), {
            method: "POST",
            headers: { "Content-Type": "application/octet-stream" },
            body: file,
          });
          imported += 1;
        } catch (reason) {
          failures.push(`${file.name}: ${reason instanceof Error ? reason.message : String(reason)}`);
        }
      }
      setFiles([]);
      await load();
      notify({
        kind: failures.length ? "warning" : "success",
        message: failures.length
          ? `Импортировано ${imported} из ${imported + failures.length}; подробности указаны в сообщении об ошибке.`
          : `Импортировано геномов: ${imported}.`,
      });
      onError(failures.length ? failures.join("\n") : "");
    } finally {
      setBusy(false);
    }
  };
  const selectedIds = selected.slice(0, 2);

  return (
    <div className="genome-catalog">
      <div className="section-head">
        <div><p className="eyebrow">ГЕНОМЫ</p><h2>Каталог стратегий</h2></div>
        <span className="count-pill">{genomes.length}</span>
      </div>
      <p className="muted small">Откройте происхождение, сравните параметры или перенесите проверенный NPZ-геном между компьютерами.</p>
      <div className="inline-form genome-toolbar">
        <Field label="Поиск"><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Имя, ветка, тип…" /></Field>
        <Field label="Импорт геномов NPZ"><input key={files.length} type="file" multiple accept=".npz,application/octet-stream" onChange={(event) => setFiles(Array.from(event.target.files ?? []))} /></Field>
        <button className="button secondary" disabled={!files.length || busy} onClick={() => void importFiles()}>{busy ? "Импорт…" : `Импортировать (${files.length})`}</button>
      </div>
      <div className="button-row">
        <span className="count-pill">{selected.length} выбрано</span>
        <button
          className="button secondary"
          disabled={!genomes.length || busy}
          aria-pressed={genomes.length > 0 && genomes.every((genome) => selected.includes(genome.genome_id))}
          onClick={() => setSelected((current) => genomes.length > 0 && genomes.every((genome) => current.includes(genome.genome_id))
            ? current.filter((id) => !genomes.some((genome) => genome.genome_id === id))
            : [...new Set([...current, ...genomes.map((genome) => genome.genome_id)])])}
        >
          {genomes.length > 0 && genomes.every((genome) => selected.includes(genome.genome_id)) ? "Снять выделение" : "Выбрать все результаты"}
        </button>
        <button className="button secondary" disabled={!selected.length || busy} onClick={() => {
          for (const id of selected) {
            const link = document.createElement("a");
            link.href = `/api/genomes/${id}/export`;
            link.download = `citysim-genome-${id.slice(0, 8)}.npz`;
            document.body.appendChild(link);
            link.click();
            link.remove();
          }
        }}>Экспорт ({selected.length})</button>
        <button className="button secondary" disabled={selected.length !== 1 || busy} onClick={async () => {
          const genome = genomes.find((item) => item.genome_id === selected[0]);
          const name = await prompt({
            title: "Клонировать геном",
            message: "Создаётся независимая копия параметров выбранной стратегии.",
            label: "Имя копии",
            initialValue: `${String(genome?.metadata.name ?? "Genome")} copy`,
          });
          if (name?.trim()) void act("Клон генома создан.", () => api(`/api/genomes/${selected[0]}/clone`, { method: "POST", body: JSON.stringify({ name: name.trim() }) }));
        }}>Клонировать</button>
        <button className="button secondary" disabled={selected.length !== 2 || busy} onClick={() => {
          void act("Сравнение готово.", async () => {
            const result = await api<Record<string, unknown>>("/api/genomes/compare", {
              method: "POST",
              body: JSON.stringify({ left_id: selectedIds[0], right_id: selectedIds[1] }),
            });
            setDetail(result);
          });
        }}>Сравнить два</button>
        <button className="button danger" disabled={!selected.length || busy} onClick={async () => {
          if (await confirm({
            title: "Переместить геномы в корзину?",
            message: `Будет перемещено объектов: ${selected.length}. Их можно восстановить из корзины.`,
            confirmLabel: "В корзину",
            danger: true,
          })) {
            const ids = [...selected];
            void act(
              "Геномы перемещены в корзину. Их можно восстановить ниже.",
              () => api("/api/data/trash", { method: "POST", body: JSON.stringify({ entity_type: "genome", ids }) }).then(() => setSelected([])),
              {
                label: "Восстановить",
                onSelect: () => void act(
                  "Геномы восстановлены.",
                  () => api("/api/data/restore", { method: "POST", body: JSON.stringify({ entity_type: "genome", ids }) }),
                ),
              },
            );
          }
        }}>В корзину ({selected.length})</button>
      </div>
      <div className="table-wrap">
        <table>
          <thead><tr><th></th><th>Название</th><th>Источник</th><th>Ветка / поколение</th><th>Действия</th></tr></thead>
          <tbody>{genomes.map((genome) => (
            <tr key={genome.genome_id}>
              <td><input aria-label={`Выбрать ${String(genome.metadata.name ?? genome.genome_id)}`} type="checkbox" checked={selected.includes(genome.genome_id)} onChange={(event) => setSelected((current) => event.target.checked ? [...current, genome.genome_id] : current.filter((id) => id !== genome.genome_id))} /></td>
              <td><b>{String(genome.metadata.name ?? `${genome.member_kind} genome`)}</b><small className="sub-cell mono">{genome.genome_id.slice(0, 10)}</small></td>
              <td>{genome.source_kind} · {genome.member_kind}{genome.member_index == null ? "" : ` #${genome.member_index}`}</td>
              <td>{genome.branch_name ?? "Импорт"}{genome.generation == null ? "" : ` · gen ${genome.generation}`}</td>
              <td><div className="button-row"><button className="button small-button secondary" disabled={busy} onClick={() => void act("Детали генома загружены.", async () => setDetail(await api(`/api/genomes/${genome.genome_id}`)))}>Открыть</button><button className="button small-button secondary" disabled={busy} onClick={async () => {
                const oldName = String(genome.metadata.name ?? "");
                const name = await prompt({
                  title: "Переименовать геном",
                  message: "Изменится только подпись в каталоге, параметры стратегии останутся прежними.",
                  label: "Название",
                  initialValue: oldName,
                });
                if (name?.trim() && name.trim() !== oldName) void act("Метаданные обновлены.", () => api(`/api/genomes/${genome.genome_id}`, { method: "PUT", body: JSON.stringify({ updates: { name: name.trim() } }) }));
              }}>Переименовать</button></div></td>
            </tr>
          ))}{!genomes.length && <tr><td colSpan={5} className="empty-cell">
            {query ? <>По запросу «{query}» ничего не найдено. <button className="text-button" onClick={() => setQuery("")}>Сбросить поиск</button></> : "Геномов пока нет. После первого checkpoint каталог заполнится автоматически; также можно импортировать NPZ-файл выше."}
          </td></tr>}</tbody>
        </table>
      </div>
      {detail && <div className="dataset-details">
        <div className="section-head compact"><div><p className="eyebrow">КАРТОЧКА ГЕНОМА / СРАВНЕНИЕ</p><h3>{String(detail.genome_id ?? "Результат сравнения")}</h3></div><button className="icon-button" aria-label="Закрыть детали" onClick={() => setDetail(null)}>×</button></div>
        <pre className="code-panel">{JSON.stringify(detail, null, 2)}</pre>
      </div>}
    </div>
  );
}

function TrashPanel({
  onError,
}: {
  onError: (message: string) => void;
}) {
  const { notify, confirm } = useFeedback();
  const [items, setItems] = useState<TrashItem[]>([]);
  const [busyId, setBusyId] = useState("");
  const load = useCallback(async () => {
    try {
      const result = await api<{ items: TrashItem[] }>("/api/data/trash");
      setItems(result.items);
    } catch (reason) {
      onError(reason instanceof Error ? reason.message : String(reason));
    }
  }, [onError]);
  useEffect(() => { void load(); }, [load]);
  const change = async (item: TrashItem, action: "restore" | "purge") => {
    const id = item.entity_type === "genome" ? item.genome_id : item.id ?? item.name;
    if (!id) return;
    if (action === "purge" && !await confirm({
      title: "Удалить объект навсегда?",
      message: `«${item.name ?? id}» будет удалён без возможности восстановления.`,
      confirmLabel: "Удалить навсегда",
      danger: true,
    })) return;
    setBusyId(`${action}:${id}`);
    try {
      await api(`/api/data/${action === "restore" ? "restore" : "purge"}`, {
        method: "POST",
        body: JSON.stringify({ entity_type: item.entity_type, ids: [id] }),
      });
      notify({ kind: action === "restore" ? "success" : "warning", message: action === "restore" ? "Объект восстановлен." : "Объект удалён окончательно." });
      onError("");
      await load();
    } catch (reason) {
      onError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setBusyId("");
    }
  };
  return (
    <section className="panel">
      <div className="section-head"><div><p className="eyebrow">ВОССТАНОВЛЕНИЕ</p><h2>Корзина</h2></div><button className="button small-button secondary" onClick={() => void load()}>Обновить</button></div>
      <p className="muted small">Удалённые объекты сохраняются до ручной безвозвратной очистки.</p>
      <div className="table-wrap"><table><thead><tr><th>Объект</th><th>Тип</th><th>Удалён</th><th>Действия</th></tr></thead><tbody>
        {items.map((item) => {
          const id = item.entity_type === "genome" ? item.genome_id : item.id ?? item.name;
          if (!id) return null;
          return <tr key={`${item.entity_type}:${id}`}><td>{item.name ?? id}</td><td>{item.entity_type}</td><td>{new Date(item.deleted_at).toLocaleString()}</td><td><div className="button-row"><button className="button small-button secondary" disabled={Boolean(busyId)} onClick={() => void change(item, "restore")}>Восстановить</button><button className="button small-button danger" disabled={Boolean(busyId)} onClick={() => void change(item, "purge")}>Удалить навсегда</button></div></td></tr>;
        })}
        {!items.length && <tr><td colSpan={4} className="empty-cell">Корзина пуста.</td></tr>}
      </tbody></table></div>
    </section>
  );
}

function RunTable({
  runs,
  exportButtons = false,
  onTrash,
  searchable = false,
}: {
  runs: Run[];
  exportButtons?: boolean;
  onTrash?: (run: Run) => void;
  searchable?: boolean;
}) {
  const [query, setQuery] = useState("");
  const [sortOrder, setSortOrder] = useState<"newest" | "oldest">("newest");
  const normalizedQuery = query.trim().toLocaleLowerCase("ru-RU");
  const visibleRuns = runs
    .filter((run) => !normalizedQuery || `${run.run_id} ${kindLabel(run.kind)} ${stateLabel(run.state)} ${parameterSummary(run.request)}`.toLocaleLowerCase("ru-RU").includes(normalizedQuery))
    .sort((left, right) => sortOrder === "newest"
      ? right.created_at.localeCompare(left.created_at)
      : left.created_at.localeCompare(right.created_at));
  const columnCount = 4 + Number(exportButtons) + Number(Boolean(onTrash));
  return (
    <>
      {searchable && <div className="run-table-toolbar">
        <Field label="Поиск запусков"><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="ID, тип, статус, параметры…" /></Field>
        <Field label="Сортировка"><select value={sortOrder} onChange={(event) => setSortOrder(event.target.value as typeof sortOrder)}><option value="newest">Сначала новые</option><option value="oldest">Сначала старые</option></select></Field>
        <span className="muted small">{visibleRuns.length} / {runs.length}</span>
      </div>}
      <div className="table-wrap">
        <table>
          <thead><tr><th>Задача</th><th>Состояние</th><th>Дата</th><th>Параметры</th>{exportButtons && <th>Экспорт</th>}{onTrash && <th>Действия</th>}</tr></thead>
          <tbody>
          {visibleRuns.map((run) => (
            <tr key={run.run_id}>
              <td><span className="run-kind"><span className={`kind-dot ${run.kind}`} />{kindLabel(run.kind)}</span><small className="sub-cell">{run.run_id.slice(0, 8)}</small></td>
              <td><span className={`badge ${run.state === "completed" ? "green" : run.state === "failed" ? "red" : ""}`}>{stateLabel(run.state)}</span></td>
              <td>{new Date(run.created_at).toLocaleString()}</td>
              <td className="muted">{parameterSummary(run.request)}</td>
              {exportButtons && (
                <td>
                  {run.state === "running" || run.state === "stopping" ? (
                    <span className="muted small">Экспорт после завершения</span>
                  ) : (
                    <>
                      <a
                        className="download-link"
                        href={`/api/data/runs/${run.run_id}/export?mode=light`}
                        title="Результаты и параметры без истории обучения"
                      >
                        Лёгкий ZIP
                      </a>
                      <a
                        className="download-link"
                        href={`/api/data/runs/${run.run_id}/export?mode=full`}
                        title="Результаты вместе с историей и checkpoint"
                      >
                        Полный ZIP
                      </a>
                    </>
                  )}
                </td>
              )}
              {onTrash && <td><button className="button small-button danger" disabled={run.state === "running" || run.state === "stopping"} onClick={() => onTrash(run)}>В корзину</button></td>}
            </tr>
          ))}
          {!visibleRuns.length && <tr><td colSpan={columnCount} className="empty-cell">{runs.length ? "По запросу ничего не найдено. Измените условия поиска." : "Запусков пока нет."}</td></tr>}
          </tbody>
        </table>
      </div>
    </>
  );
}

function kindLabel(kind: string) {
  return ({ demo: "Демо", train: "Обучение", experiment: "Эксперимент", compare: "Сравнение" } as Record<string, string>)[kind] ?? kind;
}

function stateLabel(state: string) {
  return ({ completed: "Завершён", stopped: "Остановлен", failed: "Ошибка", running: "В работе", stopping: "Остановка" } as Record<string, string>)[state] ?? state;
}

function parameterSummary(request: Record<string, unknown>) {
  const defaultRounds = request.quick === true ? 6 : 20;
  const rounds = typeof request.rounds === "number" ? request.rounds : defaultRounds;
  const values = [
    request.experiment,
    request.branch,
    request.policy,
    request.seeds,
    `${rounds} раундов`,
  ]
    .filter((value) => typeof value === "string");
  return values.length ? values.join(" · ") : "—";
}

export default App;
