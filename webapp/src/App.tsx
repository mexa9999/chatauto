import { useCallback, useEffect, useRef, useState } from "react";
import type { FormEvent, ReactNode } from "react";

type Page = "overview" | "launch" | "history" | "data" | "live";
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
  owner_grid?: number[][];
  agent_id?: number | null;
  action?: { cell: number | null; type_id: number; quality: number; demolish: boolean } | null;
  accepted?: boolean | null;
};

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers);
  if (typeof init?.body === "string") headers.set("Content-Type", "application/json");
  const response = await fetch(path, { ...init, headers });
  const body: unknown = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail =
      typeof body === "object" && body !== null && "detail" in body
        ? (body as { detail: unknown }).detail
        : response.statusText;
    throw new Error(
      typeof detail === "string" ? detail : JSON.stringify(detail),
    );
  }
  return body as T;
}

function App() {
  const [page, setPage] = useState<Page>("overview");
  const [runs, setRuns] = useState<Run[]>([]);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [branches, setBranches] = useState<Branch[]>([]);
  const [activeBranch, setActiveBranch] = useState("main");
  const [job, setJob] = useState<JobStatus>({ job_id: null, state: "idle" });
  const [error, setError] = useState("");

  const refresh = useCallback(async () => {
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
      setError("");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    }
  }, []);

  useEffect(() => {
    void refresh();
    const timer = window.setInterval(() => void refresh(), 2500);
    return () => window.clearInterval(timer);
  }, [refresh]);

  const start = async (request: Record<string, unknown>) => {
    try {
      const result = await api<{ job_id: string }>("/api/jobs", {
        method: "POST",
        body: JSON.stringify(request),
      });
      setPage(request.kind === "demo" ? "live" : "overview");
      await refresh();
      return result.job_id;
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
      return null;
    }
  };

  const stop = async () => {
    if (!job.job_id) return;
    try {
      await api(`/api/jobs/${job.job_id}/stop`, { method: "POST" });
      await refresh();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
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
          <NavButton active={page === "overview"} onClick={() => setPage("overview")} icon="⌂">
            Обзор
          </NavButton>
          <NavButton active={page === "launch"} onClick={() => setPage("launch")} icon="▶">
            Запуски
          </NavButton>
          <NavButton active={page === "history"} onClick={() => setPage("history")} icon="⌁">
            История
          </NavButton>
          <NavButton active={page === "data"} onClick={() => setPage("data")} icon="▤">
            Данные
          </NavButton>
        </nav>
        <div className="sidebar-bottom">
          <span className={`connection-dot ${job.state === "running" ? "busy" : ""}`} />
          <span>{job.state === "running" ? "Выполняется задача" : "Локальный сервер"}</span>
          <small>127.0.0.1</small>
        </div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <div className="breadcrumb">CitySim <span>/</span> {pageTitle(page)}</div>
          <div className="topbar-actions">
            <span className="branch-pill">Ветка <b>{activeBranch}</b></span>
            <button className="icon-button" title="Обновить данные" onClick={() => void refresh()}>
              ↻
            </button>
          </div>
        </header>
        {error && (
          <div className="alert" role="alert">
            <span>{error}</span>
            <button className="alert-close" onClick={() => setError("")} aria-label="Закрыть">
              ×
            </button>
          </div>
        )}
        {page === "overview" && (
          <Overview
            runs={runs}
            datasets={datasets}
            job={job}
            onNavigate={setPage}
            onStop={() => void stop()}
          />
        )}
        {page === "launch" && (
          <LaunchPage
            branches={branches}
            activeBranch={activeBranch}
            onStart={start}
            onError={setError}
          />
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
  icon: string;
  children: React.ReactNode;
  onClick: () => void;
}) {
  return (
    <button className={`nav-item ${active ? "active" : ""}`} onClick={onClick}>
      <span className="nav-icon">{icon}</span>
      {children}
    </button>
  );
}

function Overview({
  runs,
  datasets,
  job,
  onNavigate,
  onStop,
}: {
  runs: Run[];
  datasets: Dataset[];
  job: JobStatus;
  onNavigate: (page: Page) => void;
  onStop: () => void;
}) {
  const completed = runs.filter((run) => run.state === "completed").length;
  const running = job.state === "running" || job.state === "stopping";
  const progress = job.progress;
  const ratio =
    progress?.generations && progress.generations > 0
      ? Math.min((progress.generation ?? 0) / progress.generations, 1)
      : 0;
  return (
    <div className="page-content">
      <div className="page-heading">
        <div>
          <p className="eyebrow">ЛАБОРАТОРИЯ ЭКСПЕРИМЕНТОВ</p>
          <h1>Добро пожаловать в CitySim</h1>
          <p className="muted">Запускайте симуляции, сравнивайте результаты и собирайте данные с разных компьютеров.</p>
        </div>
        <button className="button primary" onClick={() => onNavigate("launch")}>
          <span>＋</span> Новый запуск
        </button>
      </div>

      <div className="stat-grid">
        <StatCard label="Всего запусков" value={runs.length} caption="В этом рабочем пространстве" icon="◷" />
        <StatCard label="Успешные" value={completed} caption="Завершились без ошибок" icon="✓" tone="green" />
        <StatCard label="Импортированные наборы" value={datasets.length} caption="Собраны с других устройств" icon="⇣" tone="purple" />
        <StatCard label="Активная задача" value={running ? "В работе" : "Нет"} caption={running ? (job.kind ?? "Задача") : "Можно запускать эксперимент"} icon="●" tone={running ? "orange" : "blue"} />
      </div>

      <section className="panel current-panel">
        <div className="section-head">
          <div>
            <p className="eyebrow">МОНИТОРИНГ</p>
            <h2>{running ? "Текущий запуск" : "Состояние системы"}</h2>
          </div>
          {job.kind === "demo" && job.job_id && (
            <button className="button secondary" onClick={() => onNavigate("live")}>
              Открыть просмотр
            </button>
          )}
          {running && job.kind !== "demo" && (
            <button className="button danger" onClick={onStop}>Остановить после сохранения</button>
          )}
        </div>
        {running ? (
          <>
            <div className="progress-meta">
              <span>{progress?.phase ?? "Запуск задачи"}</span>
              <b>{progress?.generation ?? 0} / {progress?.generations ?? "—"}</b>
            </div>
            <div className="progress-track"><span style={{ width: `${ratio * 100}%` }} /></div>
            <p className="muted small">
              {progress?.eta_seconds == null
                ? "Прогресс обновляется автоматически"
                : `Осталось примерно ${Math.ceil(progress.eta_seconds)} сек.`}
            </p>
          </>
        ) : (
          <div className="empty-inline">
            <span className="empty-icon">✦</span>
            <div><b>Готово к работе</b><p className="muted">Запустите демо, обучение или пакет экспериментов.</p></div>
            <button className="button secondary" onClick={() => onNavigate("launch")}>Открыть запуск</button>
          </div>
        )}
      </section>

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
  onStart,
  onError,
}: {
  branches: Branch[];
  activeBranch: string;
  onStart: (request: Record<string, unknown>) => Promise<string | null>;
  onError: (message: string) => void;
}) {
  const [kind, setKind] = useState<"demo" | "train" | "experiment">("experiment");
  const [branch, setBranch] = useState(activeBranch);
  const [experiment, setExperiment] = useState("e1");
  const [policy, setPolicy] = useState("manual");
  const [shock, setShock] = useState("");
  const [quick, setQuick] = useState(true);
  const [generations, setGenerations] = useState(100);
  const [episodes, setEpisodes] = useState(10);
  const [population, setPopulation] = useState(32);
  const [agents, setAgents] = useState(10);
  const [workers, setWorkers] = useState("");
  const [seed, setSeed] = useState(0);
  const [seeds, setSeeds] = useState("0-2");
  const [grid, setGrid] = useState("12x12");
  const [genomeRef, setGenomeRef] = useState("");
  const [checkpoints, setCheckpoints] = useState<DemoCheckpoint[]>([]);
  const [resume, setResume] = useState(false);
  const [busy, setBusy] = useState(false);

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
    setBusy(true);
    const result = await onStart({
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
      n_agents: agents,
      workers: workers ? Number(workers) : null,
      seed,
      seeds,
      grid,
      speed_ms: 80,
      eval_seeds: 5,
      checkpoint_every: 10,
    });
    setBusy(false);
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
              ["experiment", "Эксперимент", "Пакет E1–E11 и повторные сиды"],
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
                <input value={grid} onChange={(event) => setGrid(event.target.value)} placeholder="12x12" />
              </Field>
              <Field label="Сид"><input type="number" min="0" value={seed} onChange={(event) => setSeed(Number(event.target.value))} /></Field>
              <Field label="Сценарий">
                <select value={shock} onChange={(event) => setShock(event.target.value)}>
                  <option value="">Без сценария</option><option value="industry_removal">Удаление промышленности</option>
                  <option value="population_boom">Рост населения</option><option value="earthquake">Землетрясение</option>
                </select>
              </Field>
            </div>
            <label className="check-row"><input type="checkbox" checked={quick} onChange={(event) => setQuick(event.target.checked)} /> Быстрая конфигурация</label>
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
                  </>
                ) : (
                  <>
                    <Field label="Эксперимент">
                      <select value={experiment} onChange={(event) => setExperiment(event.target.value)}>
                        {Array.from({ length: 11 }, (_, i) => <option key={i + 1} value={`e${i + 1}`}>E{i + 1}</option>)}
                        <option value="all">Все эксперименты</option>
                      </select>
                    </Field>
                    <Field label="Сиды (например, 0-5 или 0,2,4)"><input value={seeds} onChange={(event) => setSeeds(event.target.value)} /></Field>
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
                <Field label="Поколений"><input type="number" min="1" value={generations} onChange={(event) => setGenerations(Number(event.target.value))} /></Field>
                <Field label="Эпизодов на поколение"><input type="number" min="1" value={episodes} onChange={(event) => setEpisodes(Number(event.target.value))} /></Field>
                <Field label="Размер популяции"><input type="number" min="8" step="4" value={population} onChange={(event) => setPopulation(Number(event.target.value))} /></Field>
                <Field label="Агентов в городе"><input type="number" min="1" value={agents} onChange={(event) => setAgents(Number(event.target.value))} /></Field>
                <Field label="CPU workers (пусто — авто)"><input type="number" min="1" value={workers} onChange={(event) => setWorkers(event.target.value)} placeholder="Авто" /></Field>
              </div>
              <p className="hint">Размер популяции должен быть кратен 4 и не меньше числа агентов. Для подробных параметров используйте CLI.</p>
            </div>
          </>
        )}
        <div className="form-footer">
          <span className="muted small">Задачи выполняются локально. Данные останутся в выбранном хранилище.</span>
          <button className="button primary" disabled={busy || (kind === "demo" && policy === "neural" && !genomeRef)}>
            {busy ? "Запуск…" : "▶  Запустить задачу"}
          </button>
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
    if (!job.job_id) return;
    try {
      await api(`/api/jobs/${job.job_id}/stop`, { method: "POST" });
      setPaused(false);
    } catch (reason) {
      onError(reason instanceof Error ? reason.message : String(reason));
    }
  };
  const downloadPng = async () => {
    if (!job.job_id || !frame) return;
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
        <span className={`badge ${running ? "running" : ""}`}>{running ? "Идёт прогон" : job.state}</span>
      </div>
      <section className="live-layout">
        <div className="panel live-map-panel">
          <div className="section-head">
            <div>
              <p className="eyebrow">ИЗОМЕТРИЧЕСКАЯ КАРТА</p>
              <h2>{frame?.round ? `Раунд ${frame.round}` : "Ожидание первого состояния"}</h2>
            </div>
            <button className="button secondary" disabled={!frame} onClick={() => void downloadPng()}>
              Скачать PNG
            </button>
          </div>
          <IsometricMap frame={frame} />
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
              <button className="button danger" disabled={!running} onClick={() => void stop()}>
                Остановить
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
                        : `${event.action.demolish ? "Снос" : "Строительство"} · клетка ${event.action.cell + 1}`
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

function IsometricMap({ frame }: { frame: LiveFrame | null }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    const typeGrid = frame?.type_grid;
    const qualityGrid = frame?.quality_grid;
    const ownerGrid = frame?.owner_grid;
    if (!canvas || !typeGrid || !qualityGrid || !ownerGrid) return;
    const draw = () => {
      const rect = canvas.getBoundingClientRect();
      if (!rect.width || !rect.height) return;
      const pixelRatio = window.devicePixelRatio || 1;
      canvas.width = Math.round(rect.width * pixelRatio);
      canvas.height = Math.round(rect.height * pixelRatio);
      const context = canvas.getContext("2d");
      if (!context) return;
      context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
      context.clearRect(0, 0, rect.width, rect.height);
      const height = frame.height ?? typeGrid.length;
      const width = frame.width ?? typeGrid[0]?.length ?? 0;
      const tileWidth = 34;
      const tileHeight = 17;
      const maxBuildingHeight = tileHeight * 2.3;
      const gridWidth = (height + width) * tileWidth / 2;
      const gridHeight = (height + width) * tileHeight / 2 + maxBuildingHeight;
      const scale = Math.min((rect.width - 32) / gridWidth, (rect.height - 32) / gridHeight);
      context.translate((rect.width - gridWidth * scale) / 2, (rect.height - gridHeight * scale) / 2);
      context.scale(scale, scale);
      const originX = (height + width) * tileWidth / 4;
      const originY = maxBuildingHeight + 8;
      const colors = ["#283444", "#4c78a8", "#f58518", "#9d755d", "#b6423c", "#54a24b"];
      const points = (cx: number, cy: number) => [
        [cx, cy - tileHeight / 2],
        [cx + tileWidth / 2, cy],
        [cx, cy + tileHeight / 2],
        [cx - tileWidth / 2, cy],
      ];
      const polygon = (vertices: number[][], color: string) => {
        context.beginPath();
        context.moveTo(vertices[0][0], vertices[0][1]);
        for (const [x, y] of vertices.slice(1)) context.lineTo(x, y);
        context.closePath();
        context.fillStyle = color;
        context.fill();
        context.strokeStyle = "#18212c";
        context.lineWidth = 0.7;
        context.stroke();
      };
      for (let diagonal = 0; diagonal < height + width - 1; diagonal++) {
        for (let row = Math.max(0, diagonal - width + 1); row <= Math.min(height - 1, diagonal); row++) {
          const col = diagonal - row;
          const type = typeGrid[row]?.[col] ?? 0;
          const quality = qualityGrid[row]?.[col] ?? 0;
          const owner = ownerGrid[row]?.[col] ?? -1;
          const cx = originX + (col - row) * tileWidth / 2;
          const cy = originY + (row + col) * tileHeight / 2;
          const ground = points(cx, cy);
          polygon(ground, type === 0 ? "#303b49" : "#374252");
          if (type <= 0) continue;
          const buildingHeight = type === 5
            ? 0
            : tileHeight * [0.9, 1.2, 1.05, 1.45][type - 1] * (0.75 + 0.15 * (Math.max(0, Math.min(2, quality)) + 1));
          if (buildingHeight > 0) {
            const top = points(cx, cy - buildingHeight);
            polygon([ground[3], ground[2], top[2], top[3]], "#293746");
            polygon([ground[2], ground[1], top[1], top[2]], "#202c39");
            polygon(top, colors[type]);
          } else {
            polygon(ground, colors[type]);
          }
          if (owner >= 0) {
            context.beginPath();
            context.arc(cx, cy - buildingHeight - 2, 3.2, 0, Math.PI * 2);
            context.fillStyle = agentColor(owner);
            context.fill();
          }
        }
      }
      const cell = frame.action?.cell;
      if (cell != null && cell >= 0) {
        const row = Math.floor(cell / width);
        const col = cell % width;
        const cx = originX + (col - row) * tileWidth / 2;
        const cy = originY + (row + col) * tileHeight / 2;
        polygon(points(cx, cy), "rgba(255,255,255,.42)");
      }
    };
    draw();
    const observer = new ResizeObserver(draw);
    observer.observe(canvas);
    return () => observer.disconnect();
  }, [frame]);
  return <div className="live-map"><canvas ref={canvasRef} aria-label="Изометрическая карта города" /></div>;
}

function agentColor(agentId: number) {
  return `hsl(${(agentId * 137.5 + 6) % 360} 72% 60%)`;
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
  const [maintenance, setMaintenance] = useState("");
  const [gcToken, setGcToken] = useState("");
  const [error, setError] = useState("");
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
  const mutate = async (path: string, body: unknown) => {
    try {
      await api(path, { method: "POST", body: JSON.stringify(body) });
      setError("");
      await Promise.all([load(), onRefresh()]);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    }
  };
  const verify = async () => {
    try { setMaintenance(JSON.stringify(await api("/api/verify"), null, 2)); }
    catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); }
  };
  const previewGc = async () => {
    try {
      const result = await api<{ confirmation_token: string }>("/api/gc/preview");
      setGcToken(result.confirmation_token);
      setMaintenance(JSON.stringify(result, null, 2));
    } catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); }
  };
  const compare = async () => {
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
      await onRefresh();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    }
  };
  return (
    <div className="page-content">
      <div className="page-heading"><div><p className="eyebrow">ВЕТКИ И СНИМКИ</p><h1>История обучения</h1><p className="muted">Управляйте ветками, просматривайте поколения и проверяйте целостность.</p></div><button className="button secondary" onClick={() => void onRefresh()}>↻ Обновить</button></div>
      {error && <div className="alert" role="alert">{error}</div>}
      <div className="two-column">
        <section className="panel">
          <div className="section-head"><div><p className="eyebrow">ИСТОРИЯ</p><h2>Снимки ветки</h2></div></div>
          <Field label="Выбранная ветка">
            <select value={branch} onChange={(event) => setBranch(event.target.value)}>
              {branches.map((item) => <option key={item.name} value={item.name}>{item.name} {item.generation == null ? "· пустая" : `· gen ${item.generation}`}</option>)}
            </select>
          </Field>
          <button className="button secondary" onClick={() => void mutate("/api/checkout", { branch })}>Сделать активной</button>
          <div className="inline-form">
            <Field label="Новая ветка"><input value={name} onChange={(event) => setName(event.target.value)} placeholder="experiment" /></Field>
            <Field label="Создать от (необязательно)"><input value={fromRef} onChange={(event) => setFromRef(event.target.value)} placeholder="main@40" /></Field>
            <button className="button primary" onClick={() => void mutate("/api/branches", { name, from_ref: fromRef || null })}>Создать</button>
          </div>
          <div className="inline-form reset-form">
            <Field label="Переместить указатель к"><input value={resetRef} onChange={(event) => setResetRef(event.target.value)} placeholder="main@20" /></Field>
            <button className="button secondary" onClick={() => void mutate("/api/reset", { branch, to_ref: resetRef })}>Переместить</button>
          </div>
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
          <div className="button-row"><button className="button secondary" onClick={() => void verify()}>Проверить хранилище</button><button className="button secondary" onClick={() => void previewGc()}>Предпросмотр очистки</button></div>
          {gcToken && <button className="button danger" onClick={() => {
            if (window.confirm("Удалить недостижимые снимки из предпросмотра?")) void mutate("/api/gc", { confirmation_token: gcToken });
          }}>Подтвердить очистку</button>}
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
        <button className="button primary" disabled={!refs.length || job.state === "running" || job.state === "stopping"} onClick={() => void compare()}>Сравнить снимки</button>
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
  const [file, setFile] = useState<File | null>(null);
  const [name, setName] = useState("");
  const [preview, setPreview] = useState<ImportPreview | null>(null);
  const [busy, setBusy] = useState(false);
  const [action, setAction] = useState<"skip" | "rename" | "replace">("skip");
  const [selectedDataset, setSelectedDataset] = useState("");
  const [datasetDetails, setDatasetDetails] = useState<{
    name: string;
    mode: string;
    run: Record<string, unknown>;
    summaries: { file: string; data: unknown }[];
  } | null>(null);

  const previewImport = async () => {
    if (!file) return;
    setBusy(true);
    try {
      const result = await api<ImportPreview>(`/api/data/import/preview?name=${encodeURIComponent(name || file.name.replace(/\.zip$/i, ""))}`, {
        method: "POST",
        headers: { "Content-Type": "application/zip" },
        body: file,
      });
      setPreview(result);
      setAction("skip");
      onError("");
    } catch (reason) {
      onError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setBusy(false);
    }
  };

  const commitImport = async () => {
    if (!preview) return;
    let newName: string | undefined;
    if (action === "rename") {
      newName = window.prompt("Название новой копии", `${preview.name}-copy`) ?? undefined;
      if (!newName) return;
    }
    if (action === "replace" && !window.confirm(`Заменить набор «${preview.name}»?`)) return;
    setBusy(true);
    try {
      await api("/api/data/import/commit", {
        method: "POST",
        body: JSON.stringify({ token: preview.token, action, name: newName }),
      });
      setPreview(null);
      setFile(null);
      setName("");
      await onRefresh();
    } catch (reason) {
      onError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setBusy(false);
    }
  };

  const activate = async (dataset: Dataset) => {
    if (dataset.mode !== "full") return;
    try {
      await api(`/api/data/datasets/${dataset.dataset_id}/activate`, { method: "POST" });
      await onRefresh();
    } catch (reason) {
      onError(reason instanceof Error ? reason.message : String(reason));
    }
  };

  const inspectDataset = async (dataset: Dataset) => {
    try {
      const details = await api<NonNullable<typeof datasetDetails>>(
        `/api/data/datasets/${dataset.dataset_id}`,
      );
      setSelectedDataset(dataset.dataset_id);
      setDatasetDetails(details);
    } catch (reason) {
      onError(reason instanceof Error ? reason.message : String(reason));
    }
  };

  return (
    <div className="page-content">
      <div className="page-heading">
        <div><p className="eyebrow">ОБМЕН РЕЗУЛЬТАТАМИ</p><h1>Наборы данных</h1><p className="muted">Переносите эксперименты между компьютерами с помощью ZIP-архивов.</p></div>
        <button className="button secondary" onClick={async () => {
          try {
            await api("/api/data/history/default", { method: "POST" });
            await onRefresh();
          } catch (reason) {
            onError(reason instanceof Error ? reason.message : String(reason));
          }
        }}>Локальная история</button>
      </div>
      <div className="two-column data-columns">
        <section className="panel">
          <div className="section-head"><div><p className="eyebrow">ИМПОРТ С КОМПЬЮТЕРА</p><h2>Добавить архив</h2></div><span className="stat-icon purple">⇣</span></div>
          <p className="muted small">Выберите ZIP, который скачали на другом компьютере. Архив проверяется до добавления в хранилище.</p>
          <Field label="Название набора"><input value={name} onChange={(event) => setName(event.target.value)} placeholder="Например, experiment-gpu-02" /></Field>
          <Field label="ZIP-архив">
            <input type="file" accept=".zip,application/zip" onChange={(event) => {
              const selected = event.target.files?.[0] ?? null;
              setFile(selected);
              if (selected && !name) setName(selected.name.replace(/\.zip$/i, ""));
              setPreview(null);
            }} />
          </Field>
          <button className="button primary" disabled={!file || busy} onClick={() => void previewImport()}>
            {busy ? "Проверка…" : "Проверить архив"}
          </button>
          {preview && (
            <div className="preview-box">
              <b>Архив проверен · {preview.manifest.mode === "full" ? "полный" : "легкий"}</b>
              <p className="muted small">Задача: {String(preview.manifest.run.kind ?? "эксперимент")}</p>
              {Object.keys(preview.conflicts).length > 0 ? (
                <>
                  <p className="warning-text">Найдено совпадение. Выберите действие:</p>
                  <div className="choice-row">
                    <label><input type="radio" checked={action === "skip"} onChange={() => setAction("skip")} /> Пропустить</label>
                    <label><input type="radio" checked={action === "rename"} onChange={() => setAction("rename")} /> Переименовать</label>
                    <label><input type="radio" checked={action === "replace"} onChange={() => setAction("replace")} /> Заменить</label>
                  </div>
                </>
              ) : <p className="success-text">Дубликатов не найдено. Набор будет добавлен как «{preview.name}».</p>}
              <div className="button-row">
                <button className="button primary" disabled={busy} onClick={() => void commitImport()}>{action === "skip" && Object.keys(preview.conflicts).length ? "Пропустить" : "Импортировать набор"}</button>
                <button className="button secondary" onClick={() => setPreview(null)}>Отмена</button>
              </div>
            </div>
          )}
          <div className="notice"><b>Легкий архив</b><span>JSON, CSV, изображения и параметры эксперимента.</span></div>
          <div className="notice"><b>Полный архив</b><span>Результаты плюс история обучения и контрольные точки для продолжения.</span></div>
        </section>
        <section className="panel">
          <div className="section-head"><div><p className="eyebrow">ИМПОРТИРОВАННЫЕ ДАННЫЕ</p><h2>Коллекция компьютеров</h2></div><span className="count-pill">{datasets.length}</span></div>
          <div className="dataset-list">
            {datasets.map((dataset) => (
              <article className="dataset-card" key={dataset.dataset_id}>
                <div className="dataset-avatar">{dataset.name.slice(0, 1).toUpperCase()}</div>
                <div className="dataset-info"><b>{dataset.name}</b><span>{dataset.mode === "full" ? "Полный архив · можно продолжить" : "Легкий архив · результаты"}</span><small>{new Date(dataset.created_at).toLocaleString()}</small></div>
                <button className="button small-button secondary" onClick={() => void inspectDataset(dataset)}>Данные</button>
                {dataset.active ? <span className="badge green">Активный</span> : dataset.mode === "full" ? <button className="button small-button secondary" onClick={() => void activate(dataset)}>Открыть</button> : <span className="badge">Импортирован</span>}
              </article>
            ))}
            {!datasets.length && <div className="empty-state"><span className="empty-icon">⇣</span><b>Пока нет импортированных наборов</b><p>Загрузите ZIP-архив, чтобы собрать результаты с другого компьютера.</p></div>}
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
        <div className="section-head"><div><p className="eyebrow">ЛОКАЛЬНЫЕ ЗАПУСКИ</p><h2>Архивировать результаты</h2></div></div>
        <p className="muted small">Каждый ZIP можно перенести и импортировать на основном компьютере. Полный архив включает активную историю.</p>
        <RunTable runs={runs} exportButtons />
      </section>
    </div>
  );
}

function RunTable({ runs, exportButtons = false }: { runs: Run[]; exportButtons?: boolean }) {
  return (
    <div className="table-wrap">
      <table>
        <thead><tr><th>Задача</th><th>Состояние</th><th>Дата</th><th>Параметры</th>{exportButtons && <th>Экспорт</th>}</tr></thead>
        <tbody>
          {runs.map((run) => (
            <tr key={run.run_id}>
              <td><span className="run-kind"><span className={`kind-dot ${run.kind}`} />{kindLabel(run.kind)}</span><small className="sub-cell">{run.run_id.slice(0, 8)}</small></td>
              <td><span className={`badge ${run.state === "completed" ? "green" : run.state === "failed" ? "red" : ""}`}>{stateLabel(run.state)}</span></td>
              <td>{new Date(run.created_at).toLocaleString()}</td>
              <td className="muted">{parameterSummary(run.request)}</td>
              {exportButtons && <td><a className="download-link" href={`/api/data/runs/${run.run_id}/export?mode=light`}>ZIP light</a><a className="download-link" href={`/api/data/runs/${run.run_id}/export?mode=full`}>Full</a></td>}
            </tr>
          ))}
          {!runs.length && <tr><td colSpan={exportButtons ? 5 : 4} className="empty-cell">Запусков пока нет.</td></tr>}
        </tbody>
      </table>
    </div>
  );
}

function kindLabel(kind: string) {
  return ({ demo: "Демо", train: "Обучение", experiment: "Эксперимент", compare: "Сравнение" } as Record<string, string>)[kind] ?? kind;
}

function stateLabel(state: string) {
  return ({ completed: "Завершён", stopped: "Остановлен", failed: "Ошибка", running: "В работе", stopping: "Остановка" } as Record<string, string>)[state] ?? state;
}

function parameterSummary(request: Record<string, unknown>) {
  const values = [request.experiment, request.branch, request.policy, request.seeds]
    .filter((value) => typeof value === "string");
  return values.length ? values.join(" · ") : "—";
}

export default App;
