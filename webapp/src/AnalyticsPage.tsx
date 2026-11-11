import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as echarts from "echarts/core";
import { BarChart, BoxplotChart, HeatmapChart, LineChart, PieChart, RadarChart, ScatterChart } from "echarts/charts";
import { GridComponent, LegendComponent, RadarComponent, TooltipComponent, VisualMapComponent } from "echarts/components";
import { CanvasRenderer } from "echarts/renderers";
import type { EChartsOption } from "echarts";
import type { EChartsType } from "echarts/core";
import { api } from "./api";
import { useFeedback } from "./Feedback";
import { useStoredState } from "./useStoredState";

echarts.use([
  BarChart,
  BoxplotChart,
  CanvasRenderer,
  GridComponent,
  HeatmapChart,
  LegendComponent,
  LineChart,
  PieChart,
  RadarChart,
  RadarComponent,
  ScatterChart,
  TooltipComponent,
  VisualMapComponent,
]);

type AnalyticsSource = {
  source_id: string;
  source_type: "run" | "dataset";
  label: string;
  kind: string;
  state: string;
  created_at: string;
  available: boolean;
};
type RoundRow = {
  round: number;
  elapsed_seconds?: number;
  city?: Record<string, number>;
};
type ActionRow = {
  seq: number;
  round: number;
  agent_id: number;
  outcome: string;
  damage: number;
  action: {
    kind: string;
    cell: number | null;
    type_id: number;
    quality: number;
    floors?: number;
    roof_quality?: number;
    cost: number;
    bid: number;
    demolish: boolean;
  };
};
type AnalyticsDetail = {
  source: AnalyticsSource;
  action_type_names: Record<string, string>;
  summary: {
    round_count?: number;
    selected_round_count?: number;
    action_count?: number;
    city?: Record<string, number>;
    selected_city?: Record<string, number>;
    agents?: Record<string, number | boolean>[];
    scores?: number[];
    metrics?: Record<string, Record<string, number> | null>;
    action_counts?: {
      by_agent?: Record<string, number>;
      by_round?: Record<string, number>;
      by_round_agent?: Record<string, number>;
      accepted_by_round_agent?: Record<string, number>;
      by_type?: Record<string, number>;
      by_outcome?: Record<string, number>;
    };
  };
  rounds: RoundRow[];
  actions: ActionRow[];
  action_metrics: {
    count: number;
    accepted: number;
    rejected: number;
    sample_count?: number;
    by_agent: Record<string, number>;
    by_round_agent: Record<string, number>;
    accepted_by_round_agent: Record<string, number>;
    by_type: Record<string, number>;
    building_count: number;
    floors_bins: { from: number; to: number; count: number }[];
    roof_quality_counts: Record<string, number>;
    cost_bins: { from: number; to: number; count: number }[];
    bid_bins: { from: number; to: number; count: number }[];
    damage_bins: { from: number; to: number; count: number }[];
  };
  actions_truncated: boolean;
  training: { series?: Record<string, string | number>[] } | null;
};
type LiveProgress = {
  phase?: string;
  generation?: number;
  generations?: number;
  completed_generations?: number;
  best_fitness?: number;
};

const metricLabels: Record<string, string> = {
  city_quality: "Качество города",
  mean_satisfaction: "Удовлетворённость",
  green_share: "Доля зелёных зон",
  built_cells: "Застроенные клетки",
  population: "Население",
  mean_utilization: "Загрузка инфраструктуры",
  type_entropy: "Разнообразие застройки",
  violations_total: "Нарушения",
  externality_damage_total: "Внешний ущерб",
  goods_balance: "Баланс товаров",
};

function sourceKey(source: AnalyticsSource) {
  return `${source.source_type}:${source.source_id}`;
}

function validStringList(value: unknown): value is string[] {
  return Array.isArray(value) && value.length <= 4 && value.every((item) => typeof item === "string");
}

function validRound(value: string) {
  return value === "" || (/^\d+$/.test(value) && Number.isSafeInteger(Number(value)));
}

function numeric(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function formatMetric(value: number | undefined, digits = 2) {
  return value == null ? "—" : value.toLocaleString("ru-RU", { maximumFractionDigits: digits });
}

function pearson(points: [number, number][]) {
  if (points.length < 2) return undefined;
  const meanX = points.reduce((sum, point) => sum + point[0], 0) / points.length;
  const meanY = points.reduce((sum, point) => sum + point[1], 0) / points.length;
  const covariance = points.reduce((sum, point) => sum + (point[0] - meanX) * (point[1] - meanY), 0);
  const varianceX = points.reduce((sum, point) => sum + (point[0] - meanX) ** 2, 0);
  const varianceY = points.reduce((sum, point) => sum + (point[1] - meanY) ** 2, 0);
  return varianceX && varianceY ? covariance / Math.sqrt(varianceX * varianceY) : undefined;
}

function trend(values: number[]) {
  if (values.length < 2) return "—";
  const delta = values[values.length - 1] - values[0];
  if (Math.abs(delta) < 1e-8) return "→ Без изменений";
  return `${delta > 0 ? "↑" : "↓"} ${formatMetric(Math.abs(delta), 3)}`;
}

function Sparkline({ values }: { values: number[] }) {
  const points = values.filter(Number.isFinite);
  if (points.length < 2) return <svg className="kpi-sparkline" viewBox="0 0 100 30" aria-hidden="true" />;
  const min = Math.min(...points);
  const max = Math.max(...points);
  const range = max - min || 1;
  const coords = points.map((value, index) => {
    const x = (index / (points.length - 1)) * 100;
    const y = 27 - ((value - min) / range) * 23;
    return `${x},${y}`;
  }).join(" ");
  return <svg className="kpi-sparkline" viewBox="0 0 100 30" aria-hidden="true"><polyline points={coords} /></svg>;
}

function ChartCard({
  title,
  eyebrow,
  option,
  empty,
  description,
}: {
  title: string;
  eyebrow: string;
  option: EChartsOption;
  empty: boolean;
  description?: string;
}) {
  const element = useRef<HTMLDivElement>(null);
  const chart = useRef<EChartsType | null>(null);

  useEffect(() => {
    if (!element.current) return;
    chart.current = echarts.init(element.current, "dark", { renderer: "canvas" });
    const resize = () => chart.current?.resize();
    const observer = new ResizeObserver(resize);
    observer.observe(element.current);
    return () => {
      observer.disconnect();
      chart.current?.dispose();
      chart.current = null;
    };
  }, []);

  useEffect(() => {
    chart.current?.setOption(option, { notMerge: true });
  }, [option]);

  const downloadPng = () => {
    if (!chart.current) return;
    const link = document.createElement("a");
    link.href = chart.current.getDataURL({ type: "png", pixelRatio: 2, backgroundColor: "#11151c" });
    link.download = `${title.toLocaleLowerCase("ru-RU").replaceAll(" ", "-")}.png`;
    link.click();
  };

  return (
    <article className="analytics-chart-card">
      <header className="analytics-chart-heading">
        <div><p className="eyebrow">{eyebrow}</p><h3>{title}</h3>{description && <small>{description}</small>}</div>
        <button className="icon-button" onClick={downloadPng} disabled={empty} title="Скачать график в PNG" aria-label={`Скачать график «${title}» в PNG`}>↓</button>
      </header>
      <div ref={element} className="analytics-chart" role="img" aria-label={description ? `${title}. ${description}` : title} />
      {empty && <div className="analytics-chart-empty" role="status">Недостаточно данных для этого графика</div>}
    </article>
  );
}

export default function AnalyticsPage({
  running,
  progress,
  onNavigate,
}: {
  running: boolean;
  progress?: LiveProgress | null;
  onNavigate: (page: "launch" | "data") => void;
}) {
  const { notify } = useFeedback();
  const reportStorageError = useCallback((message: string) => {
    notify({ kind: "warning", message });
  }, [notify]);
  const [sources, setSources] = useState<AnalyticsSource[]>([]);
  const [selected, setSelected] = useStoredState(
    "citysim.analytics.v1.selected",
    [] as string[],
    validStringList,
    reportStorageError,
  );
  const [details, setDetails] = useState<AnalyticsDetail[]>([]);
  const [metric, setMetric] = useStoredState(
    "citysim.analytics.v1.metric",
    "city_quality",
    (value): value is string => typeof value === "string" && Object.hasOwn(metricLabels, value),
    reportStorageError,
  );
  const [roundStart, setRoundStart] = useStoredState("citysim.analytics.v1.roundStart", "", (value): value is string => typeof value === "string" && validRound(value), reportStorageError);
  const [roundEnd, setRoundEnd] = useStoredState("citysim.analytics.v1.roundEnd", "", (value): value is string => typeof value === "string" && validRound(value), reportStorageError);
  const [agentId, setAgentId] = useStoredState("citysim.analytics.v1.agentId", "", (value): value is string => typeof value === "string" && (value === "" || /^\d+$/.test(value)), reportStorageError);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [retryKey, setRetryKey] = useState(0);
  const [appliedFilters, setAppliedFilters] = useState({ roundStart, roundEnd, agentId });

  const selectedKey = selected.join("|");
  const roundError = !validRound(roundStart) || !validRound(roundEnd)
    ? "Введите целые неотрицательные номера раундов."
    : roundStart && roundEnd && Number(roundStart) > Number(roundEnd)
      ? "Начальный раунд не может быть больше конечного."
      : "";

  useEffect(() => {
    const timer = window.setTimeout(() => setAppliedFilters({ roundStart, roundEnd, agentId }), 250);
    return () => window.clearTimeout(timer);
  }, [agentId, roundEnd, roundStart]);

  const load = useCallback(async (signal: AbortSignal) => {
    setLoading(true);
    try {
      const catalog = await api<{ sources: AnalyticsSource[] }>("/api/analytics", { signal });
      setSources(catalog.sources);
      const available = catalog.sources.filter((source) => source.available);
      const validKeys = new Set(available.map(sourceKey));
      const activeSelection = selected.filter((key) => validKeys.has(key));
      const nextSelection = activeSelection.length ? activeSelection : available.slice(0, 1).map(sourceKey);
      if (nextSelection.join("|") !== selectedKey) setSelected(nextSelection);
      const keys = new Set(nextSelection);
      const chosen = available.filter((source) => keys.has(sourceKey(source))).slice(0, 4);
      const query = new URLSearchParams();
      if (appliedFilters.roundStart) query.set("round_start", appliedFilters.roundStart);
      if (appliedFilters.roundEnd) query.set("round_end", appliedFilters.roundEnd);
      if (appliedFilters.agentId) query.set("agent_id", appliedFilters.agentId);
      const results = await Promise.all(chosen.map((source) =>
        api<AnalyticsDetail>(
          `/api/analytics/${source.source_type}/${source.source_id}?${query}`,
          { signal },
        ),
      ));
      if (signal.aborted) return;
      setDetails(results);
      setError("");
      setLastUpdated(new Date());
    } catch (reason) {
      if (!signal.aborted) setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      if (!signal.aborted) setLoading(false);
    }
  }, [appliedFilters, selected, selectedKey]);

  useEffect(() => {
    if (roundError) {
      setError(roundError);
      setDetails([]);
      setLoading(false);
      return;
    }
    let disposed = false;
    let timer: number | null = null;
    let controller: AbortController | null = null;
    let requestInFlight = false;
    const interval = running ? 2500 : 20_000;
    const scheduleRefresh = () => {
      if (timer !== null) window.clearTimeout(timer);
      timer = window.setTimeout(() => {
        timer = null;
        void refresh();
      }, interval);
    };
    const refresh = async () => {
      if (disposed || requestInFlight) return;
      if (document.visibilityState !== "visible") {
        scheduleRefresh();
        return;
      }
      requestInFlight = true;
      controller = new AbortController();
      try {
        await load(controller.signal);
      } finally {
        requestInFlight = false;
        if (!disposed) scheduleRefresh();
      }
    };
    const onVisibilityChange = () => {
      if (document.visibilityState !== "visible") return;
      if (timer !== null) window.clearTimeout(timer);
      timer = null;
      void refresh();
    };
    document.addEventListener("visibilitychange", onVisibilityChange);
    void refresh();
    return () => {
      disposed = true;
      document.removeEventListener("visibilitychange", onVisibilityChange);
      controller?.abort();
      if (timer !== null) window.clearTimeout(timer);
    };
  }, [load, retryKey, roundError, running]);

  const agentOptions = useMemo(() => {
    const ids = new Set<number>();
    for (const detail of details) {
      for (const agent of detail.summary.agents ?? []) {
        const id = numeric(agent.agent_id);
        if (id != null) ids.add(id);
      }
      for (const agent of Object.keys(detail.action_metrics.by_agent ?? {})) ids.add(Number(agent));
    }
    return [...ids].filter(Number.isFinite).sort((a, b) => a - b);
  }, [details]);

  const toggleSource = (key: string) => {
    setSelected((current) => current.includes(key)
      ? current.filter((item) => item !== key)
      : current.length < 4 ? [...current, key] : current);
  };

  const first = details[0];
  const finalCity = first?.summary.selected_city ?? first?.summary.city ?? {};
  const roundValues = (key: string) => first?.rounds
    .map((row) => numeric(row.city?.[key]))
    .filter((value): value is number => value != null) ?? [];
  const scoreValues = first?.summary.scores ?? [];
  const meanScore = scoreValues.length ? scoreValues.reduce((sum, score) => sum + score, 0) / scoreValues.length : undefined;
  const actionTypes = Object.entries(first?.action_metrics.by_type ?? {});
  const hasRoundSeries = details.some((detail) => detail.rounds.length > 0);
  const comparisonOption = useMemo<EChartsOption>(() => {
    const series: NonNullable<EChartsOption["series"]> = [];
    if (hasRoundSeries) {
      for (const detail of details) {
        series.push({
          name: detail.source.label,
          type: "line",
          smooth: true,
          showSymbol: false,
          data: detail.rounds.map((row) => [row.round, numeric(row.city?.[metric]) ?? null]),
        });
      }
    } else {
      for (const detail of details) {
        const rows = detail.training?.series ?? [];
        for (const [key, label] of [["best_ever", "Лучший результат"], ["best", "Лучший за поколение"], ["mean", "Средний fitness"]] as const) {
          const points = rows
            .map((row) => [numeric(row.gen ?? row.generation), numeric(row[key])])
            .filter((point): point is [number, number] => point[0] != null && point[1] != null);
          if (points.length) {
            series.push({
              name: `${detail.source.label} · ${label}`,
              type: "line",
              smooth: true,
              showSymbol: true,
              symbolSize: 5,
              data: points,
            });
          }
        }
      }
    }
    return {
      color: ["#8b5cf6", "#e55137", "#ff4f00", "#8bb8ff"],
      animationDuration: 550,
      tooltip: { trigger: "axis" },
      legend: { top: 4, textStyle: { color: "#b8c0ce" } },
      grid: { left: 54, right: 24, top: 42, bottom: 42 },
      xAxis: { type: "value", name: hasRoundSeries ? "Раунд" : "Поколение", nameLocation: "middle", nameGap: 28, axisLabel: { color: "#8f9aaa" }, splitLine: { lineStyle: { color: "rgba(255,255,255,.06)" } } },
      yAxis: { type: "value", name: hasRoundSeries ? metricLabels[metric] : "Fitness", min: hasRoundSeries && ["city_quality", "mean_satisfaction", "green_share", "mean_utilization", "type_entropy"].includes(metric) ? 0 : undefined, max: hasRoundSeries && ["city_quality", "mean_satisfaction", "green_share", "mean_utilization", "type_entropy"].includes(metric) ? 1 : undefined, axisLabel: { color: "#8f9aaa" }, splitLine: { lineStyle: { color: "rgba(255,255,255,.06)" } } },
      series,
    };
  }, [details, hasRoundSeries, metric]);

  const compareBars = useMemo<EChartsOption>(() => ({
    color: ["#8b5cf6", "#e55137", "#ff4f00", "#8bb8ff"],
    tooltip: { trigger: "axis", axisPointer: { type: "shadow" } },
    legend: { top: 4, textStyle: { color: "#b8c0ce" } },
    grid: { left: 54, right: 18, top: 42, bottom: 42 },
    xAxis: { type: "category", data: ["Качество", "Удовлетворённость", "Зелёные зоны"], axisLabel: { color: "#a9b2c1" } },
    yAxis: { type: "value", min: 0, max: 1, axisLabel: { color: "#8f9aaa" }, splitLine: { lineStyle: { color: "rgba(255,255,255,.06)" } } },
    series: details.map((detail) => ({
      name: detail.source.label,
      type: "bar",
      barMaxWidth: 28,
      data: ["city_quality", "mean_satisfaction", "green_share"].map((key) => numeric(detail.summary.city?.[key]) ?? 0),
    })),
  }), [details]);

  const typeOption = useMemo<EChartsOption>(() => ({
    color: ["#8b5cf6", "#e55137", "#ff4f00", "#8bb8ff", "#d9c8ff"],
    tooltip: { trigger: "item", formatter: "{b}: {c} ({d}%)" },
    legend: { bottom: 0, textStyle: { color: "#a9b2c1" } },
    series: [{
      type: "pie",
      radius: ["48%", "72%"],
      center: ["50%", "45%"],
      avoidLabelOverlap: true,
      itemStyle: { borderColor: "#151922", borderWidth: 3 },
      label: { color: "#d8deea", formatter: "{b}\n{d}%" },
      data: actionTypes.map(([name, value], index) => ({
        name: name === "pass"
          ? "Пропуск"
          : name === "demolish"
            ? "Снос"
            : ["Жильё", "Коммерция", "Смешанная застройка", "Промышленность", "Зелёная зона"][Number(name)]
              ?? first?.action_type_names[name]
              ?? `Неизвестный тип ${name}`,
        value,
        itemStyle: { color: ["#8b5cf6", "#ff6b4a", "#ff4f00", "#8bb8ff", "#d9c8ff"][index % 5] },
      })),
    }],
  }), [actionTypes]);

  const heatOption = useMemo<EChartsOption>(() => {
    const detail = first;
    const counts = detail?.action_metrics.by_round_agent ?? {};
    const accepted = detail?.action_metrics.accepted_by_round_agent ?? {};
    const rawCells = Object.entries(counts).map(([key, count]) => {
      const [round, agent] = key.split(":").map(Number);
      return { round, agent, rate: count ? (accepted[key] ?? 0) / count : 0 };
    });
    const roundLabels = [...new Set(rawCells.map(({ round }) => round))].sort((a, b) => a - b);
    const agentLabels = [...new Set(rawCells.map(({ agent }) => agent))].sort((a, b) => a - b);
    const roundIndex = new Map(roundLabels.map((round, index) => [round, index]));
    const agentIndex = new Map(agentLabels.map((agent, index) => [agent, index]));
    const cells = rawCells.map(({ round, agent, rate }) => [
      roundIndex.get(round) ?? 0,
      agentIndex.get(agent) ?? 0,
      rate,
      round,
      agent,
    ]);
    return {
      tooltip: { position: "top", formatter: (params: unknown) => {
        const item = params as { value: [number, number, number, number, number] };
        return `Раунд ${item.value[3]} · агент ${item.value[4]}<br/>Принято: ${(item.value[2] * 100).toFixed(1)}%`;
      } },
      grid: { left: 58, right: 20, top: 18, bottom: 42 },
      xAxis: { type: "category", name: "Раунд", data: roundLabels.map(String), axisLabel: { color: "#9da8ba" } },
      yAxis: { type: "category", name: "Агент", data: agentLabels.map(String), axisLabel: { color: "#9da8ba" } },
      visualMap: { min: 0, max: 1, dimension: 2, calculable: false, orient: "horizontal", left: "center", bottom: 0, inRange: { color: ["#211a31", "#6600ff", "#ff4f00"] }, textStyle: { color: "#9da8ba" } },
      series: [{ type: "heatmap", data: cells, emphasis: { itemStyle: { shadowBlur: 9, shadowColor: "rgba(255,255,255,.35)" } } }],
    };
  }, [first]);

  const scatterOption = useMemo<EChartsOption>(() => ({
    color: ["#8b5cf6", "#e55137", "#ff4f00", "#8bb8ff"],
    tooltip: { trigger: "item", formatter: (params: unknown) => {
      const point = params as { value: [number, number, number, number] };
      return `Раунд ${point.value[2]} · агент ${point.value[3]}<br/>Ставка: ${formatMetric(point.value[0])}<br/>Ущерб: ${formatMetric(point.value[1])}`;
    } },
    legend: { top: 4, textStyle: { color: "#b8c0ce" } },
    grid: { left: 58, right: 24, top: 42, bottom: 46 },
    xAxis: { name: "Ставка / стоимость", type: "value", axisLabel: { color: "#9da8ba" }, splitLine: { lineStyle: { color: "rgba(255,255,255,.06)" } } },
    yAxis: { name: "Ущерб", type: "value", axisLabel: { color: "#9da8ba" }, splitLine: { lineStyle: { color: "rgba(255,255,255,.06)" } } },
    series: details.flatMap((detail) => {
      const points = detail.actions.map((action) => [
        action.action.bid || action.action.cost,
        action.damage,
        action.round,
        action.agent_id,
      ] as [number, number, number, number]);
      const xy = points.map(([x, y]) => [x, y] as [number, number]);
      if (xy.length < 2) {
        return [{ name: detail.source.label, type: "scatter" as const, symbolSize: 8, data: points }];
      }
      const meanX = xy.reduce((sum, point) => sum + point[0], 0) / xy.length;
      const meanY = xy.reduce((sum, point) => sum + point[1], 0) / xy.length;
      const varianceX = xy.reduce((sum, point) => sum + (point[0] - meanX) ** 2, 0);
      const slope = varianceX
        ? xy.reduce((sum, point) => sum + (point[0] - meanX) * (point[1] - meanY), 0) / varianceX
        : 0;
      const minX = Math.min(...xy.map((point) => point[0]));
      const maxX = Math.max(...xy.map((point) => point[0]));
      const correlation = pearson(xy);
      return [
        { name: detail.source.label, type: "scatter" as const, symbolSize: 8, data: points },
        {
          name: `${detail.source.label} · тренд r=${formatMetric(correlation, 2)}`,
          type: "line" as const,
          showSymbol: false,
          lineStyle: { type: "dashed" as const, opacity: .6 },
          data: [[minX, meanY + slope * (minX - meanX)], [maxX, meanY + slope * (maxX - meanX)]],
        },
      ];
    }),
  }), [details]);

  const scatterCorrelation = pearson(details.flatMap((detail) =>
    detail.actions.map((action) => [
      action.action.bid || action.action.cost,
      action.damage,
    ] as [number, number]),
  ));
  const histogramOption = useMemo<EChartsOption>(() => {
    const costBins = first?.action_metrics.cost_bins ?? [];
    const bidBins = first?.action_metrics.bid_bins ?? [];
    const labelBin = (prefix: string, bin: { from: number; to: number }) =>
      `${prefix} ${formatMetric(bin.from, 1)}–${formatMetric(bin.to, 1)}`;
    return {
      tooltip: { trigger: "axis" },
      legend: { top: 4, textStyle: { color: "#b8c0ce" } },
      grid: { left: 58, right: 20, top: 42, bottom: 42 },
      xAxis: {
        type: "category",
        name: "Диапазоны",
        data: [...costBins.map((bin) => labelBin("Стоимость", bin)), ...bidBins.map((bin) => labelBin("Ставка", bin))],
        axisLabel: { color: "#9da8ba", rotate: 20 },
      },
      yAxis: { type: "value", name: "Число действий", axisLabel: { color: "#9da8ba" } },
      series: [
        { name: "Стоимость", type: "bar", itemStyle: { color: "#8b5cf6" }, data: [...costBins.map((bin) => bin.count), ...bidBins.map(() => 0)] },
        { name: "Ставка", type: "bar", itemStyle: { color: "#ff4f00" }, data: [...costBins.map(() => 0), ...bidBins.map((bin) => bin.count)] },
      ],
    };
  }, [first]);
  const actionDimensionsOption = useMemo<EChartsOption>(() => {
    const floorBins = first?.action_metrics.floors_bins ?? [];
    const roofCounts = first?.action_metrics.roof_quality_counts ?? {};
    const roofLabels = ["без озеленения", "базовое", "улучшенное", "премиальное"];
    const roofEntries = Object.entries(roofCounts).sort(
      ([a], [b]) => Number(a) - Number(b),
    );
    const labels = [
      ...floorBins.map((bin) =>
        bin.from === bin.to
          ? `${formatMetric(bin.from, 0)} этаж.`
          : `${formatMetric(bin.from, 1)}–${formatMetric(bin.to, 1)} этаж.`,
      ),
      ...roofEntries.map(([level]) => roofLabels[Number(level)] ?? `уровень ${level}`),
    ];
    return {
      tooltip: { trigger: "axis" },
      legend: { top: 4, textStyle: { color: "#b8c0ce" } },
      grid: { left: 58, right: 20, top: 42, bottom: 42 },
      xAxis: {
        type: "category",
        name: "Параметр действия",
        data: labels,
        axisLabel: { color: "#9da8ba", rotate: 20 },
      },
      yAxis: { type: "value", name: "Число построек", axisLabel: { color: "#9da8ba" } },
      series: [
        {
          name: "Этажность",
          type: "bar",
          itemStyle: { color: "#8b5cf6" },
          data: [
            ...floorBins.map((bin) => bin.count),
            ...roofEntries.map(() => 0),
          ],
        },
        {
          name: "Качество зелёной крыши",
          type: "bar",
          itemStyle: { color: "#54a24b" },
          data: [
            ...floorBins.map(() => 0),
            ...roofEntries.map(([, count]) => count),
          ],
        },
      ],
    };
  }, [first]);

  const boxOption = useMemo<EChartsOption>(() => {
    const sourcesWithValues = details.map((detail) => ({
      name: detail.source.label,
      values: detail.rounds.map((row) => numeric(row.city?.[metric])).filter((value): value is number => value != null).sort((a, b) => a - b),
    })).filter((item) => item.values.length);
    const quantile = (values: number[], p: number) => values[Math.min(values.length - 1, Math.floor((values.length - 1) * p))] ?? 0;
    return {
      tooltip: { trigger: "item" },
      grid: { left: 54, right: 20, top: 20, bottom: 42 },
      xAxis: { type: "category", data: sourcesWithValues.map((item) => item.name), axisLabel: { color: "#a9b2c1", interval: 0, rotate: 12 } },
      yAxis: { type: "value", name: metricLabels[metric], axisLabel: { color: "#8f9aaa" }, splitLine: { lineStyle: { color: "rgba(255,255,255,.06)" } } },
      series: [{ type: "boxplot", data: sourcesWithValues.map(({ values }) => [values[0] ?? 0, quantile(values, .25), quantile(values, .5), quantile(values, .75), values.at(-1) ?? 0]) }],
    };
  }, [details, metric]);

  const radarOption = useMemo<EChartsOption>(() => ({
    color: ["#8b5cf6", "#e55137", "#ff4f00", "#8bb8ff"],
    tooltip: {},
    legend: { bottom: 0, textStyle: { color: "#b8c0ce" } },
    radar: { indicator: [
      { name: "Качество", max: 1 }, { name: "Удовлетворённость", max: 1 },
      { name: "Зелёные зоны", max: 1 }, { name: "Доступность", max: 1 },
      { name: "Совместимость", max: 1 },
    ], axisName: { color: "#b8c0ce" }, splitLine: { lineStyle: { color: "rgba(255,255,255,.1)" } }, splitArea: { areaStyle: { color: ["rgba(255,255,255,.01)", "rgba(255,255,255,.035)"] } } },
    series: [{
      type: "radar",
      data: details.map((detail) => ({
        name: detail.source.label,
        value: ["city_quality", "mean_satisfaction", "green_share", "accessibility", "compatibility"].map((key) => numeric(detail.summary.city?.[key]) ?? 0),
        areaStyle: { opacity: .09 },
      })),
    }],
  }), [details]);

  const trainingOption = useMemo<EChartsOption>(() => {
    const series: NonNullable<EChartsOption["series"]> = [];
    for (const detail of details) {
      const rows = detail.training?.series ?? [];
      for (const [key, name] of [["best_ever", "Лучший результат"], ["best", "Лучший за поколение"], ["mean", "Средний fitness"]] as const) {
        const values = rows.map((row) => [numeric(row.gen ?? row.generation), numeric(row[key])]).filter((item): item is [number, number] => item[0] != null && item[1] != null);
        if (values.length) series.push({ name: `${detail.source.label} · ${name}`, type: "line", smooth: true, showSymbol: true, symbolSize: 5, data: values });
      }
    }
    return {
      color: ["#8b5cf6", "#e55137", "#ff4f00", "#8bb8ff"],
      tooltip: { trigger: "axis" },
      legend: { top: 4, textStyle: { color: "#b8c0ce" } },
      grid: { left: 58, right: 22, top: 46, bottom: 42 },
      xAxis: { type: "value", name: "Поколение", axisLabel: { color: "#9da8ba" } },
      yAxis: { type: "value", name: "Fitness", axisLabel: { color: "#9da8ba" }, splitLine: { lineStyle: { color: "rgba(255,255,255,.06)" } } },
      series,
    };
  }, [details]);

  const availableAgents = first?.summary.agents ?? [];
  const rankedAgents = [...availableAgents].sort((a, b) =>
    (numeric(first?.summary.scores?.[numeric(b.agent_id) ?? -1]) ?? -Infinity)
    - (numeric(first?.summary.scores?.[numeric(a.agent_id) ?? -1]) ?? -Infinity),
  );
  const metricStatistics = Object.entries(first?.summary.metrics ?? {})
    .filter((entry): entry is [string, Record<string, number>] => entry[1] != null)
    .sort(([keyA], [keyB]) => (metricLabels[keyA] ?? keyA).localeCompare(metricLabels[keyB] ?? keyB, "ru"));

  return (
    <div className="page-content analytics-page">
      <div className="page-heading">
        <div><p className="eyebrow">АНАЛИТИКА ЭКСПЕРИМЕНТОВ</p><h1>Результаты и динамика</h1><p className="muted">Сравнивайте прогоны, поведение агентов и ход обучения. Выберите до четырёх запусков.</p>{lastUpdated && <small className="analytics-updated">Обновлено: <time dateTime={lastUpdated.toISOString()}>{lastUpdated.toLocaleTimeString("ru-RU")}</time>{loading ? " · обновление…" : ""}</small>}</div>
      </div>
      {running && (
        <section className="analytics-live panel" role="status">
          <div><span className="connection-dot busy" /><div><b>Выполняется задача</b><small>{progress?.phase ?? "Данные обновляются по мере поступления"}</small></div></div>
          <div className="analytics-live-progress"><b>{progress?.best_fitness == null ? "Обновление данных…" : `Лучший fitness ${formatMetric(progress.best_fitness, 3)}`}</b><span>{progress?.completed_generations ?? progress?.generation ?? 0} / {progress?.generations ?? "—"} поколений</span></div>
        </section>
      )}
      <section className="analytics-filters panel">
        <div className="analytics-filter-sources">
          <div className="section-head compact"><div><p className="eyebrow">ИСТОЧНИКИ</p><h2>Прогоны для сравнения</h2></div><span className="badge">{selected.length} / 4</span></div>
          <div className="analytics-source-list">
            {sources.map((source) => (
              <label key={sourceKey(source)} className={`analytics-source-option ${selected.includes(sourceKey(source)) ? "selected" : ""} ${!source.available ? "unavailable" : ""}`}>
                <input type="checkbox" checked={selected.includes(sourceKey(source))} disabled={!source.available || (!selected.includes(sourceKey(source)) && selected.length >= 4)} onChange={() => toggleSource(sourceKey(source))} />
                <span><b>{source.label}</b><small>{source.kind} · {source.state} · {new Date(source.created_at).toLocaleDateString()}</small></span>
                <i className={`analytics-source-dot ${source.source_type}`} />
              </label>
            ))}
            {!sources.length && !loading && <p className="muted small">Пока нет запусков. Создайте симуляцию или загрузите архив на вкладке «Данные».</p>}
            {sources.length > 0 && !sources.some((source) => source.available) && <p className="muted small">Для старых архивов доступны только сводки. Новые запуски будут содержать полный журнал.</p>}
          </div>
        </div>
        <div className="analytics-filter-controls">
          <label className="field"><span>Метрика динамики</span><select value={metric} onChange={(event) => setMetric(event.target.value)}>{Object.entries(metricLabels).map(([key, label]) => <option key={key} value={key}>{label}</option>)}</select></label>
          <label className="field"><span>Агент</span><select value={agentId} onChange={(event) => setAgentId(event.target.value)}><option value="">Все агенты</option>{agentOptions.map((id) => <option key={id} value={id}>Агент {id}</option>)}</select></label>
          <div className="analytics-round-filter"><span>Диапазон раундов</span><div><input aria-label="Первый раунд" aria-invalid={Boolean(roundError)} aria-describedby={roundError ? "round-range-error" : undefined} type="number" min="0" step="1" placeholder="с начала" value={roundStart} onChange={(event) => setRoundStart(event.target.value)} /><i aria-hidden="true">—</i><input aria-label="Последний раунд" aria-invalid={Boolean(roundError)} aria-describedby={roundError ? "round-range-error" : undefined} type="number" min="0" step="1" placeholder="до конца" value={roundEnd} onChange={(event) => setRoundEnd(event.target.value)} /></div>{roundError && <small className="field-error" id="round-range-error">{roundError}</small>}</div>
          <p className="hint">Полный журнал действий сохраняется в артефактах и входит в переносимый ZIP.</p>
        </div>
      </section>
      {error && <div className="alert analytics-error" role="alert"><span>{error}{details.length > 0 && !roundError ? " Показана последняя успешно загруженная версия данных." : ""}</span><button className="button small-button secondary" disabled={loading || Boolean(roundError)} onClick={() => setRetryKey((current) => current + 1)}>Повторить</button></div>}
      {loading && !details.length ? <div className="analytics-loading panel"><span className="analytics-loader" />Загружаем метрики и историю прогонов…</div> : null}
      {!details.length && !loading && <div className="panel analytics-empty"><b>{sources.length ? "Выберите запуск с данными" : "Пока нет доступных запусков"}</b><p>{sources.length ? "Для архивных прогонов без подробных логов доступны параметры и итоговые сводки." : "Запустите симуляцию или добавьте архив, чтобы построить графики."}</p><div className="button-row"><button className="button primary" onClick={() => onNavigate("launch")}>Создать запуск</button><button className="button secondary" onClick={() => onNavigate("data")}>Импортировать архив</button></div></div>}
      {details.length > 0 && (
        <>
          <section className="analytics-kpis">
            {[
              { label: "Качество города", value: numeric(finalCity.city_quality), metric: "city_quality", icon: "◈", suffix: "" },
              { label: "Удовлетворённость", value: numeric(finalCity.mean_satisfaction), metric: "mean_satisfaction", icon: "◎", suffix: "" },
              { label: "Действия", value: first?.action_metrics.count, metric: "", icon: "⌁", suffix: "" },
              { label: "Средний fitness", value: meanScore, metric: "", icon: "↗", suffix: "" },
            ].map((kpi) => (
              <article className="analytics-kpi" key={kpi.label}>
                <span className="analytics-kpi-icon">{kpi.icon}</span>
                <small>{kpi.label}</small>
                <strong>{formatMetric(kpi.value, kpi.label === "Действия" ? 0 : 3)}{kpi.suffix}</strong>
                <span className="analytics-kpi-foot">{kpi.label === "Действия" ? `${first?.action_metrics.accepted ?? 0} приняты · ${first?.action_metrics.rejected ?? 0} пропущены` : `${kpi.metric ? trend(roundValues(kpi.metric)) + " · " : ""}Раундов: ${first?.summary.selected_round_count ?? 0}`}</span>
                <Sparkline values={kpi.metric ? roundValues(kpi.metric) : []} />
              </article>
            ))}
          </section>
          <section className="analytics-grid">
            <ChartCard title={hasRoundSeries ? "Динамика по раундам" : "Динамика обучения"} eyebrow={hasRoundSeries ? "ВРЕМЕННОЙ РЯД" : "ПОКОЛЕНИЯ"} option={comparisonOption} empty={!hasRoundSeries && !details.some((item) => item.training?.series?.length)} />
            <ChartCard title="Сравнение прогонов" eyebrow="СВОДНЫЕ KPI" option={compareBars} empty={details.length === 0} />
            <ChartCard title="Распределение действий" eyebrow="СТРУКТУРА" option={typeOption} empty={!actionTypes.length} />
            <ChartCard title="Активность агентов" eyebrow="ТЕПЛОВАЯ КАРТА" option={heatOption} empty={!Object.keys(first?.action_metrics.by_round_agent ?? {}).length} />
            <ChartCard title="Ставки и внешний ущерб" eyebrow="КОРРЕЛЯЦИЯ" option={scatterOption} empty={!details.some((item) => item.actions.length)} description={`Корреляция Пирсона: ${formatMetric(scatterCorrelation, 3)}`} />
            <ChartCard title="Распределение ставок и стоимости" eyebrow="ГИСТОГРАММА" option={histogramOption} empty={!first?.action_metrics.cost_bins.length && !first?.action_metrics.bid_bins.length} />
            <ChartCard title="Этажность и зелёные крыши" eyebrow="ФАКТОРНЫЕ ДЕЙСТВИЯ" option={actionDimensionsOption} empty={!first?.action_metrics.building_count} />
            <ChartCard title="Разброс метрики" eyebrow="РАСПРЕДЕЛЕНИЕ" option={boxOption} empty={!details.some((item) => item.rounds.length)} />
            <ChartCard title="Профиль города" eyebrow="МНОГОМЕРНОЕ СРАВНЕНИЕ" option={radarOption} empty={!details.some((item) => item.summary.city)} />
            <ChartCard title="Кривая обучения" eyebrow="ПОКОЛЕНИЯ" option={trainingOption} empty={!details.some((item) => item.training?.series?.length)} />
          </section>
          <section className="panel analytics-table-panel">
            <div className="section-head compact"><div><p className="eyebrow">РАУНДОВЫЕ МЕТРИКИ</p><h2>Статистика и тренды</h2></div><span className="muted small">{first?.source.label}</span></div>
            <div className="table-wrap analytics-table-wrap">
              <table><thead><tr><th>Метрика</th><th>Среднее</th><th>Медиана</th><th>Мин.</th><th>Макс.</th><th>Станд. отклонение</th><th>P10</th><th>P90</th><th>Тренд / раунд</th></tr></thead>
                <tbody>{metricStatistics.map(([key, stats]) => (
                  <tr key={key}><td>{metricLabels[key] ?? key}</td><td>{formatMetric(stats.mean, 3)}</td><td>{formatMetric(stats.median, 3)}</td><td>{formatMetric(stats.min, 3)}</td><td>{formatMetric(stats.max, 3)}</td><td>{formatMetric(stats.std, 3)}</td><td>{formatMetric(stats.p10, 3)}</td><td>{formatMetric(stats.p90, 3)}</td><td>{formatMetric(stats.trend_per_round, 4)}</td></tr>
                ))}{!metricStatistics.length && <tr><td colSpan={9} className="empty-cell">Нет раундовых статистик для выбранного запуска.</td></tr>}</tbody>
              </table>
            </div>
          </section>
          <section className="panel analytics-table-panel">
            <div className="section-head compact"><div><p className="eyebrow">АГЕНТЫ</p><h2>Итоговые показатели</h2></div><span className="muted small">{first?.source.label}</span></div>
            <div className="table-wrap analytics-table-wrap">
              <table><thead><tr><th>Агент</th><th>Fitness</th><th>Прибыль</th><th>Постройки</th><th>Пропуски</th><th>Ущерб</th><th>Нарушения</th><th>Состояние</th></tr></thead>
                <tbody>{rankedAgents.map((agent) => {
                  const id = numeric(agent.agent_id) ?? -1;
                  const score = first?.summary.scores?.[id];
                  return <tr key={id}><td>Агент {id}</td><td>{formatMetric(score, 3)}</td><td>{formatMetric(numeric(agent.profit), 1)}</td><td>{formatMetric(numeric(agent.builds), 0)}</td><td>{formatMetric(numeric(agent.skips), 0)}</td><td>{formatMetric(numeric(agent.damage_caused), 1)}</td><td>{formatMetric(numeric(agent.violations), 1)}</td><td><span className={`badge ${agent.bankrupt ? "red" : "green"}`}>{agent.bankrupt ? "Банкротство" : "Активен"}</span></td></tr>;
                })}{!availableAgents.length && <tr><td colSpan={8} className="empty-cell">Нет итоговых данных по агентам для выбранных источников.</td></tr>}</tbody>
              </table>
            </div>
            {first?.actions_truncated && <p className="hint analytics-sampling-note">Точки диаграммы рассеяния: {first.actions.length.toLocaleString("ru-RU")} из {first.action_metrics.count.toLocaleString("ru-RU")} действий. Гистограммы рассчитаны по выборке из {first.action_metrics.sample_count?.toLocaleString("ru-RU") ?? "—"} действий.</p>}
            {first && <div className="analytics-downloads"><a className="button secondary" href={`/api/analytics/${first.source.source_type}/${first.source.source_id}/export?format=csv${roundStart ? `&round_start=${roundStart}` : ""}${roundEnd ? `&round_end=${roundEnd}` : ""}${agentId ? `&agent_id=${agentId}` : ""}`}>Экспорт действий CSV</a><a className="button secondary" href={`/api/analytics/${first.source.source_type}/${first.source.source_id}/export?format=json${roundStart ? `&round_start=${roundStart}` : ""}${roundEnd ? `&round_end=${roundEnd}` : ""}${agentId ? `&agent_id=${agentId}` : ""}`}>Экспорт аналитики JSON</a></div>}
          </section>
        </>
      )}
    </div>
  );
}
