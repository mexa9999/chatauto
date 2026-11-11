import { useEffect, useRef, useState } from "react";
import { api } from "./api";

type Branch = { name: string; head_commit: string | null; generation: number | null };
type TrainingRun = {
  key: string;
  label: string;
  checkpoint_count: number;
  first_generation: number;
  last_generation: number;
};
type Checkpoint = {
  commit_id: string;
  generation: number;
  best_fitness: number | null;
  complete: boolean;
};
type GenomeGroup = { member_kind: string; count: number };
type Genome = {
  genome_id: string;
  member_index: number | null;
  metadata: Record<string, unknown>;
};
type GenomePage = { genomes: Genome[]; has_more: boolean };
type PageState = { genomes: Genome[]; nextOffset: number; hasMore: boolean };

const groupLabels: Record<string, string> = {
  best: "Лучший геном",
  population: "Популяция",
  hall_of_fame: "Зал славы",
  map_elites: "MAP-Elites",
};

export function StrategyCatalog({
  branches,
  onError,
  refreshKey,
}: {
  branches: Branch[];
  onError: (message: string) => void;
  refreshKey: string;
}) {
  const [runs, setRuns] = useState<Record<string, TrainingRun[]>>({});
  const [checkpoints, setCheckpoints] = useState<Record<string, Checkpoint[]>>({});
  const [groups, setGroups] = useState<Record<string, GenomeGroup[]>>({});
  const [pages, setPages] = useState<Record<string, PageState>>({});
  const [genomeDetails, setGenomeDetails] = useState<Record<string, Record<string, unknown>>>({});
  const [loading, setLoading] = useState<Set<string>>(() => new Set());
  const previousRefreshKey = useRef(refreshKey);
  const branchRef = useRef(branches);
  const runsRef = useRef(runs);
  const checkpointsRef = useRef(checkpoints);
  branchRef.current = branches;
  runsRef.current = runs;
  checkpointsRef.current = checkpoints;

  useEffect(() => {
    if (previousRefreshKey.current === refreshKey) return;
    previousRefreshKey.current = refreshKey;
    let active = true;
    for (const branch of branchRef.current) {
      if (!Object.hasOwn(runsRef.current, branch.name)) continue;
      api<{ runs: TrainingRun[] }>(
        `/api/branches/${encodeURIComponent(branch.name)}/strategy-runs`,
      ).then(async (data) => {
        if (!active) return;
        setRuns((current) => ({ ...current, [branch.name]: data.runs }));
        const checkpointUpdates: Record<string, Checkpoint[]> = {};
        await Promise.all(data.runs.map(async (run) => {
          const key = `${branch.name}:${run.key}`;
          if (!Object.hasOwn(checkpointsRef.current, key)) return;
          const result = await api<{ checkpoints: Checkpoint[] }>(
            `/api/branches/${encodeURIComponent(branch.name)}/strategy-runs/${encodeURIComponent(run.key)}/checkpoints`,
          );
          checkpointUpdates[key] = result.checkpoints;
        }));
        if (active) setCheckpoints((current) => ({ ...current, ...checkpointUpdates }));
      }).catch((reason: unknown) => {
        if (active) onError(reason instanceof Error ? reason.message : String(reason));
      });
    }
    return () => { active = false; };
  }, [onError, refreshKey]);

  const loadOnce = async <T,>(
    key: string,
    loaded: boolean,
    path: string,
    save: (value: T) => void,
  ) => {
    if (loaded || loading.has(key)) return;
    setLoading((current) => new Set(current).add(key));
    try {
      save(await api<T>(path));
    } catch (reason) {
      onError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setLoading((current) => {
        const next = new Set(current);
        next.delete(key);
        return next;
      });
    }
  };

  const loadGenomes = async (commitId: string, memberKind: string, offset: number) => {
    const key = `${commitId}:${memberKind}`;
    if (loading.has(key)) return;
    setLoading((current) => new Set(current).add(key));
    try {
      const params = new URLSearchParams({
        commit_id: commitId,
        member_kind: memberKind,
        limit: "50",
        offset: String(offset),
      });
      const result = await api<GenomePage>(`/api/genomes?${params}`);
      setPages((current) => {
        const previous = offset === 0 ? [] : current[key]?.genomes ?? [];
        return {
          ...current,
          [key]: {
            genomes: [...previous, ...result.genomes],
            nextOffset: offset + result.genomes.length,
            hasMore: result.has_more,
          },
        };
      });
    } catch (reason) {
      onError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setLoading((current) => {
        const next = new Set(current);
        next.delete(key);
        return next;
      });
    }
  };

  const loadGenomeDetails = async (genomeId: string) => {
    if (genomeDetails[genomeId] || loading.has(genomeId)) return;
    setLoading((current) => new Set(current).add(genomeId));
    try {
      const details = await api<Record<string, unknown>>(`/api/genomes/${genomeId}`);
      setGenomeDetails((current) => ({
        ...current,
        [genomeId]: details,
      }));
    } catch (reason) {
      onError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setLoading((current) => {
        const next = new Set(current);
        next.delete(genomeId);
        return next;
      });
    }
  };

  return (
    <div className="strategy-catalog">
      <div className="section-head">
        <div><p className="eyebrow">ДЕРЕВО СТРАТЕГИЙ</p><h2>Обучения и геномы</h2></div>
      </div>
      <p className="muted small">Откройте ветку, затем запуск, поколение и тип геномов. Списки геномов загружаются страницами по 50 записей.</p>
      <div className="strategy-tree">
        {branches.map((branch) => {
          const branchRuns = runs[branch.name] ?? [];
          const runKey = (key: string) => `${branch.name}:${key}`;
          return (
            <details
              className="strategy-level branch-level"
              key={branch.name}
              onToggle={(event) => {
                event.stopPropagation();
                if (event.currentTarget.open) {
                  void loadOnce<{ runs: TrainingRun[] }>(
                    `runs:${branch.name}`,
                    Object.hasOwn(runs, branch.name),
                    `/api/branches/${encodeURIComponent(branch.name)}/strategy-runs`,
                    (data) => setRuns((current) => ({ ...current, [branch.name]: data.runs })),
                  );
                }
              }}
            >
              <summary>
                <b>{branch.name}</b>
                <span>{branch.head_commit ? `до поколения ${branch.generation ?? "—"}` : "пустая ветка"}</span>
                <small>{branchRuns.length} запусков</small>
              </summary>
              {branchRuns.length ? branchRuns.map((run) => {
                const checkpointsForRun = checkpoints[runKey(run.key)] ?? [];
                return (
                  <details
                    className="strategy-level run-level"
                    key={run.key}
                    onToggle={(event) => {
                      event.stopPropagation();
                      if (event.currentTarget.open) {
                        void loadOnce<{ checkpoints: Checkpoint[] }>(
                          `checkpoints:${runKey(run.key)}`,
                          Object.hasOwn(checkpoints, runKey(run.key)),
                          `/api/branches/${encodeURIComponent(branch.name)}/strategy-runs/${encodeURIComponent(run.key)}/checkpoints`,
                          (data) => setCheckpoints((current) => ({ ...current, [runKey(run.key)]: data.checkpoints })),
                        );
                      }
                    }}
                  >
                    <summary>
                      <b>{run.label}</b>
                      <span>поколения {run.first_generation}–{run.last_generation}</span>
                      <small>{run.checkpoint_count} снимков</small>
                    </summary>
                    {checkpointsForRun.map((checkpoint) => {
                      const checkpointGroups = groups[checkpoint.commit_id] ?? [];
                      return (
                        <details
                          className="strategy-level checkpoint-level"
                          key={checkpoint.commit_id}
                          onToggle={(event) => {
                            event.stopPropagation();
                            if (event.currentTarget.open) {
                              void loadOnce<{ groups: GenomeGroup[] }>(
                                `groups:${checkpoint.commit_id}`,
                                Object.hasOwn(groups, checkpoint.commit_id),
                                `/api/history/commits/${checkpoint.commit_id}/genome-groups`,
                                (data) => setGroups((current) => ({ ...current, [checkpoint.commit_id]: data.groups })),
                              );
                            }
                          }}
                        >
                          <summary>
                            <b>Поколение {checkpoint.generation}</b>
                            <span>fitness {checkpoint.best_fitness?.toFixed(3) ?? "—"}</span>
                            <small>{checkpoint.complete ? "завершено" : "прервано"}</small>
                          </summary>
                          {checkpointGroups.map((group) => {
                            const key = `${checkpoint.commit_id}:${group.member_kind}`;
                            const page = pages[key];
                            return (
                              <details
                                className="strategy-level genome-group-level"
                                key={group.member_kind}
                                onToggle={(event) => {
                                  event.stopPropagation();
                                  if (event.currentTarget.open && !page) {
                                    void loadGenomes(checkpoint.commit_id, group.member_kind, 0);
                                  }
                                }}
                              >
                                <summary>
                                  <b>{groupLabels[group.member_kind] ?? group.member_kind}</b>
                                  <span>{group.count.toLocaleString("ru-RU")} геномов</span>
                                </summary>
                                {page?.genomes.map((genome) => (
                                  <details
                                    className="strategy-genome"
                                    key={genome.genome_id}
                                    onToggle={(event) => {
                                      event.stopPropagation();
                                      if (event.currentTarget.open) void loadGenomeDetails(genome.genome_id);
                                    }}
                                  >
                                    <summary>
                                      Геном {genome.member_index == null ? "" : `№${genome.member_index + 1} · `}
                                      <code>{genome.genome_id.slice(0, 12)}</code>
                                    </summary>
                                    {genomeDetails[genome.genome_id] ? (
                                      <pre className="code-panel">{JSON.stringify(genomeDetails[genome.genome_id], null, 2)}</pre>
                                    ) : <span className="muted small">{loading.has(genome.genome_id) ? "Загрузка…" : "Откройте, чтобы посмотреть параметры"}</span>}
                                  </details>
                                ))}
                                {page?.hasMore && (
                                  <button
                                    className="button small-button secondary"
                                    disabled={loading.has(key)}
                                    onClick={() => void loadGenomes(checkpoint.commit_id, group.member_kind, page.nextOffset)}
                                  >
                                    {loading.has(key) ? "Загрузка…" : "Показать ещё 50"}
                                  </button>
                                )}
                              </details>
                            );
                          })}
                          {!checkpointGroups.length && <p className="muted small">Для этого снимка геномы ещё не проиндексированы.</p>}
                        </details>
                      );
                    })}
                  </details>
                );
              }) : branch.head_commit ? (
                <p className="muted small">{loading.has(`runs:${branch.name}`) ? "Загрузка запусков…" : "Для этой ветки не найдено ID запусков."}</p>
              ) : <p className="muted small">Запустите обучение в этой ветке, чтобы здесь появились поколения и геномы.</p>}
            </details>
          );
        })}
      </div>
    </div>
  );
}
