import { useEffect, useMemo, useState } from "react";
import { api } from "./api";

type Branch = {
  name: string;
  head_commit: string | null;
  generation: number | null;
  archived_at?: string | null;
};

type Commit = {
  commit_id: string;
  parent_id: string | null;
  branch_name: string;
  branch_names: string[];
  generation: number;
  best_fitness: number | null;
  complete: boolean;
};

type Graph = { branches: Branch[]; commits: Commit[] };

export function HistoryGraph({
  onError,
  refreshKey,
  onSelectCommit,
}: {
  onError: (message: string) => void;
  refreshKey: string;
  onSelectCommit: (branch: string, generation: number) => void;
}) {
  const [graph, setGraph] = useState<Graph>({ branches: [], commits: [] });

  useEffect(() => {
    let active = true;
    api<Graph>("/api/history/graph")
      .then((result) => {
        if (active) setGraph(result);
      })
      .catch((reason: unknown) => {
        if (active) onError(reason instanceof Error ? reason.message : String(reason));
      });
    return () => { active = false; };
  }, [onError, refreshKey]);

  const layout = useMemo(() => {
    const branchY = new Map(graph.branches.map((branch, index) => [branch.name, 50 + index * 76]));
    const positions = new Map(graph.commits.map((commit) => [
      commit.commit_id,
      {
        x: 190 + Math.max(0, commit.generation - 1) * 92,
        y: branchY.get(
          commit.branch_names.includes(commit.branch_name)
            ? commit.branch_name
            : commit.branch_names[0] ?? commit.branch_name,
        ) ?? 45,
      },
    ]));
    const maxX = Math.max(800, ...[...positions.values()].map(({ x }) => x + 40));
    const maxY = Math.max(110, 30 + graph.branches.length * 76);
    return { branchY, positions, maxX, maxY };
  }, [graph]);

  return (
    <section className="panel history-graph-panel">
      <div className="section-head">
        <div><p className="eyebrow">КАРТА ИСТОРИИ</p><h2>Ветки и поколения</h2></div>
        <span className="count-pill">{graph.branches.length} веток · {graph.commits.length} снимков</span>
      </div>
      {graph.branches.length ? (
        <div className="history-graph-scroll">
          <svg
            className="history-graph"
            role="img"
            aria-label="Граф веток и снимков истории обучения"
            viewBox={`0 0 ${layout.maxX} ${layout.maxY}`}
            width={layout.maxX}
            height={layout.maxY}
          >
            {graph.branches.map((branch) => (
              <g key={branch.name}>
                <line x1="0" x2={layout.maxX} y1={layout.branchY.get(branch.name)} y2={layout.branchY.get(branch.name)} className="history-graph-lane" />
                <text x="12" y={(layout.branchY.get(branch.name) ?? 0) + 4} className="history-graph-branch">
                  {branch.name}{branch.archived_at ? " · архив" : ""}
                </text>
                {!branch.head_commit && (
                  <text x="190" y={(layout.branchY.get(branch.name) ?? 0) + 4} className="history-graph-empty">
                    пустая ветка
                  </text>
                )}
              </g>
            ))}
            {graph.commits.map((commit) => {
              const position = layout.positions.get(commit.commit_id);
              const parent = commit.parent_id ? layout.positions.get(commit.parent_id) : undefined;
              if (!position) return null;
              return (
                <g key={`edge:${commit.commit_id}`}>
                  {parent && <path d={`M ${parent.x} ${parent.y} C ${parent.x + 42} ${parent.y}, ${position.x - 42} ${position.y}, ${position.x} ${position.y}`} className="history-graph-edge" />}
                </g>
              );
            })}
            {graph.commits.map((commit) => {
              const position = layout.positions.get(commit.commit_id);
              if (!position) return null;
              const isHead = graph.branches.some((branch) => branch.head_commit === commit.commit_id);
              return (
                <g
                  key={commit.commit_id}
                  className="history-graph-node"
                  role="button"
                  tabIndex={0}
                  aria-label={`Выбрать ${commit.branch_name}, поколение ${commit.generation}`}
                  onClick={() => onSelectCommit(
                    commit.branch_names.includes(commit.branch_name)
                      ? commit.branch_name
                      : commit.branch_names[0] ?? commit.branch_name,
                    commit.generation,
                  )}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" || event.key === " ") {
                      event.preventDefault();
                      onSelectCommit(
                        commit.branch_names.includes(commit.branch_name)
                          ? commit.branch_name
                          : commit.branch_names[0] ?? commit.branch_name,
                        commit.generation,
                      );
                    }
                  }}
                >
                  <title>{`${commit.branch_name} · поколение ${commit.generation} · fitness ${commit.best_fitness?.toFixed(3) ?? "—"}`}</title>
                  <circle cx={position.x} cy={position.y} r={isHead ? 11 : 8} className={isHead ? "head" : commit.complete ? "complete" : "incomplete"} />
                  <text x={position.x} y={position.y + 27} textAnchor="middle" className="history-graph-generation">
                    {commit.generation}
                  </text>
                </g>
              );
            })}
          </svg>
        </div>
      ) : <p className="muted small">Пока нет веток истории.</p>}
    </section>
  );
}
