import { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

const AGENTS = [
  { key: "search", name: "Search", description: "Finds sources on the web" },
  {
    key: "reader",
    name: "Reader",
    description: "Scrapes and extracts source evidence",
  },
  {
    key: "writer",
    name: "Writer",
    description: "Drafts the cited research report",
  },
  {
    key: "critic",
    name: "Critic",
    description: "Reviews the report and requests revisions",
  },
];

const AGENT_STATUS = {
  pending: { label: "Waiting", dot: "bg-slate-300", text: "text-slate-400" },
  running: { label: "Running", dot: "bg-amber-500", text: "text-amber-700" },
  done: { label: "Done", dot: "bg-emerald-500", text: "text-emerald-700" },
  error: { label: "Error", dot: "bg-red-500", text: "text-red-700" },
  skipped: { label: "Not run", dot: "bg-slate-200", text: "text-slate-400" },
};

const JOB_STATUS = {
  queued: {
    label: "Queued",
    classes: "bg-slate-100 text-slate-600",
    dot: "bg-slate-400",
  },
  running: {
    label: "Running",
    classes: "bg-amber-50 text-amber-700",
    dot: "bg-amber-500",
  },
  completed: {
    label: "Complete",
    classes: "bg-emerald-50 text-emerald-700",
    dot: "bg-emerald-500",
  },
  failed: {
    label: "Failed",
    classes: "bg-red-50 text-red-700",
    dot: "bg-red-500",
  },
};

function isHttpUrl(value) {
  return typeof value === "string" && /^https?:\/\//i.test(value);
}

/**
 * Turn inline [N] citations into links to the matching source URL.
 * Reference-list lines and existing markdown links are left untouched.
 */
function linkifyCitations(markdown, sources) {
  const urlByIndex = new Map(
    sources
      .filter((source) => isHttpUrl(source.url))
      .map((source) => [source.index, source.url]),
  );

  return markdown
    .split("\n")
    .map((line) => {
      if (/^\s*(?:\d+\.\s*)?\[\d+\][\s:]/.test(line)) {
        return line;
      }
      return line.replace(/\[(\d+)\](?!\(|:|\[)/g, (match, rawIndex) => {
        const url = urlByIndex.get(Number(rawIndex));
        return url ? `[${rawIndex}](${url})` : match;
      });
    })
    .join("\n");
}

function ReportLink({ node: _node, href, children, ...props }) {
  const external = isHttpUrl(href);
  return (
    <a
      href={external ? href : undefined}
      target={external ? "_blank" : undefined}
      rel={external ? "noopener noreferrer" : undefined}
      {...props}
    >
      {children}
    </a>
  );
}

function SourceLink({ source }) {
  if (!isHttpUrl(source.url)) {
    return <span>{source.title || "Untitled source"}</span>;
  }
  return (
    <a
      href={source.url}
      target="_blank"
      rel="noopener noreferrer"
      className="break-words font-medium text-blue-700 underline underline-offset-2 hover:text-blue-900"
    >
      {source.title || source.url}
    </a>
  );
}

export default function App() {
  const [topic, setTopic] = useState("");
  const [job, setJob] = useState(null);
  const [error, setError] = useState("");
  const [starting, setStarting] = useState(false);

  const isActive = job?.status === "queued" || job?.status === "running";
  const busy = starting || isActive;

  useEffect(() => {
    if (!job || !isActive) return undefined;

    let cancelled = false;

    async function poll() {
      try {
        const response = await fetch(`/api/jobs/${job.id}`);
        if (!response.ok) {
          if (response.status === 404) {
            throw new Error(
              "This research job was not found. The server may have restarted.",
            );
          }
          throw new Error("Unable to load the research progress.");
        }
        const data = await response.json();
        if (!cancelled) {
          setJob(data);
          setError("");
        }
      } catch (err) {
        if (!cancelled) {
          setError(
            err.message ||
              "Could not reach the research server. Is it running on port 8000?",
          );
        }
      }
    }

    poll();
    const timer = window.setInterval(poll, 1200);

    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [job?.id, job?.status, isActive]);

  async function startResearch(event) {
    event.preventDefault();
    const trimmed = topic.trim();
    if (!trimmed) {
      setError("Enter a research topic first.");
      return;
    }
    if (busy) return;

    setStarting(true);
    setError("");
    setJob(null);

    try {
      const response = await fetch("/api/research", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ topic: trimmed }),
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(
          typeof data.detail === "string"
            ? data.detail
            : "Could not start the research job.",
        );
      }
      setJob(data);
    } catch (err) {
      setError(
        err.message ||
          "Could not start the research job. Is the backend running on port 8000?",
      );
    } finally {
      setStarting(false);
    }
  }

  const report = job?.status === "completed" ? job.report : null;
  const sources = job?.sources ?? [];
  const critic = job?.critic;
  const warnings = job?.warnings ?? [];
  const failure = error || (job?.status === "failed" ? job.error : "");
  const jobBadge = JOB_STATUS[job?.status];

  return (
    <div className="min-h-screen bg-white text-slate-900">
      <main className="mx-auto w-full max-w-3xl px-5 py-12 sm:py-16">
        <header>
          <h1 className="text-3xl font-semibold tracking-tight text-slate-900 sm:text-4xl">
            AI Research Assistant
          </h1>
          <p className="mt-3 max-w-2xl text-sm leading-6 text-slate-600">
            Enter a topic and run your local research agents. The report is
            written from live web sources and reviewed by the critic agent.
          </p>
        </header>

        <form onSubmit={startResearch} className="mt-8">
          <div className="rounded-xl border border-slate-200 p-5">
            <label
              htmlFor="topic"
              className="block text-sm font-medium text-slate-700"
            >
              Research topic
            </label>
            <div className="mt-2 flex flex-col gap-3 sm:flex-row">
              <input
                id="topic"
                type="text"
                value={topic}
                onChange={(event) => setTopic(event.target.value)}
                placeholder="e.g. The impact of AI on healthcare"
                maxLength={300}
                disabled={busy}
                className="w-full rounded-lg border border-slate-300 bg-white px-3.5 py-2.5 text-sm text-slate-900 placeholder:text-slate-400 focus:border-slate-500 focus:outline-none focus:ring-1 focus:ring-slate-500 disabled:bg-slate-50 sm:flex-1"
              />
              <button
                type="submit"
                disabled={busy}
                className="w-full rounded-lg bg-slate-900 px-5 py-2.5 text-sm font-medium text-white hover:bg-slate-700 focus:outline-none focus:ring-2 focus:ring-slate-500 focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 sm:w-auto"
              >
                {busy ? "Researching…" : "Start Research"}
              </button>
            </div>
            <p className="mt-2 text-xs text-slate-500">
              Uses the local LM Studio model. Research jobs can take a few
              minutes.
            </p>
          </div>
        </form>

        {failure && (
          <div
            role="alert"
            className="mt-4 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800"
          >
            <p className="font-medium">
              {job?.status === "failed" ? "Research failed" : "Request failed"}
            </p>
            <p className="mt-1 leading-6">{failure}</p>
          </div>
        )}

        {job && (
          <section className="mt-8 rounded-xl border border-slate-200">
            <div className="flex items-center justify-between gap-4 border-b border-slate-200 px-5 py-4">
              <h2 className="text-base font-semibold text-slate-900">
                Agent progress
              </h2>
              {jobBadge && (
                <span
                  className={`inline-flex shrink-0 items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium ${jobBadge.classes}`}
                >
                  <span className={`h-1.5 w-1.5 rounded-full ${jobBadge.dot}`} />
                  {jobBadge.label}
                </span>
              )}
            </div>

            <ul className="divide-y divide-slate-100">
              {AGENTS.map((agent) => {
                const state = job.agents?.[agent.key] ?? {
                  status: "pending",
                  detail: "",
                };
                const status =
                  AGENT_STATUS[state.status] ?? AGENT_STATUS.pending;
                return (
                  <li
                    key={agent.key}
                    className="flex items-start justify-between gap-4 px-5 py-3.5"
                  >
                    <div className="min-w-0">
                      <div className="flex items-center gap-2">
                        <span
                          className={`h-2 w-2 shrink-0 rounded-full ${status.dot}`}
                        />
                        <p className="text-sm font-medium text-slate-900">
                          {agent.name}
                        </p>
                      </div>
                      <p className="mt-1 pl-4 text-xs leading-5 text-slate-500">
                        {state.detail || agent.description}
                      </p>
                    </div>
                    <span
                      className={`mt-0.5 shrink-0 text-xs font-medium ${status.text}`}
                    >
                      {status.label}
                    </span>
                  </li>
                );
              })}
            </ul>

            {isActive && (
              <p className="border-t border-slate-100 px-5 py-3 text-xs text-slate-500">
                The report will appear here as soon as all agents finish.
              </p>
            )}
          </section>
        )}

        {warnings.length > 0 && (
          <div className="mt-4 rounded-lg border border-amber-200 bg-amber-50 px-4 py-3">
            <p className="text-xs font-medium uppercase tracking-wide text-amber-800">
              Warnings
            </p>
            <ul className="mt-1.5 space-y-1 text-sm leading-6 text-amber-800">
              {warnings.map((warning, index) => (
                <li key={index}>• {warning}</li>
              ))}
            </ul>
          </div>
        )}

        {report && (
          <section className="mt-8 rounded-xl border border-slate-200">
            <div className="flex flex-col gap-3 border-b border-slate-200 px-5 py-4 sm:flex-row sm:items-center sm:justify-between">
              <div className="min-w-0">
                <h2 className="text-base font-semibold text-slate-900">
                  Research Report
                </h2>
                <p className="mt-0.5 break-words text-xs text-slate-500">
                  {job.topic}
                </p>
              </div>
              <a
                href={`/api/jobs/${job.id}/report`}
                download
                className="inline-flex w-fit shrink-0 items-center rounded-lg border border-slate-300 px-3 py-1.5 text-xs font-medium text-slate-700 hover:bg-slate-50"
              >
                Download report
              </a>
            </div>

            {critic && (
              <div className="border-b border-slate-200 bg-slate-50 px-5 py-4">
                <div className="flex flex-wrap items-center gap-2 text-xs">
                  <span
                    className={`inline-flex items-center rounded-full px-2 py-0.5 font-medium ${
                      critic.decision === "pass"
                        ? "bg-emerald-100 text-emerald-800"
                        : "bg-amber-100 text-amber-800"
                    }`}
                  >
                    {critic.decision === "pass"
                      ? "Passed critic review"
                      : "Needs revision"}
                  </span>
                  <span className="text-slate-600">
                    Score {critic.score}/10
                  </span>
                  <span className="text-slate-500">
                    · {job.revisions} revision
                    {job.revisions === 1 ? "" : "s"}
                  </span>
                </div>
                {critic.summary && (
                  <p className="mt-2 text-sm leading-6 text-slate-700">
                    {critic.summary}
                  </p>
                )}
                {critic.issues?.length > 0 && (
                  <ul className="mt-2 space-y-1 text-sm leading-6 text-slate-600">
                    {critic.issues.map((issue, index) => (
                      <li key={index}>• {issue}</li>
                    ))}
                  </ul>
                )}
              </div>
            )}

            <div className="report px-5 py-4">
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={{ a: ReportLink }}
              >
                {linkifyCitations(report, sources)}
              </ReactMarkdown>
            </div>

            {sources.length > 0 && (
              <div className="border-t border-slate-200 px-5 py-4">
                <h3 className="text-sm font-semibold text-slate-900">
                  Sources
                </h3>
                <ul className="mt-2 space-y-3">
                  {sources.map((source) => (
                    <li
                      key={`${source.index}-${source.url}`}
                      className="flex gap-3"
                    >
                      <span className="mt-0.5 shrink-0 text-xs font-medium text-slate-400">
                        [{source.index}]
                      </span>
                      <div className="min-w-0 text-sm">
                        <SourceLink source={source} />
                        <p className="mt-0.5 text-xs text-slate-500">
                          {source.source_domain}
                          {source.published_date
                            ? ` · ${source.published_date}`
                            : ""}
                          {source.scrape_success
                            ? ""
                            : " · page content unavailable"}
                        </p>
                      </div>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </section>
        )}

        <footer className="mt-12 border-t border-slate-200 pt-5 text-xs text-slate-400">
          Local pipeline · LM Studio{" "}
          <span className="font-mono">qwen3.5-2b</span> · Search, Reader,
          Writer, and Critic agents
        </footer>
      </main>
    </div>
  );
}
