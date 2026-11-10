export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly path: string,
  ) {
    super(`${message} (HTTP ${status})`);
    this.name = "ApiError";
  }
}

function responseMessage(payload: unknown, fallback: string) {
  if (typeof payload === "object" && payload !== null && "detail" in payload) {
    const detail = (payload as { detail: unknown }).detail;
    if (typeof detail === "string") return detail;
    if (detail != null) return JSON.stringify(detail);
  }
  if (typeof payload === "string" && payload.trim()) return payload;
  return fallback;
}

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers);
  if (typeof init?.body === "string") headers.set("Content-Type", "application/json");
  const method = (init?.method ?? "GET").toUpperCase();
  const retryable = method === "GET" && !init?.signal?.aborted;

  for (let attempt = 0; ; attempt += 1) {
    let response: Response;
    try {
      response = await fetch(path, { ...init, headers });
    } catch (reason) {
      if (retryable && attempt === 0 && !init?.signal?.aborted && reason instanceof TypeError) {
        await new Promise((resolve) => window.setTimeout(resolve, 250));
        continue;
      }
      throw reason;
    }

    const text = await response.text();
    let body: unknown = {};
    if (text) {
      try {
        body = JSON.parse(text);
      } catch {
        body = text;
      }
    }
    if (!response.ok) {
      if (retryable && attempt === 0 && (response.status === 502 || response.status === 503)) {
        await new Promise((resolve) => window.setTimeout(resolve, 250));
        continue;
      }
      throw new ApiError(
        responseMessage(body, response.statusText || "Ошибка сервера"),
        response.status,
        path,
      );
    }
    return body as T;
  }
}
