import type { HealthResponse, OptimizationResponse } from "../types";

const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";
const TIMEOUT_MS = 8000;

export class ApiError extends Error {
  status: number;
  code: "unavailable" | "invalid" | "failed";

  constructor(message: string, status = 0, code: ApiError["code"] = "failed") {
    super(message);
    this.status = status;
    this.code = code;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), TIMEOUT_MS);

  try {
    const response = await fetch(`${API_URL}${path}`, {
      ...init,
      signal: controller.signal,
      headers: {
        "Content-Type": "application/json",
        ...(init?.headers ?? {}),
      },
    });

    if (!response.ok) {
      let detail = "Request failed.";
      try {
        const body = (await response.json()) as { detail?: string };
        if (typeof body.detail === "string") detail = body.detail;
      } catch {
        /* ignore malformed error payloads */
      }
      if (response.status === 400) {
        throw new ApiError(detail, 400, "invalid");
      }
      throw new ApiError("Optimization request failed.", response.status, "failed");
    }

    return (await response.json()) as T;
  } catch (error) {
    if (error instanceof ApiError) throw error;
    throw new ApiError("Backend unavailable.", 0, "unavailable");
  } finally {
    window.clearTimeout(timer);
  }
}

export async function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/api/health");
}

export async function analyzeTac(tac: string): Promise<OptimizationResponse> {
  return request<OptimizationResponse>("/api/analyze", {
    method: "POST",
    body: JSON.stringify({ tac }),
  });
}

export async function optimizeTac(
  tac: string,
  strategy?: string,
): Promise<OptimizationResponse> {
  return request<OptimizationResponse>("/api/optimize", {
    method: "POST",
    body: JSON.stringify({ tac, strategy }),
  });
}

export async function simulateTac(tac: string): Promise<OptimizationResponse> {
  return request<OptimizationResponse>("/api/simulate", {
    method: "POST",
    body: JSON.stringify({ tac }),
  });
}
