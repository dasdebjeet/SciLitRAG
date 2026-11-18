
/*
 * SciLitRAG API service.
 *
 * Centralizes HTTP communication so React components do not need
 * to know the details of request construction or error handling.
 */

import type { AskRequest, AskResponse, Paper } from "../types/api";

/**
 * Submit a research question to the FastAPI backend.
 *
 * @throws Error when the backend returns an unsuccessful response
 *         or the response cannot be parsed.
 */
export async function askQuestion(
  request: AskRequest,
): Promise<AskResponse> {
  const response = await fetch("/api/v1/ask", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Accept": "application/json",
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    let errorMessage = `Request failed (${response.status})`;

    try {
      const errorData: { detail?: string } = await response.json();

      if (errorData.detail) {
        errorMessage = errorData.detail;
      }
    } catch {
      // Keep the fallback message if the error body isn't valid JSON.
    }

    throw new Error(errorMessage);
  }

  return (await response.json()) as AskResponse;
}

/**
 * Fetch all papers available in the SciLitRAG library.
 *
 * @throws Error when the backend returns an unsuccessful response
 *         or the response cannot be parsed.
 */
export async function getPapers(): Promise<Paper[]> {
  const response = await fetch("/api/v1/papers", {
    method: "GET",
    headers: {
      "Accept": "application/json",
    },
  });

  if (!response.ok) {
    let errorMessage = `Request failed (${response.status})`;

    try {
      const errorData: { detail?: string } = await response.json();

      if (errorData.detail) {
        errorMessage = errorData.detail;
      }
    } catch {
      // Retain the fallback message if the response isn't valid JSON.
    }

    throw new Error(errorMessage);
  }

  return (await response.json()) as Paper[];
}
