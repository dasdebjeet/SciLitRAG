
/*
 * SciLitRAG API service.
 *
 * Centralizes HTTP communication so React components do not need
 * to know the details of request construction or error handling.
 */

import type {
  AskRequest,
  AskResponse,
  CompareRequest,
  CompareResponse,
  Paper,
} from "../types/api";

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
 * Compare two or three selected research papers.
 *
 * The backend independently retrieves and reranks evidence for each
 * paper before generating a consolidated, evidence-grounded comparison.
 *
 * @throws Error when the backend returns an unsuccessful response
 *         or the response cannot be parsed.
 */
export async function comparePapers(
  request: CompareRequest,
): Promise<CompareResponse> {
  const response = await fetch("/api/v1/compare", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Accept": "application/json",
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    let errorMessage = `Comparison failed (${response.status})`;

    try {
      const errorData: { detail?: string } = await response.json();

      if (errorData.detail) {
        errorMessage = errorData.detail;
      }
    } catch {
      // Preserve the fallback message if the error body isn't valid JSON.
    }

    throw new Error(errorMessage);
  }

  return (await response.json()) as CompareResponse;
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
