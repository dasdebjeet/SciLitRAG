
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
  UploadResponse,
} from "../types/api";

/**
 * Extract a useful error message from an unsuccessful API response.
 *
 * The fallback message is retained when the response body is not
 * valid JSON or does not contain a string detail field.
 */
async function getErrorMessage(
  response: Response,
  fallback: string,
): Promise<string> {
  try {
    const errorData: { detail?: string } = await response.json();

    if (typeof errorData.detail === "string" && errorData.detail.trim()) {
      return errorData.detail;
    }
  } catch {
    // Preserve the fallback when the response body is not valid JSON.
  }

  return fallback;
}

/**
 * Submit a research question to the FastAPI backend.
 *
 * The request may optionally include an arXiv retrieval scope or
 * an uploaded-document ID. These scopes are mutually exclusive.
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
    const errorMessage = await getErrorMessage(
      response,
      `Request failed (${response.status})`,
    );

    throw new Error(errorMessage);
  }

  return (await response.json()) as AskResponse;
}

/**
 * Upload a research PDF to the FastAPI backend.
 *
 * The browser constructs the multipart/form-data request boundary
 * automatically. Do not set Content-Type manually when sending FormData.
 *
 * @throws Error when the backend rejects the file or the response
 *         cannot be parsed.
 */
export async function uploadResearchPaper(
  file: File,
): Promise<UploadResponse> {
  const formData = new FormData();

  // The field name must match the UploadFile parameter in the backend route.
  formData.append("file", file);

  const response = await fetch("/api/v1/uploads", {
    method: "POST",
    headers: {
      "Accept": "application/json",
    },
    body: formData,
  });

  if (!response.ok) {
    const errorMessage = await getErrorMessage(
      response,
      `Upload failed (${response.status})`,
    );

    throw new Error(errorMessage);
  }

  return (await response.json()) as UploadResponse;
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
    const errorMessage = await getErrorMessage(
      response,
      `Comparison failed (${response.status})`,
    );

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
    const errorMessage = await getErrorMessage(
      response,
      `Request failed (${response.status})`,
    );

    throw new Error(errorMessage);
  }

  return (await response.json()) as Paper[];
}
