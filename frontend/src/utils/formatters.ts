
/**
 * Shared presentation helpers for bibliographic metadata.
 *
 * Keep canonical arXiv category identifiers in application state and API
 * requests. These helpers only transform values for human-readable display.
 */

const CATEGORY_LABELS: Record<string, string> = {
  "astro-ph.IM": "Astrophysical Instrumentation",
  "cond-mat.mtrl-sci": "Materials Science",
  "cond-mat.soft": "Soft Condensed Matter",
  "cs.AI": "Artificial Intelligence",
  "cs.AR": "Computer Architecture",
  "cs.CE": "Computational Engineering and Science",
  "cs.CG": "Computational Geometry",
  "cs.CL": "Natural Language Processing",
  "cs.CR": "Cryptography and Security",
  "cs.CV": "Computer Vision",
  "cs.CY": "Computers and Society",
  "cs.DC": "Distributed and Parallel Computing",
  "cs.DL": "Digital Libraries",
  "cs.DS": "Data Structures and Algorithms",
  "cs.FL": "Formal Languages and Automata Theory",
  "cs.GR": "Computer Graphics",
  "cs.GT": "Computer Science and Game Theory",
  "cs.HC": "Human-Computer Interaction",
  "cs.IR": "Information Retrieval",
  "cs.IT": "Information Theory",
  "cs.LG": "Machine Learning",
  "cs.LO": "Logic in Computer Science",
  "cs.MA": "Multiagent Systems",
  "cs.NE": "Neural and Evolutionary Computing",
  "cs.NI": "Networking and Internet Architecture",
  "cs.PF": "Computer Performance",
  "cs.RO": "Robotics",
  "cs.SD": "Sound and Music Computing",
  "cs.SE": "Software Engineering",
  "cs.SI": "Social and Information Networks",
  "eess.AS": "Audio and Speech Processing",
  "eess.IV": "Image and Video Processing",
  "eess.SP": "Signal Processing",
  "eess.SY": "Systems and Control",
  "math-ph": "Mathematical Physics",
  "math.DS": "Dynamical Systems",
  "math.OC": "Optimization and Control",
  "math.PR": "Probability",
  "math.RT": "Representation Theory",
  "math.ST": "Statistics",
  "nlin.PS": "Pattern Formation and Solitons",
  "physics.flu-dyn": "Fluid Dynamics",
  "q-bio.BM": "Biomolecules",
  "q-bio.NC": "Neurons and Cognition",
  "q-bio.PE": "Populations and Evolution",
  "q-bio.QM": "Quantitative Methods in Biology",
  "quant-ph": "Quantum Physics",
  "stat.CO": "Statistical Computation",
  "stat.ME": "Statistical Methodology",
  "stat.ML": "Machine Learning",
};

/**
 * Convert an arXiv category identifier into a readable label.
 * Unknown categories remain visible rather than being silently discarded.
 */
export function formatCategory(category: string): string {
  return CATEGORY_LABELS[category] ?? category;
}

/**
 * Format an ISO timestamp as a readable calendar date in the user's locale
 * and local timezone. Invalid or missing values fall back safely.
 */
export function formatDate(value: string): string {
  if (!value) {
    return "Date unavailable";
  }

  const date = new Date(value);

  if (Number.isNaN(date.getTime())) {
    return value;
  }

  return new Intl.DateTimeFormat(undefined, {
    day: "numeric",
    month: "short",
    year: "numeric",
  }).format(date);
}
