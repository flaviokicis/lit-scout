export type LitScoutOutcome = 'shown' | 'queued' | 'superseded' | 'yielded' | 'none' | 'deduped' | 'backoff' | 'error'

export type LitScoutAnswer = 'check' | 'know' | 'dismiss' | 'expired'

/** What the reader may conclude from the papers it read. */
export type LitScoutReaderVerdict = 'NEW' | 'PARTLY DONE' | 'ALREADY DONE' | 'WELL ESTABLISHED'

/** NO MATCHES FOUND: the search gave nothing to read, which says nothing about novelty (a poor query, a source down). */
export type LitScoutVerdictWord = LitScoutReaderVerdict | 'NO MATCHES FOUND'

export type LitScoutAccess = 'full text' | 'abstract only'

/** A model call's token counts, as the API spells them. */
export type LitScoutUsage = {
  input_tokens: number
  output_tokens: number
  cache_read_input_tokens: number
  cache_creation_input_tokens: number
}

/** An idea offered for a check: the classifier's literature search query for it. */
export type LitScoutOffer = {
  id: string
  query: string
  promptsSince: number
}

/** The three searches a check runs for one idea, written by Haiku from the idea and the recent turns. */
export type LitScoutQueries = {
  /** The idea in plain words. */
  plain: string
  /** The terms a paper on it would use: the method's likely name, the field's jargon. */
  jargon: string
  /** An adjacent problem or method family it may be filed under. */
  neighbor: string
}

/** One search list that returned a result: which query (its index in the order plain, jargon, neighbor), which source, at which rank (1 first). */
export type LitScoutHit = { query: number; source: string; rank: number }

/** A paper the check read: its full text, or its abstract only and why. */
export type LitScoutPaper = {
  title: string
  year: number | null
  url: string
  access: LitScoutAccess
  /** Why the full text was not read; absent for full text. */
  reason?: string
  charsKept: number
  truncated: boolean
  /** Every search list that returned it. */
  foundBy: LitScoutHit[]
}

/**
 * A paper the reader kept: how it connects to the researcher's work, and the passage it quoted as support. Verified
 * when the quote is in the text the reader saw (the full text, or the abstract only) and every number in the
 * connection is in that text or the conversation; else `problem` says what the check found, after one repair round.
 */
export type LitScoutFinding = LitScoutPaper & { connection: string; quote: string; isVerified: boolean; problem?: string }

/** How the kept papers' lines fared against their texts: first pass, and the one repair round of the failed ones. */
export type LitScoutVerification = {
  /** Kept papers whose lines passed on the first check. */
  verified: number
  /** Failed lines the repair round fixed. */
  repaired: number
  /** Failed lines the reader dropped in the repair round. */
  dropped: number
  /** Kept papers still failing after the repair: shown as unverified. */
  unverified: number
}

/** How the session's model read the papers: a fork over the main thread, or a completion over the recent turns. */
export type LitScoutReader = 'fork' | 'complete'

export type LitScoutReport = {
  /** The idea checked. */
  query: string
  queries: LitScoutQueries
  verdict: LitScoutVerdictWord
  nextStep: string
  /** What the reader took the researcher's goal to be; empty when nothing was read. */
  goal: string
  /** The papers with a real connection to the researcher's work, in search rank order. */
  papers: LitScoutFinding[]
  /** The papers read and dropped as unrelated. */
  dropped: LitScoutPaper[]
  /** How the kept papers together bear on the plan; empty when nothing was read. */
  overall: string
  sourceStatus: Record<string, string>
}

/** How long each stage of a deep check took, in milliseconds. */
export type LitScoutTimings = {
  queries: number
  search: number
  /** Each source's own time over all queries, inside `search`. */
  sources: Record<string, number>
  fetch: number
  /** Each read paper's own fetch time, in the order read, inside `fetch`. */
  papers: number[]
  reader: number
  /** The check of every kept paper's quote and numbers; absent in checks recorded before it. */
  verify?: number
  /** The repair round, its re-check included; 0 when every line passed. */
  repair?: number
  total: number
}

/** What one deep check read, cost and took. */
export type LitScoutDeepCheck = {
  time: string
  query: string
  fullText: number
  abstractOnly: number
  charsSent: number
  reader: LitScoutReader
  model: string
  usage: LitScoutUsage
  /** Absent in checks recorded before timings were. */
  timings?: LitScoutTimings
  /** Absent in checks recorded before quotes were checked. */
  verification?: LitScoutVerification
}

export type LitScoutStage =
  | { kind: 'searching' }
  | { kind: 'fetching'; papers: number }
  | { kind: 'reading'; papers: number; model: string }
  | { kind: 'verifying'; papers: number }

export type LitScoutBand =
  | { kind: 'idle' }
  | { kind: 'offer'; offer: LitScoutOffer }
  | { kind: 'searching'; id: string; query: string; loadId: string; stage: LitScoutStage }
  | { kind: 'report'; report: LitScoutReport; promptsSince: number }

export type LitScoutSession = {
  band: LitScoutBand
  /** The newest idea that came while the band was taken; shown when it frees. */
  pending: LitScoutOffer | null
  ignoredInRow: number
  /** How many more ideas to drop unoffered, after a run of ignored offers. */
  ideasToSkip: number
}

export type LitScoutLogEntry = {
  id: string
  time: string
  haikuLabel: string
  /** The classifier's search query, when it answered IDEA. */
  query?: string
  /** Absent in entries written before usage was logged. */
  haikuUsage?: LitScoutUsage
  outcome: LitScoutOutcome
  answer: LitScoutAnswer | null
  promptHead: string
  detail?: string
}

declare module 'claude-code' {
  interface PluginState {
    'lit-scout': { session: Shaped<LitScoutSession> }
  }
}
