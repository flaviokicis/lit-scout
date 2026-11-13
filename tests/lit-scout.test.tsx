import type { ModelCompleteInput, ModelForkRequest, On, SessionMessage } from 'claude-code'
import { describe, expect, mock, test } from 'claude-code/testing'
import type { Engine, MockClock } from 'claude-code/testing'

import { CLASSIFIER_CALL, parseClassifier, QUERY_CALL, recentContext } from '../hooks/prompts'
import type { LitScoutDeepCheck, LitScoutLogEntry } from '../types'

const ROOT = '/work'
const SESSION_ID = 's1'
const LOG_KEY = `log:${SESSION_ID}`
const USAGE = { input_tokens: 10, output_tokens: 2, cache_read_input_tokens: 0, cache_creation_input_tokens: 0 }

const BAND = {
  plugin: 'lit-scout',
  component: 'AbovePrompt',
  requestId: 'band',
  props: {
    hasSurvey: false,
    isCollapsed: false,
    isWorking: false,
    maxRows: 20,
    bodyColumns: 100,
    scroll: { offset: 0, bodyRows: 20 },
    view: {},
  },
} as const

const SESSION_MODEL = 'claude-opus-5-5'
const FORK_USAGE = { input_tokens: 900, output_tokens: 1500, cache_read_input_tokens: 120_000, cache_creation_input_tokens: 40_000 }

const QUERIES = ['PLAIN: replace reward model with contrastive pairs', 'JARGON: contrastive preference learning without reward model', 'NEIGHBOR: direct preference optimization'].join('\n')

const RATE_LIMITED = { status: 'rate-limited', detail: 'HTTP 429 after 4 retries' }

const SEARCH_REPORT = {
  keyless: ['semantic_scholar'],
  queries: ['replace reward model with contrastive pairs', 'contrastive preference learning without reward model', 'direct preference optimization'],
  sourceMs: { openalex: 600, semantic_scholar: 8000 },
  sources: {
    openalex: [
      { status: 'ok', count: 2 },
      { status: 'ok', count: 0 },
      { status: 'ok', count: 0 },
    ],
    semantic_scholar: [{ status: 'ok', count: 3 }, RATE_LIMITED, RATE_LIMITED],
  },
  results: [
    {
      title: 'Contrastive Preference Learning',
      year: 2023,
      url: 'http://arxiv.org/abs/2310.13639v3',
      abstract: 'We learn from contrastive pairs.',
      arxivId: '2310.13639v3',
      pdfUrl: null,
      foundBy: [
        { query: 1, source: 'arxiv', rank: 1 },
        { query: 0, source: 'openalex', rank: 2 },
      ],
    },
    {
      title: 'Pairs in a Paywalled Journal',
      year: 2022,
      url: 'https://doi.org/10/pay',
      abstract: 'Pairs beat a reward model on two tasks.',
      arxivId: null,
      pdfUrl: null,
      foundBy: [{ query: 0, source: 'openalex', rank: 1 }],
    },
    { title: 'someone/pairs', year: 2024, url: 'https://github.com/someone/pairs', abstract: '', arxivId: null, pdfUrl: null, foundBy: [{ query: 2, source: 'github', rank: 1 }] },
  ],
}

const PAYWALLED = 'no arXiv id and no open-access link (paywalled or not a paper)'

const FETCHED = [
  { url: SEARCH_REPORT.results[0]?.url, access: 'full text', text: 'CPL FULL TEXT: we set beta to 0.5.', charsKept: 34, truncated: true, ms: 900 },
  { url: SEARCH_REPORT.results[1]?.url, access: 'abstract only', reason: PAYWALLED, text: '', charsKept: 0, truncated: false, ms: 0 },
  { url: SEARCH_REPORT.results[2]?.url, access: 'abstract only', reason: PAYWALLED, text: '', charsKept: 0, truncated: false, ms: 0 },
]

const CPL_CONNECTION = 'CPL drops the reward model the way your plan does; reuse its loss and its beta of 0.5.'
const CPL_QUOTE = 'we set beta to 0.5.'
const PAIRS_QUOTE = 'Pairs beat a reward model on two tasks.'

const READING = [
  'VERDICT: PARTLY DONE',
  'NEXT: Start from the CPL code and its beta of 0.5.',
  'GOAL: drop the noisy reward model.',
  `[1] ${CPL_CONNECTION}`,
  `QUOTE: "${CPL_QUOTE}"`,
  '[2] Its two tasks match yours,',
  '    so its result is your baseline (abstract only).',
  `QUOTE: ${PAIRS_QUOTE}`,
  '[3] DROP',
  'OVERALL: Pairs without a reward model are done [1, 3]; the noisy-labels angle is open [3].',
].join('\n')

/** What the quote check script answers for one paper. */
type QuoteCheck = { verified: boolean; method: string | null; coverage: number; problems: string[] }

const PASSED: QuoteCheck = { verified: true, method: 'exact', coverage: 1, problems: [] }

/** What the quote check script is asked: the conversation's text and each kept paper's text and line. */
type CheckRequest = { context: string; papers: { source: string; title: string; year: number | null; sentence: string; quote: string }[] }

/** What a model answers: its text and how many tokens it spent. */
type Reply = { text: string; outputTokens: number }

type Hold = { done: Promise<void>; release: () => void }

type World = {
  classifier: Reply
  queryWriter: Reply
  searchReport: typeof SEARCH_REPORT
  reader: Reply
  /** What the reader answers when asked to repair the lines that failed the quote check. */
  repairer: Reply
  /** False when the session has no main-thread snapshot to fork. */
  isForkable: boolean
  window: { window: number; tokens?: number }
  stored: Map<string, unknown>
  history: SessionMessage[]
  hasTreeBeneath: boolean
  files: Map<string, string>
  classifierCalls: ModelCompleteInput[]
  queryCalls: ModelCompleteInput[]
  readerCompletions: ModelCompleteInput[]
  forks: ModelForkRequest[]
  scripts: { argv: readonly string[]; stdin?: string; env?: Record<string, string> }[]
  /** The quote check's answers, one list per run, in order; past the last, every paper passes. */
  quoteChecks: QuoteCheck[][]
  checkRequests: CheckRequest[]
  toasts: string[]
  /** The prompt box's draft; null where the session draws no box, so a fill is refused. */
  draft: string | null
  clock: MockClock
  /** Holds the search script, the fetch script or the fork until the test releases it. */
  holds: { search?: Hold; fetch?: Hold; fork?: Hold }
}

function hold(): Hold {
  const held: { release?: () => void } = {}
  const done = new Promise<void>(resolve => {
    held.release = resolve
  })

  return { done, release: () => held.release?.() }
}

/** Seats the engine beneath the plugin: models, the scripts, the chat, files and the store, answered from memory. */
function worldOf(on: On, classifierText: string, initial: Record<string, unknown> = {}): World {
  const answer = (text: string, outputTokens: number) => ({ value: { isAnswered: true as const, text, usage: { ...USAGE, output_tokens: outputTokens } } })
  const pathOf = (path: string) => (path.startsWith('/') ? path : `${ROOT}/${path}`)
  const world: World = {
    classifier: { text: classifierText, outputTokens: USAGE.output_tokens },
    queryWriter: { text: QUERIES, outputTokens: USAGE.output_tokens },
    searchReport: SEARCH_REPORT,
    reader: { text: READING, outputTokens: FORK_USAGE.output_tokens },
    repairer: { text: '', outputTokens: 0 },
    isForkable: true,
    window: { window: 1_000_000, tokens: 100_000 },
    stored: new Map<string, unknown>(Object.entries(initial)),
    history: [],
    hasTreeBeneath: false,
    files: new Map<string, string>(),
    classifierCalls: [],
    queryCalls: [],
    readerCompletions: [],
    forks: [],
    scripts: [],
    quoteChecks: [],
    checkRequests: [],
    toasts: [],
    draft: '',
    clock: mock.clock(on, { now: Date.UTC(2026, 10, 10) }),
    holds: {},
  }

  on('store.get', ($, e) => ({ value: world.stored.get(e.key) }))
  on('store.set', ($, e) => {
    world.stored.set(e.key, e.value)

    return { value: undefined }
  })
  on('store.keys', () => ({ value: [...world.stored.keys()] }))
  on('session.id', () => ({ value: SESSION_ID }))
  on('session.root', () => ({ value: ROOT }))
  on('session.model', () => ({ value: SESSION_MODEL }))
  on('session.usage', () => ({ value: { startedAt: 0, context: world.window, rateLimits: [] } }))
  on('session.messages', () => ({ value: world.history }))
  on('model.complete', ($, e) => {
    if (e.model === 'haiku' && e.prompt.includes('IDEA TO CHECK:')) {
      world.queryCalls.push(e)
      return answer(world.queryWriter.text, world.queryWriter.outputTokens)
    }

    if (e.model === 'haiku') {
      world.classifierCalls.push(e)
      return answer(world.classifier.text, world.classifier.outputTokens)
    }

    world.readerCompletions.push(e)
    return answer(world.reader.text, world.reader.outputTokens)
  })
  on('model.fork', async ($, e) => {
    world.forks.push(e)
    await world.holds.fork?.done

    const reply = e.prompt.includes('LINES THAT FAILED') ? world.repairer : world.reader

    return world.isForkable
      ? { value: { isAnswered: true as const, text: reply.text, usage: FORK_USAGE } }
      : { value: { isAnswered: false as const, reason: 'nothing-to-fork' as const } }
  })
  on('process.run', async ($, e) => {
    if (e.argv[1]?.endsWith('/scripts/check_quotes.py')) {
      const request = JSON.parse(e.init?.stdin ?? '{}') as CheckRequest
      world.checkRequests.push(request)
      const checks = world.quoteChecks.shift() ?? request.papers.map(() => PASSED)

      return { value: { exitCode: 0, stdout: JSON.stringify(checks), stderr: '', isStdoutTruncated: false, isStderrTruncated: false } }
    }

    const command = e.argv[2]
    world.scripts.push({ argv: e.argv, ...(e.init?.stdin === undefined ? {} : { stdin: e.init.stdin }), ...(e.init?.env === undefined ? {} : { env: e.init.env }) })
    await (command === 'search' ? world.holds.search : world.holds.fetch)?.done
    const stdout = JSON.stringify(command === 'search' ? world.searchReport : FETCHED)

    return { value: { exitCode: 0, stdout, stderr: '', isStdoutTruncated: false, isStderrTruncated: false } }
  })
  on('fs.exists', ($, e) => ({ value: world.files.has(pathOf(e.path)) }))
  on('fs.read', ($, e) => {
    const text = world.files.get(pathOf(e.path))

    return text === undefined ? { deny: `ENOENT: ${e.path}` } : { value: text }
  })
  on('fs.write', ($, e) => {
    world.files.set(pathOf(e.path), e.text)

    return { value: undefined }
  })
  on('ui.log', () => ({ value: undefined }))
  on('ui.toast', ($, e) => {
    world.toasts.push(e.text)

    return { value: undefined }
  })
  on('prompt.submit', ($, e) => ({ text: e.text }))
  on('prompt.read', () => ({ value: { text: world.draft ?? '', cursor: (world.draft ?? '').length } }))
  on('prompt.fill', ($, e) => {
    if (world.draft === null) {
      return { isFilled: false, refusal: 'no_composer' as const }
    }

    world.draft = e.mode === 'append' ? world.draft + e.text : e.text

    return { isFilled: true }
  })
  on('ui.render', ($, e, next) => {
    if (!world.hasTreeBeneath) {
      return next(e)
    }

    const { Text } = $.ui.resolve(e)

    return <Text>you should know this</Text>
  })

  return world
}

function logOf(world: World): LitScoutLogEntry[] {
  return (world.stored.get(LOG_KEY) ?? []) as LitScoutLogEntry[]
}

function said(role: 'user' | 'assistant', text: string, extra: Partial<SessionMessage> = {}): SessionMessage {
  return { role, text, toolUses: [], ...extra }
}

async function submit($: Engine, world: World, text: string): Promise<void> {
  await $.prompt.submit({ text, wait: false, origin: { kind: 'composer' } })
  await world.clock.settle()
}

const LIT = 'LIT: contrastive pairs reward model'

const STATS = { command: 'lit-scout', args: '', origin: { kind: 'composer' }, presentation: { isFullscreen: false, columns: 100 } } as const

const lit = (query: string) => `LIT: ${query}`

describe('reading the models', () => {
  test('the replies are parsed, and a reply of neither form fails loud', () => {
    expect(parseClassifier('LIT: sparse attention for long context')).toEqual({ label: 'LIT', query: 'sparse attention for long context' })
    expect(parseClassifier('OTHER')).toEqual({ label: 'OTHER' })
    expect(() => parseClassifier('Maybe?')).toThrow(/neither LIT nor OTHER/)
  })

  test('the context holds the last three exchanges as visible text, keeps the newest and stays under the cap', () => {
    const history = [
      said('user', 'exchange one'),
      said('assistant', 'reply one'),
      ...[2, 3, 4].flatMap(n => [said('user', `exchange ${n}`), said('assistant', `reply ${n} ${'x'.repeat(20_000)}`)]),
      said('user', '', { toolResults: [{ tool_use_id: 't1', text: 'RAW TOOL OUTPUT', isError: false, result: null }] }),
      said('user', 'newest'),
    ]
    const context = recentContext(history, 'newest')

    expect(context).not.toContain('exchange one')
    expect(context).not.toContain('RAW TOOL OUTPUT')
    expect(context).toContain('exchange 4')
    expect(context).toMatch(/NEWEST USER PROMPT:\nnewest$/)
    expect(context.split('newest').length, 'the newest prompt, already in the chat, is not repeated').toBe(2)
    expect(context.length).toBeLessThanOrEqual(32_100)
    expect(recentContext([], 'p'.repeat(50_000)).length, 'a long pasted prompt is cut in the middle').toBeLessThanOrEqual(16_200)
  })
})

describe('the trigger pipeline', () => {
  test('Haiku reads the recent turns at low effort, and its LIT query goes straight to the offer', async ($, on) => {
    const world = worldOf(on, LIT)
    world.history = [said('user', 'the reward model is noisy'), said('assistant', 'We could replace the reward model with contrastive pairs.')]

    await submit($, world, 'yes, do that')

    const call = world.classifierCalls[0]
    expect(call?.prompt).toContain('We could replace the reward model with contrastive pairs.')
    expect(call?.prompt).toContain('NEWEST USER PROMPT:\nyes, do that')
    expect([call?.effort, call?.maxTokens]).toEqual(['low', CLASSIFIER_CALL.maxTokens])
    const log = logOf(world)
    expect(log.map(entry => [entry.haikuLabel, entry.query, entry.outcome])).toEqual([['LIT', 'contrastive pairs reward model', 'shown']])
    expect(log[0]?.haikuUsage).toEqual(USAGE)

    for (const surface of ['terminal', 'desktop'] as const) {
      const ui = await $.ui.mount({ ...BAND, surface })
      expect((await ui.find({ type: 'Text', text: /Prior art\? contrastive pairs reward model/ })) !== undefined).toBe(true)
      expect((await ui.find({ key: 'check' }))?.text).toBe('Check prior art')
      await ui.unmount()
    }
  })

  test('OTHER makes no offer and logs the check with its Haiku cost', async ($, on) => {
    const world = worldOf(on, 'OTHER')

    await submit($, world, 'run the tests and fix what fails')

    const log = logOf(world)
    expect(log.map(entry => [entry.outcome, entry.promptHead, entry.haikuUsage?.input_tokens])).toEqual([['none', 'run the tests and fix what fails', 10]])
    const stats = (await $.command.run(STATS)).text
    expect(stats).toContain('run the tests and fix what fails')
    expect(stats).toContain('Haiku calls: 1; tokens: input 10')
  })

  test('an empty reply, or one cut at the token cap, is an error with a toast, never OTHER', async ($, on) => {
    const world = worldOf(on, '')

    await submit($, world, 'what if we prune heads')
    world.classifier = { text: 'LIT: head pruning', outputTokens: CLASSIFIER_CALL.maxTokens }
    await submit($, world, 'what if we prune heads by entropy')

    expect(logOf(world).map(entry => [entry.haikuLabel, entry.outcome, entry.detail])).toEqual([
      ['ERROR', 'error', 'the reply was empty'],
      ['ERROR', 'error', `the reply hit the ${CLASSIFIER_CALL.maxTokens}-token cap`],
    ])
    expect(world.toasts.length).toBe(2)
    const ui = await $.ui.mount({ ...BAND, surface: 'terminal' })
    expect(await ui.find({ key: 'check' })).toBeUndefined()
    await ui.unmount()
  })
})

describe('anti-spam', () => {
  test('an idea offered before is deduped and not shown', async ($, on) => {
    const world = worldOf(on, LIT, { offered: ['contrastive pairs reward model'] })

    await submit($, world, 'contrastive pairs over a reward model')

    expect(logOf(world).map(entry => entry.outcome)).toEqual(['deduped'])
    const ui = await $.ui.mount({ ...BAND, surface: 'terminal' })
    expect(await ui.find({ key: 'check' })).toBeUndefined()
    await ui.unmount()
  })

  test('ideas checked or marked "I know the lit" are listed to Haiku on every later prompt, and never offered again', async ($, on) => {
    const world = worldOf(on, lit('idea one'))
    const ui = await $.ui.mount({ ...BAND, surface: 'terminal' })

    await submit($, world, 'what if idea one')
    await ui.press({ key: 'know' })
    world.classifier = { text: lit('Idea Two'), outputTokens: 5 }
    await submit($, world, 'what if idea two')
    await ui.press({ key: 'check' })
    await world.clock.settle()
    world.classifier = { text: lit('idea one'), outputTokens: 5 }
    await submit($, world, 'lets run idea one')
    await ui.unmount()

    const heading = 'IDEAS ALREADY CHECKED THIS SESSION (a prior-art check ran, or the researcher said they know the literature):'
    expect(world.classifierCalls[0]?.prompt).toContain(`${heading}\n(none)\n`)
    expect(world.classifierCalls[2]?.prompt).toContain(`${heading}\n- idea one\n- idea two\n`)
    expect(world.stored.get(`checked:${SESSION_ID}`)).toEqual(['idea one', 'idea two'])
    expect(logOf(world).map(entry => `${entry.outcome}:${entry.answer ?? '-'}`)).toEqual(['shown:know', 'shown:check', 'deduped:-'])
  })

  test('an unanswered offer expires after two later prompts', async ($, on) => {
    const world = worldOf(on, LIT)

    await submit($, world, 'idea one')
    world.classifier = { text: 'OTHER', outputTokens: 2 }
    await submit($, world, 'status please')
    await submit($, world, 'status again')

    const log = logOf(world)
    expect(log[0]?.answer).toBe('expired')
    expect(log[0]?.outcome, 'never drawn: the band yielded the whole time').toBe('yielded')
    const ui = await $.ui.mount({ ...BAND, surface: 'terminal' })
    expect(await ui.find({ key: 'check' }), 'the band is empty again').toBeUndefined()
    await ui.unmount()
  })

  test('after three dismissed offers in a row the next idea is skipped as backoff, then offers resume', async ($, on) => {
    const world = worldOf(on, LIT)
    const ui = await $.ui.mount({ ...BAND, surface: 'terminal' })

    for (const round of [1, 2, 3]) {
      world.classifier = { text: lit(`idea ${round}`), outputTokens: 5 }
      await submit($, world, `what if idea ${round}`)
      await ui.press({ key: 'dismiss' })
    }

    for (const round of [4, 5]) {
      world.classifier = { text: lit(`idea ${round}`), outputTokens: 5 }
      await submit($, world, `what if idea ${round}`)
    }
    await ui.unmount()

    expect(logOf(world).map(entry => `${entry.outcome}:${entry.answer ?? '-'}`)).toEqual([
      'shown:dismiss',
      'shown:dismiss',
      'shown:dismiss',
      'backoff:-',
      'shown:-',
    ])
  })
})

describe('sharing the band', () => {
  test('the band yields to a survey, and stacks with the row of another plugin', async ($, on) => {
    const world = worldOf(on, LIT)
    await submit($, world, 'what if contrastive pairs')

    const survey = await $.ui.mount({ ...BAND, surface: 'terminal', props: { ...BAND.props, hasSurvey: true } })
    expect(await survey.find({ key: 'check' })).toBeUndefined()
    await survey.unmount()

    world.hasTreeBeneath = true
    const shared = await $.ui.mount({ ...BAND, surface: 'terminal', requestId: 'band-2' })
    expect((await shared.find({ type: 'Text', text: /you should know/ })) !== undefined, 'the other plugin\'s row stays').toBe(true)
    expect((await shared.find({ key: 'check' })) !== undefined, 'the engine asks each plugin for its own row, so the offer draws too').toBe(true)
    await shared.unmount()
  })

  test('an idea found while the band is taken waits, only the newest is kept, and it shows when the band frees', async ($, on) => {
    const world = worldOf(on, LIT)
    world.holds.search = hold()
    const ui = await $.ui.mount({ ...BAND, surface: 'terminal' })

    await submit($, world, 'what if contrastive pairs')
    await ui.press({ key: 'check' })
    expect((await ui.find({ type: 'Text', text: /Searching prior art/ })) !== undefined).toBe(true)

    world.classifier = { text: lit('idea B'), outputTokens: 5 }
    await submit($, world, 'what about B')
    world.classifier = { text: lit('idea C'), outputTokens: 5 }
    await submit($, world, 'what about C')
    expect(await ui.find({ type: 'Text', text: /idea C/ }), 'C waits while the search holds the band').toBeUndefined()

    await ui.press({ key: 'stop' })
    expect((await ui.find({ type: 'Text', text: /idea C/ })) !== undefined).toBe(true)
    expect(await ui.find({ type: 'Text', text: /idea B/ })).toBeUndefined()

    world.holds.search.release()
    await world.clock.settle()
    expect((await ui.find({ type: 'Text', text: /idea C/ })) !== undefined, 'the dismissed search does not take the band back').toBe(true)
    expect(world.files.get(`${ROOT}/PRIOR_ART.md`)).toContain('contrastive pairs reward model')
    await ui.unmount()

    expect(logOf(world).map(entry => `${entry.query}:${entry.outcome}`)).toEqual([
      'contrastive pairs reward model:shown',
      'idea B:superseded',
      'idea C:shown',
    ])
  })
})

describe('the store', () => {
  test('prompts checked at once all keep their log entries, under this session\'s key, beside other sessions\'', async ($, on) => {
    const other: LitScoutLogEntry = { id: 'x', time: '2026-11-09T00:00:00.000Z', haikuLabel: 'OTHER', outcome: 'none', answer: null, promptHead: 'from another session' }
    const world = worldOf(on, 'OTHER', { 'log:s0': [other] })

    await Promise.all(['first', 'second', 'third'].map(text => $.prompt.submit({ text, wait: false, origin: { kind: 'composer' } })))
    await world.clock.settle()

    expect(logOf(world).map(entry => entry.promptHead).sort()).toEqual(['first', 'second', 'third'])
    expect(world.stored.get('log:s0')).toEqual([other])
    const stats = (await $.command.run(STATS)).text
    expect(stats).toContain('lit-scout: 4 checks')
  })
})

describe('the deep check', () => {
  test('pressing 1 writes three queries, searches them, fetches the top results, has a fork of the session connect them to the work, shows each stage and only the relevant papers; 1 adds the findings, 2 the links', async ($, on) => {
    const world = worldOf(on, LIT)
    const session = mock.session(on)
    world.holds = { fetch: hold(), fork: hold() }
    world.history = [said('user', 'the reward model is noisy'), said('assistant', 'We could learn from preference pairs directly.')]
    await submit($, world, 'what if contrastive pairs')

    const ui = await $.ui.mount({ ...BAND, surface: 'terminal' })
    await ui.press({ key: 'check' })
    expect((await ui.find({ type: 'Text', text: /Fetching 3 papers for: contrastive pairs reward model/ })) !== undefined).toBe(true)
    world.holds.fetch?.release()
    await world.clock.settle()
    expect((await ui.find({ type: 'Text', text: new RegExp(`Reading 3 papers with ${SESSION_MODEL}`) })) !== undefined).toBe(true)
    world.holds.fork?.release()
    await world.clock.settle()

    const queryCall = world.queryCalls[0]
    expect([world.queryCalls.length, queryCall?.effort, queryCall?.maxTokens], 'one Haiku call per check, at low effort').toEqual([1, 'low', QUERY_CALL.maxTokens])
    expect(queryCall?.prompt).toContain('IDEA TO CHECK: contrastive pairs reward model')
    expect(queryCall?.prompt).toContain('We could learn from preference pairs directly.')
    const [search, fetch] = world.scripts
    expect(search?.argv.slice(2), 'the plain, jargon and neighbor queries, in that order').toEqual(['search', ...SEARCH_REPORT.queries])
    expect(fetch?.argv.slice(2), 'a 1M window with room to spare caps each paper at 25k characters').toEqual(['fetch', '--max-chars', '25000'])
    expect(JSON.parse(fetch?.stdin ?? '[]'), 'the top results by search rank, links only, with no model picking among them').toEqual(
      SEARCH_REPORT.results.map(({ url, arxivId, pdfUrl }) => ({ url, arxivId, pdfUrl })),
    )
    expect(world.readerCompletions, 'no model call besides the query writer and the reader').toEqual([])
    const fork = world.forks[0]
    expect(fork?.effort, 'the fork runs at the session effort, so it reads the cached conversation').toBeUndefined()
    expect(fork?.prompt).toContain('CPL FULL TEXT: we set beta to 0.5.')
    expect(fork?.prompt).toContain(`ABSTRACT ONLY (${PAYWALLED})\nURL: https://doi.org/10/pay\nPairs beat a reward model on two tasks.`)
    expect(fork?.prompt).toContain('Never invent results')
    expect(fork?.prompt).toContain('as much of the conversation above as you can see')
    expect(world.forks.length, 'every quote passed, so no repair round').toBe(1)
    expect(
      world.checkRequests[0]?.papers,
      'the check gets each kept paper\'s text exactly as the reader saw it (the abstract for an abstract-only paper), its line and its quote',
    ).toEqual([
      { source: 'CPL FULL TEXT: we set beta to 0.5.', title: 'Contrastive Preference Learning', year: 2023, sentence: CPL_CONNECTION, quote: CPL_QUOTE },
      { source: 'Pairs beat a reward model on two tasks.', title: 'Pairs in a Paywalled Journal', year: 2022, sentence: 'Its two tasks match yours, so its result is your baseline (abstract only).', quote: PAIRS_QUOTE },
    ])
    expect(world.checkRequests[0]?.context, 'numbers may come from the conversation').toContain('We could learn from preference pairs directly.')

    const findings = [
      '**PARTLY DONE**: Start from the CPL code and its beta of 0.5.',
      '',
      'Read 3 papers, 2 relevant:',
      '1. [Contrastive Preference Learning](http://arxiv.org/abs/2310.13639v3) (2023)',
      `   - ${CPL_CONNECTION}`,
      `   - *"${CPL_QUOTE}"*`,
      '2. [Pairs in a Paywalled Journal](https://doi.org/10/pay) (2022, abstract only)',
      '   - Its two tasks match yours, so its result is your baseline (abstract only).',
      `   - *"${PAIRS_QUOTE}"* (from the abstract)`,
    ].join('\n')
    const drawn = (await ui.findAll({})).filter(element => element.type === 'Markdown' || element.type === 'Button')
    expect(
      drawn.map(element => element.key),
      'the buttons sit right under the verdict, before the paper list, so a long list can never push them out of the band and disarm the hotkeys',
    ).toEqual(['verdict', 'add', 'links', 'close', 'papers'])
    expect(
      drawn.filter(element => element.type === 'Markdown').map(element => element.text).join('\n\n'),
      'the band: verdict and next step, then each relevant paper with how it connects; the dropped one is not shown',
    ).toBe(findings)

    const priorArt = world.files.get(`${ROOT}/PRIOR_ART.md`) ?? ''
    expect(priorArt).toMatch(/^# Prior art checks/)
    expect(priorArt).toContain('## 2026-11-10T00:00:00.000Z: contrastive pairs reward model')
    expect(priorArt).toContain(
      'Queries: plain `replace reward model with contrastive pairs`; jargon `contrastive preference learning without reward model`; neighbor `direct preference optimization`',
    )
    expect(priorArt).toContain('Goal as understood: drop the noisy reward model.')
    expect(priorArt).toContain('(2023; found by arxiv #1 for jargon, openalex #2 for plain)')
    expect(priorArt).toContain(`(2022, abstract only: ${PAYWALLED}; found by openalex #1 for plain)`)
    expect(priorArt, 'citations keep only the kept papers').toContain('Overall: Pairs without a reward model are done [1]; the noisy-labels angle is open.')
    expect(priorArt).toContain('Read and dropped as unrelated:\n1. [someone/pairs](https://github.com/someone/pairs)')
    expect(priorArt).toContain(
      'Sources: openalex ok (2/0/0); semantic_scholar plain ok 3, jargon rate-limited: HTTP 429 after 4 retries, neighbor rate-limited: HTTP 429 after 4 retries [no API key: shared rate pool]',
    )

    const buttons = (await ui.findAll({ type: 'Button' })).map(button => [button.props.hotkey, button.props.label])
    expect(buttons, 'the report band\'s buttons, in order').toEqual([
      ['1', 'Add findings to chat'],
      ['2', 'Add links only'],
      ['0', 'Dismiss'],
    ])
    await ui.press({ key: 'add' })
    expect(await ui.find({ type: 'Markdown' })).toBeUndefined()
    await ui.unmount()

    const stats = (await $.command.run(STATS)).text
    expect(stats).toContain(`Deep checks: 1; papers read in full text 1, from the abstract only 2; characters sent ${fork?.prompt.length}.`)
    expect(stats).toContain('Reader calls: 1; tokens: input 900, output 1500, cache read 120000, cache write 40000.')
    expect(stats, 'each deep check shows its total time').toMatch(/40000\/1500 tokens, \d+ s: contrastive pairs reward model/)
    expect(logOf(world)[0]?.answer).toBe('check')
    expect(world.toasts).toEqual([])

    expect(world.draft, 'Add findings to chat pastes what the band shows into the prompt box, where the person sees it and sends it').toBe(
      `Prior-art check by the lit-scout mod for "contrastive pairs reward model":\n\n${findings}\n\n`,
    )
    expect(session.appended(), 'no hidden row the person never sees and the model reads only on some later turn').toEqual([])
  })

  test('a line whose quote fails goes back to the reader once; a fixed line that passes replaces it, one that does not change stays and is marked unverified', async ($, on) => {
    const world = worldOf(on, LIT)
    mock.session(on)
    const repoLine = 'The repo trains on pairs as your plan does.'
    const repoQuote = 'someone/pairs trains a policy on preference pairs alone'
    const notFound = 'the quote is not in the paper text (the closest passage matches 40% of it, and any changed word or number fails)'
    world.reader = {
      text: READING.replace('[3] DROP', `[3] ${repoLine}\nQUOTE: ${repoQuote}`).replace(`QUOTE: ${PAIRS_QUOTE}`, 'QUOTE: Pairs beat a reward model on three tasks.'),
      outputTokens: 100,
    }
    world.repairer = { text: [`[2] Its two tasks match yours.`, `QUOTE: ${PAIRS_QUOTE}`, `[3] ${repoLine}`, `QUOTE: ${repoQuote}`].join('\n'), outputTokens: 50 }
    const failed = (problem: string): QuoteCheck => ({ verified: false, method: null, coverage: 0.4, problems: [problem] })
    world.quoteChecks = [[PASSED, failed(notFound), failed(notFound)]]
    await submit($, world, 'what if contrastive pairs')

    const ui = await $.ui.mount({ ...BAND, surface: 'terminal' })
    await ui.press({ key: 'check' })
    await world.clock.settle()

    const repair = world.forks[1]?.prompt ?? ''
    expect(world.forks.length, 'one repair round').toBe(2)
    expect(repair, 'only the failed papers go back, with their texts and what the check found').toContain(`CHECK: ${notFound}`)
    expect(repair).toContain('=== [2] Pairs in a Paywalled Journal')
    expect(repair).toContain('=== [3] someone/pairs')
    expect(repair).not.toContain('CPL FULL TEXT')
    expect(
      world.checkRequests.map(request => request.papers.map(paper => paper.quote)),
      'the re-check takes only the changed line; the unchanged one is not checked again',
    ).toEqual([[CPL_QUOTE, 'Pairs beat a reward model on three tasks.', repoQuote], [PAIRS_QUOTE]])

    const band = (await ui.findAll({ type: 'Markdown' })).map(element => element.text).join('\n\n')
    await ui.unmount()
    expect(band).toContain(`2. [Pairs in a Paywalled Journal](https://doi.org/10/pay) (2022, abstract only)\n   - Its two tasks match yours.\n   - *"${PAIRS_QUOTE}"* (from the abstract)`)
    expect(band).toContain(`3. [someone/pairs](https://github.com/someone/pairs) (2024, abstract only, unverified)\n   - ${repoLine}`)
    expect(world.files.get(`${ROOT}/PRIOR_ART.md`)).toContain(`(2024, abstract only: ${PAYWALLED}, unverified: ${notFound}; found by github #1 for neighbor)`)
    const [check] = world.stored.get(`deep:${SESSION_ID}`) as LitScoutDeepCheck[]
    expect(check?.verification).toEqual({ verified: 1, repaired: 1, dropped: 0, unverified: 1 })
    expect(check?.usage.output_tokens, 'the repair round counts toward the reader\'s cost').toBe(FORK_USAGE.output_tokens * 2)
  })

  test('pressing 2 on a report adds only the relevant papers\' links, after what the person has typed', async ($, on) => {
    const world = worldOf(on, LIT)
    const session = mock.session(on)
    await submit($, world, 'what if contrastive pairs')
    world.draft = 'which of these should I start from?'

    const ui = await $.ui.mount({ ...BAND, surface: 'terminal' })
    await ui.press({ key: 'check' })
    await world.clock.settle()
    await ui.press({ key: 'links' })
    await ui.unmount()

    expect(world.draft).toBe(
      [
        'which of these should I start from?',
        '',
        'Prior-art papers found by the lit-scout mod for "contrastive pairs reward model":',
        '',
        '- [Contrastive Preference Learning](http://arxiv.org/abs/2310.13639v3)',
        '- [Pairs in a Paywalled Journal](https://doi.org/10/pay)',
        '',
        '',
      ].join('\n'),
    )
    expect(session.appended()).toEqual([])
  })

  test('where no prompt box takes the findings, the report stays and a toast says why', async ($, on) => {
    const world = worldOf(on, LIT)
    mock.session(on)
    await submit($, world, 'what if contrastive pairs')
    world.draft = null

    const ui = await $.ui.mount({ ...BAND, surface: 'terminal' })
    await ui.press({ key: 'check' })
    await world.clock.settle()
    await ui.press({ key: 'add' })
    await world.clock.settle()

    expect(world.toasts).toEqual([expect.stringContaining('lit-scout: adding the findings failed: the prompt box did not take the report')])
    expect((await ui.find({ key: 'add' })) !== undefined, 'the report band stays, so the person can retry').toBe(true)
    await ui.unmount()
  })

  test('API keys set in the plugin options reach the search script as env', { options: { openalexKey: ' oa-test ', s2Key: '' } }, async ($, on) => {
    const world = worldOf(on, LIT)
    mock.session(on)
    await submit($, world, 'what if contrastive pairs')
    const ui = await $.ui.mount({ ...BAND, surface: 'terminal' })
    await ui.press({ key: 'check' })
    await world.clock.settle()
    expect(world.scripts[0]?.env, 'a set key is trimmed and passed; an empty one is left out').toEqual({ OPENALEX_API_KEY: 'oa-test' })
    expect(world.files.get(`${ROOT}/PRIOR_ART.md`) ?? '', 'a source that ran without a key says so in the report').toContain('[no API key: shared rate pool]')
    await ui.unmount()
  })

  test('with no main-thread snapshot to fork, the session model reads by completion over the recent turns; a full window shrinks each paper', async ($, on) => {
    const world = worldOf(on, 'OTHER')
    world.isForkable = false
    world.window = { window: 200_000, tokens: 150_000 }
    world.history = [said('user', 'the reward model is noisy, plan: replace it with pairs')]

    const started = await $.command.run({ ...STATS, args: 'check contrastive pairs reward model' })
    await world.clock.settle()

    expect(started.text).toMatch(/checking prior art/)
    expect(world.scripts[1]?.argv.slice(2), '10k tokens of room over three papers at 3 characters a token').toEqual(['fetch', '--max-chars', '10000'])
    const completion = world.readerCompletions[0]
    expect(completion?.prompt).toContain('the reward model is noisy, plan: replace it with pairs')
    expect(completion?.prompt).toContain('CPL FULL TEXT')
    expect(world.files.get(`${ROOT}/PRIOR_ART.md`)).toContain('**PARTLY DONE**')
    expect((await $.command.run(STATS)).text).toContain('Deep checks by reader: complete 1')
    expect(world.toasts).toEqual([])
  })

  test('a search with no hits reads nothing and says NO MATCHES FOUND, never NEW, with the query, each source and how to rephrase', async ($, on) => {
    const world = worldOf(on, LIT)
    world.searchReport = { ...SEARCH_REPORT, sources: { ...SEARCH_REPORT.sources, semantic_scholar: [RATE_LIMITED, RATE_LIMITED, RATE_LIMITED] }, results: [] }
    await submit($, world, 'what if contrastive pairs')

    const ui = await $.ui.mount({ ...BAND, surface: 'terminal' })
    await ui.press({ key: 'check' })
    await world.clock.settle()

    const band = (await ui.findAll({ type: 'Markdown' })).map(element => element.text).join('\n\n')
    await ui.unmount()
    const priorArt = world.files.get(`${ROOT}/PRIOR_ART.md`) ?? ''
    for (const shown of [band, priorArt]) {
      expect(shown).toContain('**NO MATCHES FOUND**')
      expect(shown).not.toContain('**NEW**')
      expect(shown).toContain('/lit-scout check <other words>')
      expect(shown).toContain('Idea: `contrastive pairs reward model`')
      expect(shown).toContain('Queries: plain `replace reward model with contrastive pairs`;')
      expect(shown).toContain('Sources: openalex ok (2/0/0); semantic_scholar rate-limited: HTTP 429 after 4 retries')
    }
    expect([world.scripts.length, world.forks.length], 'no fetch, no read').toEqual([1, 0])
    expect((await $.command.run(STATS)).text).toContain('Deep checks: 0;')
  })

  test('a query writer reply without the three query lines fails loud before any search', async ($, on) => {
    const world = worldOf(on, LIT)
    world.queryWriter = { text: 'PLAIN: pairs instead of a reward model', outputTokens: 10 }
    await submit($, world, 'what if contrastive pairs')

    const ui = await $.ui.mount({ ...BAND, surface: 'terminal' })
    await ui.press({ key: 'check' })
    await world.clock.settle()
    await ui.unmount()

    expect(world.toasts.some(text => /lacks a PLAIN, JARGON or NEIGHBOR line/.test(text))).toBe(true)
    expect(world.scripts).toEqual([])
    expect(logOf(world)[0]?.detail).toMatch(/scout error: the query writer/)
  })

  test('a reader reply with no verdict fails loud and frees the band', async ($, on) => {
    const world = worldOf(on, LIT)
    world.reader = { text: 'I think it is new.', outputTokens: 10 }
    await submit($, world, 'what if contrastive pairs')

    const ui = await $.ui.mount({ ...BAND, surface: 'terminal' })
    await ui.press({ key: 'check' })
    await world.clock.settle()

    expect(world.toasts.some(text => /lacks a known VERDICT/.test(text))).toBe(true)
    expect(await ui.find({ type: 'Markdown' })).toBeUndefined()
    expect(await ui.find({ type: 'Text', text: /Reading/ })).toBeUndefined()
    expect(logOf(world)[0]?.detail).toMatch(/scout error: .*VERDICT/)
    expect(world.files.has(`${ROOT}/PRIOR_ART.md`)).toBe(false)
    await ui.unmount()
  })
})
