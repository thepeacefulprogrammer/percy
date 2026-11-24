const STORAGE_KEY = '531-companion-state-v1';
const APP_TITLE = '5/3/1 Companion';
const LIFT_ORDER = ['press', 'bench', 'squat', 'deadlift'];
const LIFT_LABELS = {
  press: 'Press',
  bench: 'Bench Press',
  squat: 'Squat',
  deadlift: 'Deadlift',
};
const LIFT_CATEGORY = {
  press: 'upper',
  bench: 'upper',
  squat: 'lower',
  deadlift: 'lower',
};
const ASSISTANCE_CATEGORIES = ['push', 'pull', 'single_leg_core'];
const ASSISTANCE_LABELS = {
  push: 'Push',
  pull: 'Pull',
  single_leg_core: 'Single leg / core',
};
const ASSISTANCE_GUIDANCE = {
  push: {
    summary: 'Pick 1 push exercise and chip away at the reps in clean, easy-to-moderate sets.',
    examples: ['Push-ups', 'Dips', 'DB bench', 'DB incline', 'Triceps pushdowns'],
  },
  pull: {
    summary: 'Balance the pressing with rows, pull-ups, or pulldowns. Smooth reps matter more than load.',
    examples: ['Pull-ups', 'Chin-ups', 'DB rows', 'Cable rows', 'Lat pulldowns'],
  },
  single_leg_core: {
    summary: 'Finish with single-leg or trunk work that builds stability without wrecking recovery.',
    examples: ['Lunges', 'Split squats', 'Back raises', 'Hanging leg raises', 'Ab wheel'],
  },
};
const ASSISTANCE_TEMPLATE_NOTES = {
  vanilla: 'Classic 5/3/1: do one movement from each assistance category and stop with a few reps left in the tank.',
  beginner: 'Beginner 5/3/1 already includes two main lifts. Keep assistance simple and easy—one exercise per category is enough.',
  fsl: 'FSL already adds 5x5 supplemental work, so keep assistance moderate and avoid grinding.',
  bbb: 'BBB already adds a lot of volume. Stay near the low end of the assistance rep range and keep it very manageable.',
};
const TAB_ITEMS = [
  { key: 'home', label: 'Home', icon: '⌂' },
  { key: 'workout', label: 'Lift', icon: '🏋' },
  { key: 'history', label: 'History', icon: '↺' },
  { key: 'program', label: 'Program', icon: '▤' },
  { key: 'settings', label: 'Settings', icon: '⚙' },
];
const WEEK_WAVES = [
  {
    key: '5s',
    label: 'Week 1 · 5s',
    shortLabel: '5s',
    sets: [
      { pct: 0.65, reps: 5, style: 'fixed' },
      { pct: 0.75, reps: 5, style: 'fixed' },
      { pct: 0.85, reps: 5, style: 'plus' },
    ],
  },
  {
    key: '3s',
    label: 'Week 2 · 3s',
    shortLabel: '3s',
    sets: [
      { pct: 0.7, reps: 3, style: 'fixed' },
      { pct: 0.8, reps: 3, style: 'fixed' },
      { pct: 0.9, reps: 3, style: 'plus' },
    ],
  },
  {
    key: '531',
    label: 'Week 3 · 5/3/1',
    shortLabel: '5/3/1',
    sets: [
      { pct: 0.75, reps: 5, style: 'fixed' },
      { pct: 0.85, reps: 3, style: 'fixed' },
      { pct: 0.95, reps: 1, style: 'plus' },
    ],
  },
];
const RECOVERY_WAVES = {
  deload: {
    key: 'deload',
    label: 'Recovery week · Deload',
    shortLabel: 'Deload',
    sets: [
      { pct: 0.4, reps: 5, style: 'fixed' },
      { pct: 0.5, reps: 5, style: 'fixed' },
      { pct: 0.6, reps: 5, style: 'fixed' },
    ],
  },
  tm_test: {
    key: 'tm_test',
    label: 'Recovery week · TM test',
    shortLabel: 'TM test',
    sets: [
      { pct: 0.7, reps: 5, style: 'fixed' },
      { pct: 0.8, reps: 3, style: 'fixed' },
      { pct: 0.9, reps: 1, style: 'plus' },
    ],
  },
};
const ALL_WAVES = [...WEEK_WAVES, ...Object.values(RECOVERY_WAVES)];
const DEFAULT_ASSISTANCE_FAVORITES = {
  push: '',
  pull: '',
  single_leg_core: '',
};
const DEFAULT_REST_TIMER_SECONDS = {
  main: 180,
  supplemental: 120,
  assistance: 75,
};
const TEMPLATE_PRESETS = {
  vanilla: {
    slug: 'vanilla',
    name: 'Vanilla 5/3/1',
    summary: 'Classic main work only, plus assistance categories.',
    mainStyle: 'plus',
    supplementalStyle: 'none',
    jokerSets: false,
  },
  beginner: {
    slug: 'beginner',
    name: 'Beginner 5/3/1',
    summary: 'Two main lifts per session with simple 5x5 support.',
    mainStyle: 'plus',
    supplementalStyle: 'beginner_secondary',
    jokerSets: false,
  },
  fsl: {
    slug: 'fsl',
    name: 'FSL 5x5',
    summary: 'Classic main work plus First Set Last 5x5.',
    mainStyle: 'plus',
    supplementalStyle: 'fsl',
    jokerSets: false,
  },
  bbb: {
    slug: 'bbb',
    name: 'BBB 5x10',
    summary: 'Classic main work plus Boring But Big 5x10.',
    mainStyle: 'plus',
    supplementalStyle: 'bbb',
    jokerSets: false,
  },
  five_pro_fsl: {
    slug: 'five_pro_fsl',
    name: "5's Pro + FSL",
    summary: 'All main sets for 5 reps, plus First Set Last 5x5.',
    mainStyle: 'five_pro',
    supplementalStyle: 'fsl',
    jokerSets: false,
  },
  five_pro_bbb: {
    slug: 'five_pro_bbb',
    name: "5's Pro + BBB",
    summary: 'All main sets for 5 reps, plus Boring But Big 5x10.',
    mainStyle: 'five_pro',
    supplementalStyle: 'bbb',
    jokerSets: false,
  },
  leader_fsl: {
    slug: 'leader_fsl',
    name: 'Leader · 5s Pro FSL',
    summary: 'Conservative leader setup with 5s Pro and First Set Last 5x5.',
    mainStyle: 'five_pro',
    supplementalStyle: 'fsl',
    jokerSets: false,
  },
  anchor_fsl: {
    slug: 'anchor_fsl',
    name: 'Anchor · Jokers FSL',
    summary: 'Anchor-style setup with plus sets, joker singles, and First Set Last 5x5.',
    mainStyle: 'plus',
    supplementalStyle: 'fsl',
    jokerSets: true,
  },
  joker_fsl: {
    slug: 'joker_fsl',
    name: 'Jokers + FSL',
    summary: 'Classic 5/3/1 with joker singles and First Set Last 5x5.',
    mainStyle: 'plus',
    supplementalStyle: 'fsl',
    jokerSets: true,
  },
};

const app = document.getElementById('app');
let state = loadState();

function defaultDraft() {
  return {
    units: 'lb',
    roundingIncrement: 5,
    experienceLevel: 'beginner',
    daysPerWeek: 4,
    templateSlug: 'fsl',
    showWarmups: true,
    lifts: {
      press: { sourceType: 'estimated_1rm', sourceValue: 95, trainingMaxOverride: '' },
      bench: { sourceType: 'estimated_1rm', sourceValue: 185, trainingMaxOverride: '' },
      squat: { sourceType: 'estimated_1rm', sourceValue: 225, trainingMaxOverride: '' },
      deadlift: { sourceType: 'estimated_1rm', sourceValue: 275, trainingMaxOverride: '' },
    },
  };
}

function getDefaultUserSettings() {
  return {
    showWarmups: true,
    assistanceFavorites: { ...DEFAULT_ASSISTANCE_FAVORITES },
    recoveryWeekType: 'none',
    recoveryWeekEveryCycles: 3,
    restSecondsMain: DEFAULT_REST_TIMER_SECONDS.main,
    restSecondsSupplemental: DEFAULT_REST_TIMER_SECONDS.supplemental,
    restSecondsAssistance: DEFAULT_REST_TIMER_SECONDS.assistance,
  };
}

function getDefaultAssistanceTargets() {
  return ASSISTANCE_CATEGORIES.map((category, index) => ({
    category,
    targetMin: 50,
    targetMax: 100,
    displayOrder: index,
  }));
}

function getDefaultWorkoutFlags() {
  return {
    difficulty: 'normal',
    barSpeed: 'okay',
    pain: 'no',
  };
}

function getDefaultConditioningLog() {
  return {
    completed: 'no',
    minutes: '',
    note: '',
  };
}

function normalizeWorkoutRecord(workout) {
  if (!workout) return workout;

  return {
    ...workout,
    assistanceTargets: Array.isArray(workout.assistanceTargets) ? workout.assistanceTargets : getDefaultAssistanceTargets(),
    assistanceLog: {
      push: '',
      pull: '',
      single_leg_core: '',
      ...(workout.assistanceLog || {}),
    },
    assistanceExercises: {
      ...DEFAULT_ASSISTANCE_FAVORITES,
      ...(workout.assistanceExercises || {}),
    },
    flags: {
      ...getDefaultWorkoutFlags(),
      ...(workout.flags || {}),
    },
    conditioning: {
      ...getDefaultConditioningLog(),
      ...(workout.conditioning || {}),
    },
    recovery: workout.recovery || 'okay',
  };
}

function normalizeCycleRecord(cycle) {
  if (!cycle) return cycle;
  return {
    ...cycle,
    workouts: Array.isArray(cycle.workouts) ? cycle.workouts.map((workout) => normalizeWorkoutRecord(workout)) : [],
  };
}

function createInitialState() {
  return {
    version: 1,
    ui: {
      tab: 'home',
      selectedWorkoutId: null,
      restTimer: {
        endAt: null,
        durationSeconds: 0,
        label: '',
        phase: 'main',
      },
    },
    onboarding: defaultDraft(),
    user: null,
    liftProfiles: null,
    activeCycle: null,
    cycleHistory: [],
    reviewSelections: {},
    backups: {
      lastExportedAt: null,
      lastImportedAt: null,
    },
  };
}

function normalizeState(candidate) {
  const initialState = createInitialState();
  const parsed = candidate && typeof candidate === 'object' ? candidate : {};
  const onboarding = parsed.onboarding && typeof parsed.onboarding === 'object' ? parsed.onboarding : {};
  const backups = parsed.backups && typeof parsed.backups === 'object' ? parsed.backups : {};
  const defaultOnboarding = defaultDraft();
  const defaultUserSettings = getDefaultUserSettings();
  const parsedUser = parsed.user && typeof parsed.user === 'object'
    ? {
        ...defaultUserSettings,
        ...parsed.user,
        assistanceFavorites: {
          ...DEFAULT_ASSISTANCE_FAVORITES,
          ...((parsed.user && parsed.user.assistanceFavorites) || {}),
        },
      }
    : null;

  return {
    ...initialState,
    ...parsed,
    ui: {
      ...initialState.ui,
      ...(parsed.ui || {}),
      restTimer: {
        ...initialState.ui.restTimer,
        ...((parsed.ui && parsed.ui.restTimer) || {}),
      },
    },
    onboarding: {
      ...defaultOnboarding,
      ...onboarding,
      lifts: LIFT_ORDER.reduce((acc, liftKey) => {
        acc[liftKey] = {
          ...defaultOnboarding.lifts[liftKey],
          ...((onboarding.lifts && onboarding.lifts[liftKey]) || {}),
        };
        return acc;
      }, {}),
    },
    user: parsedUser,
    activeCycle: normalizeCycleRecord(parsed.activeCycle),
    cycleHistory: Array.isArray(parsed.cycleHistory)
      ? parsed.cycleHistory.map((cycle) => normalizeCycleRecord(cycle))
      : initialState.cycleHistory,
    reviewSelections:
      parsed.reviewSelections && typeof parsed.reviewSelections === 'object'
        ? parsed.reviewSelections
        : initialState.reviewSelections,
    backups: {
      ...initialState.backups,
      ...backups,
    },
  };
}

function loadState() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return createInitialState();
    const parsed = JSON.parse(raw);
    return normalizeState(parsed);
  } catch (error) {
    console.error('Failed to load state', error);
    return createInitialState();
  }
}

function saveState() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
}

async function importStateFile(file) {
  if (!file) return;

  const hasExistingData = Boolean(state.user || state.activeCycle || state.cycleHistory?.length);
  if (hasExistingData) {
    const confirmed = window.confirm('Importing a backup will replace the current data on this device. Continue?');
    if (!confirmed) return;
  }

  try {
    const raw = await file.text();
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== 'object') {
      throw new Error('The selected file is not a valid 5/3/1 Companion export.');
    }

    state = normalizeState({
      ...parsed,
      backups: {
        ...(parsed.backups || {}),
        lastImportedAt: new Date().toISOString(),
      },
    });
    saveState();
    render();
    window.alert(`Import complete. ${file.name || 'Your backup'} was restored on this device.`);
  } catch (error) {
    console.error('Failed to import state', error);
    window.alert(error?.message || 'Import failed. Please select a valid JSON export.');
  }
}

function updateState(mutator) {
  mutator(state);
  saveState();
  render();
}

function uid(prefix) {
  return `${prefix}_${Math.random().toString(36).slice(2, 10)}`;
}

function roundWeight(value, increment) {
  if (!increment || increment <= 0) return Math.round(value);
  return Math.round(value / increment) * increment;
}

function formatNumber(value) {
  if (value == null || Number.isNaN(Number(value))) return '—';
  const numeric = Number(value);
  return Number.isInteger(numeric) ? String(numeric) : numeric.toFixed(1);
}

function formatPlateNumber(value, units) {
  if (units !== 'lb' || value == null || Number.isNaN(Number(value))) return formatNumber(value);

  const numeric = Number(value);
  if (Number.isInteger(numeric)) return String(numeric);

  const absolute = Math.abs(numeric);
  const whole = Math.floor(absolute);
  const roundedQuarter = Math.round((absolute - whole) * 4);
  const quarterValue = whole + roundedQuarter / 4;

  if (Math.abs(absolute - quarterValue) > 0.001) return formatNumber(value);

  const fractionGlyph = ['', '¼', '½', '¾'][roundedQuarter] || '';
  const sign = numeric < 0 ? '-' : '';

  if (!fractionGlyph) return `${sign}${whole}`;
  return `${sign}${whole > 0 ? `${whole}${fractionGlyph}` : fractionGlyph}`;
}

function formatWeight(value, units) {
  return `${formatNumber(value)} ${units}`;
}

function formatDateTime(value) {
  if (!value) return 'Never';
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return 'Never';
  return parsed.toLocaleString();
}

function formatPlateWeight(value, units) {
  return `${formatPlateNumber(value, units)} ${units}`;
}

function escapeHtml(value) {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function estimateTrainingMax(sourceType, sourceValue, experienceLevel, roundingIncrement) {
  const numericValue = Number(sourceValue || 0);
  if (!numericValue) return 0;
  if (sourceType === 'training_max') return roundWeight(numericValue, roundingIncrement);
  const conservativePercent = experienceLevel === 'beginner' ? 0.85 : 0.9;
  return roundWeight(numericValue * conservativePercent, roundingIncrement);
}

function resolveTrainingMax(liftDraft, experienceLevel, roundingIncrement) {
  const suggestedTM = estimateTrainingMax(
    liftDraft.sourceType,
    liftDraft.sourceValue,
    experienceLevel,
    roundingIncrement,
  );
  const overrideValue = Number(liftDraft.trainingMaxOverride || 0);
  const plannedTM = overrideValue > 0 ? roundWeight(overrideValue, roundingIncrement) : suggestedTM;

  return {
    suggestedTM,
    plannedTM,
    usingOverride: overrideValue > 0,
  };
}

function getOnboardingLiftGuardrail(liftDraft, experienceLevel, roundingIncrement) {
  const sourceValue = Number(liftDraft.sourceValue || 0);
  const { suggestedTM, plannedTM } = resolveTrainingMax(liftDraft, experienceLevel, roundingIncrement);

  if (!sourceValue || plannedTM <= 0) {
    return 'Enter a valid max above 0 to build your program.';
  }

  if (liftDraft.sourceType === 'training_max' && plannedTM > sourceValue) {
    return 'This override is above the current TM you entered.';
  }

  if (liftDraft.sourceType !== 'training_max') {
    const aggressiveRatio = experienceLevel === 'beginner' ? 0.9 : 0.925;
    if (plannedTM / sourceValue >= aggressiveRatio) {
      return 'Aggressive TM for a conservative start. Consider backing it down slightly.';
    }
    if (plannedTM > suggestedTM) {
      return 'Above the conservative suggestion. Keep it only if your source max is very solid.';
    }
  }

  return '';
}

function getCadenceNote(templateSlug, daysPerWeek) {
  if (Number(daysPerWeek) === 3 && templateSlug !== 'beginner') {
    return '3-day single-lift templates use a rolling 4-workout rotation, so one 5s/3s/5/3/1 wave can spill across calendar weeks.';
  }

  if (Number(daysPerWeek) === 3 && templateSlug === 'beginner') {
    return '3-day Beginner 5/3/1 rotates three sessions across the wave while keeping the main lifts paired.';
  }

  return '4-day schedules keep each wave aligned to four main training sessions.';
}

function getStandardIncrement(liftKey, units) {
  const isUpper = LIFT_CATEGORY[liftKey] === 'upper';
  if (units === 'kg') return isUpper ? 2.5 : 5;
  return isUpper ? 5 : 10;
}

function getDefaultBarbellWeight(units) {
  return units === 'kg' ? 20 : 45;
}

function getAvailablePlates(units) {
  if (units === 'kg') {
    return [25, 20, 15, 10, 5, 2.5, 1.25];
  }
  return [45, 35, 25, 10, 5, 2.5, 1, 0.75, 0.5, 0.25];
}

function getPlateBreakdown(totalWeight, units, barbellWeight = getDefaultBarbellWeight(units)) {
  const normalizedWeight = Number(totalWeight || 0);
  const normalizedBarbellWeight = Number(barbellWeight || getDefaultBarbellWeight(units));
  const perSideTarget = Math.max(0, (normalizedWeight - normalizedBarbellWeight) / 2);
  let remaining = perSideTarget;
  const plates = [];

  for (const plate of getAvailablePlates(units)) {
    const count = Math.floor((remaining + 0.0001) / plate);
    for (let index = 0; index < count; index += 1) {
      plates.push(plate);
      remaining -= plate;
    }
  }

  remaining = Math.round(remaining * 100) / 100;

  return {
    barbellWeight: normalizedBarbellWeight,
    perSideTarget,
    plates,
    remainder: remaining,
  };
}

function renderPlateBreakdown(totalWeight) {
  const units = state.user?.units || 'lb';
  const barbellWeight = Number(state.user?.barbellWeight || getDefaultBarbellWeight(units));
  const normalizedWeight = Number(totalWeight || 0);

  if (normalizedWeight > 0 && normalizedWeight < barbellWeight) {
    const perHandWeight = Math.round((normalizedWeight / 2) * 100) / 100;
    return `
      <div class="plate-breakdown plate-breakdown--dumbbell">
        <div class="plate-breakdown__stack plate-breakdown__stack--dumbbell" role="img" aria-label="Use dumbbells instead, ${formatWeight(perHandWeight, units)} each hand">
          <span class="dumbbell-pair" aria-hidden="true">
            <span class="dumbbell-pair__item"><span class="dumbbell-pair__handle"></span></span>
            <span class="dumbbell-pair__item"><span class="dumbbell-pair__handle"></span></span>
          </span>
        </div>
        <div class="plate-breakdown__text">
          Dumbbells · ${formatWeight(perHandWeight, units)} each hand
          <span class="muted"> · lighter than a ${formatWeight(barbellWeight, units)} bar</span>
        </div>
      </div>
    `;
  }

  const breakdown = getPlateBreakdown(totalWeight, units, barbellWeight);
  const platesOutsideToInside = [...breakdown.plates].sort((left, right) => left - right);
  const hasPlates = normalizedWeight > barbellWeight && platesOutsideToInside.length > 0;
  const plateDescription = hasPlates ? platesOutsideToInside.map((plate) => formatPlateNumber(plate, units)).join(' · ') : 'bar only';
  const stackLabel = hasPlates
    ? `Per-side stack showing ${platesOutsideToInside.map((plate) => formatPlateWeight(plate, units)).join(', ')}`
    : `Bar only with a ${formatWeight(barbellWeight, units)} bar`;
  const plateMarkup = hasPlates
    ? platesOutsideToInside
      .map(
        (plate) =>
          `<span class="plate-chip" data-plate-size="${String(plate).replace('.', '_')}" aria-hidden="true" title="${formatPlateWeight(plate, units)} plate"><span class="plate-chip__label">${formatPlateNumber(plate, units)}</span></span>`,
      )
      .join('')
    : '';

  return `
    <div class="plate-breakdown">
      <div class="plate-breakdown__stack" role="img" aria-label="${stackLabel}">
        <span class="plate-breakdown__bar" aria-hidden="true"></span>
        <span class="plate-breakdown__collar" aria-hidden="true"></span>
        <span class="plate-breakdown__plates${hasPlates ? '' : ' plate-breakdown__plates--empty'}" aria-hidden="true">${plateMarkup}</span>
      </div>
      <div class="plate-breakdown__text">
        ${hasPlates ? `Per side · ${plateDescription}` : 'Bar only'}
        <span class="muted"> · assumes a ${formatWeight(barbellWeight, units)} bar</span>
      </div>
    </div>
  `;
}

function getWeekWave(weekType) {
  return ALL_WAVES.find((wave) => wave.key === weekType) || WEEK_WAVES[0];
}

function getCycleWaves(cycleNumber, recoveryWeekType, recoveryWeekEveryCycles) {
  const waves = [...WEEK_WAVES];
  if (
    recoveryWeekType
    && recoveryWeekType !== 'none'
    && recoveryWeekEveryCycles > 0
    && cycleNumber % recoveryWeekEveryCycles === 0
    && RECOVERY_WAVES[recoveryWeekType]
  ) {
    waves.push(RECOVERY_WAVES[recoveryWeekType]);
  }
  return waves;
}

function buildMainSets(trainingMax, weekType, roundingIncrement, mainStyle = 'plus') {
  const wave = getWeekWave(weekType);
  return wave.sets.map((set, index) => {
    const rawWeight = trainingMax * set.pct;
    const roundedWeight = roundWeight(rawWeight, roundingIncrement);
    const prescribedReps = mainStyle === 'five_pro' ? 5 : set.reps;
    const repStyle = mainStyle === 'five_pro' ? 'fixed' : set.style;
    return {
      id: uid('set'),
      phase: 'main',
      groupLabel: 'main_531',
      setIndex: index + 1,
      percentageOfTrainingMax: set.pct,
      prescribedWeightRaw: rawWeight,
      prescribedWeight: roundedWeight,
      prescribedReps,
      repStyle,
      completed: false,
      actualReps: prescribedReps,
      actualWeight: roundedWeight,
      painFlag: false,
      notes: '',
    };
  });
}

function buildRepeatedSets({
  trainingMax,
  percentage,
  sets,
  reps,
  roundingIncrement,
  groupLabel,
  phase,
}) {
  const rawWeight = trainingMax * percentage;
  const roundedWeight = roundWeight(rawWeight, roundingIncrement);
  return Array.from({ length: sets }, (_, index) => ({
    id: uid('set'),
    phase,
    groupLabel,
    setIndex: index + 1,
    percentageOfTrainingMax: percentage,
    prescribedWeightRaw: rawWeight,
    prescribedWeight: roundedWeight,
    prescribedReps: reps,
    repStyle: 'fixed',
    completed: false,
    actualReps: reps,
    actualWeight: roundedWeight,
    painFlag: false,
    notes: '',
  }));
}

function buildJokerSection(trainingMax, weekType, roundingIncrement) {
  const jokerPercentages = {
    '5s': [0.9, 0.95],
    '3s': [0.95, 1.0],
    '531': [1.0, 1.05],
  };
  const percentages = jokerPercentages[weekType];
  if (!percentages) return null;

  return {
    key: 'jokers',
    title: 'Joker singles',
    description: 'Only take these if the top set moved well and recovery feels solid.',
    sets: percentages.map((percentage, index) => {
      const rawWeight = trainingMax * percentage;
      const roundedWeight = roundWeight(rawWeight, roundingIncrement);
      return {
        id: uid('set'),
        phase: 'joker',
        groupLabel: 'joker',
        setIndex: index + 1,
        percentageOfTrainingMax: percentage,
        prescribedWeightRaw: rawWeight,
        prescribedWeight: roundedWeight,
        prescribedReps: 1,
        repStyle: 'fixed',
        completed: false,
        actualReps: 1,
        actualWeight: roundedWeight,
        painFlag: false,
        notes: '',
      };
    }),
  };
}

function buildSupplementalSection(templateSlug, trainingMax, weekType, roundingIncrement) {
  const preset = TEMPLATE_PRESETS[templateSlug] || TEMPLATE_PRESETS.fsl;
  const supplementalStyle = preset.supplementalStyle;

  if (supplementalStyle === 'fsl') {
    const firstSetPct = getWeekWave(weekType).sets[0].pct;
    return {
      key: 'supplemental',
      title: 'First Set Last 5x5',
      description: 'Repeat the first work-set percentage for 5 sets of 5.',
      sets: buildRepeatedSets({
        trainingMax,
        percentage: firstSetPct,
        sets: 5,
        reps: 5,
        roundingIncrement,
        groupLabel: 'fsl',
        phase: 'supplemental',
      }),
    };
  }

  if (supplementalStyle === 'bbb') {
    return {
      key: 'supplemental',
      title: 'Boring But Big 5x10',
      description: 'Use a conservative 50% of TM for volume.',
      sets: buildRepeatedSets({
        trainingMax,
        percentage: 0.5,
        sets: 5,
        reps: 10,
        roundingIncrement,
        groupLabel: 'bbb',
        phase: 'supplemental',
      }),
    };
  }

  return null;
}

function buildBlock({
  liftKey,
  role,
  trainingMax,
  weekType,
  roundingIncrement,
  templateSlug,
  secondary = false,
}) {
  const sections = [];
  const preset = TEMPLATE_PRESETS[templateSlug] || TEMPLATE_PRESETS.fsl;
  if (secondary) {
    const firstSetPct = getWeekWave(weekType).sets[0].pct;
    sections.push({
      key: 'secondary',
      title: 'Secondary lift 5x5',
      description: 'Simple 5x5 support for the second main lift.',
      sets: buildRepeatedSets({
        trainingMax,
        percentage: firstSetPct,
        sets: 5,
        reps: 5,
        roundingIncrement,
        groupLabel: 'secondary_fsl',
        phase: 'secondary_main',
      }),
    });
  } else {
    const mainWaveTitle = weekType === 'deload'
      ? 'Deload wave'
      : weekType === 'tm_test'
        ? 'TM test wave'
        : preset.mainStyle === 'five_pro'
          ? "5's Pro wave"
          : 'Main wave';
    const mainWaveDescription = weekType === 'deload'
      ? 'Keep every rep smooth and leave plenty in the tank.'
      : weekType === 'tm_test'
        ? 'Use this week to check whether the training max still feels honest.'
        : preset.mainStyle === 'five_pro'
          ? 'Use TM-based percentages for sets of 5 with no plus set.'
          : 'Use TM-based percentages. The final set is a plus set.';

    sections.push({
      key: 'main',
      title: mainWaveTitle,
      description: mainWaveDescription,
      sets: buildMainSets(trainingMax, weekType, roundingIncrement, preset.mainStyle),
    });
    const jokerSection = preset.jokerSets ? buildJokerSection(trainingMax, weekType, roundingIncrement) : null;
    if (jokerSection) sections.push(jokerSection);
    const supplemental = buildSupplementalSection(templateSlug, trainingMax, weekType, roundingIncrement);
    if (supplemental) sections.push(supplemental);
  }

  return {
    id: uid('block'),
    liftKey,
    role,
    trainingMaxAtTime: trainingMax,
    sections,
  };
}

function buildWorkout({
  cycleNumber,
  weekIndex,
  weekType,
  dayIndex,
  title,
  templateSlug,
  blocks,
  assistanceFavorites = DEFAULT_ASSISTANCE_FAVORITES,
}) {
  const totalSets = blocks.reduce(
    (sum, block) => sum + block.sections.reduce((sectionSum, section) => sectionSum + section.sets.length, 0),
    0,
  );

  return {
    id: uid('workout'),
    cycleNumber,
    weekIndex,
    weekType,
    dayIndex,
    title,
    templateSlug,
    status: 'planned',
    completedAt: null,
    blocks,
    assistanceTargets: getDefaultAssistanceTargets(),
    assistanceLog: {
      push: '',
      pull: '',
      single_leg_core: '',
    },
    assistanceExercises: {
      ...DEFAULT_ASSISTANCE_FAVORITES,
      ...assistanceFavorites,
    },
    flags: getDefaultWorkoutFlags(),
    conditioning: getDefaultConditioningLog(),
    recovery: 'okay',
    notes: '',
    totalSets,
  };
}

function generateCycle({
  cycleNumber,
  templateSlug,
  daysPerWeek,
  liftProfiles,
  units,
  roundingIncrement,
  recoveryWeekType = 'none',
  recoveryWeekEveryCycles = 0,
  assistanceFavorites = DEFAULT_ASSISTANCE_FAVORITES,
}) {
  const workouts = [];
  const cycleWaves = getCycleWaves(cycleNumber, recoveryWeekType, recoveryWeekEveryCycles);

  if (templateSlug === 'beginner') {
    const weeklySplit = daysPerWeek === 4
      ? [
          { title: 'Squat + Bench', primary: 'squat', secondary: 'bench' },
          { title: 'Press + Deadlift', primary: 'press', secondary: 'deadlift' },
          { title: 'Bench + Squat', primary: 'bench', secondary: 'squat' },
          { title: 'Deadlift + Press', primary: 'deadlift', secondary: 'press' },
        ]
      : [
          { title: 'Squat + Bench', primary: 'squat', secondary: 'bench' },
          { title: 'Deadlift + Press', primary: 'deadlift', secondary: 'press' },
          { title: 'Bench + Squat', primary: 'bench', secondary: 'squat' },
        ];

    cycleWaves.forEach((wave, weekOffset) => {
      weeklySplit.forEach((day, dayOffset) => {
        const primaryTM = liftProfiles[day.primary].currentTrainingMax;
        const secondaryTM = liftProfiles[day.secondary].currentTrainingMax;
        workouts.push(
          buildWorkout({
            cycleNumber,
            weekIndex: weekOffset + 1,
            weekType: wave.key,
            dayIndex: dayOffset + 1,
            title: day.title,
            templateSlug,
            assistanceFavorites,
            blocks: [
              buildBlock({
                liftKey: day.primary,
                role: 'main',
                trainingMax: primaryTM,
                weekType: wave.key,
                roundingIncrement,
                templateSlug,
              }),
              buildBlock({
                liftKey: day.secondary,
                role: 'secondary_main',
                trainingMax: secondaryTM,
                weekType: wave.key,
                roundingIncrement,
                templateSlug,
                secondary: true,
              }),
            ],
          }),
        );
      });
    });
  } else {
    const split = ['press', 'deadlift', 'bench', 'squat'];
    cycleWaves.forEach((wave, weekOffset) => {
      split.forEach((liftKey, dayOffset) => {
        workouts.push(
          buildWorkout({
            cycleNumber,
            weekIndex: weekOffset + 1,
            weekType: wave.key,
            dayIndex: dayOffset + 1,
            title: LIFT_LABELS[liftKey],
            templateSlug,
            assistanceFavorites,
            blocks: [
              buildBlock({
                liftKey,
                role: 'main',
                trainingMax: liftProfiles[liftKey].currentTrainingMax,
                weekType: wave.key,
                roundingIncrement,
                templateSlug,
              }),
            ],
          }),
        );
      });
    });
  }

  return {
    id: uid('cycle'),
    cycleNumber,
    templateSlug,
    units,
    roundingIncrement,
    daysPerWeek,
    recoveryWeekType,
    recoveryWeekEveryCycles,
    status: 'active',
    createdAt: new Date().toISOString(),
    workouts,
  };
}

function getNextWorkout(cycle) {
  if (!cycle) return null;
  return cycle.workouts.find((workout) => workout.status !== 'completed') || null;
}

function getWorkoutById(cycle, workoutId) {
  if (!cycle || !workoutId) return null;
  return cycle.workouts.find((workout) => workout.id === workoutId) || null;
}

function getSelectedWorkout() {
  return getWorkoutById(state.activeCycle, state.ui.selectedWorkoutId) || getNextWorkout(state.activeCycle);
}

function isWorkoutStarted(workout) {
  if (!workout || workout.status === 'completed') return false;

  const completion = getWorkoutCompletion(workout);
  if (completion.completedSets > 0) return true;

  const hasLoggedSetValues = getWorkoutSets(workout).some(
    (set) => set.actualReps !== '' || set.actualWeight !== '',
  );
  const hasAssistance = Object.values(workout.assistanceLog || {}).some((value) => Number(value || 0) > 0);
  const hasNotes = Boolean(String(workout.notes || '').trim());
  const flags = workout.flags || {};
  const conditioning = workout.conditioning || {};
  const hasConditioning = conditioning.completed === 'yes' || Number(conditioning.minutes || 0) > 0 || Boolean(String(conditioning.note || '').trim());
  const hasNonDefaultFlags = flags.difficulty !== 'normal' || flags.barSpeed !== 'okay' || flags.pain !== 'no' || workout.recovery !== 'okay';

  return hasLoggedSetValues || hasAssistance || hasNotes || hasConditioning || hasNonDefaultFlags;
}

function getInProgressWorkout(cycle) {
  if (!cycle) return null;
  return cycle.workouts.find((workout) => workout.status === 'in_progress' || isWorkoutStarted(workout)) || null;
}

function isCycleComplete(cycle) {
  return Boolean(cycle && cycle.workouts.length && cycle.workouts.every((workout) => workout.status === 'completed'));
}

function getCompletedWorkouts(cycle) {
  if (!cycle) return [];
  return cycle.workouts.filter((workout) => workout.status === 'completed');
}

function getWorkoutCompletion(workout) {
  const allSets = workout.blocks.flatMap((block) => block.sections.flatMap((section) => section.sets));
  const completedSets = allSets.filter((set) => set.completed).length;
  return {
    completedSets,
    totalSets: allSets.length,
  };
}

function getCycleProgress(cycle) {
  if (!cycle) return { completed: 0, total: 0, percent: 0 };
  const total = cycle.workouts.length;
  const completed = cycle.workouts.filter((workout) => workout.status === 'completed').length;
  return {
    completed,
    total,
    percent: total ? Math.round((completed / total) * 100) : 0,
  };
}

function getUpcomingWorkouts(cycle, limit = 4) {
  if (!cycle) return [];
  return cycle.workouts.filter((workout) => workout.status !== 'completed').slice(0, limit);
}

function getLastCompletedWorkout(cycle) {
  if (!cycle) return null;
  const completed = cycle.workouts.filter((workout) => workout.status === 'completed');
  return completed[completed.length - 1] || null;
}

function getWorkoutLiftNames(workout) {
  return workout.blocks.map((block) => LIFT_LABELS[block.liftKey]).join(' + ');
}

function getTopSetFromSection(section) {
  if (!section?.sets?.length) return null;
  return section.sets.find((set) => set.repStyle === 'plus') || section.sets[section.sets.length - 1] || null;
}

function getTopSetSummary(workout) {
  if (!workout) return 'No completed workout yet';
  const topSet = workout.blocks
    .flatMap((block) => block.sections)
    .map((section) => getTopSetFromSection(section))
    .find(Boolean);
  if (!topSet) return `${getWorkoutCompletion(workout).completedSets} sets logged`;
  return `${formatNumber(topSet.actualReps)} reps @ ${formatWeight(topSet.actualWeight, state.user.units)}`;
}

function getWorkoutSets(workout) {
  if (!workout) return [];
  return workout.blocks.flatMap((block) => block.sections.flatMap((section) => section.sets));
}

function getNextIncompleteSet(workout) {
  return getWorkoutSets(workout).find((set) => !set.completed) || null;
}

function getRemainingSetCount(workout) {
  return getWorkoutSets(workout).filter((set) => !set.completed).length;
}

function getSetLocationLabel(workout, setId) {
  for (const block of workout.blocks) {
    for (const section of block.sections) {
      for (const set of section.sets) {
        if (set.id === setId) {
          return `${LIFT_LABELS[block.liftKey]} · ${section.title} · Set ${set.setIndex}`;
        }
      }
    }
  }
  return getWorkoutLiftNames(workout);
}

function summarizeSection(section, units) {
  const repPattern = section.sets
    .map((set) => `${set.prescribedReps}${set.repStyle === 'plus' ? '+' : ''}`)
    .join(' / ');
  const weightLabels = [...new Set(section.sets.map((set) => formatWeight(set.prescribedWeight, units)))];
  const weightPattern = weightLabels.length === 1
    ? weightLabels[0]
    : `${weightLabels[0]} → ${weightLabels[weightLabels.length - 1]}`;

  return `${section.sets.length} sets · ${repPattern} reps · ${weightPattern}`;
}

function getSectionCompletion(section) {
  const total = section.sets.length;
  const completed = section.sets.filter((set) => set.completed).length;
  return {
    completed,
    total,
    remaining: total - completed,
  };
}

function getLastCompletedSet(workout) {
  const completedSets = getWorkoutSets(workout).filter((set) => set.completed);
  return completedSets[completedSets.length - 1] || null;
}

function getTopSetResultsForLift(cycle, liftKey) {
  if (!cycle) return [];
  const results = [];

  cycle.workouts.forEach((workout) => {
    workout.blocks.forEach((block) => {
      if (block.liftKey !== liftKey) return;
      block.sections.forEach((section) => {
        const topSet = getTopSetFromSection(section);
        if (topSet && workout.status === 'completed' && ['main', 'secondary_main'].includes(topSet.phase)) {
          results.push({
            workoutId: workout.id,
            workoutTitle: workout.title,
            weekType: workout.weekType,
            prescribedReps: topSet.prescribedReps,
            actualReps: Number(topSet.actualReps || 0),
            actualWeight: Number(topSet.actualWeight || topSet.prescribedWeight || 0),
            difficulty: workout.flags.difficulty,
            barSpeed: workout.flags.barSpeed,
            pain: workout.flags.pain,
          });
        }
      });
    });
  });

  return results;
}

function estimateE1RM(weight, reps) {
  if (!weight || !reps) return 0;
  return weight * (1 + reps / 30);
}

function getLiftReview(cycle, liftKey, liftProfile, user) {
  const topSets = getTopSetResultsForLift(cycle, liftKey);
  if (!topSets.length) {
    return {
      liftKey,
      action: 'hold',
      nextTrainingMax: liftProfile.currentTrainingMax,
      reason: 'No completed plus sets yet.',
      topSets,
    };
  }

  const missed = topSets.some((set) => set.actualReps < set.prescribedReps);
  const pain = topSets.some((set) => set.pain === 'yes');
  const grindy = topSets.some((set) => set.barSpeed === 'grindy' || set.difficulty === 'hard');
  const repSurplus = topSets.reduce((sum, set) => sum + (set.actualReps - set.prescribedReps), 0);
  const increment = getStandardIncrement(liftKey, user.units);

  if (missed || pain) {
    return {
      liftKey,
      action: 'reset',
      nextTrainingMax: roundWeight(liftProfile.currentTrainingMax * 0.9, user.roundingIncrement),
      reason: missed
        ? 'Missed prescribed reps on at least one plus set.'
        : 'Pain flag detected during this cycle.',
      topSets,
    };
  }

  if (grindy || repSurplus < 3) {
    return {
      liftKey,
      action: 'hold',
      nextTrainingMax: liftProfile.currentTrainingMax,
      reason: 'Work was completed, but bar speed or recovery looked marginal.',
      topSets,
    };
  }

  return {
    liftKey,
    action: 'increase',
    nextTrainingMax: roundWeight(liftProfile.currentTrainingMax + increment, user.roundingIncrement),
    reason: 'Completed work cleanly with enough reps in reserve to keep progressing.',
    topSets,
  };
}

function getCycleReviews() {
  if (!state.activeCycle || !state.user || !state.liftProfiles) return [];
  return LIFT_ORDER.map((liftKey) =>
    getLiftReview(state.activeCycle, liftKey, state.liftProfiles[liftKey], state.user),
  );
}

function getReviewSelection(liftKey, defaultAction) {
  return state.reviewSelections[liftKey] || defaultAction;
}

function getProjectedTM(liftKey, selectedAction) {
  const profile = state.liftProfiles[liftKey];
  if (!profile || !state.user) return 0;
  const increment = getStandardIncrement(liftKey, state.user.units);
  if (selectedAction === 'increase') return roundWeight(profile.currentTrainingMax + increment, state.user.roundingIncrement);
  if (selectedAction === 'reset') return roundWeight(profile.currentTrainingMax * 0.9, state.user.roundingIncrement);
  return profile.currentTrainingMax;
}

function getActionBadgeClass(action) {
  if (action === 'increase') return 'badge--success';
  if (action === 'reset') return 'badge--danger';
  return 'badge--muted';
}

function getActionLabel(action) {
  if (action === 'increase') return 'Increase';
  if (action === 'reset') return 'Reset';
  return 'Hold';
}

function getBestTopSet(topSets) {
  if (!topSets?.length) return null;
  return [...topSets].sort(
    (left, right) => estimateE1RM(right.actualWeight, right.actualReps) - estimateE1RM(left.actualWeight, left.actualReps),
  )[0] || null;
}

function getCycleReviewSummary(cycle, reviews) {
  if (!cycle) return null;

  const completedWorkouts = getCompletedWorkouts(cycle);
  const allTopSets = reviews.flatMap((review) => review.topSets || []);
  const bestTopSet = getBestTopSet(allTopSets);
  const increaseCount = reviews.filter((review) => review.action === 'increase').length;
  const holdCount = reviews.filter((review) => review.action === 'hold').length;
  const resetCount = reviews.filter((review) => review.action === 'reset').length;
  const painCount = completedWorkouts.filter((workout) => workout.flags?.pain === 'yes').length;
  const hardCount = completedWorkouts.filter(
    (workout) => workout.flags?.difficulty === 'hard' || workout.flags?.barSpeed === 'grindy',
  ).length;
  const attentionCount = completedWorkouts.filter(
    (workout) => workout.flags?.pain === 'yes'
      || workout.flags?.difficulty === 'hard'
      || workout.flags?.barSpeed === 'grindy',
  ).length;

  let headline = 'Finish the current cycle to unlock next-cycle recommendations.';
  if (isCycleComplete(cycle)) {
    if (resetCount > 0) {
      headline = 'At least one lift needs a reset before you move on.';
    } else if (holdCount > 0) {
      headline = 'Some lifts look better held steady for one more cycle.';
    } else {
      headline = 'Everything looks ready to move up next cycle.';
    }
  }

  return {
    completedWorkoutCount: completedWorkouts.length,
    totalWorkoutCount: cycle.workouts.length,
    allTopSets,
    bestTopSet,
    increaseCount,
    holdCount,
    resetCount,
    painCount,
    hardCount,
    attentionCount,
    headline,
    nextCycleNumber: cycle.cycleNumber + 1,
  };
}

function createProgramFromDraft() {
  const draft = state.onboarding;
  const user = {
    ...getDefaultUserSettings(),
    units: draft.units,
    roundingIncrement: Number(draft.roundingIncrement),
    barbellWeight: getDefaultBarbellWeight(draft.units),
    experienceLevel: draft.experienceLevel,
    daysPerWeek: Number(draft.daysPerWeek),
    templateSlug: draft.templateSlug,
    showWarmups: draft.showWarmups !== false,
    createdAt: new Date().toISOString(),
  };

  const invalidLifts = LIFT_ORDER.filter((liftKey) => {
    const liftDraft = draft.lifts[liftKey];
    const { plannedTM } = resolveTrainingMax(liftDraft, draft.experienceLevel, user.roundingIncrement);
    return Number(liftDraft.sourceValue || 0) <= 0 || plannedTM <= 0;
  });

  if (invalidLifts.length) {
    window.alert(`Enter valid max numbers for: ${invalidLifts.map((liftKey) => LIFT_LABELS[liftKey]).join(', ')}.`);
    return;
  }

  const liftProfiles = LIFT_ORDER.reduce((acc, liftKey) => {
    const liftDraft = draft.lifts[liftKey];
    const { plannedTM } = resolveTrainingMax(liftDraft, draft.experienceLevel, user.roundingIncrement);

    acc[liftKey] = {
      liftKey,
      currentTrainingMax: plannedTM,
      sourceType: liftDraft.sourceType,
      sourceValue: Number(liftDraft.sourceValue),
      incrementPerCycle: getStandardIncrement(liftKey, draft.units),
      lastEstimatedOneRepMax:
        liftDraft.sourceType === 'training_max' ? null : Number(liftDraft.sourceValue),
    };
    return acc;
  }, {});

  const activeCycle = generateCycle({
    cycleNumber: 1,
    templateSlug: draft.templateSlug,
    daysPerWeek: user.daysPerWeek,
    liftProfiles,
    units: user.units,
    roundingIncrement: user.roundingIncrement,
    recoveryWeekType: user.recoveryWeekType,
    recoveryWeekEveryCycles: user.recoveryWeekEveryCycles,
    assistanceFavorites: user.assistanceFavorites,
  });

  updateState((draftState) => {
    draftState.user = user;
    draftState.liftProfiles = liftProfiles;
    draftState.activeCycle = activeCycle;
    draftState.cycleHistory = [];
    draftState.reviewSelections = {};
    draftState.ui.tab = 'home';
    draftState.ui.selectedWorkoutId = activeCycle.workouts[0]?.id || null;
  });
}

function fillUncompletedSets(workout) {
  workout.blocks.forEach((block) => {
    block.sections.forEach((section) => {
      section.sets.forEach((set) => {
        if (!set.completed) {
          set.completed = true;
          set.actualReps = Number(set.actualReps || set.prescribedReps);
          set.actualWeight = Number(set.actualWeight || set.prescribedWeight);
        }
      });
    });
  });
}

function saveWorkout(workoutId) {
  const workout = getWorkoutById(state.activeCycle, workoutId);
  if (!workout) return;

  const remainingSets = getRemainingSetCount(workout);
  if (remainingSets > 0) {
    const confirmed = window.confirm(
      `You still have ${remainingSets} unchecked ${remainingSets === 1 ? 'set' : 'sets'}. Complete the workout anyway and mark the remaining sets done?`,
    );
    if (!confirmed) return;
  }

  updateState((draftState) => {
    const draftWorkout = getWorkoutById(draftState.activeCycle, workoutId);
    if (!draftWorkout) return;

    fillUncompletedSets(draftWorkout);
    draftWorkout.status = 'completed';
    draftWorkout.completedAt = new Date().toISOString();

    if (isCycleComplete(draftState.activeCycle)) {
      draftState.activeCycle.status = 'completed';
      draftState.ui.tab = 'program';
      draftState.ui.selectedWorkoutId = draftWorkout.id;
    } else {
      const nextWorkout = getNextWorkout(draftState.activeCycle);
      draftState.ui.tab = 'home';
      draftState.ui.selectedWorkoutId = nextWorkout?.id || null;
    }
  });
}

function applyNextCycle() {
  updateState((draftState) => {
    if (!draftState.activeCycle || !draftState.user || !draftState.liftProfiles) return;
    const completedCycle = JSON.parse(JSON.stringify(draftState.activeCycle));
    const reviews = LIFT_ORDER.map((liftKey) => {
      const defaultReview = getLiftReview(completedCycle, liftKey, draftState.liftProfiles[liftKey], draftState.user);
      const selectedAction = draftState.reviewSelections[liftKey] || defaultReview.action;
      const nextTrainingMax = getProjectedTM(liftKey, selectedAction);
      return {
        liftKey,
        selectedAction,
        nextTrainingMax,
        reason: defaultReview.reason,
      };
    });

    completedCycle.review = reviews;
    completedCycle.completedAt = new Date().toISOString();
    draftState.cycleHistory.unshift(completedCycle);

    reviews.forEach((review) => {
      draftState.liftProfiles[review.liftKey].currentTrainingMax = review.nextTrainingMax;
    });

    draftState.activeCycle = generateCycle({
      cycleNumber: completedCycle.cycleNumber + 1,
      templateSlug: draftState.user.templateSlug,
      daysPerWeek: draftState.user.daysPerWeek,
      liftProfiles: draftState.liftProfiles,
      units: draftState.user.units,
      roundingIncrement: draftState.user.roundingIncrement,
      recoveryWeekType: draftState.user.recoveryWeekType,
      recoveryWeekEveryCycles: draftState.user.recoveryWeekEveryCycles,
      assistanceFavorites: draftState.user.assistanceFavorites,
    });
    draftState.reviewSelections = {};
    draftState.ui.tab = 'home';
    draftState.ui.selectedWorkoutId = draftState.activeCycle.workouts[0]?.id || null;
  });
}

function resetApp() {
  if (!window.confirm('Reset the entire 5/3/1 Companion app? This will erase local data.')) return;
  state = createInitialState();
  saveState();
  render();
}

function exportState() {
  const exportedAt = new Date().toISOString();
  state = normalizeState({
    ...state,
    backups: {
      ...(state.backups || {}),
      lastExportedAt: exportedAt,
    },
  });
  saveState();
  render();

  const exportDate = exportedAt.slice(0, 10);
  const blob = new Blob([JSON.stringify(state, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = `531-companion-backup-${exportDate}.json`;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

function readSettingsFormValues() {
  const roundingInput = document.querySelector('[data-settings-field="roundingIncrement"]');
  const barbellInput = document.querySelector('[data-settings-field="barbellWeight"]');
  const warmupInput = document.querySelector('[data-settings-field="showWarmups"]');
  const recoveryTypeInput = document.querySelector('[data-settings-field="recoveryWeekType"]');
  const recoveryEveryInput = document.querySelector('[data-settings-field="recoveryWeekEveryCycles"]');
  const roundingIncrement = roundingInput ? Number(roundingInput.value) : Number(state.user?.roundingIncrement || 0);
  const barbellWeight = barbellInput
    ? Number(barbellInput.value)
    : Number(state.user?.barbellWeight || getDefaultBarbellWeight(state.user?.units || 'lb'));
  const showWarmups = warmupInput ? warmupInput.checked : state.user?.showWarmups !== false;
  const recoveryWeekType = recoveryTypeInput ? recoveryTypeInput.value : state.user?.recoveryWeekType || 'none';
  const recoveryWeekEveryCycles = recoveryEveryInput
    ? Number(recoveryEveryInput.value || 0)
    : Number(state.user?.recoveryWeekEveryCycles || 0);
  const restSecondsMain = Number(document.querySelector('[data-settings-field="restSecondsMain"]')?.value || state.user?.restSecondsMain || DEFAULT_REST_TIMER_SECONDS.main);
  const restSecondsSupplemental = Number(document.querySelector('[data-settings-field="restSecondsSupplemental"]')?.value || state.user?.restSecondsSupplemental || DEFAULT_REST_TIMER_SECONDS.supplemental);
  const restSecondsAssistance = Number(document.querySelector('[data-settings-field="restSecondsAssistance"]')?.value || state.user?.restSecondsAssistance || DEFAULT_REST_TIMER_SECONDS.assistance);
  const assistanceFavorites = Object.fromEntries(
    ASSISTANCE_CATEGORIES.map((category) => {
      const input = document.querySelector(`[data-settings-assistance="${category}"]`);
      return [category, input ? String(input.value || '').trim() : String(state.user?.assistanceFavorites?.[category] || '')];
    }),
  );
  const trainingMaxes = Object.fromEntries(
    LIFT_ORDER.map((liftKey) => {
      const input = document.querySelector(`[data-settings-lift="${liftKey}"][data-field="currentTrainingMax"]`);
      return [liftKey, input ? Number(input.value) : Number(state.liftProfiles?.[liftKey]?.currentTrainingMax || 0)];
    }),
  );

  return {
    roundingIncrement,
    barbellWeight,
    showWarmups,
    recoveryWeekType,
    recoveryWeekEveryCycles,
    restSecondsMain,
    restSecondsSupplemental,
    restSecondsAssistance,
    assistanceFavorites,
    trainingMaxes,
  };
}

function regenerateUnfinishedWorkouts(cycle, user, liftProfiles) {
  if (!cycle || !user || !liftProfiles) return cycle;

  const regeneratedCycle = generateCycle({
    cycleNumber: cycle.cycleNumber,
    templateSlug: user.templateSlug,
    daysPerWeek: user.daysPerWeek,
    liftProfiles,
    units: user.units,
    roundingIncrement: user.roundingIncrement,
    recoveryWeekType: user.recoveryWeekType,
    recoveryWeekEveryCycles: user.recoveryWeekEveryCycles,
    assistanceFavorites: user.assistanceFavorites,
  });

  regeneratedCycle.id = cycle.id;
  regeneratedCycle.createdAt = cycle.createdAt;
  regeneratedCycle.status = cycle.status;
  regeneratedCycle.workouts = cycle.workouts.map((existingWorkout, index) => {
    if (existingWorkout.status === 'completed') return existingWorkout;

    const nextWorkout = regeneratedCycle.workouts[index];
    return {
      ...nextWorkout,
      id: existingWorkout.id,
      status: existingWorkout.status,
      completedAt: existingWorkout.completedAt,
      assistanceTargets: existingWorkout.assistanceTargets,
      assistanceLog: existingWorkout.assistanceLog,
      flags: existingWorkout.flags,
      notes: existingWorkout.notes,
    };
  });

  return regeneratedCycle;
}

function applyProgramSettings() {
  if (!state.user || !state.liftProfiles || !state.activeCycle) return;

  const {
    roundingIncrement,
    barbellWeight,
    showWarmups,
    recoveryWeekType,
    recoveryWeekEveryCycles,
    restSecondsMain,
    restSecondsSupplemental,
    restSecondsAssistance,
    assistanceFavorites,
    trainingMaxes,
  } = readSettingsFormValues();
  const invalidLifts = LIFT_ORDER.filter((liftKey) => Number(trainingMaxes[liftKey] || 0) <= 0);

  if (roundingIncrement <= 0) {
    window.alert('Enter a rounding increment above 0.');
    return;
  }

  if (barbellWeight <= 0) {
    window.alert('Enter a barbell weight above 0.');
    return;
  }

  if (invalidLifts.length) {
    window.alert(`Enter valid training maxes for: ${invalidLifts.map((liftKey) => LIFT_LABELS[liftKey]).join(', ')}.`);
    return;
  }

  if (recoveryWeekType !== 'none' && recoveryWeekEveryCycles <= 0) {
    window.alert('Choose how often to run the recovery week.');
    return;
  }

  if ([restSecondsMain, restSecondsSupplemental, restSecondsAssistance].some((value) => value <= 0)) {
    window.alert('Rest timer defaults must be above 0 seconds.');
    return;
  }

  const hadIncompleteWorkouts = getRemainingSetCount(getNextWorkout(state.activeCycle) || { blocks: [] }) > 0
    || state.activeCycle.workouts.some((workout) => workout.status !== 'completed');

  updateState((draftState) => {
    draftState.user.roundingIncrement = roundingIncrement;
    draftState.user.barbellWeight = barbellWeight;
    draftState.user.showWarmups = showWarmups;
    draftState.user.recoveryWeekType = recoveryWeekType;
    draftState.user.recoveryWeekEveryCycles = recoveryWeekEveryCycles;
    draftState.user.restSecondsMain = restSecondsMain;
    draftState.user.restSecondsSupplemental = restSecondsSupplemental;
    draftState.user.restSecondsAssistance = restSecondsAssistance;
    draftState.user.assistanceFavorites = assistanceFavorites;

    LIFT_ORDER.forEach((liftKey) => {
      draftState.liftProfiles[liftKey].currentTrainingMax = roundWeight(trainingMaxes[liftKey], roundingIncrement);
      draftState.liftProfiles[liftKey].incrementPerCycle = getStandardIncrement(liftKey, draftState.user.units);
    });

    draftState.activeCycle = regenerateUnfinishedWorkouts(draftState.activeCycle, draftState.user, draftState.liftProfiles);
    draftState.activeCycle.roundingIncrement = roundingIncrement;
    draftState.activeCycle.recoveryWeekType = recoveryWeekType;
    draftState.activeCycle.recoveryWeekEveryCycles = recoveryWeekEveryCycles;
  });

  window.alert(
    hadIncompleteWorkouts
      ? 'Program settings updated. Unfinished workouts were regenerated with the new training maxes and rounding.'
      : 'Program settings updated.',
  );
}

function updateOnboardingField(field, value) {
  updateState((draftState) => {
    draftState.onboarding[field] = value;
    if (field === 'units') {
      draftState.onboarding.roundingIncrement = value === 'kg' ? 2.5 : 5;
    }
  });
}

function getWarmupSetsForWeight(targetWeight, units, barbellWeight, roundingIncrement) {
  const normalizedTarget = Number(targetWeight || 0);
  const normalizedBarbellWeight = Number(barbellWeight || getDefaultBarbellWeight(units));
  if (normalizedTarget <= 0) return [];

  const ramp = [
    { percentage: 0, reps: 8, label: 'Empty bar' },
    { percentage: 0.4, reps: 5 },
    { percentage: 0.55, reps: 3 },
    { percentage: 0.7, reps: 1 },
  ];

  const warmups = ramp
    .map((step) => {
      const weight = step.percentage === 0
        ? normalizedBarbellWeight
        : Math.max(normalizedBarbellWeight, roundWeight(normalizedTarget * step.percentage, roundingIncrement));
      return {
        ...step,
        weight,
      };
    })
    .filter((step, index, steps) => {
      if (step.weight >= normalizedTarget) return false;
      return index === 0 || step.weight > steps[index - 1].weight;
    });

  if (!warmups.length && normalizedTarget > normalizedBarbellWeight) {
    return [{ weight: normalizedBarbellWeight, reps: 8, label: 'Empty bar' }];
  }

  return warmups;
}

function getSetContext(workout, setId) {
  if (!workout || !setId) return null;

  for (const block of workout.blocks) {
    for (const section of block.sections) {
      for (const set of section.sets) {
        if (set.id === setId) {
          return { block, section, set };
        }
      }
    }
  }

  return null;
}

function getWarmupPlanForSet(workout, set) {
  if (!workout || !set || !state.user || state.user.showWarmups === false) return null;

  const context = getSetContext(workout, set.id);
  if (!context) return null;

  const sectionHasStarted = context.section.sets.some((sectionSet) => sectionSet.completed);
  const supportsWarmups = ['main', 'secondary_main'].includes(set.phase);
  if (sectionHasStarted || !supportsWarmups || set.setIndex !== 1) return null;

  const warmupSets = getWarmupSetsForWeight(
    set.prescribedWeight,
    state.user.units,
    state.user.barbellWeight,
    state.user.roundingIncrement,
  );

  if (!warmupSets.length) return null;

  return {
    title: `${LIFT_LABELS[context.block.liftKey]} warm-up`,
    warmupSets,
  };
}

function renderWarmupPlan(warmupPlan) {
  if (!warmupPlan) return '';

  return `
    <div class="warmup-plan">
      <div class="warmup-plan__header">
        <strong>${escapeHtml(warmupPlan.title)}</strong>
        <span class="muted">Do these before the first work set.</span>
      </div>
      <div class="warmup-plan__steps">
        ${warmupPlan.warmupSets.map((step) => `
          <div class="warmup-chip">
            <span class="warmup-chip__weight">${formatWeight(step.weight, state.user.units)}</span>
            <span class="warmup-chip__reps">× ${step.reps}</span>
            ${step.label ? `<small>${escapeHtml(step.label)}</small>` : ''}
          </div>
        `).join('')}
      </div>
    </div>
  `;
}

function updateOnboardingLift(liftKey, field, value) {
  updateState((draftState) => {
    draftState.onboarding.lifts[liftKey][field] = value;
  });
}

function updateWorkoutSet(workoutId, setId, field, value, options = {}) {
  const { renderApp = true } = options;
  const applyUpdate = (draftState) => {
    const workout = getWorkoutById(draftState.activeCycle, workoutId);
    if (!workout) return;

    workout.blocks.forEach((block) => {
      block.sections.forEach((section) => {
        section.sets.forEach((set) => {
          if (set.id !== setId) return;
          if (field === 'completed' || field === 'painFlag') {
            set[field] = Boolean(value);
          } else {
            set[field] = value === '' ? '' : Number(value);
          }
        });
      });
    });

    if (workout.status !== 'completed') {
      workout.status = isWorkoutStarted(workout) ? 'in_progress' : 'planned';
    }
  };

  if (renderApp) {
    updateState(applyUpdate);
    return;
  }

  applyUpdate(state);
  saveState();
}

function toggleWorkoutSetCompletion(workoutId, setId) {
  updateState((draftState) => {
    const workout = getWorkoutById(draftState.activeCycle, workoutId);
    if (!workout) return;

    workout.blocks.forEach((block) => {
      block.sections.forEach((section) => {
        section.sets.forEach((set) => {
          if (set.id !== setId) return;

          const wasCompleted = Boolean(set.completed);
          set.completed = !wasCompleted;

          if (set.completed) {
            set.actualReps = Number(set.actualReps || set.prescribedReps);
            set.actualWeight = Number(set.actualWeight || set.prescribedWeight);
          } else if (workout.status === 'completed') {
            workout.status = 'planned';
            workout.completedAt = null;
            draftState.activeCycle.status = 'active';
          }
        });
      });
    });

    if (workout.status !== 'completed') {
      workout.status = isWorkoutStarted(workout) ? 'in_progress' : 'planned';
    }
  });
}

function adjustWorkoutSetValue(workoutId, setId, field, delta) {
  updateState((draftState) => {
    const workout = getWorkoutById(draftState.activeCycle, workoutId);
    if (!workout) return;

    workout.blocks.forEach((block) => {
      block.sections.forEach((section) => {
        section.sets.forEach((set) => {
          if (set.id !== setId) return;

          if (field === 'actualReps') {
            const nextValue = Math.max(0, Number(set.actualReps || 0) + Number(delta || 0));
            set.actualReps = nextValue;
            return;
          }

          if (field === 'actualWeight') {
            const roundingIncrement = Number(draftState.user?.roundingIncrement || 0.5);
            const nextValue = Math.max(0, Number(set.actualWeight || 0) + Number(delta || 0));
            set.actualWeight = roundWeight(nextValue, roundingIncrement);
          }
        });
      });
    });

    if (workout.status !== 'completed') {
      workout.status = isWorkoutStarted(workout) ? 'in_progress' : 'planned';
    }
  });
}

function updateWorkoutMeta(workoutId, field, value) {
  updateState((draftState) => {
    const workout = getWorkoutById(draftState.activeCycle, workoutId);
    if (!workout) return;
    if (field.startsWith('assistance.')) {
      const category = field.split('.')[1];
      workout.assistanceLog[category] = value;
    } else if (field.startsWith('conditioning.')) {
      const conditioningField = field.split('.')[1];
      workout.conditioning[conditioningField] = value;
    } else if (field.startsWith('flags.')) {
      const flag = field.split('.')[1];
      workout.flags[flag] = value;
    } else {
      workout[field] = value;
    }

    if (workout.status !== 'completed') {
      workout.status = isWorkoutStarted(workout) ? 'in_progress' : 'planned';
    }
  });
}

function adjustWorkoutAssistanceLog(workoutId, category, delta) {
  updateState((draftState) => {
    const workout = getWorkoutById(draftState.activeCycle, workoutId);
    if (!workout) return;
    const current = Number(workout.assistanceLog?.[category] || 0);
    workout.assistanceLog[category] = Math.max(0, current + Number(delta || 0));

    if (workout.status !== 'completed') {
      workout.status = isWorkoutStarted(workout) ? 'in_progress' : 'planned';
    }
  });
}

function getRestSecondsForPhase(phase) {
  if (!state.user) return DEFAULT_REST_TIMER_SECONDS.main;
  if (phase === 'supplemental' || phase === 'joker') return Number(state.user.restSecondsSupplemental || DEFAULT_REST_TIMER_SECONDS.supplemental);
  return Number(state.user.restSecondsMain || DEFAULT_REST_TIMER_SECONDS.main);
}

function getRestTimerDisplay() {
  const timer = state.ui?.restTimer;
  if (!timer?.endAt) {
    return {
      countdown: 'Off',
      detail: 'Start automatically after you mark a set done, or use the controls here.',
      running: false,
      finished: false,
    };
  }

  const secondsLeft = Math.max(0, Math.ceil((timer.endAt - Date.now()) / 1000));
  const minutes = Math.floor(secondsLeft / 60);
  const seconds = String(secondsLeft % 60).padStart(2, '0');
  return {
    countdown: `${minutes}:${seconds}`,
    detail: secondsLeft > 0 ? (timer.label || 'Resting') : 'Rest complete',
    running: secondsLeft > 0,
    finished: secondsLeft <= 0,
  };
}

function syncRestTimerDisplay() {
  const countdownNode = app.querySelector('[data-rest-timer-countdown]');
  const detailNode = app.querySelector('[data-rest-timer-detail]');
  const timerNode = app.querySelector('[data-rest-timer]');
  if (!countdownNode || !detailNode || !timerNode) return;

  const display = getRestTimerDisplay();
  countdownNode.textContent = display.countdown;
  detailNode.textContent = display.detail;
  timerNode.dataset.timerState = display.finished ? 'finished' : display.running ? 'running' : 'idle';
}

function startRestTimer(seconds, label, phase = 'main') {
  updateState((draftState) => {
    draftState.ui.restTimer = {
      endAt: Date.now() + Number(seconds || 0) * 1000,
      durationSeconds: Number(seconds || 0),
      label,
      phase,
    };
  });
}

function clearRestTimer() {
  updateState((draftState) => {
    draftState.ui.restTimer = {
      endAt: null,
      durationSeconds: 0,
      label: '',
      phase: 'main',
    };
  });
}

function adjustRestTimer(deltaSeconds) {
  updateState((draftState) => {
    const timer = draftState.ui.restTimer || {};
    if (!timer.endAt) {
      const baseSeconds = Math.max(30, Number(timer.durationSeconds || state.user?.restSecondsMain || DEFAULT_REST_TIMER_SECONDS.main) + Number(deltaSeconds || 0));
      draftState.ui.restTimer = {
        endAt: Date.now() + baseSeconds * 1000,
        durationSeconds: baseSeconds,
        label: timer.label || 'Rest timer',
        phase: timer.phase || 'main',
      };
      return;
    }

    const remainingSeconds = Math.max(0, Math.ceil((timer.endAt - Date.now()) / 1000));
    const nextSeconds = Math.max(0, remainingSeconds + Number(deltaSeconds || 0));
    draftState.ui.restTimer = {
      ...timer,
      endAt: nextSeconds > 0 ? Date.now() + nextSeconds * 1000 : null,
      durationSeconds: nextSeconds > 0 ? nextSeconds : 0,
    };
  });
}

function restartRestTimer() {
  const timer = state.ui?.restTimer;
  if (!timer?.durationSeconds) return;
  startRestTimer(timer.durationSeconds, timer.label || 'Rest timer', timer.phase || 'main');
}

function updateReviewSelection(liftKey, value) {
  updateState((draftState) => {
    draftState.reviewSelections[liftKey] = value;
  });
}

function getPreviewRows() {
  const draft = state.onboarding;
  return LIFT_ORDER.map((liftKey) => {
    const liftDraft = draft.lifts[liftKey];
    const { suggestedTM, plannedTM, usingOverride } = resolveTrainingMax(
      liftDraft,
      draft.experienceLevel,
      Number(draft.roundingIncrement),
    );
    const weekOneTopSet = roundWeight(plannedTM * 0.85, Number(draft.roundingIncrement));
    const guardrail = getOnboardingLiftGuardrail(
      liftDraft,
      draft.experienceLevel,
      Number(draft.roundingIncrement),
    );

    return {
      liftKey,
      sourceType: liftDraft.sourceType,
      sourceValue: Number(liftDraft.sourceValue),
      suggestedTM,
      plannedTM,
      usingOverride,
      weekOneTopSet,
      increment: getStandardIncrement(liftKey, draft.units),
      guardrail,
    };
  });
}

function renderOnboarding() {
  const draft = state.onboarding;
  const previewRows = getPreviewRows();
  const template = TEMPLATE_PRESETS[draft.templateSlug];
  const cadenceNote = getCadenceNote(draft.templateSlug, draft.daysPerWeek);
  const invalidRows = previewRows.filter((row) => row.plannedTM <= 0 || row.sourceValue <= 0);

  return `
    <main class="shell shell--auth">
      <section class="hero card">
        <div>
          <p class="eyebrow">Start here</p>
          <h1>${APP_TITLE}</h1>
          <p class="lead">Set your lifts, choose your schedule, and get clear work sets you can use right away in the gym.</p>
        </div>
      </section>

      <section class="grid grid--2">
        <article class="card">
          <h2>1. Setup</h2>
          <div class="form-grid">
            <label>
              <span>Units</span>
              <select data-onboarding-field="units">
                <option value="lb" ${draft.units === 'lb' ? 'selected' : ''}>lb</option>
                <option value="kg" ${draft.units === 'kg' ? 'selected' : ''}>kg</option>
              </select>
            </label>

            <label>
              <span>Rounding increment</span>
              <input type="number" min="0.5" step="0.5" value="${escapeHtml(draft.roundingIncrement)}" data-onboarding-field="roundingIncrement" />
            </label>

            <label>
              <span>Experience</span>
              <select data-onboarding-field="experienceLevel">
                <option value="beginner" ${draft.experienceLevel === 'beginner' ? 'selected' : ''}>Beginner</option>
                <option value="intermediate" ${draft.experienceLevel === 'intermediate' ? 'selected' : ''}>Intermediate</option>
                <option value="advanced" ${draft.experienceLevel === 'advanced' ? 'selected' : ''}>Advanced</option>
              </select>
            </label>

            <label>
              <span>Days per week</span>
              <select data-onboarding-field="daysPerWeek">
                <option value="3" ${Number(draft.daysPerWeek) === 3 ? 'selected' : ''}>3</option>
                <option value="4" ${Number(draft.daysPerWeek) === 4 ? 'selected' : ''}>4</option>
              </select>
            </label>

            <label class="field field--full">
              <span>Template</span>
              <select data-onboarding-field="templateSlug">
                ${Object.values(TEMPLATE_PRESETS)
                  .map(
                    (preset) => `<option value="${preset.slug}" ${preset.slug === draft.templateSlug ? 'selected' : ''}>${preset.name}</option>`,
                  )
                  .join('')}
              </select>
              <small>${template.summary}</small>
            </label>

            <label class="checkbox-field field--full">
              <input type="checkbox" data-onboarding-field="showWarmups" ${draft.showWarmups !== false ? 'checked' : ''} />
              <span>Show recommended warm-up sets before each main lift</span>
            </label>
          </div>
          <p class="field-note">${escapeHtml(cadenceNote)}</p>
        </article>

        <article class="card">
          <h2>2. Starting weight preview</h2>
          <p class="muted">Enter a best lift or current training max and the app will suggest a conservative starting weight for each lift. You can adjust it before you begin.</p>
          <p class="field-note">Your work sets are calculated from the training max shown here.</p>
          ${invalidRows.length ? `<p class="field-note field-note--warning">Enter valid numbers for ${invalidRows.map((row) => LIFT_LABELS[row.liftKey]).join(', ')} before starting the program.</p>` : ''}
          <div class="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Lift</th>
                  <th>Suggested training max</th>
                  <th>Starting training max</th>
                  <th>Wave 1 top set</th>
                  <th>Cycle jump</th>
                </tr>
              </thead>
              <tbody>
                ${previewRows
                  .map(
                    (row) => `
                    <tr>
                      <td>${LIFT_LABELS[row.liftKey]}</td>
                      <td>${formatWeight(row.suggestedTM, draft.units)}</td>
                      <td>
                        <strong>${formatWeight(row.plannedTM, draft.units)}</strong>
                        ${row.usingOverride ? '<div class="table-cell-note">Custom starting weight</div>' : ''}
                        ${row.guardrail ? `<div class="table-cell-note table-cell-note--warning">${escapeHtml(row.guardrail)}</div>` : ''}
                      </td>
                      <td>${formatWeight(row.weekOneTopSet, draft.units)}</td>
                      <td>+${formatNumber(row.increment)} ${draft.units}</td>
                    </tr>`,
                  )
                  .join('')}
              </tbody>
            </table>
          </div>
        </article>
      </section>

      <section class="card">
        <h2>3. Lift inputs</h2>
        <div class="grid grid--2">
          ${LIFT_ORDER.map((liftKey) => {
            const lift = draft.lifts[liftKey];
            const preview = previewRows.find((row) => row.liftKey === liftKey);
            return `
              <div class="lift-card">
                <h3>${LIFT_LABELS[liftKey]}</h3>
                <div class="form-grid">
                  <label>
                    <span>Source type</span>
                    <select data-lift-source="${liftKey}" data-field="sourceType">
                      <option value="true_1rm" ${lift.sourceType === 'true_1rm' ? 'selected' : ''}>True 1RM</option>
                      <option value="estimated_1rm" ${lift.sourceType === 'estimated_1rm' ? 'selected' : ''}>Estimated 1RM</option>
                      <option value="training_max" ${lift.sourceType === 'training_max' ? 'selected' : ''}>Current TM</option>
                    </select>
                  </label>
                  <label>
                    <span>${lift.sourceType === 'training_max' ? 'Current TM' : 'Best known max'}</span>
                    <input type="number" min="0" step="1" value="${escapeHtml(lift.sourceValue)}" data-lift-source="${liftKey}" data-field="sourceValue" />
                  </label>
                  <label class="field--full">
                    <span>Starting training max override (optional)</span>
                    <input type="number" min="0" step="0.5" value="${escapeHtml(lift.trainingMaxOverride)}" placeholder="Use suggested starting weight" data-lift-source="${liftKey}" data-field="trainingMaxOverride" />
                    <small>Suggested training max: ${formatWeight(preview?.suggestedTM || 0, draft.units)}</small>
                  </label>
                </div>
                ${preview?.guardrail ? `<p class="field-note field-note--warning">${escapeHtml(preview.guardrail)}</p>` : '<p class="field-note">Your work sets will use the starting training max shown above.</p>'}
              </div>
            `;
          }).join('')}
        </div>
        <div class="stack-row stack-row--end">
          <button class="button" data-action="start-program" ${invalidRows.length ? 'disabled' : ''}>Start cycle 1</button>
        </div>
      </section>
    </main>
  `;
}

function renderHome() {
  const cycle = state.activeCycle;
  const inProgressWorkout = getInProgressWorkout(cycle);
  const nextWorkout = getNextWorkout(cycle);
  const focusWorkout = inProgressWorkout || nextWorkout;
  const lastCompletedWorkout = getLastCompletedWorkout(cycle);
  const upcomingWorkouts = getUpcomingWorkouts(cycle, 4);
  const progress = getCycleProgress(cycle);
  const reviews = getCycleReviews();
  const bestSetByLift = reviews.map((review) => {
    const best = [...review.topSets].sort((a, b) => estimateE1RM(b.actualWeight, b.actualReps) - estimateE1RM(a.actualWeight, a.actualReps))[0];
    return { liftKey: review.liftKey, best };
  });
  const focusCompletion = focusWorkout ? getWorkoutCompletion(focusWorkout) : { completedSets: 0, totalSets: 0 };
  const focusRemainingSets = focusWorkout ? getRemainingSetCount(focusWorkout) : 0;
  const focusNextSet = focusWorkout ? getNextIncompleteSet(focusWorkout) : null;
  const focusPercent = focusWorkout
    ? Math.round((focusCompletion.completedSets / Math.max(focusCompletion.totalSets, 1)) * 100)
    : 0;
  const focusAssistanceStarted = focusWorkout
    ? Object.values(focusWorkout.assistanceLog || {}).some((value) => Number(value || 0) > 0)
    : false;

  return `
    <section class="panel-stack">
      <div class="grid grid--3">
        <article class="card stat-card">
          <span class="eyebrow">Current cycle</span>
          <h2>Cycle ${cycle.cycleNumber}</h2>
          <p>${TEMPLATE_PRESETS[cycle.templateSlug].name}</p>
          <div class="progress">
            <div class="progress__bar" style="width:${progress.percent}%"></div>
          </div>
          <small>${progress.completed} of ${progress.total} workouts complete</small>
          <div class="table-cell-note">${escapeHtml(getCadenceNote(cycle.templateSlug, cycle.daysPerWeek))}</div>
        </article>

        <article class="card stat-card">
          <span class="eyebrow">${inProgressWorkout ? 'Resume session' : 'Next session'}</span>
          <h2>${focusWorkout ? escapeHtml(focusWorkout.title) : 'Cycle review ready'}</h2>
          <p>${focusWorkout ? `${getWeekWave(focusWorkout.weekType).label} · ${getWorkoutLiftNames(focusWorkout)}${inProgressWorkout ? ` · ${focusCompletion.completedSets}/${focusCompletion.totalSets} sets logged` : ''}` : 'All workouts complete. Review your lifts and generate the next cycle.'}</p>
          <div class="stack-row">
            ${focusWorkout ? `<button class="button" data-action="open-workout" data-workout-id="${focusWorkout.id}">${inProgressWorkout ? 'Resume workout' : 'Start workout'}</button>` : `<button class="button" data-tab="program">Review cycle</button>`}
          </div>
          ${focusWorkout ? `<div class="table-cell-note">${focusNextSet ? `Next up · ${escapeHtml(getSetLocationLabel(focusWorkout, focusNextSet.id))}` : 'All sets checked off. Log assistance and finish the session.'}</div>` : ''}
        </article>

        <article class="card stat-card">
          <span class="eyebrow">Recent session</span>
          <h2>${lastCompletedWorkout ? escapeHtml(lastCompletedWorkout.title) : 'Nothing logged yet'}</h2>
          <p>${lastCompletedWorkout ? getTopSetSummary(lastCompletedWorkout) : 'Complete one workout to start building history and PR highlights.'}</p>
          <div class="table-cell-note">Keep the training max conservative and let the PRs come to you.</div>
        </article>
      </div>

      <article class="card">
        <div class="section-heading section-heading--space-between">
          <div>
            <span class="eyebrow">${inProgressWorkout ? 'In progress' : 'Today focus'}</span>
            <h2>${focusWorkout ? escapeHtml(focusWorkout.title) : 'Cycle review ready'}</h2>
            <p class="muted">${focusWorkout ? `${getWeekWave(focusWorkout.weekType).label} · ${getWorkoutLiftNames(focusWorkout)} · ${focusRemainingSets} ${focusRemainingSets === 1 ? 'set' : 'sets'} left` : 'No workouts remain in this cycle. Head to Program to review each lift and generate the next cycle.'}</p>
          </div>
          <div class="stack-row">
            ${focusWorkout ? `<span class="badge ${inProgressWorkout ? '' : 'badge--muted'}">${inProgressWorkout ? 'Resume' : 'Planned'}</span><button class="button" data-action="open-workout" data-workout-id="${focusWorkout.id}">${inProgressWorkout ? 'Resume workout' : 'Open workout'}</button>` : `<button class="button" data-tab="program">Review cycle</button>`}
          </div>
        </div>

        ${focusWorkout ? `
          <div class="grid grid--3">
            <div class="metric-card">
              <span class="metric-card__label">Session progress</span>
              <strong>${focusCompletion.completedSets}/${focusCompletion.totalSets} sets</strong>
              <small>${focusPercent}% complete · ${focusRemainingSets} left</small>
            </div>
            <div class="metric-card">
              <span class="metric-card__label">${focusNextSet ? 'Next set' : 'Main work'}</span>
              <strong>${focusNextSet ? formatWeight(focusNextSet.prescribedWeight, state.user.units) : 'All sets logged'}</strong>
              <small>${focusNextSet ? `${getSetLocationLabel(focusWorkout, focusNextSet.id)} · ${focusNextSet.repStyle === 'plus' ? `${focusNextSet.prescribedReps}+` : focusNextSet.prescribedReps} reps` : 'Review assistance and session notes, then finish the workout.'}</small>
            </div>
            <div class="metric-card">
              <span class="metric-card__label">Assistance</span>
              <strong>${focusAssistanceStarted ? 'Started' : 'Ready to log'}</strong>
              <small>${focusWorkout.assistanceTargets.map((target) => `${ASSISTANCE_LABELS[target.category]} ${target.targetMin}-${target.targetMax}`).join(' · ')}</small>
            </div>
          </div>

          <div class="stack-list compact-list">
            ${focusWorkout.blocks.flatMap((block) => block.sections.map((section) => `
              <div class="list-row">
                <div>
                  <strong>${LIFT_LABELS[block.liftKey]} · ${section.title}</strong>
                  <div class="muted">${section.description}</div>
                </div>
                <div class="list-row__meta">${summarizeSection(section, state.user.units)}</div>
              </div>
            `)).join('')}
          </div>
        ` : ''}
      </article>

      <article class="card">
        <div class="section-heading">
          <div>
            <h2>Training maxes</h2>
            <p class="muted">Separate from true or estimated 1RM. ${state.user.daysPerWeek} days/week · ${state.user.units} plates.</p>
          </div>
        </div>
        <div class="grid grid--4">
          ${LIFT_ORDER.map((liftKey) => `
            <div class="metric-card">
              <span class="metric-card__label">${LIFT_LABELS[liftKey]}</span>
              <strong>${formatWeight(state.liftProfiles[liftKey].currentTrainingMax, state.user.units)}</strong>
              <small>+${formatNumber(state.liftProfiles[liftKey].incrementPerCycle)} ${state.user.units} / cycle</small>
            </div>
          `).join('')}
        </div>
      </article>

      <div class="grid grid--2">
        <article class="card">
          <div class="section-heading">
            <div>
              <h2>Upcoming queue</h2>
              <p class="muted">The next few sessions in your rolling rotation.</p>
            </div>
          </div>
          <div class="stack-list">
            ${upcomingWorkouts.map((workout, index) => `
              <button class="list-row list-row--button" data-action="open-workout" data-workout-id="${workout.id}">
                <div>
                  <strong>${index === 0 && !inProgressWorkout ? 'Up next · ' : ''}${escapeHtml(workout.title)}</strong>
                  <div class="muted">${getWeekWave(workout.weekType).label} · ${getWorkoutLiftNames(workout)}</div>
                </div>
                <div class="list-row__meta">
                  <span class="badge ${workout.status === 'in_progress' || isWorkoutStarted(workout) ? '' : 'badge--muted'}">${workout.status === 'in_progress' || isWorkoutStarted(workout) ? 'Resume' : 'Planned'}</span>
                </div>
              </button>
            `).join('') || '<p class="muted">No upcoming workouts left in this cycle.</p>'}
          </div>
        </article>

        <article class="card">
          <div class="section-heading">
            <div>
              <h2>Recent best plus sets</h2>
              <p class="muted">Simple PR highlights based on estimated 1RM.</p>
            </div>
          </div>
          <div class="stack-list">
            ${bestSetByLift.map(({ liftKey, best }) => `
              <div class="list-row">
                <div>
                  <strong>${LIFT_LABELS[liftKey]}</strong>
                  <div class="muted">${best ? escapeHtml(best.workoutTitle) : 'No completed top sets yet'}</div>
                </div>
                <div class="list-row__meta">
                  ${best ? `${formatNumber(best.actualReps)} reps @ ${formatWeight(best.actualWeight, state.user.units)}` : '—'}
                </div>
              </div>
            `).join('')}
          </div>
        </article>
      </div>
    </section>
  `;
}

function renderSetRow(workout, set, isActive = false) {
  const weightDelta = Number(state.user?.roundingIncrement || 0.5);
  const displayedPlateWeight = Number(set.actualWeight || set.prescribedWeight || 0);
  const platePreviewMarkup = renderPlateBreakdown(displayedPlateWeight);
  const rowClassNames = [
    'set-row',
    isActive ? 'set-row--active' : '',
    set.completed ? 'set-row--done' : '',
    set.repStyle === 'plus' ? 'set-row--plus' : '',
  ]
    .filter(Boolean)
    .join(' ');
  const setMeta = [
    `Set ${set.setIndex}`,
    set.repStyle === 'plus' ? 'top set' : null,
    isActive && !set.completed ? 'next up' : null,
    set.completed ? 'logged' : null,
  ]
    .filter(Boolean)
    .join(' · ');

  return `
    <div class="${rowClassNames}">
      <div class="set-row__summary">
        <small class="set-row__meta">${setMeta}</small>
        <strong class="set-row__weight">${formatWeight(set.prescribedWeight, state.user.units)}</strong>
        <div class="set-row__prescription-line">
          <span>${set.repStyle === 'plus' ? `${set.prescribedReps}+` : set.prescribedReps} reps</span>
          <span>·</span>
          <span>${Math.round(set.percentageOfTrainingMax * 100)}% TM</span>
        </div>
      </div>
      <div class="set-row__actions">
        <button
          type="button"
          class="button ${set.completed ? 'button--secondary' : ''}"
          data-action="toggle-set-complete"
          data-workout-id="${workout.id}"
          data-set-id="${set.id}"
        >
          ${set.completed ? 'Undo' : 'Mark done'}
        </button>
      </div>
      <div class="set-row__plate-preview" data-plate-preview-workout-id="${workout.id}" data-plate-preview-set-id="${set.id}">${platePreviewMarkup}</div>
      <div class="set-row__fields">
        <label class="set-field">
          <span>Reps done</span>
          <div class="set-stepper">
            <button type="button" class="stepper-button" data-action="adjust-set-value" data-workout-id="${workout.id}" data-set-id="${set.id}" data-field="actualReps" data-delta="-1">−</button>
            <input
              type="number"
              min="0"
              step="1"
              value="${escapeHtml(set.actualReps)}"
              data-action="set-field"
              data-workout-id="${workout.id}"
              data-set-id="${set.id}"
              data-field="actualReps"
            />
            <button type="button" class="stepper-button" data-action="adjust-set-value" data-workout-id="${workout.id}" data-set-id="${set.id}" data-field="actualReps" data-delta="1">+</button>
          </div>
        </label>
        <label class="set-field">
          <span>Load</span>
          <div class="set-stepper">
            <button type="button" class="stepper-button" data-action="adjust-set-value" data-workout-id="${workout.id}" data-set-id="${set.id}" data-field="actualWeight" data-delta="-${weightDelta}">−</button>
            <input
              type="number"
              min="0"
              step="0.5"
              value="${escapeHtml(set.actualWeight)}"
              data-action="set-field"
              data-workout-id="${workout.id}"
              data-set-id="${set.id}"
              data-field="actualWeight"
            />
            <button type="button" class="stepper-button" data-action="adjust-set-value" data-workout-id="${workout.id}" data-set-id="${set.id}" data-field="actualWeight" data-delta="${weightDelta}">+</button>
          </div>
        </label>
      </div>
    </div>
  `;
}

function renderWorkout() {
  const workout = getSelectedWorkout();
  if (!workout) {
    return `
      <section class="card empty-state">
        <h2>No workout selected</h2>
        <p>Finish onboarding or generate a cycle to start training.</p>
      </section>
    `;
  }

  const completion = getWorkoutCompletion(workout);
  const remainingSets = getRemainingSetCount(workout);
  const nextSet = getNextIncompleteSet(workout);
  const lastCompletedSet = getLastCompletedSet(workout);
  const completeButtonLabel = remainingSets > 0 ? `Finish workout (${remainingSets} left)` : 'Complete workout';
  const warmupPlan = nextSet ? getWarmupPlanForSet(workout, nextSet) : null;
  const restTimerDisplay = getRestTimerDisplay();

  return `
    <section class="panel-stack">
      <article class="card">
        <div class="section-heading section-heading--space-between">
          <div>
            <p class="eyebrow">${getWeekWave(workout.weekType).label}</p>
            <h2>${escapeHtml(workout.title)}</h2>
            <p class="muted">${TEMPLATE_PRESETS[workout.templateSlug].name} · ${completion.completedSets}/${completion.totalSets} sets checked off · ${remainingSets} left</p>
            <small class="field-note">${escapeHtml(getWorkoutLiftNames(workout))} · ${Math.round((completion.completedSets / Math.max(completion.totalSets, 1)) * 100)}% complete</small>
          </div>
          <div class="stack-row">
            ${workout.status !== 'completed' ? `<button class="button" data-action="save-workout" data-workout-id="${workout.id}">${completeButtonLabel}</button>` : `<span class="badge badge--success">Completed</span>`}
          </div>
        </div>
      </article>

      <article class="card workout-focus-card">
        <div class="section-heading section-heading--space-between">
          <div>
            <p class="eyebrow">Now lifting</p>
            <h3>${nextSet ? 'Next set ready' : 'Main work logged'}</h3>
            <p class="muted">${nextSet ? `${getSetLocationLabel(workout, nextSet.id)} · ${formatWeight(nextSet.prescribedWeight, state.user.units)} for ${nextSet.repStyle === 'plus' ? `${nextSet.prescribedReps}+` : nextSet.prescribedReps}` : 'All sets are checked off. Review assistance and notes, then finish the workout.'}</p>
          </div>
          <div class="stack-row">
            ${lastCompletedSet && workout.status !== 'completed' ? `<button class="button button--secondary" data-action="toggle-set-complete" data-workout-id="${workout.id}" data-set-id="${lastCompletedSet.id}">Undo last set</button>` : ''}
            ${nextSet ? `<button class="button" data-action="toggle-set-complete" data-workout-id="${workout.id}" data-set-id="${nextSet.id}">Mark next set done</button>` : ''}
            ${workout.status !== 'completed' ? `<button class="button button--secondary" data-action="save-workout" data-workout-id="${workout.id}">${completeButtonLabel}</button>` : ''}
          </div>
        </div>
        <div class="rest-timer" data-rest-timer data-timer-state="${restTimerDisplay.finished ? 'finished' : restTimerDisplay.running ? 'running' : 'idle'}">
          <div>
            <small class="set-row__meta">Rest timer</small>
            <strong class="rest-timer__countdown" data-rest-timer-countdown>${restTimerDisplay.countdown}</strong>
            <div class="muted" data-rest-timer-detail>${restTimerDisplay.detail}</div>
          </div>
          <div class="stack-row">
            <button class="button button--secondary" type="button" data-action="rest-timer-adjust" data-seconds="-30">−30s</button>
            <button class="button button--secondary" type="button" data-action="rest-timer-restart">Restart</button>
            <button class="button button--secondary" type="button" data-action="rest-timer-adjust" data-seconds="30">+30s</button>
            <button class="button button--secondary" type="button" data-action="rest-timer-clear">Clear</button>
          </div>
        </div>
        ${warmupPlan ? renderWarmupPlan(warmupPlan) : ''}
        ${nextSet ? renderSetRow(workout, nextSet, true) : `<div class="list-row"><div><strong>Main work complete</strong><div class="muted">${lastCompletedSet ? escapeHtml(getSetLocationLabel(workout, lastCompletedSet.id)) : 'All prescribed sets are logged.'}</div></div><div class="list-row__meta">${lastCompletedSet ? `${formatNumber(lastCompletedSet.actualReps)} reps @ ${formatWeight(lastCompletedSet.actualWeight || lastCompletedSet.prescribedWeight, state.user.units)}` : 'Ready for assistance'}</div></div>`}
      </article>

      <article class="card">
        <div class="section-heading">
          <div>
            <h3>Workout flow</h3>
            <p class="muted">Stay focused on the next set. The rest of the session is summarized here.</p>
          </div>
        </div>
        <div class="stack-list compact-list">
          ${workout.blocks.flatMap((block) => block.sections.map((section) => {
            const sectionCompletion = getSectionCompletion(section);
            const sectionNextSet = section.sets.find((set) => !set.completed) || null;
            return `
              <div class="list-row">
                <div>
                  <strong>${LIFT_LABELS[block.liftKey]} · ${section.title}</strong>
                  <div class="muted">${section.description}</div>
                </div>
                <div class="list-row__meta">
                  <div>${sectionCompletion.completed}/${sectionCompletion.total} done</div>
                  <small>${sectionNextSet ? `${formatWeight(sectionNextSet.prescribedWeight, state.user.units)} × ${sectionNextSet.repStyle === 'plus' ? `${sectionNextSet.prescribedReps}+` : sectionNextSet.prescribedReps} next` : 'Section complete'}</small>
                </div>
              </div>
            `;
          })).join('')}
        </div>
      </article>

      <article class="card">
        <div class="section-heading">
          <div>
            <h3>Assistance + session notes</h3>
            <p class="muted">Log assistance by category to keep the main workout fast.</p>
          </div>
        </div>
        <div class="list-row">
          <div>
            <strong>What to do today</strong>
            <div class="muted">${ASSISTANCE_TEMPLATE_NOTES[workout.templateSlug] || ASSISTANCE_TEMPLATE_NOTES.vanilla}</div>
          </div>
          <div class="list-row__meta">50-100 reps per category</div>
        </div>
        <div class="grid grid--3">
          ${workout.assistanceTargets.map((target) => {
            const guide = ASSISTANCE_GUIDANCE[target.category];
            const favorite = workout.assistanceExercises?.[target.category] || state.user?.assistanceFavorites?.[target.category] || '';
            return `
              <div class="lift-card assistance-card">
                <h4>${ASSISTANCE_LABELS[target.category]}</h4>
                <p class="muted">${guide.summary}</p>
                <small class="field-note">Try: ${guide.examples.join(' · ')}</small>
                ${favorite ? `<small class="field-note">Favorite: ${escapeHtml(favorite)}</small>` : ''}
              </div>
            `;
          }).join('')}
        </div>
        <div class="grid grid--3">
          ${workout.assistanceTargets.map((target) => `
            <div class="assist-log-card">
              <label>
                <span>${ASSISTANCE_LABELS[target.category]} reps (${target.targetMin}-${target.targetMax})</span>
                <input
                  type="number"
                  min="0"
                  step="1"
                  value="${escapeHtml(workout.assistanceLog[target.category])}"
                  data-action="workout-meta"
                  data-workout-id="${workout.id}"
                  data-field="assistance.${target.category}"
                />
              </label>
              <div class="assist-log-buttons">
                <button type="button" class="button button--secondary" data-action="adjust-assistance-reps" data-workout-id="${workout.id}" data-category="${target.category}" data-delta="25">+25</button>
                <button type="button" class="button button--secondary" data-action="adjust-assistance-reps" data-workout-id="${workout.id}" data-category="${target.category}" data-delta="50">+50</button>
                <button type="button" class="button button--secondary" data-action="adjust-assistance-reps" data-workout-id="${workout.id}" data-category="${target.category}" data-delta="75">+75</button>
                <button type="button" class="button button--secondary" data-action="adjust-assistance-reps" data-workout-id="${workout.id}" data-category="${target.category}" data-delta="100">+100</button>
              </div>
            </div>
          `).join('')}
        </div>

        <div class="grid grid--3">
          <label>
            <span>Recovery</span>
            <select data-action="workout-meta" data-workout-id="${workout.id}" data-field="recovery">
              <option value="fresh" ${workout.recovery === 'fresh' ? 'selected' : ''}>Fresh</option>
              <option value="okay" ${workout.recovery === 'okay' ? 'selected' : ''}>Okay</option>
              <option value="beat_up" ${workout.recovery === 'beat_up' ? 'selected' : ''}>Beat up</option>
            </select>
          </label>
          <label>
            <span>Conditioning done</span>
            <select data-action="workout-meta" data-workout-id="${workout.id}" data-field="conditioning.completed">
              <option value="no" ${workout.conditioning?.completed === 'no' ? 'selected' : ''}>No</option>
              <option value="yes" ${workout.conditioning?.completed === 'yes' ? 'selected' : ''}>Yes</option>
            </select>
          </label>
          <label>
            <span>Conditioning minutes</span>
            <input type="number" min="0" step="1" value="${escapeHtml(workout.conditioning?.minutes || '')}" data-action="workout-meta" data-workout-id="${workout.id}" data-field="conditioning.minutes" />
          </label>
        </div>

        <div class="grid grid--3">
          <label>
            <span>Difficulty</span>
            <select data-action="workout-meta" data-workout-id="${workout.id}" data-field="flags.difficulty">
              <option value="easy" ${workout.flags.difficulty === 'easy' ? 'selected' : ''}>Easy</option>
              <option value="normal" ${workout.flags.difficulty === 'normal' ? 'selected' : ''}>Normal</option>
              <option value="hard" ${workout.flags.difficulty === 'hard' ? 'selected' : ''}>Hard</option>
            </select>
          </label>
          <label>
            <span>Bar speed</span>
            <select data-action="workout-meta" data-workout-id="${workout.id}" data-field="flags.barSpeed">
              <option value="fast" ${workout.flags.barSpeed === 'fast' ? 'selected' : ''}>Fast</option>
              <option value="okay" ${workout.flags.barSpeed === 'okay' ? 'selected' : ''}>Okay</option>
              <option value="grindy" ${workout.flags.barSpeed === 'grindy' ? 'selected' : ''}>Grindy</option>
            </select>
          </label>
          <label>
            <span>Pain flag</span>
            <select data-action="workout-meta" data-workout-id="${workout.id}" data-field="flags.pain">
              <option value="no" ${workout.flags.pain === 'no' ? 'selected' : ''}>No pain</option>
              <option value="yes" ${workout.flags.pain === 'yes' ? 'selected' : ''}>Pain noted</option>
            </select>
          </label>
        </div>

        <div class="grid grid--2">
          <label class="field field--full">
            <span>Conditioning note</span>
            <input type="text" value="${escapeHtml(workout.conditioning?.note || '')}" data-action="workout-meta" data-workout-id="${workout.id}" data-field="conditioning.note" placeholder="Bike, sled, jumps, easy run, etc." />
          </label>
          <label class="field field--full">
            <span>Notes</span>
            <textarea rows="4" data-action="workout-meta" data-workout-id="${workout.id}" data-field="notes">${escapeHtml(workout.notes)}</textarea>
          </label>
        </div>
      </article>
    </section>
  `;
}

function renderHistoryCycle(cycle) {
  const completed = getCompletedWorkouts(cycle);
  return `
    <article class="card">
      <div class="section-heading section-heading--space-between">
        <div>
          <h3>Cycle ${cycle.cycleNumber}</h3>
          <p class="muted">${TEMPLATE_PRESETS[cycle.templateSlug].name}</p>
        </div>
        <small class="field-note">${completed.length}/${cycle.workouts.length} complete</small>
      </div>
      <div class="stack-list">
        ${completed.map((workout) => {
          const topSet = workout.blocks
            .flatMap((block) => block.sections)
            .map((section) => getTopSetFromSection(section))
            .find(Boolean);
          return `
            <div class="list-row">
              <div>
                <strong>${escapeHtml(workout.title)}</strong>
                <div class="muted">${new Date(workout.completedAt).toLocaleDateString()}</div>
              </div>
              <div class="list-row__meta">
                ${topSet ? `${formatNumber(topSet.actualReps)} @ ${formatWeight(topSet.actualWeight, state.user.units)}` : 'Logged'}
              </div>
            </div>
          `;
        }).join('') || '<p class="muted">No completed workouts yet.</p>'}
      </div>
    </article>
  `;
}

function renderHistory() {
  const activeCompleted = state.activeCycle ? getCompletedWorkouts(state.activeCycle) : [];
  return `
    <section class="panel-stack">
      <article class="card">
        <div class="section-heading">
          <div>
            <h2>Workout history</h2>
            <p class="muted">Simple audit trail of what actually got done.</p>
          </div>
        </div>
        <div class="stack-list">
          ${activeCompleted.map((workout) => `
            <div class="list-row">
              <div>
                <strong>${escapeHtml(workout.title)}</strong>
                <div class="muted">Cycle ${workout.cycleNumber} · ${getWeekWave(workout.weekType).shortLabel}</div>
              </div>
              <div class="list-row__meta">${workout.completedAt ? new Date(workout.completedAt).toLocaleDateString() : '—'}</div>
            </div>
          `).join('') || '<p class="muted">No workouts completed in the active cycle yet.</p>'}
        </div>
      </article>

      ${state.cycleHistory.map((cycle) => renderHistoryCycle(cycle)).join('') || '<article class="card"><p class="muted">Completed cycles will show up here after review and rollover.</p></article>'}
    </section>
  `;
}

function getLiftTopSetsAcrossCycles(liftKey) {
  return [state.activeCycle, ...state.cycleHistory]
    .filter(Boolean)
    .flatMap((cycle) => getTopSetResultsForLift(cycle, liftKey));
}

function getLiftProgressSummary(liftKey) {
  const currentTM = state.liftProfiles?.[liftKey]?.currentTrainingMax || 0;
  const previousReview = state.cycleHistory[0]?.review?.find((review) => review.liftKey === liftKey) || null;
  const topSets = getLiftTopSetsAcrossCycles(liftKey);
  const bestTopSet = getBestTopSet(topSets);
  return {
    currentTM,
    previousTM: previousReview ? previousReview.nextTrainingMax : null,
    bestTopSet,
    bestEstimatedOneRepMax: bestTopSet ? estimateE1RM(bestTopSet.actualWeight, bestTopSet.actualReps) : 0,
  };
}

function getProgramGuardrails(reviews) {
  const notes = [];
  const completedWorkouts = getCompletedWorkouts(state.activeCycle);
  const assistanceMisses = completedWorkouts.filter((workout) =>
    Object.values(workout.assistanceLog || {}).every((value) => Number(value || 0) === 0),
  ).length;
  const conditioningMisses = completedWorkouts.filter((workout) => workout.conditioning?.completed !== 'yes').length;

  if (reviews.some((review) => review.action === 'reset')) {
    notes.push('At least one lift is asking for a reset. Keep the next cycle conservative and clean.');
  }

  if (state.user?.recoveryWeekType === 'none' && state.cycleHistory.length >= 2) {
    notes.push('You have run multiple cycles without a recovery week. Consider turning on a deload or TM test week in Settings.');
  }

  if ((state.user?.templateSlug === 'bbb' || state.user?.templateSlug === 'five_pro_bbb') && conditioningMisses >= 2) {
    notes.push('BBB usually feels better with regular easy conditioning. You have skipped conditioning on multiple completed workouts.');
  }

  if (assistanceMisses >= 2) {
    notes.push('Assistance work has been skipped on several completed workouts. Save favorite movements in Settings to make logging easier.');
  }

  if (!notes.length) {
    notes.push('No major red flags right now. Keep progressing with smooth reps and honest training maxes.');
  }

  return notes;
}

function renderProgram() {
  const cycle = state.activeCycle;
  const reviews = getCycleReviews();
  const complete = isCycleComplete(cycle);
  const reviewSummary = getCycleReviewSummary(cycle, reviews);
  const guardrails = getProgramGuardrails(reviews);

  return `
    <section class="panel-stack">
      <article class="card">
        <div class="section-heading section-heading--space-between">
          <div>
            <h2>Program</h2>
            <p class="muted">${TEMPLATE_PRESETS[state.user.templateSlug].name} · ${state.user.daysPerWeek} days/week</p>
            <small class="field-note">${escapeHtml(getCadenceNote(state.user.templateSlug, state.user.daysPerWeek))}</small>
          </div>
          <small class="field-note">${complete ? 'Cycle ready for review' : 'Cycle in progress'}</small>
        </div>
        <div class="grid grid--4">
          ${LIFT_ORDER.map((liftKey) => {
            const progress = getLiftProgressSummary(liftKey);
            return `
              <div class="metric-card">
                <span class="metric-card__label">${LIFT_LABELS[liftKey]}</span>
                <strong>${formatWeight(progress.currentTM, state.user.units)}</strong>
                <small>${progress.bestEstimatedOneRepMax ? `Best e1RM ${formatWeight(roundWeight(progress.bestEstimatedOneRepMax, state.user.roundingIncrement), state.user.units)}` : `Source: ${state.liftProfiles[liftKey].sourceType.replaceAll('_', ' ')}`}</small>
              </div>
            `;
          }).join('')}
        </div>
      </article>

      <article class="card">
        <div class="section-heading">
          <div>
            <h3>Progress snapshot</h3>
            <p class="muted">See how each lift is trending before you make the next cycle decision.</p>
          </div>
        </div>
        <div class="grid grid--2">
          ${LIFT_ORDER.map((liftKey) => {
            const progress = getLiftProgressSummary(liftKey);
            return `
              <div class="lift-card">
                <h4>${LIFT_LABELS[liftKey]}</h4>
                <div class="grid grid--2 compact-grid">
                  <div class="metric-card metric-card--compact">
                    <span class="metric-card__label">Current TM</span>
                    <strong>${formatWeight(progress.currentTM, state.user.units)}</strong>
                    <small>${progress.previousTM ? `Last saved TM ${formatWeight(progress.previousTM, state.user.units)}` : 'No completed cycle yet'}</small>
                  </div>
                  <div class="metric-card metric-card--compact">
                    <span class="metric-card__label">Best top set</span>
                    <strong>${progress.bestTopSet ? `${formatNumber(progress.bestTopSet.actualReps)} reps` : '—'}</strong>
                    <small>${progress.bestTopSet ? `${formatWeight(progress.bestTopSet.actualWeight, state.user.units)} · ${escapeHtml(progress.bestTopSet.workoutTitle)}` : 'No recorded top set yet'}</small>
                  </div>
                </div>
              </div>
            `;
          }).join('')}
        </div>
      </article>

      <article class="card">
        <div class="section-heading">
          <div>
            <h3>Workout lineup</h3>
            <p class="muted">See every session in your current cycle and jump into the one you want to train.</p>
          </div>
        </div>
        <div class="stack-list">
          ${cycle.workouts.map((workout) => `
            <button class="list-row list-row--button" data-action="open-workout" data-workout-id="${workout.id}">
              <div>
                <strong>${escapeHtml(workout.title)}</strong>
                <div class="muted">${getWeekWave(workout.weekType).label} · ${workout.blocks.map((block) => LIFT_LABELS[block.liftKey]).join(' + ')} · ${workout.status}</div>
              </div>
            </button>
          `).join('')}
        </div>
      </article>

      <article class="card">
        <div class="section-heading">
          <div>
            <h3>Coach notes</h3>
            <p class="muted">Simple guardrails based on your recent cycle, recovery settings, and logged habits.</p>
          </div>
        </div>
        <div class="stack-list compact-list">
          ${guardrails.map((note) => `
            <div class="list-row review-callout">
              <div>
                <strong>Keep in mind</strong>
                <div class="muted">${escapeHtml(note)}</div>
              </div>
            </div>
          `).join('')}
        </div>
      </article>

      <article class="card">
        <div class="section-heading section-heading--space-between">
          <div>
            <h3>Next cycle recommendations</h3>
            <p class="muted">Review each lift, then choose whether to increase, hold, or reset before starting the next cycle.</p>
          </div>
          ${complete ? `<button class="button" data-action="apply-next-cycle">Start cycle ${reviewSummary.nextCycleNumber}</button>` : ''}
        </div>

        <div class="grid grid--4">
          <div class="metric-card">
            <span class="metric-card__label">Cycle status</span>
            <strong>${reviewSummary.completedWorkoutCount}/${reviewSummary.totalWorkoutCount}</strong>
            <small>${reviewSummary.headline}</small>
          </div>
          <div class="metric-card">
            <span class="metric-card__label">Recommendations</span>
            <strong>${reviewSummary.increaseCount} / ${reviewSummary.holdCount} / ${reviewSummary.resetCount}</strong>
            <small>Increase / Hold / Reset</small>
          </div>
          <div class="metric-card">
            <span class="metric-card__label">Flags this cycle</span>
            <strong>${reviewSummary.attentionCount}</strong>
            <small>${reviewSummary.painCount} pain · ${reviewSummary.hardCount} hard or grindy</small>
          </div>
          <div class="metric-card">
            <span class="metric-card__label">Best top set</span>
            <strong>${reviewSummary.bestTopSet ? formatWeight(reviewSummary.bestTopSet.actualWeight, state.user.units) : '—'}</strong>
            <small>${reviewSummary.bestTopSet ? `${formatNumber(reviewSummary.bestTopSet.actualReps)} reps in ${escapeHtml(reviewSummary.bestTopSet.workoutTitle)}` : 'Complete top sets to see a highlight.'}</small>
          </div>
        </div>

        ${complete ? `
          <div class="list-row review-callout">
            <div>
              <strong>Ready for cycle ${reviewSummary.nextCycleNumber}</strong>
              <div class="muted">Your selections below will be saved into history, then the next cycle will be built with the updated training maxes.</div>
            </div>
            <div class="list-row__meta">
              <span class="badge ${reviewSummary.resetCount > 0 ? 'badge--danger' : reviewSummary.holdCount > 0 ? 'badge--muted' : 'badge--success'}">${reviewSummary.resetCount > 0 ? 'Needs adjustment' : reviewSummary.holdCount > 0 ? 'Mixed' : 'Ready to progress'}</span>
            </div>
          </div>
        ` : `
          <div class="list-row review-callout">
            <div>
              <strong>Finish the current cycle first</strong>
              <div class="muted">Once every workout is complete, you can review each lift and start the next cycle in one tap.</div>
            </div>
            <div class="list-row__meta"><span class="badge badge--muted">Locked</span></div>
          </div>
        `}

        <div class="grid grid--2">
          ${reviews.map((review) => {
            const selectedAction = getReviewSelection(review.liftKey, review.action);
            const projectedTM = getProjectedTM(review.liftKey, selectedAction);
            const bestTopSet = getBestTopSet(review.topSets);
            return `
              <div class="lift-card">
                <div class="section-heading section-heading--space-between">
                  <div>
                    <h4>${LIFT_LABELS[review.liftKey]}</h4>
                    <p class="muted">${review.reason}</p>
                  </div>
                  <span class="badge ${getActionBadgeClass(review.action)}">Suggested: ${getActionLabel(review.action)}</span>
                </div>
                <div class="form-grid">
                  <label>
                    <span>Next action</span>
                    <select ${complete ? '' : 'disabled'} data-action="review-selection" data-lift-key="${review.liftKey}">
                      <option value="increase" ${selectedAction === 'increase' ? 'selected' : ''}>Increase</option>
                      <option value="hold" ${selectedAction === 'hold' ? 'selected' : ''}>Hold</option>
                      <option value="reset" ${selectedAction === 'reset' ? 'selected' : ''}>Reset</option>
                    </select>
                  </label>
                  <label>
                    <span>Projected next training max</span>
                    <input type="text" value="${formatWeight(projectedTM, state.user.units)}" disabled />
                  </label>
                </div>
                <div class="stack-list compact-list">
                  ${bestTopSet ? `
                    <div class="list-row">
                      <div>
                        <strong>Best top set</strong>
                        <div class="muted">${escapeHtml(bestTopSet.workoutTitle)} · ${getWeekWave(bestTopSet.weekType).shortLabel}</div>
                      </div>
                      <div class="list-row__meta">${formatNumber(bestTopSet.actualReps)} reps @ ${formatWeight(bestTopSet.actualWeight, state.user.units)}</div>
                    </div>
                  ` : ''}
                  ${review.topSets.map((set) => `
                    <div class="list-row">
                      <div>
                        <strong>${escapeHtml(set.workoutTitle)}</strong>
                        <div class="muted">${getWeekWave(set.weekType).shortLabel}</div>
                      </div>
                      <div class="list-row__meta">${formatNumber(set.actualReps)} reps @ ${formatWeight(set.actualWeight, state.user.units)}</div>
                    </div>
                  `).join('') || '<p class="muted">No top sets logged yet.</p>'}
                </div>
              </div>
            `;
          }).join('')}
        </div>
      </article>
    </section>
  `;
}

function renderSettings() {
  return `
    <section class="panel-stack">
      <article class="card">
        <div class="section-heading">
          <div>
            <h2>Program settings</h2>
            <p class="muted">Update your bar weight, rounding, and training maxes whenever you need to.</p>
            <small class="field-note">Completed workouts stay logged. Unfinished workouts will refresh with the updated numbers.</small>
          </div>
        </div>

        <div class="grid grid--3">
          <div class="metric-card">
            <span class="metric-card__label">Units</span>
            <strong>${state.user.units}</strong>
            <small>Changing units does not convert your saved numbers automatically.</small>
          </div>
          <label>
            <span>Rounding increment</span>
            <input type="number" min="0.5" step="0.5" value="${escapeHtml(state.user.roundingIncrement)}" data-settings-field="roundingIncrement" />
          </label>
          <label>
            <span>Barbell weight</span>
            <input type="number" min="1" step="0.5" value="${escapeHtml(state.user.barbellWeight || getDefaultBarbellWeight(state.user.units))}" data-settings-field="barbellWeight" />
          </label>
        </div>

        <label class="checkbox-field">
          <input type="checkbox" data-settings-field="showWarmups" ${state.user.showWarmups !== false ? 'checked' : ''} />
          <span>Show recommended warm-up sets before each main lift</span>
        </label>

        <div class="grid grid--3">
          <label>
            <span>Recovery week</span>
            <select data-settings-field="recoveryWeekType">
              <option value="none" ${state.user.recoveryWeekType === 'none' ? 'selected' : ''}>No recovery week</option>
              <option value="deload" ${state.user.recoveryWeekType === 'deload' ? 'selected' : ''}>Deload week</option>
              <option value="tm_test" ${state.user.recoveryWeekType === 'tm_test' ? 'selected' : ''}>TM test week</option>
            </select>
          </label>
          <label>
            <span>Run it every</span>
            <select data-settings-field="recoveryWeekEveryCycles">
              <option value="2" ${Number(state.user.recoveryWeekEveryCycles) === 2 ? 'selected' : ''}>2 cycles</option>
              <option value="3" ${Number(state.user.recoveryWeekEveryCycles) === 3 ? 'selected' : ''}>3 cycles</option>
              <option value="4" ${Number(state.user.recoveryWeekEveryCycles) === 4 ? 'selected' : ''}>4 cycles</option>
            </select>
          </label>
          <div class="metric-card metric-card--compact">
            <span class="metric-card__label">Current template</span>
            <strong>${TEMPLATE_PRESETS[state.user.templateSlug].name}</strong>
            <small>Choose 5's Pro, Joker, Leader, or Anchor variants during setup.</small>
          </div>
        </div>

        <div class="grid grid--3">
          <label>
            <span>Main lift rest timer</span>
            <input type="number" min="30" step="15" value="${escapeHtml(state.user.restSecondsMain)}" data-settings-field="restSecondsMain" />
          </label>
          <label>
            <span>Supplemental / joker rest timer</span>
            <input type="number" min="30" step="15" value="${escapeHtml(state.user.restSecondsSupplemental)}" data-settings-field="restSecondsSupplemental" />
          </label>
          <label>
            <span>Assistance rest timer</span>
            <input type="number" min="15" step="15" value="${escapeHtml(state.user.restSecondsAssistance)}" data-settings-field="restSecondsAssistance" />
          </label>
        </div>

        <div class="grid grid--3">
          ${ASSISTANCE_CATEGORIES.map((category) => `
            <label>
              <span>${ASSISTANCE_LABELS[category]} favorite</span>
              <input type="text" value="${escapeHtml(state.user.assistanceFavorites?.[category] || '')}" data-settings-assistance="${category}" placeholder="Example: ${escapeHtml(ASSISTANCE_GUIDANCE[category].examples[0])}" />
            </label>
          `).join('')}
        </div>

        <div class="grid grid--2">
          ${LIFT_ORDER.map((liftKey) => `
            <label>
              <span>${LIFT_LABELS[liftKey]} training max</span>
              <input
                type="number"
                min="0"
                step="0.5"
                value="${escapeHtml(state.liftProfiles[liftKey].currentTrainingMax)}"
                data-settings-lift="${liftKey}"
                data-field="currentTrainingMax"
              />
            </label>
          `).join('')}
        </div>

        <div class="stack-row stack-row--end">
          <button class="button" data-action="apply-program-settings">Apply changes</button>
        </div>
      </article>

      <article class="card">
        <div class="section-heading">
          <div>
            <h2>Data & backup</h2>
            <p class="muted">Your training log stays on this device unless you export a backup file.</p>
            <small class="field-note">Export a backup before switching phones, clearing browser data, or trying risky changes.</small>
          </div>
        </div>
        <div class="grid grid--2">
          <div class="metric-card">
            <span class="metric-card__label">Last export</span>
            <strong>${formatDateTime(state.backups?.lastExportedAt)}</strong>
            <small>JSON backup you can keep anywhere you trust.</small>
          </div>
          <div class="metric-card">
            <span class="metric-card__label">Last import</span>
            <strong>${formatDateTime(state.backups?.lastImportedAt)}</strong>
            <small>Imports replace the current data on this device.</small>
          </div>
        </div>
        <div class="stack-list compact-list">
          <div class="list-row review-callout">
            <div>
              <strong>Recommended backup rhythm</strong>
              <div class="muted">Export after setting up a new cycle, after finishing a cycle, or before changing training maxes in bulk.</div>
            </div>
            <div class="list-row__meta"><span class="badge badge--muted">Local-first</span></div>
          </div>
        </div>
        <div class="stack-row">
          <button class="button" data-action="export-data">Export backup</button>
          <button class="button button--secondary" data-action="import-data">Import backup</button>
          <button class="button button--danger" data-action="reset-app">Reset app</button>
        </div>
      </article>
    </section>
  `;
}

function renderShell() {
  const activeTab = state.ui.tab;
  const views = {
    home: renderHome(),
    workout: renderWorkout(),
    history: renderHistory(),
    program: renderProgram(),
    settings: renderSettings(),
  };

  return `
    <main class="shell">
      <header class="app-header card">
        <div class="app-header__top">
          <div>
            <p class="eyebrow">Training companion</p>
            <h1>${APP_TITLE}</h1>
            <p class="muted">${TEMPLATE_PRESETS[state.user.templateSlug].name} · Cycle ${state.activeCycle.cycleNumber} · ${state.user.daysPerWeek} days/week · ${state.user.units}</p>
          </div>
        </div>
        <nav class="tabs" aria-label="Primary">
          ${TAB_ITEMS.map((tab) => `
            <button class="tab ${activeTab === tab.key ? 'tab--active' : ''}" data-tab="${tab.key}" aria-label="${tab.label}">
              <span class="tab__icon" aria-hidden="true">${tab.icon}</span>
              <span class="tab__label">${tab.label}</span>
            </button>
          `).join('')}
        </nav>
      </header>
      ${views[activeTab]}
      <input id="importDataInput" type="file" accept="application/json" hidden />
    </main>
  `;
}

function syncWorkoutSetPlatePreview(workoutId, setId, totalWeight) {
  const platePreview = app.querySelector(
    `[data-plate-preview-workout-id="${workoutId}"][data-plate-preview-set-id="${setId}"]`,
  );
  if (!platePreview) return;
  platePreview.innerHTML = renderPlateBreakdown(totalWeight);
}

function render() {
  app.innerHTML = state.user ? renderShell() : renderOnboarding();
  syncRestTimerDisplay();
}

window.setInterval(() => {
  syncRestTimerDisplay();
}, 1000);

document.addEventListener('click', (event) => {
  const target = event.target.closest('[data-action], [data-tab]');
  if (!target) return;

  if (target.dataset.tab) {
    updateState((draftState) => {
      draftState.ui.tab = target.dataset.tab;
      if (target.dataset.tab === 'workout') {
        draftState.ui.selectedWorkoutId = getSelectedWorkout()?.id || getNextWorkout(draftState.activeCycle)?.id || null;
      }
    });
    return;
  }

  const { action } = target.dataset;
  if (action === 'start-program') {
    createProgramFromDraft();
  } else if (action === 'open-next-workout') {
    updateState((draftState) => {
      draftState.ui.selectedWorkoutId = getNextWorkout(draftState.activeCycle)?.id || null;
      draftState.ui.tab = 'workout';
    });
  } else if (action === 'open-workout') {
    updateState((draftState) => {
      draftState.ui.selectedWorkoutId = target.dataset.workoutId;
      draftState.ui.tab = 'workout';
    });
  } else if (action === 'save-workout') {
    saveWorkout(target.dataset.workoutId);
  } else if (action === 'toggle-set-complete') {
    const workout = getWorkoutById(state.activeCycle, target.dataset.workoutId);
    const context = workout ? getSetContext(workout, target.dataset.setId) : null;
    const shouldStartRest = Boolean(context && !context.set.completed);
    const restLabel = context ? `${getSetLocationLabel(workout, target.dataset.setId)} rest` : 'Rest timer';
    const restPhase = context?.set?.phase || 'main';
    toggleWorkoutSetCompletion(target.dataset.workoutId, target.dataset.setId);
    if (shouldStartRest) {
      startRestTimer(getRestSecondsForPhase(restPhase), restLabel, restPhase);
    }
  } else if (action === 'adjust-assistance-reps') {
    adjustWorkoutAssistanceLog(target.dataset.workoutId, target.dataset.category, Number(target.dataset.delta || 0));
  } else if (action === 'rest-timer-adjust') {
    adjustRestTimer(Number(target.dataset.seconds || 0));
  } else if (action === 'rest-timer-restart') {
    restartRestTimer();
  } else if (action === 'rest-timer-clear') {
    clearRestTimer();
  } else if (action === 'adjust-set-value') {
    adjustWorkoutSetValue(
      target.dataset.workoutId,
      target.dataset.setId,
      target.dataset.field,
      Number(target.dataset.delta || 0),
    );
  } else if (action === 'apply-next-cycle') {
    applyNextCycle();
  } else if (action === 'apply-program-settings') {
    applyProgramSettings();
  } else if (action === 'reset-app') {
    resetApp();
  } else if (action === 'export-data') {
    exportState();
  } else if (action === 'import-data') {
    document.getElementById('importDataInput')?.click();
  }
});

document.addEventListener('input', (event) => {
  const target = event.target;

  if (target.dataset.action === 'set-field' && target.dataset.field === 'actualWeight') {
    updateWorkoutSet(target.dataset.workoutId, target.dataset.setId, target.dataset.field, target.value, { renderApp: false });
    syncWorkoutSetPlatePreview(target.dataset.workoutId, target.dataset.setId, target.value);
  }
});

document.addEventListener('change', (event) => {
  const target = event.target;

  if (target.dataset.onboardingField) {
    const value = target.type === 'checkbox'
      ? target.checked
      : target.type === 'number'
        ? Number(target.value)
        : target.value;
    updateOnboardingField(target.dataset.onboardingField, value);
    return;
  }

  if (target.dataset.liftSource) {
    const value = target.type === 'number'
      ? (target.value === '' ? '' : Number(target.value))
      : target.value;
    updateOnboardingLift(target.dataset.liftSource, target.dataset.field, value);
    return;
  }

  if (target.dataset.action === 'set-field') {
    updateWorkoutSet(target.dataset.workoutId, target.dataset.setId, target.dataset.field, target.value);
    return;
  }

  if (target.dataset.action === 'workout-meta') {
    updateWorkoutMeta(target.dataset.workoutId, target.dataset.field, target.value);
    return;
  }

  if (target.dataset.action === 'review-selection') {
    updateReviewSelection(target.dataset.liftKey, target.value);
    return;
  }

  if (target.id === 'importDataInput') {
    importStateFile(target.files?.[0] || null);
    target.value = '';
  }
});

render();
