# 5/3/1 Companion Data Model

This document proposes a practical domain model for a 5/3/1 tracking app.

---

## 1. Design goals

The data model should:

- separate **true/e1RM** from **training max**
- support multiple templates without rewriting the workout engine
- preserve both prescribed and actual performance
- allow per-lift progression decisions
- support offline-first local storage

---

## 2. Core entities

### User
Represents the athlete using the app.

Suggested fields:

- `id`
- `displayName`
- `units` (`lb | kg`)
- `roundingIncrement`
- `experienceLevel`
- `createdAt`
- `updatedAt`

### Lift
Represents one of the core lifts.

Suggested values:

- `press`
- `bench`
- `squat`
- `deadlift`

Suggested fields:

- `id`
- `userId`
- `name`
- `category` (`upper | lower`)
- `isActive`

### LiftProfile
Stores long-lived settings and current numbers for a lift.

Suggested fields:

- `id`
- `userId`
- `liftId`
- `currentTrainingMax`
- `trainingMaxSource` (`manual | from_true_1rm | from_estimated_1rm | imported`)
- `sourceMaxValue`
- `sourceMaxType` (`true_1rm | estimated_1rm | training_max`)
- `lastEstimatedOneRepMax`
- `lastRepRecordWeight`
- `lastRepRecordReps`
- `incrementPerCycle`
- `lastReviewedAt`
- `updatedAt`

### ProgramTemplate
Defines the structure of a training style.

Examples:

- Vanilla 5/3/1
- Beginner 5/3/1
- FSL 5x5
- BBB 5x10

Suggested fields:

- `id`
- `name`
- `slug`
- `daysPerWeek`
- `mainSetStyle` (`classic | fives_pro | custom`)
- `supplementalStyle` (`none | fsl | bbb | custom`)
- `assistanceStyle` (`none | push_pull_single_leg_core | custom`)
- `deloadStrategy` (`classic_week_4 | configurable | tm_test`)
- `isSystemTemplate`

### ProgramInstance
Represents one user's active plan.

Suggested fields:

- `id`
- `userId`
- `templateId`
- `name`
- `startDate`
- `status` (`active | paused | completed | archived`)
- `daysPerWeek`
- `notes`
- `createdAt`
- `updatedAt`

### Cycle
Represents a single cycle within a program instance.

Suggested fields:

- `id`
- `programInstanceId`
- `cycleNumber`
- `startDate`
- `endDate`
- `status` (`planned | active | completed`)
- `progressionMode` (`standard | hold_some_lifts | reset_some_lifts`)

### Week
Represents a week inside a cycle.

Suggested fields:

- `id`
- `cycleId`
- `weekNumber`
- `weekType` (`5s | 3s | 531 | deload | tm_test`)
- `startDate`
- `endDate`
- `status` (`planned | active | completed | skipped`)

### Workout
Represents a scheduled training session.

Suggested fields:

- `id`
- `weekId`
- `programInstanceId`
- `scheduledDate`
- `completedDate`
- `dayIndex`
- `title`
- `status` (`planned | in_progress | completed | skipped`)
- `templateSnapshot`
- `notes`

---

## 3. Workout composition entities

### WorkoutLift
A main-lift assignment inside a workout.

Use this so a workout can support one or two main lifts, which is useful for Beginner 5/3/1.

Suggested fields:

- `id`
- `workoutId`
- `liftId`
- `displayOrder`
- `role` (`main | secondary_main | supplemental_base`)
- `trainingMaxAtTime`
- `mainSetStyle`
- `supplementalStyle`

### PrescribedSet
A generated set the user is expected to perform.

Suggested fields:

- `id`
- `workoutLiftId`
- `phase` (`warmup | main | supplemental | assistance`)
- `setIndex`
- `percentageOfTrainingMax`
- `prescribedWeightRaw`
- `prescribedWeightRounded`
- `prescribedReps`
- `repStyle` (`fixed | plus | amrap | range`)
- `groupLabel` (`main_531 | fsl | bbb | warmup`)

### CompletedSet
What the user actually did.

Suggested fields:

- `id`
- `prescribedSetId`
- `actualWeight`
- `actualReps`
- `completed`
- `perceivedDifficulty` (`easy | normal | hard`)
- `barSpeed` (`fast | okay | grindy`)
- `painFlag`
- `notes`
- `completedAt`

### AssistanceCategoryTarget
Stores planned assistance categories for the workout.

Suggested fields:

- `id`
- `workoutId`
- `category` (`push | pull | single_leg_core`)
- `targetRepsMin`
- `targetRepsMax`
- `displayOrder`

### AssistanceEntry
Stores actual assistance exercise performance.

Suggested fields:

- `id`
- `workoutId`
- `category`
- `exerciseName`
- `sets`
- `reps`
- `load`
- `totalReps`
- `notes`

---

## 4. Progression and review entities

### LiftCycleReview
Stores a lift-specific review at cycle end.

Suggested fields:

- `id`
- `cycleId`
- `liftId`
- `currentTrainingMax`
- `recommendedAction` (`increase | hold | reset`)
- `confirmedAction` (`increase | hold | reset`)
- `nextTrainingMax`
- `reasonSummary`
- `createdAt`
- `confirmedAt`

### PerformanceRecord
Stores notable outcomes for analytics.

Suggested fields:

- `id`
- `userId`
- `liftId`
- `workoutId`
- `recordType` (`rep_pr | estimated_1rm_pr | volume_pr`)
- `weight`
- `reps`
- `estimatedOneRepMax`
- `recordedAt`

### BodyMetric
Optional body composition / bodyweight tracking.

Suggested fields:

- `id`
- `userId`
- `metricType` (`bodyweight | waist | sleep | resting_hr`)
- `value`
- `unit`
- `recordedAt`

---

## 5. Suggested calculation model

The app should treat calculations as derived data, not only stored data.

### Inputs used by the calculation engine
- current TM for the lift
- template configuration
- week type
- rounding increment
- unit system

### Outputs produced by the engine
- prescribed main sets
- prescribed supplemental sets
- plate-ready rounded loads
- cycle-to-cycle next TM suggestions

This allows the app to regenerate workouts if a template or TM changes.

---

## 6. Suggested relationships

- one `User` has many `LiftProfile` records
- one `ProgramTemplate` can be used by many `ProgramInstance` records
- one `ProgramInstance` has many `Cycle` records
- one `Cycle` has many `Week` records
- one `Week` has many `Workout` records
- one `Workout` has many `WorkoutLift` records
- one `WorkoutLift` has many `PrescribedSet` records
- one `PrescribedSet` has zero or one `CompletedSet`
- one `Workout` has many `AssistanceEntry` records
- one `Cycle` has many `LiftCycleReview` records

---

## 7. Example JSON shape

```json
{
  "workout": {
    "id": "w_102",
    "title": "Week 2 Day 1 - Press",
    "status": "planned",
    "weekType": "3s",
    "lifts": [
      {
        "lift": "press",
        "trainingMaxAtTime": 135,
        "mainSets": [
          { "pct": 0.7, "weight": 95, "reps": 3, "style": "fixed" },
          { "pct": 0.8, "weight": 110, "reps": 3, "style": "fixed" },
          { "pct": 0.9, "weight": 120, "reps": 3, "style": "plus" }
        ],
        "supplemental": {
          "style": "fsl",
          "sets": 5,
          "reps": 5,
          "weight": 95
        }
      }
    ],
    "assistanceTargets": [
      { "category": "push", "min": 50, "max": 100 },
      { "category": "pull", "min": 50, "max": 100 },
      { "category": "single_leg_core", "min": 50, "max": 100 }
    ]
  }
}
```

---

## 8. Storage recommendation

For an MVP:

- local SQLite or equivalent persistent store
- sync layer optional later
- derived calculations generated in app logic

For a multi-device version later:

- keep calculation engine deterministic
- sync raw user inputs and completed performance
- allow server or client regeneration of prescribed workouts

---

## 9. Product-specific recommendation

Do not model 5/3/1 as a generic exercise log first.

Instead, model it as:

- **program -> cycle -> week -> workout -> lift block -> set**

That structure matches how 5/3/1 users think and will make the UI, analytics, and progression logic much easier to keep correct.
