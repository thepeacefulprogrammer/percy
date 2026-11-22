# 5/3/1 Companion Product Requirements

## Product summary

**5/3/1 Companion** is a simple training app for lifters following Jim Wendler-style 5/3/1 programs.

It helps users:

- choose a sensible starting training max
- generate each workout automatically
- log plus sets, supplemental work, and assistance work
- progress each lift correctly across cycles
- avoid common mistakes like starting too heavy or adding too much extra work

---

## 1. Problem

Many 5/3/1 lifters currently use:

- paper notebooks
- ad hoc spreadsheets
- generic workout apps
- mental math in the gym

These approaches usually break down in one or more ways:

- incorrect training max calculations
- confusion between 1RM and training max
- poor cycle-to-cycle progression tracking
- no structured way to review whether a lift should increase, hold, or reset
- no support for assistance categories and template-specific logic

The result is friction, errors, and avoidable stalls.

---

## 2. Target user

### Primary user
A beginner-to-intermediate lifter who:

- trains 3 to 4 days per week
- wants long-term strength progress
- prefers simple, repeatable structure
- is willing to log top-set reps and basic workout completion
- does not want to build or maintain a spreadsheet

### Secondary user
A more experienced 5/3/1 lifter who:

- runs variants like FSL or BBB
- wants quick calculations and trend tracking
- understands the methodology but wants less bookkeeping

---

## 3. Product principles

1. **Conservative by default**  
   The app should encourage sustainable progression, not ego lifting.

2. **Simple during the workout**  
   The workout screen should answer: what do I lift now, for how many reps, and what comes next?

3. **Flexible between workouts**  
   Templates, assistance choices, and schedule structure should be configurable.

4. **Guardrails over gimmicks**  
   The app's value is correctness and consistency, not novelty.

5. **Template aware**  
   The app should understand the difference between vanilla, FSL, BBB, and beginner structures.

---

## 4. MVP goals

### Goal 1: Fast onboarding
User can set up the app in under 10 minutes.

### Goal 2: Accurate daily prescription
User sees exact weights and sets for today's session without manual math.

### Goal 3: Useful logging
User can record the key things that matter for 5/3/1 progression.

### Goal 4: Cycle management
At the end of a cycle, the app recommends next steps per lift.

### Goal 5: Low-friction adherence
The app makes it easier to stay on the program than to improvise away from it.

---

## 5. Core MVP features

### 5.1 Onboarding

### Inputs
- unit system: lb or kg
- gym rounding increment: e.g. 5 lb, 2.5 kg, 1 kg
- experience level: beginner / intermediate / advanced
- days per week: 3 or 4
- chosen template:
  - vanilla 5/3/1
  - beginner 5/3/1
  - FSL 5x5
  - BBB 5x10
- per-lift source number:
  - true 1RM
  - estimated 1RM from reps
  - current training max

### Onboarding calculations
- estimate 1RM when needed
- compute suggested TM using conservative defaults
- preview first cycle weights for all 4 lifts
- allow the user to adjust TM downward before confirming

### Guardrails
- explain that working percentages use **TM, not 1RM**
- warn when user enters an unusually aggressive TM
- suggest lighter BBB starting percentages for new users

---

### 5.2 Home / dashboard

The home screen should show:

- current cycle and week
- today's scheduled lift(s)
- next workout date
- current TM for each main lift
- recent PR / rep record highlights
- cycle adherence summary

Optional widgets:

- bodyweight trend
- conditioning streak
- recovery note trend

---

### 5.3 Workout screen

This is the most important screen.

### Must show
- primary lift for the day
- current week type: 5s / 3s / 5-3-1
- exact prescribed sets, reps, and working weights
- indicator for top-set plus work or fixed-rep work
- supplemental block (FSL / BBB / none)
- assistance categories to complete
- notes field

### Logging actions
- mark set complete
- adjust actual reps performed on top set
- log actual load if different from prescribed
- mark supplemental sets complete
- log assistance rep totals by category
- record quick subjective flags:
  - easy / normal / hard
  - fast / okay / grindy
  - pain / no pain

### Nice-to-have behavior
- plate breakdown for each set
- auto rest timer
- supersets for assistance logging
- offline-first behavior in the gym

---

### 5.4 Cycle progression workflow

At the end of each cycle, the app should run a review per lift.

### Inputs into review
- top-set reps achieved
- missed reps or skipped work
- subjective rep quality
- pain flags
- consistency across the cycle

### Recommended outcomes
- **Increase**
- **Hold**
- **Reset**

### Default recommendation logic
- increase if prescribed work was completed and top-set quality stayed acceptable
- hold if work was technically completed but quality deteriorated or recovery was poor
- reset if reps were missed, quality broke down badly, or TM appears too high

The app should never make progression feel mysterious.

---

### 5.5 Template support

### Vanilla 5/3/1
- one main lift per day
- main wave only
- optional assistance categories

### Beginner 5/3/1
- two main lifts on training days
- FSL support
- assistance categories

### FSL 5x5
- same main work as vanilla
- supplemental 5x5 at first-set percentage

### BBB 5x10
- same main work as vanilla
- supplemental 5x10 at selected BBB percentage
- recovery warning copy for lower-body days

---

### 5.6 History and analytics

### MVP analytics
- rep PRs on top sets
- workout completion streak
- cycle completion rate
- TM history by lift
- volume summary by week

### Non-MVP later analytics
- e1RM trend line by lift
- fatigue vs performance correlation
- assistance category balance
- recovery pattern insights

---

## 6. Recommended user flows

### Flow A: New user setup
1. Choose units and schedule
2. Select template
3. Enter best known lift numbers
4. Review suggested training maxes
5. Preview first week
6. Start program

### Flow B: Today's workout
1. Open app
2. See today's session
3. Complete main work set by set
4. Log plus-set reps
5. Complete supplemental block
6. Log assistance totals
7. Save workout and recovery note

### Flow C: End-of-cycle review
1. App detects cycle completion
2. User reviews each lift summary
3. App recommends increase / hold / reset
4. User confirms
5. Next cycle is generated automatically

---

## 7. Information architecture

Primary navigation:

- **Home**
- **Workout**
- **History**
- **Program**
- **Settings**

### Program section
Should include:

- current template
- current week and cycle
- training maxes
- schedule
- assistance defaults
- progression settings

---

## 8. Decision rules the product should encode

### Training max handling
- keep TM as a first-class entity
- show its source and last-updated date
- allow manual override with confirmation

### Rounding
- every prescribed weight should be rounded based on gym equipment settings
- raw and rounded values should both be stored internally when useful

### Assistance tracking
- allow category completion without requiring identical exercises every week
- user can swap exercises while preserving push/pull/single-leg-core intent

### Advanced options hidden by default
Examples:
- joker-style add-on sets
- advanced TM review settings
- alternate supplemental pairings

These should be opt-in so beginners are not overloaded.

---

## 9. MVP non-functional requirements

- works offline for active workouts
- starts fast in the gym
- touch targets easy to use with chalky hands
- minimal typing during sessions
- persistent local storage
- exportable workout history

---

## 10. Future roadmap

### Phase 2
- conditioning planner
- warm-up generator
- shared coach view
- wearable / health integrations
- Apple Health / Google Fit bodyweight sync

### Phase 3
- community template library
- intelligent TM suggestions from trend data
- advanced leader / anchor planning
- coaching notes and review workflows

---

## 11. Open product questions

1. Should onboarding default to 85% TM for true beginners and 90% for others?
2. Should plus sets be optional per template or globally configurable?
3. How much freedom should users have to edit workouts mid-cycle?
4. Should the app support kg-first rounding quirks for gyms with limited plates?
5. Should the first release be mobile-first only, or mobile plus web?

---

## 12. Strong MVP recommendation

If building this as software, the first shipped version should optimize for **clarity and trust**:

- obvious calculations
- clean workout execution
- cycle progression review
- no spreadsheet dependency

That is enough to make the product genuinely useful to real 5/3/1 users.
