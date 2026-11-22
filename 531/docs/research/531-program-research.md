# 5/3/1 Program Research

## Goal of this research

Summarize the publicly available parts of the 5/3/1 system so an app can help a lifter run the method well without overcomplicating it.

---

## 1. Core program concepts

### Main lifts
Most 5/3/1 variants revolve around four barbell lifts:

- Overhead press
- Bench press
- Squat
- Deadlift

### Training max (TM)
The program is built from a **training max**, not from the user's true 1RM on every set.

Common public guidance:

- Use **85% to 90% of a true or estimated 1RM** as the initial training max
- If unsure, start lighter
- Progress comes from slowly increasing the TM across cycles

Why this matters for the app:

- TM setup is the most important onboarding step
- The app should default to a conservative TM and explain why
- The app should treat TM as a separate value from e1RM or true 1RM

### Classic main-set wave
Publicly available sources consistently describe the classic 3-week wave as percentages of the TM:

- **Week 1:** 65% x 5, 75% x 5, 85% x 5+
- **Week 2:** 70% x 3, 80% x 3, 90% x 3+
- **Week 3:** 75% x 5, 85% x 3, 95% x 1+

The final set is usually a **plus / AMRAP set**, but not to technical failure.

### Progression between cycles
Public summaries consistently note the standard monthly progression:

- **Upper body lifts:** +5 lb per cycle
- **Lower body lifts:** +10 lb per cycle

Why this matters for the app:

- Progression should be automated
- Users should not have to recalculate every cycle manually
- The app should still allow a hold or reset when performance quality is poor

---

## 2. Assistance and supplemental work

A useful app needs to distinguish between three layers:

1. **Main work** — the core 5/3/1 percentages
2. **Supplemental work** — extra work based on the main lift or related lift
3. **Assistance work** — flexible push / pull / single-leg-core work

### First Set Last (FSL)
A common public supplemental option:

- Uses the first work-set percentage again
- Commonly programmed as **5 x 5**
- Generally easier to recover from than higher-volume options

Publicly described FSL percentages:

- Week 1: 65%
- Week 2: 70%
- Week 3: 75%

Implication for the app:

- FSL is a strong default for beginners and general strength users
- The app should compute it automatically from the same TM

### Boring But Big (BBB)
A popular high-volume supplemental template:

- Typically **5 x 10** after the main work
- Commonly around **50% to 60% of TM**, with guidance to start even lighter when new
- Lower-body BBB volume can become very fatiguing quickly

Public guidance also recommends ramping BBB conservatively, for example:

- Cycle 1: 5 x 10 @ 30% TM
- Cycle 2: 5 x 10 @ 45% TM
- Cycle 3: 5 x 10 @ 60% TM

Implication for the app:

- BBB should be selectable as a template
- The app should warn about recovery cost
- The app should support both fixed BBB percentages and progressive ramp-in

### Assistance work structure
Public sources commonly describe assistance as one movement from each category:

- **Push**
- **Pull**
- **Single leg / core**

Total reps are usually flexible, often in the **50-100 total reps** range per category.

Implication for the app:

- The user should log assistance by category, not just by exercise name
- The app should allow rep-bank completion instead of forcing strict set schemes

---

## 3. Recovery, deloads, and TM management

### Rep quality matters
Wendler's public beginner guidance emphasizes:

- no grinding reps
- no sloppy reps
- do not increase TM if rep quality is poor

Implication for the app:

- logging should include rep quality, bar speed feel, or an "earned progression" check
- progression should not be purely automatic without user confirmation

### Deloads and test weeks
Public summaries note that older 5/3/1 content often includes a fixed 4th-week deload, while newer guidance is more nuanced.

Publicly discussed deload / TM-test ideas include:

- lighter deload work such as 40/50/60% x 5
- periodic TM test weeks
- evaluating whether the TM is still appropriate after a few cycles

Implication for the app:

- the app should support both older and newer scheduling styles
- deload logic should be configurable at the template level
- TM review should be a first-class workflow, not an afterthought

---

## 4. Common user mistakes an app should prevent

The app should actively prevent the mistakes most often mentioned in public explanations of 5/3/1.

### Mistake 1: Starting too heavy
Symptoms:

- user sets TM equal to true max
- AMRAP sets turn into grinders immediately
- progress stalls by cycle 2 or 3

App response:

- conservative defaults
- onboarding warnings
- optional e1RM calculator from submax reps
- explicit explanation that percentages use TM, not true max

### Mistake 2: Confusing TM with 1RM
Symptoms:

- percentages are calculated off the user's best-ever max
- workout weights feel about 10% too heavy

App response:

- show both numbers separately: true/e1RM and TM
- label every working set as "% of TM"

### Mistake 3: Overdoing assistance
Symptoms:

- turning the workout into a second bodybuilding session
- poor recovery, especially with BBB

App response:

- assistance cap suggestions
- recovery warning banners when supplemental and assistance volume are both high
- template-specific defaults

### Mistake 4: Customizing too early
Symptoms:

- mixing multiple templates at once
- adding joker-style heavy work without a clear plan

App response:

- guided template selection
- discourage stacking multiple supplemental schemes on one day
- advanced features hidden unless enabled

### Mistake 5: Progressing when performance quality is poor
Symptoms:

- automatically adding weight after every cycle regardless of bar speed or missed reps

App response:

- post-cycle review asking:
  - did all prescribed work complete?
  - did top sets stay technically sound?
  - any failed reps?
- recommend **increase / hold / reset** per lift

---

## 5. What an effective 5/3/1 app must do

### The app should be excellent at:

1. **Onboarding conservatively**
   - estimate 1RM
   - calculate TM
   - choose a template
   - set rounding rules and units

2. **Showing today's workout simply**
   - warm-up sets
   - main work
   - supplemental work
   - assistance categories
   - conditioning reminder

3. **Logging the important things**
   - actual reps on plus sets
   - completed supplemental sets
   - assistance rep totals
   - subjective readiness / soreness / bar speed
   - notes and pain flags

4. **Managing progression**
   - auto-build the next cycle
   - allow hold/reset decisions
   - detect trend problems early

5. **Reducing cognitive load**
   - lifter should not need a spreadsheet during training
   - app should minimize manual calculations

---

## 6. Product implications distilled

### Best initial target user
A beginner-to-intermediate lifter who:

- trains 3-4 days per week
- wants straightforward strength progression
- is currently using notes, spreadsheets, or memory
- wants guardrails against doing too much

### Best initial templates for MVP
For a first product version, support:

- Vanilla 5/3/1
- Beginner 5/3/1
- FSL 5x5
- BBB 5x10

Avoid trying to ship every obscure variant on day one.

### Key product insight
A 5/3/1 app is not mainly a workout timer. It is a **progression governor**.

Its biggest value is helping the user:

- start lighter than ego wants
- hit the right weights today
- keep progression sustainable
- know when to increase, hold, or reset

---

## 7. Source notes

Primary and secondary sources reviewed:

1. Jim Wendler, **The 5/3/1 Philosophy for Beginners**  
   https://www.jimwendler.com/blogs/jimwendler-com/101065094-5-3-1-for-a-beginner

2. Jim Wendler, **Boring But Big**  
   https://www.jimwendler.com/blogs/jimwendler-com/101077382-boring-but-big

3. The Fitness Wiki, **5/3/1 Primer**  
   https://thefitness.wiki/5-3-1-primer/

4. T Nation / Jim Wendler, **5/3/1: How to Build Pure Strength**  
   https://t-nation.com/t/5-3-1-how-to-build-pure-strength/281694

## 8. Boundary for app design

This research is intentionally based on publicly accessible explanations and summaries. The product should support user-entered templates and public program structures, and should avoid pretending to reproduce proprietary book-only programming details that are not freely published.
