# UrbanZero — Final Video Script (≤ 3:00)

**Rubric target** (10 pts total):
- Introduction + contributions ........ 2 pts
- ≤ 3:00 time limit ................... 2 pts
- **Lessons learned / journey** ....... **3 pts**
- Polish, spelling, grammar ........... 1 pt
- Conclusion summarizes work .......... 2 pts

**Spoken target:** ~400–440 words at conversational pace (~150 wpm).
Hard cap 3:00 — TAs may stop grading at the limit.

**Record setup:** OBS or QuickTime, screen-capture the CARLA eval
rollout + the four paper figures + the GitHub repo. Voiceover on top.
Town01 clip of the best-episode (ep 16, 32.9 % RC) makes the strongest
B-roll if available; otherwise use figure PNGs full-screen.

---

## 0:00 – 0:30  Hook + problem + contributions  *(2 pts)*

**[On screen:** UrbanZero title card → CARLA semantic-seg camera clip
from eval, ego following a route. **]**

> "I'm Aaditya Mishra. For my CSC 591 Software for Robotics project,
> UrbanZero, I tried to train a reinforcement-learning agent to drive
> urban routes in the CARLA simulator — from a single camera,
> completely from scratch, no expert demonstrations.
>
> Over about seventy-two hours I ran six training runs. None of them
> learned to drive. What I want to show you in the next three minutes
> is: *why* they failed, how I eventually found the actual root cause,
> and what I shipped instead — because I think the failure is more
> interesting than a success would have been."

---

## 0:30 – 1:30  Six failure modes + root cause  *(3 pts — lessons-learned)*

**[On screen:** cut to Figure 1 (failure-timeline bar chart), hold
while narrating. Then Figure 2 (beacon endpoints). Then a GitHub
Issues view showing issues #7–#13. **]**

> "The design was textbook CARLA-PPO: CaRL-style minimal reward,
> Ng-Harada-Russell potential shaping, Andrychowicz entropy band. I
> pre-registered a falsification criterion for each run before
> launching it.
>
> Every run falsified its own hypothesis. Run one found the sit-still
> attractor — the agent learned that sitting was cheaper than
> crashing. Run two pinned policy-sigma at the clamp, effectively
> emitting pure noise. Run three hit a sparse-steering-gradient
> plateau. I then pivoted to Behavior Cloning warmstart. Runs four
> and five saw PPO shred the BC prior with approx-KL around 0.08 —
> five times the healthy target. Run six, by contrast, looked
> textbook-healthy on every PPO metric — KL at 0.0015, clip fraction
> 0.08, explained variance 0.85 — yet the actor was frozen.
>
> That's when I stopped trusting PPO's diagnostics. I wrote a
> hundred-and-fifty-line standalone script that spawned the same
> traffic my training used and just measured how many NPCs actually
> moved. In ten minutes it told me seventy percent of them were
> frozen, every run, every BC collection. The culprit was a single
> TrafficManager call — `set_hybrid_physics_mode` — that I had added
> weeks earlier as a *defensive* fix. It was the bug itself."

---

## 1:30 – 2:30  Final result + comparison to pure-RL  *(2 pts conclusion)*

**[On screen:** Figure 3 (BC-eval distribution, N=20). Overlay the
aggregate numbers: 7.42 % mean, 32.93 % max. Then the eval JSON in
`eval/bc_final_20260424_0059.json`. **]**

> "After removing the bug I froze the Behavior Cloning policy and
> evaluated it deterministically on twenty episodes.
>
> The results: 7.42 percent mean route completion, 32.93 percent on
> the best individual episode, average speed 5.4 meters per second.
> That beats the pure-RL plateau by more than two percentage points —
> BC genuinely learned *some* of BehaviorAgent's driving behavior.
> Zero episodes completed a full route. Seventy-five percent ended
> in collision.
>
> That seventy-five percent collision number is the interesting part.
> It's the Codevilla-2019 distribution-shift signature. My BC dataset
> was collected *with* the hybrid-physics bug active — so the policy
> learned to drive through a world of frozen cars. At evaluation time,
> with moving traffic restored, it drives at expert speed directly
> into vehicles it has no internal model of. It's not a policy
> problem; it's a dataset problem, invisible until deployment."

---

## 2:30 – 3:00  Lessons learned + takeaway  *(journey + polish)*

**[On screen:** Figure 4 (reward economics) briefly, then the
`PROJECT_NOTES.md` file scrolling — every failure with its citation.
Close on the GitHub repo URL card. **]**

> "Four lessons I'm taking out of this: change one axis at a time.
> PPO metrics don't describe the environment. Pre-register your
> falsifiers. And when the training loop stops telling you the
> truth, leave the loop and ask the environment a direct question.
>
> The final artifact — the frozen BC policy, six documented failure
> modes with citations, and the standalone diagnostic — is public at
> github.com/AadityaMishra1/UrbanZero. Thanks for watching."

---

## Timing check

Approximate word counts (conversational pace ~150 wpm):

| Segment        | Words | Target minutes |
|----------------|-------|----------------|
| 0:00 – 0:30    | ~70   | 0:28           |
| 0:30 – 1:30    | ~170  | 1:08           |
| 1:30 – 2:30    | ~135  | 0:54           |
| 2:30 – 3:00    | ~75   | 0:30           |
| **Total**      | ~450  | **~3:00**      |

**Practice twice with a stopwatch before recording.** If any segment
runs over, cut on the run-by-run narration in §0:30–1:30 first (drop
to three named runs: v1 sit-still, v2 std-pin, v6 actor-freeze) —
that preserves the root-cause discovery which is the highest-scoring
section (3 pts).

## Recording checklist

1. Close anything that might make noise (Slack, email, Discord).
2. OBS or QuickTime → new screen recording, 1920×1080.
3. Window layout: CARLA eval clip top-left, paper figure lower-right,
   GitHub repo as final transition. Test switch speed.
4. Record a 10-second test, play back, check audio level and
   plosives.
5. Do one full take, one safety take. Review both.
6. Export MP4 at ≤ 500 MB. Upload via the video submission Google
   Form linked in `docs/COURSE_REQUIREMENTS.md`.

## Post-production

- Title card and end card: one slide each, 2 seconds max.
- Add captions burned-in (improves "polish, spelling, grammar"
  rubric item). Auto-transcribe via macOS Live Captions or Whisper
  if time permits.
- Verify total runtime ≤ 2:55 to give 5-second buffer.

## Fallback — if recording runs over 3:00

Cut this sentence first (redundant with paper):

> "I pre-registered a falsification criterion for each run before
> launching it."

Then this one:

> "Zero episodes completed a full route."

Then this clause:

> "— KL at 0.0015, clip fraction 0.08, explained variance 0.85 —"
