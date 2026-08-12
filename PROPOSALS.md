# Matcha scorer: what went wrong, and what to try instead

## What happened

PR #1 replaced the (unreliable, RNG-feeling) 41-feature SVR headline with a
"transparent" 6-term perceptual scorer `perceptualScore()`. On real matcha
latte photos it was a severe regression — excellent matchas scored 1/5.

**Root cause.** I calibrated the new scorer against the in-app "Ideal 5/5"
swatch `RGB(100,165,60)` using **synthetic uniform-color patches**. But that
swatch is *matcha powder*, not *a matcha latte*. A great latte is matcha cut
with milk — a creamy, muted green that sits far from vivid ceremonial powder
in Lab space. My scorer punished distance from the powder-ideal (one-sided
ΔE term) and required powder-level chroma (`chroma ≈ 45–55`) and green
dominance (`G−R ≈ 65`) to score well. A real 5/5 latte photo lands around
`a* ≈ −15`, `chroma ≈ 22`, `G−R ≈ 25` — so every "wrong feature" threshold I
guessed from the powder swatch voted ~2–3, and the shaping curve compressed it
toward 1. The SVR, trained on *real latte photos*, already "knew" creamy =
good; the perceptual scorer didn't.

**Methodological mistake.** "Verified on synthetic patches" verified nothing
about real photos. The whole calibration was done in `/tmp` against 9 flat
colors, never against a real latte image. That gap hid the latte-vs-powder
distinction completely.

**The skew caveat (generalised).** Any change to `extractFeatures()` — even
"obvious" bug fixes like the circular hue average or the foam-biased white
balance — shifts the feature distribution the SVR was trained on. Shipping
those without retraining causes train/serve skew, the same failure mode that
just happened. **Anything that touches the feature pipeline must be
retrained and validated on real photos before it ships.** That's why this
revert is full, not partial.

---

## Alternatives to consider

Each is described with its real cost and its risk, given the lessons above.
None should ship without cross-validation on the 90-photo set (or more).

### A. Learned reference color (fixes the mistake that broke PR #1)
Don't hardcode the ideal. Compute the **mean Lab color of the user's own
5-rated photos** (or community 5-ratings already in Firebase), and use that
centroid as the ΔE target. Set each sub-score's thresholds from the rating
distribution (e.g. `a*` at p10 of 1★ vs p90 of 5★) rather than guessing them
from a powder swatch.
- **Pro:** keeps the transparency and the "right features" idea; rewards
  "looks like great *lattes*" not "looks like powder."
- **Con:** needs a labeled subset; quality bounded by that data.
- **Retrain needed?** No — `perceptualScore` is rule-based. But thresholds
  must be fit to data, not invented.

### B. Augment, don't replace, the SVR features (lowest-risk accuracy gain)
Keep the 41 legacy features; **append** the genuinely new ones — green
dominance `mean(G−R)`, Lab chroma, ΔE-to-learned-reference (per A), circular
hue distance — to ~45 features, then **retrain the SVR** and confirm via
5-fold CV that MAE drops before shipping.
- **Pro:** directly addresses the original "wrong features" complaint without
  throwing away what the SVR learned; the model decides how much the new
  features help.
- **Con:** Python retrain; still only ~90 photos (~2/feature), so prefer
  *few additive* features over many.
- **Retrain needed?** Yes — mandatory (new feature dimensions).

### C. Hybrid with an uncertainty flag (attacks the "RNG feel" with zero risk)
Keep the SVR as the **primary, unchanged** headline. Compute a robust
perceptual score in parallel as a *consistency check*. When the two disagree
by more than N (say 1.5) **or** segmentation is weak (`greenFraction < 0.25`),
show a "low confidence — lighting/framing may be off" badge and reveal both
numbers. No override, no retrain.
- **Pro:** pure UI, no model change, zero regression risk; makes the score's
  uncertainty **honest instead of silent** — which is the actual complaint
  ("it feels like an RNG").
- **Con:** doesn't make scores more accurate; only more transparent.
- **Retrain needed?** No.

### D. Segment the cup properly (the largest cross-framing variance source)
The green-pixel detector + `greenFraction` feature + circular fallback let
foam, cup rim, and latte-art bleed into "matcha" pixels, so the same drink
framed differently scores differently — a direct RNG source. Replace the
fixed 12% center crop with circle detection (Hough / largest green blob) and
the RGB thresholds with a Lab-hue-cone segmenter, **then retrain**.
- **Pro:** removes a real, large framing-driven variance source; the
  "right region" is a precondition for any feature being meaningful.
- **Con:** more engineering; retrain required; risk of new edge cases
  (cups with green rims, etc.).
- **Retrain needed?** Yes.

### E. Grow the labeled set (the fundamental constraint)
~90 photos for ~41 features is ~2 samples/feature — below the ~10/feature
rule of thumb for stable regression, so the SVR is high-variance by
construction. The community-rating Firebase path already exists; seeding it
and retraining on the current 41 features may improve stability more than any
feature rewrite.
- **Pro:** attacks the root cause the user originally identified (RNG);
  preserves the existing pipeline.
- **Con:** time + engagement to collect; marginal gains until ~hundreds of
  labeled photos exist.
- **Retrain needed?** Yes (on the larger set).

---

## Recommended path

1. **Now:** merge this revert. The SVR is the baseline again.
2. **Quick win (C):** ship the *uncertainty flag* as a tiny, zero-risk PR —
   it directly addresses "feels like an RNG" by surfacing disagreement
   instead of hiding it, with no model risk.
3. **Foundational (E):** push the community-rating flow to collect more
   labeled latte photos (not powder swatches).
4. **Then (B + A together):** once there's a usable labeled set, retrain the
   SVR augmented with a *learned-reference* ΔE feature and green dominance,
   and gate shipping on a CV MAE improvement.
5. **If CV still shows high framing variance (D):** improve segmentation and
   retrain again.

The standalone diagnostic tool from PR #1 (the in-page sanity harness that
scores synthetic patches) is still worth resurrecting as a *developer-only*
check — but it must never be the basis for a scoring decision on real
photos, which is exactly how it was misused. That's the lesson.
