That sounds like a very strong structure honestly. The sequencing is good scientifically *and* operationally.

You are basically organizing the benchmark so that each campaign answers a progressively harder question:

---

# Campaign 1 — Simple, Cheap, Reproducible Baselines

Problems:

* OUP
* GLM
* GLU
* Ricker

Missingness:

* MCAR
* MAR
* MNAR

Fractions:

* 10%
* 25%
* 50%

Methods:

1. Full-data NPE
2. Zero-imputation NPE
3. Mean-imputation NPE
4. Mean-imputation + mask augmentation

This is excellent because:

* everything is easy to reproduce,
* mostly pure `sbi`,
* very low engineering burden,
* gives immediate benchmark value,
* establishes calibration baselines.

This alone is already publishable as a benchmark/data paper if done cleanly.

Especially because:

* almost nobody evaluates SBI missing-data methods systematically with SBC/TARP,
* almost nobody compares MCAR/MAR/MNAR carefully,
* many papers only show posterior samples visually.

So even Campaign 1 has real value.

---

# Campaign 2 — Representation Learning Methods

Methods:
5. Learned-imputation + mask (Lueckmann-style)
6. RISE
7. Transformer embedding + NPE

Same 4 core problems.

This is the correct place for them because:

* now you already have calibration baselines,
* now the question becomes:

  “Do learned representations actually help?”

instead of:

“Can we even benchmark missing-data SBI?”

Very important distinction.

Also:

* this isolates the computational cost discussion,
* lets you quantify calibration-vs-compute tradeoffs,
* avoids exploding complexity early.

And honestly, the transformer method may become one of the strongest results in the paper if:

* it is simpler than RISE,
* cheaper than RISE,
* and calibration is competitive.

That is very plausible.

---

# Campaign 3 — Hard / Structured Scientific Problems

Problems:

* Spatial SIR
* Hodgkin–Huxley

Methods:
all previous methods.

This is where the benchmark becomes much more convincing scientifically.

Because then you can show:

* low-dimensional smooth problems,
* nonlinear chaotic-ish problems,
* neuroscience-style problems,
* spatially structured problems.

At this stage, reviewers stop seeing it as:

> “toy benchmark paper”

and start seeing:

> “this is becoming a standard SBI missing-data benchmark suite.”

Very important transition.

---

# Campaign 4 — Simformer

This is smart because Simformer is:

* expensive,
* operationally heavier,
* likely more fragile,
* harder to tune fairly.

So delaying it is the correct decision.

Also:
Simformer becomes MUCH more meaningful once:

* you already have calibrated baselines,
* compute measurements,
* standardized datasets,
* established diagnostics.

Then you can ask:

> “Does a large generative foundation-style SBI model actually outperform simpler specialized methods under missingness?”

That is a strong paper question.

---

The really important thing is that your campaigns are now:

* modular,
* publishable independently,
* incrementally extensible,
* easy to parallelize later with collaborators.

That is exactly how benchmark projects survive long-term instead of collapsing under scope.

Also, strategically:
Campaign 1 alone is feasible locally on your MacBook.
Campaign 2 maybe partly local + cluster.
Campaigns 3–4 are where collaborators/compute become useful.

That is a very healthy scaling path.
