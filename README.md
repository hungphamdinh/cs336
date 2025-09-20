Sinusoidal PE — Quick Summary

What problem it solves

Self-attention is order-agnostic. PE injects position into token representations so the model can reason about order and relative offsets.

How it’s constructed
	•	Assume even model dim D=2m. For position t and pair index i\in\{0,\dots,m-1\}:
\omega_i = 10000^{-2i/D},\qquad
\theta_{t,i} = t\,\omega_i.
	•	Fill channels:
\mathrm{PE}(t,2i)=\sin(\theta_{t,i}),\quad
\mathrm{PE}(t,2i+1)=\cos(\theta_{t,i}).
	•	Matrix view: with positions \mathbf p=[0,\dots,T-1]^\top and frequency row \boldsymbol\omega^\top, angles are \Theta=\mathbf p\,\boldsymbol\omega^\top and you apply \sin/\cos per column, interleaving the pairs.

Why sin & cos (not just one)
	•	Shifts become rotations (linear):
\big[\sin(\omega(t+\Delta)),\cos(\omega(t+\Delta))\big]^\top = R(\omega\Delta)\big[\sin(\omega t),\cos(\omega t)\big]^\top.
Relative distance \Delta is a linear transform—perfect for dot-products/linear layers.
	•	Constant energy: \sin^2+\cos^2=1 per pair ⇒ stable magnitude across positions.
	•	Distance-aware similarity:
\sin a\sin b+\cos a\cos b=\cos(a-b) ⇒ inner products depend only on t-s.
	•	Phase is unambiguous: using both sin & cos preserves phase (a single sine would lose sign/phase info).
	•	Smooth, bounded, Fourier features: good gradients; multi-frequency basis for translation-invariant patterns.

Why the div term (\omega_i schedule)
	•	\omega_i=10000^{-2i/D} is log-uniform across pairs → multi-scale coverage.
	•	Large \omega: fast oscillation ⇒ sensitive to local offsets (1–2 tokens).
Small \omega: slow oscillation ⇒ captures global/long-range structure.
	•	Mixed frequencies reduce aliasing (two different offsets rarely match across all bands).
	•	Spacing tied to D: doubling D inserts more bands between same extremes.

How attention uses PE

With z_t=x_t+\mathrm{PE}(t), attention logits contain:
	•	position–position terms ≈ sums of \cos(\omega_i (t-s)) (and \sin via skew combos) ⇒ learnable offset preferences;
	•	content–position cross terms ⇒ distance preferences that depend on content (“look two tokens back if it’s a noun”).

Practical notes
	•	Use even D (one sin/cos pair per 2 dims). If D is odd, drop the last cosine or trim the frequency vector.
	•	Common to scale embeddings by \sqrt{D} before adding PE to balance magnitudes.
	•	Deterministic (no learned table) ⇒ extrapolates to lengths beyond training.
	•	Values are in [-1,1]; PE changes direction, not scale.

Tiny numeric feel (e.g., D{=}8)
	•	\omega = [1,\,0.1,\,0.01,\,0.001].
	•	At t=5: angles [5,\,0.5,\,0.05,\,0.005]; PE interleaves \sin and \cos of these.

⸻

Bottom line: PE encodes each position as a multi-scale phase code where relative shifts are linear rotations and similarity depends on the offset. This matches dot-product attention’s algebra, making relative-position rules easy to learn, stable to train, and able to generalize to longer sequences.