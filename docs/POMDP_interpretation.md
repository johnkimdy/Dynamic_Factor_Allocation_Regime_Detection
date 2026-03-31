# Helix and the Partially Observable Markov Decision Process (POMDP)

Our implementation and strategy **can be interpreted and explained** using a **Partially Observable Markov Decision Process (POMDP)**. We do not *solve* a POMDP numerically; we implement a **belief → action** pipeline that fits naturally into the POMDP framework.

---

## 1. POMDP in one paragraph

A **POMDP** has: (1) a **hidden state** the agent never sees directly; (2) **observations** that depend stochastically on the state; (3) **actions** the agent chooses; (4) **rewards** that depend on state and action; (5) **state transitions** (e.g. Markov). The agent maintains a **belief** (distribution over states) given the history of observations. The goal is to choose a **policy** that maps beliefs to actions so as to maximize expected (discounted) reward. So: *belief* is the sufficient statistic for the history, and the decision problem is a *belief-state MDP*.

---

## 2. Mapping Helix to a POMDP

| POMDP element | Helix interpretation |
|---------------|----------------------|
| **State** \(s_t\) | **Regime** (e.g. which of K regimes the market/factor is in). We never observe this directly. |
| **Observation** \(o_t\) | **Features** (EWMA, RSI, %K, MACD, VIX, yields, etc.) and/or **returns**. Noisy signals that depend on the underlying regime. |
| **Belief** \(b_t\) | **Posterior over regime** given the history of features (and returns). In our code this is produced by the SJM: either a **point estimate** (the last regime from online inference) or, in principle, the model’s **soft assignments** (probabilities over regimes). |
| **Action** \(a_t\) | **Portfolio weights** over the 7 assets (SPY + 6 factor ETFs). Chosen once per day (or at rebalance). |
| **Reward** \(r_t\) | **Portfolio return** (or risk-adjusted return) over the period the weights are applied. Stochastic given state and action. |
| **Transition** \(P(s_{t+1} \mid s_t)\) | **Regime dynamics.** We do not write an explicit transition matrix; the **jump model** encourages **persistence** (penalty for switching), so the inferred regime sequence is consistent with a sticky Markov process. |

So: the **true regime** is the hidden state; we only see **observations** (features and returns); we use the **SJM** to turn the history of observations into a **belief** (in practice, a point estimate of the current regime); we then choose an **action** (weights) from that belief; and we receive a **reward** (realized return). The regime is assumed to evolve over time (transitions), with a preference for not switching too often.

---

## 3. The belief → action pipeline in Helix

In standard POMDP terms, the optimal policy would be a function \(\pi(b)\) that maps the *full* belief state to an action, often computed by value iteration or similar. **Helix does not compute \(\pi(b)\).** Instead it uses a **heuristic belief → action** map:

1. **Belief:** From features (and possibly returns) we get the **current regime** per factor via the SJM (online inference over a lookback window with fixed centroids). So \(b_t \approx\) “we are in regime \(k\) for each factor.”
2. **Expected returns:** For each factor, expected active return = **historical average return in that regime** (capped ±5% p.a.). So we treat the regime as summarizing “what kind of environment we’re in” and use that to set **views**.
3. **Action:** **Black–Litterman** combines the benchmark (prior) with these views and the covariance (EWMA) to produce **weights**. So \(a_t = \text{BL}(b_t)\) in a loose notation.

So the **control** is: **belief (regime) → views → BL → weights.** That is a **myopic, one-step** use of the belief: we do not optimize over future beliefs or multi-period returns. Within the POMDP story, we are using the **current belief** to pick **today’s action**, and we do not explicitly solve the full POMDP.

---

## 4. Why this is “partial observability”

We never see the regime. We see:

- Past and current **returns** (noisy, regime-dependent).
- **Features** built from returns and market data (VIX, yields, etc.), which are also noisy functions of the regime.

So the **state** (regime) is **partially observable**: we only have indirect, noisy **observations**. The SJM’s job is to turn the observation history into a **belief** (or a point estimate of the regime) so we can decide **actions** (weights).

---

## 5. Summary

- **Yes:** The setup is **relevant to and explainable by** a POMDP: hidden state = regime, observations = features/returns, belief = SJM output, action = weights, reward = return, transition = regime dynamics (implicit in the jump penalty).
- **Caveat:** We do **not** solve the POMDP (no value iteration, no optimal \(\pi(b)\)). We use a **hand-designed** mapping: belief (regime) → expected returns → BL → weights. So the strategy is **POMDP-inspired** and **POMDP-interpretable**, but the control rule is heuristic (one-step BL given current belief), not the optimal POMDP policy.

A natural extension would be to **learn or optimize** the belief→action map (e.g. approximate dynamic programming on the belief state, or reinforcement learning with the belief as state) instead of fixing it to “regime → historical mean → BL.” That would move closer to a true POMDP solution.
