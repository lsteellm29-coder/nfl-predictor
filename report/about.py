# general-audience "How SharpLine Works" transparency page
"""Master Honing Plan round 2, item #5: a simplified, general-audience
version of the technical rigor already documented for developers (walk-
forward validation, empirical parameter sweeps, honest calibration
audits) -- written for a visitor who has never read a line of this
project's source and isn't going to. The goal is the same one the design
system already serves visually (three-tier confidence colors, "no line
available" cards instead of a fabricated one): showing the discipline
behind the numbers, not just asserting "trust us."

Every claim on this page has to be true of the actual, current codebase,
not aspirational -- see model/train.py, model/calibration.py,
model/td_ensemble.py's own module docstrings for the real methodology
and the real numbers this page paraphrases in plain language.
"""

HOW_IT_WORKS_STYLE = """
.about-section { max-width: 72ch; }
.about-section h3 {
  font-family: 'Oswald', sans-serif; font-weight: 700; font-size: 17px; letter-spacing: 0.01em;
  color: var(--ink); margin: 32px 0 10px;
}
.about-section h3:first-of-type { margin-top: 8px; }
.about-section p { font-size: 15px; line-height: 1.7; color: var(--ink); margin: 0 0 14px; }
.about-section p.lead { font-size: 16px; color: var(--muted); }
.about-rules { display: grid; gap: 10px; margin: 14px 0 24px; }
.about-rule {
  background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
  padding: 12px 14px; display: flex; gap: 12px; align-items: baseline;
}
.about-rule .rule-mark { color: var(--accent); font-family: 'Oswald', sans-serif; font-weight: 700; font-size: 13px; flex-shrink: 0; }
.about-rule p { margin: 0; font-size: 14px; color: var(--muted); line-height: 1.55; }
.about-rule p b { color: var(--ink); font-weight: 600; }
"""

HOW_IT_WORKS_HTML = """<div class="section-head" id="how-it-works"><span class="accent">&#9679;</span> How SharpLine Works</div>
<div class="about-section">
<p class="lead">SharpLine isn't a gut call dressed up with a percentage sign. Every number on this site comes from a model that's tested the same way a real forecaster would be tested -- against outcomes it never saw coming.</p>

<h3>Two models, doing two different jobs</h3>
<p>Win probability comes from an ensemble of two models -- a logistic regression and a gradient-boosted tree (XGBoost) -- trained on ten years of team-level stats: scoring efficiency, EPA per play, third-down and red-zone rates, turnover margin, injury reports, Elo ratings that track hot and cold streaks faster than a season average can, and the current Vegas line itself. Anytime-touchdown odds come from a completely separate model: a player's share of their team's real red-zone opportunities, adjusted for the specific opponent's defense, blended with a second model trained to catch patterns the first one's math misses. Neither model touches the other's numbers.</p>

<h3>Tested on games it never trained on</h3>
<p>Before any version of this model gets used for real, it's checked with what's called walk-forward validation: train on several years of history, test on the very next season it's never seen, then slide the whole window forward and repeat. That's the only fair test -- a model can look brilliant on data it already memorized and still fail on next Sunday's games. The number this site quotes as "model accuracy" is always from that kind of held-out test, never from games the model got to study first.</p>

<h3>Every setting was earned, not guessed</h3>
<p>Dozens of small decisions go into a model like this -- how much to trust a player's last four games versus their whole season, how hard to regress a hot streak toward the average, how much weight to give a second opinion when two models disagree. None of those numbers were picked because they sounded reasonable. Each one was tested across a real range of values against real outcomes, and only kept if it actually made the model's real-world track record better. Plenty of ideas that sounded promising -- and a few that are written up openly on this site's own technical pages -- were tried, tested, and thrown out because the numbers came back worse, not better.</p>

<h3>Checked for honesty, not just accuracy</h3>
<p>A model that says "70% confident" should be right about 70% of the time it says that -- not 55%, not 90%. That's a different question from whether it's accurate overall, and it's checked separately, every time the model retrains. If a smarter-sounding adjustment doesn't clearly improve that honesty by a real margin, it's left out rather than shipped on faith.</p>

<h3>What that looks like on the page</h3>
<div class="about-rules">
  <div class="about-rule"><span class="rule-mark">01</span><p><b>Color means something.</b> A pick only turns green or red once the model is genuinely confident (60%+). Anything closer than that shows up as an honest yellow toss-up -- never dressed up to look like a sure thing.</p></div>
  <div class="about-rule"><span class="rule-mark">02</span><p><b>No line, no fake number.</b> When a sportsbook hasn't posted a price for a player yet, the card says so plainly instead of hiding the player or inventing an edge that doesn't exist.</p></div>
  <div class="about-rule"><span class="rule-mark">03</span><p><b>The track record is public.</b> Every graded result -- good weeks and bad ones -- shows up on the <a href="#track-record" style="color: var(--accent);">Track Record page</a>, not just the numbers that make the model look good.</p></div>
  <div class="about-rule"><span class="rule-mark">04</span><p><b>News and storylines never move the number.</b> Headlines, injury chatter, and matchup narratives are shown for context, but they're generated after the model has already made its prediction -- they explain the pick, they never influence it.</p></div>
</div>

<p>None of this is investment or betting advice -- it's an honest attempt to build a model that says what it actually knows, and nothing more.</p>
</div>"""
