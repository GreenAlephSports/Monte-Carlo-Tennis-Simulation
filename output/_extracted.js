
(function () {
  "use strict";
  const DATA = {
    atp: JSON.parse(document.getElementById("data-atp").textContent),
    wta: JSON.parse(document.getElementById("data-wta").textContent),
  };

  // How far the model sits from the market, relative to the market's own price - e.g. model 40%
  // vs market 20% is "+100%", not "+20 points" - rather than the flat percentage-point gap, which
  // treats a 20pt swing on a coin-flip match the same as a 20pt swing on a near-lock.
  [DATA.atp, DATA.wta].forEach((d) => {
    d.matches.forEach((m) => {
      const rel = m.market_prob_a ? ((m.model_prob_a - m.market_prob_a) / m.market_prob_a) * 100 : null;
      m.rel_diff_pct = rel === null ? null : rel;
      m.rel_diff_abs = rel === null ? -1 : Math.abs(rel);
    });
  });

  let state = { tour: "atp", view: "draw", sortKey: "rel_diff_abs", sortDir: "desc", search: "", status: "all", mkt: "all" };

  const pct = (x) => x === null || x === undefined ? null : (x * 100);
  const fmtPct = (x, digits) => x === null || x === undefined ? "—" : x.toFixed(digits === undefined ? 1 : digits) + "%";
  const esc = (s) => (s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  function findMatch(data, nameA, nameB) {
    return data.matches.find((m) =>
      (m.player_a === nameA && m.player_b === nameB) || (m.player_a === nameB && m.player_b === nameA)
    ) || null;
  }

  function renderMeta() {
    const d = DATA[state.tour];
    const el = document.getElementById("head-meta");
    el.innerHTML =
      `<span>generated <span class="hl">${esc(d.meta.generated_at)}</span></span>` +
      `<span>${d.meta.current_iterations.toLocaleString()} sims · seed ${d.meta.current_seed}</span>` +
      `<span>${d.players.filter(p => p.alive).length} of ${d.draw_size} still alive</span>`;
  }

  // Named by true tournament round (r counts from Round 1 of the whole draw), never by a side's
  // own local match count - a 128-draw's Round 1 has 64 matches total but only 32 per side, so
  // naming off the local count mislabeled every round (e.g. Round 1 read "Round 6").
  const ROUND_NAME = (r, numRounds) => {
    if (r === numRounds) return "Final";
    if (r === numRounds - 1) return "Semifinals";
    if (r === numRounds - 2) return "Quarterfinals";
    return `Round ${r}`;
  };

  function matchProbsFor(match, playerName) {
    if (!match) return { model: null, market: null };
    const isA = match.player_a === playerName;
    const model = match.model_prob_a === null || match.model_prob_a === undefined ? null : (isA ? match.model_prob_a : 1 - match.model_prob_a);
    const market = match.market_prob_a === null || match.market_prob_a === undefined ? null : (isA ? match.market_prob_a : 1 - match.market_prob_a);
    return { model: model === null ? null : model * 100, market: market === null ? null : market * 100 };
  }

  function matchStatus(m) {
    if (!m) return "not_started";
    if (m.decided) return "decided";
    if (m.live_state === "in") return "live";
    return "not_started";
  }

  function liveBadge(match) {
    if (!match || match.live_state !== "in") return "";
    return `<span class="live-dot"></span><span class="live-tag">LIVE</span>${match.score ? " " + esc(match.score) : ""} · `;
  }

  function oddsLine(model, market, match) {
    const badge = liveBadge(match);
    if (model === null && market === null) return badge;
    const parts = [];
    if (model !== null) parts.push(`<span class="m">M</span> ${model.toFixed(1)}%`);
    if (market !== null) parts.push(`<span class="k">Mk</span> ${market.toFixed(1)}%`);
    else parts.push(`<span class="k">Mk</span> —`);
    return badge + parts.join(" · ");
  }

  // Round-1 leaves feed forward round by round; a real match for any cell is found purely by
  // name-pair lookup (draw position determines the pairing exactly, independent of the
  // "which round is this checkpoint" label the underlying data can mis-tag - see the artifact's
  // footer note), so this recursion is correct even before every earlier round is fully decided.
  // `leaves` is one half of the draw (32 players -> 5 rounds) when building a mirrored side-tree,
  // or the whole draw when there's only one round left (the Final, built separately - see below).
  function buildRounds(d, leaves) {
    const numRounds = Math.round(Math.log2(leaves.length));
    let prevWinners = leaves.map((p) => ({ name: p.player, bye: p.bye }));
    const rounds = [];
    for (let r = 1; r <= numRounds; r++) {
      const count = prevWinners.length / 2;
      const cells = [];
      for (let i = 0; i < count; i++) {
        const a = prevWinners[2 * i], b = prevWinners[2 * i + 1];
        let match = null, winnerName = null, decided = false, isBye = false;
        if (a && a.bye && a.name && !(b && b.bye)) { winnerName = b ? b.name : null; decided = !!winnerName; isBye = true; }
        else if (b && b.bye && b.name && !(a && a.bye)) { winnerName = a ? a.name : null; decided = !!winnerName; isBye = true; }
        else if (a && b && a.name && b.name) {
          match = findMatch(d, a.name, b.name);
          if (match && match.decided) { winnerName = match.winner; decided = true; }
        }
        cells.push({ a, b, match, winnerName, decided, isBye });
      }
      rounds.push(cells);
      prevWinners = cells.map((c) => ({ name: c.winnerName, bye: false }));
    }
    return rounds;
  }

  const LEAF_H = 40, BOX_W = 182, COL_GAP = 44, COL_STEP = BOX_W + COL_GAP;

  function makeRound1Cell(cell, x, top, byName) {
    const box = document.createElement("div");
    box.className = "cell" + (cell.match ? " has-match" : "") + (cell.match && cell.match.live_state === "in" ? " live" : "");
    box.style.left = x + "px";
    box.style.top = top + "px";
    box.style.width = BOX_W + "px";
    box.style.height = (2 * LEAF_H - 4) + "px";
    [cell.a, cell.b].forEach((leaf) => {
      if (!leaf || !leaf.name) {
        const row = document.createElement("div");
        row.className = "leaf-row";
        row.innerHTML = `<span class="leaf-name" style="color:var(--ink-faint);">—</span>`;
        box.appendChild(row);
        return;
      }
      const p = byName[leaf.name] || { player: leaf.name, seed: null, current: null, alive: true };
      const isWinner = cell.decided && cell.winnerName === leaf.name;
      const isLoser = cell.decided && cell.winnerName !== leaf.name;
      const row = document.createElement("div");
      row.className = "leaf-row" + (isWinner ? " winner" : "") + (isLoser ? " loser" : "");
      const champ = p.current ? pct(p.current.p_champ) : (p.pretournament ? pct(p.pretournament.p_champ) : null);
      const { model, market } = matchProbsFor(cell.match, leaf.name);
      const oddsHtml = cell.isBye ? `<span style="font-style:italic;">bye</span>` : oddsLine(model, market, cell.match);
      row.innerHTML =
        `<div class="leaf-top"><div class="leaf-name-wrap"><span class="leaf-seed">${p.seed ? "#" + p.seed : ""}</span>` +
        `<span class="leaf-name">${esc(leaf.name)}</span></div>` +
        `<span class="leaf-champ num">${champ !== null ? champ.toFixed(1) + "%" : "—"}</span></div>` +
        `<div class="leaf-odds num">${oddsHtml}</div>`;
      box.appendChild(row);
    });
    box.addEventListener("click", () => {
      if (!cell.a || !cell.b || !cell.a.name || !cell.b.name) return;
      const pa = byName[cell.a.name] || { player: cell.a.name };
      const pb = byName[cell.b.name] || { player: cell.b.name };
      openDetail(pa, pb, cell.match);
    });
    return box;
  }

  function makeAdvCell(cell, x, top, isFinal, byName) {
    const box = document.createElement("div");
    box.className = "cell" + (cell.match ? " has-match" : "") + (!cell.decided && !cell.match ? " tbd" : "") +
      (isFinal && cell.decided ? " final-cell" : "") + (cell.match && cell.match.live_state === "in" ? " live" : "");
    box.style.left = x + "px";
    box.style.top = top + "px";
    box.style.width = BOX_W + "px";
    box.style.height = (LEAF_H - 6) + "px";

    if (cell.decided && cell.winnerName) {
      const p = byName[cell.winnerName] || { player: cell.winnerName, current: null };
      const champ = p.current ? pct(p.current.p_champ) : (p.pretournament ? pct(p.pretournament.p_champ) : null);
      const { model, market } = matchProbsFor(cell.match, cell.winnerName);
      box.innerHTML =
        `<div class="adv-cell"><div class="adv-top"><span class="adv-name">${esc(cell.winnerName)}</span>` +
        `<span class="adv-champ num">${champ !== null ? champ.toFixed(1) + "%" : "—"}</span></div>` +
        `<div class="adv-odds num">${oddsLine(model, market, cell.match)}</div></div>`;
      box.addEventListener("click", () => {
        const pa = byName[cell.match.player_a] || { player: cell.match.player_a };
        const pb = byName[cell.match.player_b] || { player: cell.match.player_b };
        openDetail(pa, pb, cell.match);
      });
    } else if (cell.match && cell.a && cell.b) {
      const { model } = matchProbsFor(cell.match, cell.a.name);
      const favName = model !== null && model >= 50 ? cell.a.name : cell.b.name;
      const statusLine = cell.match.live_state === "in"
        ? `${liveBadge(cell.match)}fav ${esc(favName)}`
        : `pending · fav ${esc(favName)}`;
      box.innerHTML =
        `<div class="adv-cell"><div class="adv-top"><span class="adv-name">${esc(cell.a.name)} / ${esc(cell.b.name)}</span></div>` +
        `<div class="adv-odds num">${statusLine}</div></div>`;
      box.addEventListener("click", () => {
        const pa = byName[cell.a.name] || { player: cell.a.name };
        const pb = byName[cell.b.name] || { player: cell.b.name };
        openDetail(pa, pb, cell.match);
      });
    } else {
      box.innerHTML = `<div class="adv-cell"><span class="adv-tbd">TBD</span></div>`;
    }
    return box;
  }

  function drawConnectorSet(svg, sideRounds, xOf, centerYFn, mirrored) {
    for (let r = 2; r <= sideRounds.length; r++) {
      const prevCount = sideRounds[r - 2].length;
      for (let i = 0; i < prevCount / 2; i++) {
        const yA = centerYFn(r - 1, 2 * i), yB = centerYFn(r - 1, 2 * i + 1), yM = centerYFn(r, i);
        const feederAnchorX = mirrored ? xOf(r - 1) : xOf(r - 1) + BOX_W;
        const cellAnchorX = mirrored ? xOf(r) + BOX_W : xOf(r);
        const midX = (feederAnchorX + cellAnchorX) / 2;
        [[feederAnchorX, yA, midX, yA], [feederAnchorX, yB, midX, yB], [midX, yA, midX, yB], [midX, yM, cellAnchorX, yM]]
          .forEach(([x1, y1, x2, y2]) => {
            const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
            line.setAttribute("x1", x1); line.setAttribute("y1", y1);
            line.setAttribute("x2", x2); line.setAttribute("y2", y2);
            svg.appendChild(line);
          });
      }
    }
  }

  // Two mirrored half-trees (left = quarters 1-2, right = quarters 3-4) grow toward the center and
  // meet at a single Final cell - the standard two-sided draw-sheet layout, rather than one long
  // single-direction ladder.
  function renderDraw() {
    const d = DATA[state.tour];
    const byName = {}; d.players.forEach((p) => { byName[p.player] = p; });
    const sorted = d.players.slice().sort((a, b) => a.position - b.position);
    const half = d.draw_size / 2;
    const leftLeaves = sorted.slice(0, half);
    const rightLeaves = sorted.slice(half);
    const leftRounds = buildRounds(d, leftLeaves);
    const rightRounds = buildRounds(d, rightLeaves);
    const sideRounds = leftRounds.length; // == rightRounds.length
    const totalRounds = sideRounds + 1; // + the Final

    const headerH = 34;
    const totalH = half * LEAF_H + headerH + 8;
    const leftX = (r) => (r - 1) * COL_STEP;
    const finalX = leftX(sideRounds) + BOX_W + COL_GAP;
    const rightX = (r) => finalX + BOX_W + COL_GAP + (sideRounds - r) * COL_STEP;
    const totalW = rightX(1) + BOX_W + 12;

    const root = document.getElementById("bracket-tree");
    root.style.width = totalW + "px";
    root.style.height = totalH + "px";
    root.innerHTML = "";

    const centerY = (r, i) => headerH + (i + 0.5) * Math.pow(2, r) * LEAF_H;

    // quarter tags + seam, one per side (each side is exactly 2 quarters)
    [["Q1", "Q2", -2, "left"], ["Q3", "Q4", totalW - 14, "right"]].forEach(([topLabel, bottomLabel, tagX]) => {
      [topLabel, bottomLabel].forEach((label, qi) => {
        const tag = document.createElement("div");
        tag.className = "quarter-tag";
        tag.style.left = tagX + "px";
        tag.style.top = (headerH + qi * (half / 2) * LEAF_H + (half / 2) * LEAF_H / 2 - 30) + "px";
        tag.textContent = label;
        root.appendChild(tag);
      });
    });

    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("class", "connectors");
    svg.setAttribute("width", totalW);
    svg.setAttribute("height", totalH);
    drawConnectorSet(svg, leftRounds, leftX, centerY, false);
    drawConnectorSet(svg, rightRounds, rightX, centerY, true);
    // both semifinal winners into the Final, dead center
    const midY = centerY(sideRounds, 0);
    [[leftX(sideRounds) + BOX_W, midY, finalX, midY], [rightX(sideRounds), midY, finalX + BOX_W, midY]]
      .forEach(([x1, y1, x2, y2]) => {
        const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
        line.setAttribute("x1", x1); line.setAttribute("y1", y1);
        line.setAttribute("x2", x2); line.setAttribute("y2", y2);
        svg.appendChild(line);
      });
    // quarter seam dashed dividers, over each side's round-1 column only
    [[0, BOX_W], [rightX(1), rightX(1) + BOX_W]].forEach(([x1, x2]) => {
      const y = headerH + (half / 2) * LEAF_H;
      const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
      line.setAttribute("class", "seam");
      line.setAttribute("x1", x1); line.setAttribute("y1", y);
      line.setAttribute("x2", x2); line.setAttribute("y2", y);
      svg.appendChild(line);
    });
    root.appendChild(svg);

    // round headers: left grows 1..sideRounds rightward, right grows 1..sideRounds leftward, Final centered
    for (let r = 1; r <= sideRounds; r++) {
      const headL = document.createElement("div");
      headL.className = "round-head";
      headL.style.left = leftX(r) + "px";
      headL.innerHTML = `${ROUND_NAME(r, totalRounds)}<span>${leftRounds[r - 1].length} match${leftRounds[r - 1].length === 1 ? "" : "es"}</span>`;
      root.appendChild(headL);

      const headR = document.createElement("div");
      headR.className = "round-head";
      headR.style.left = rightX(r) + "px";
      headR.innerHTML = `${ROUND_NAME(r, totalRounds)}<span>${rightRounds[r - 1].length} match${rightRounds[r - 1].length === 1 ? "" : "es"}</span>`;
      root.appendChild(headR);
    }
    const headF = document.createElement("div");
    headF.className = "round-head";
    headF.style.left = finalX + "px";
    headF.innerHTML = `${ROUND_NAME(totalRounds, totalRounds)}<span>1 match</span>`;
    root.appendChild(headF);

    // round 1 cells (both sides)
    leftRounds[0].forEach((cell, i) => root.appendChild(makeRound1Cell(cell, leftX(1), centerY(1, i) - LEAF_H, byName)));
    rightRounds[0].forEach((cell, i) => root.appendChild(makeRound1Cell(cell, rightX(1), centerY(1, i) - LEAF_H, byName)));

    // round 2..sideRounds cells (both sides)
    for (let r = 2; r <= sideRounds; r++) {
      leftRounds[r - 1].forEach((cell, i) => root.appendChild(makeAdvCell(cell, leftX(r), centerY(r, i) - (LEAF_H - 6) / 2, false, byName)));
      rightRounds[r - 1].forEach((cell, i) => root.appendChild(makeAdvCell(cell, rightX(r), centerY(r, i) - (LEAF_H - 6) / 2, false, byName)));
    }

    // the Final itself - fed by each side's own last-round winner
    const leftWinner = leftRounds[sideRounds - 1][0];
    const rightWinner = rightRounds[sideRounds - 1][0];
    let finalCell = { a: null, b: null, match: null, winnerName: null, decided: false };
    if (leftWinner.winnerName && rightWinner.winnerName) {
      const a = { name: leftWinner.winnerName, bye: false }, b = { name: rightWinner.winnerName, bye: false };
      const match = findMatch(d, a.name, b.name);
      finalCell = { a, b, match, winnerName: match && match.decided ? match.winner : null, decided: !!(match && match.decided) };
    }
    root.appendChild(makeAdvCell(finalCell, finalX, midY - (LEAF_H - 6) / 2, true, byName));
  }

  function renderMatches() {
    const d = DATA[state.tour];
    let rows = d.matches.slice();

    if (state.search) {
      const s = state.search.toLowerCase();
      rows = rows.filter((m) => m.player_a.toLowerCase().includes(s) || m.player_b.toLowerCase().includes(s));
    }
    if (state.status !== "all") rows = rows.filter((m) => matchStatus(m) === state.status);
    if (state.mkt === "priced") rows = rows.filter((m) => m.market_prob_a !== null);

    const key = state.sortKey;
    const dir = state.sortDir === "asc" ? 1 : -1;
    rows.sort((x, y) => {
      let vx = x[key], vy = y[key];
      if (key === "decided") { vx = vx ? 1 : 0; vy = vy ? 1 : 0; }
      if (typeof vx === "string") return vx.localeCompare(vy) * dir;
      vx = vx === null || vx === undefined ? -Infinity : vx;
      vy = vy === null || vy === undefined ? -Infinity : vy;
      return (vx - vy) * dir;
    });

    document.getElementById("match-count").textContent = `${rows.length} of ${d.matches.length} matches`;

    const tbody = document.getElementById("matches-body");
    tbody.innerHTML = "";
    const maxGap = Math.max(1, ...d.matches.map((m) => m.rel_diff_abs > 0 ? m.rel_diff_abs : 0));

    rows.forEach((m) => {
      const tr = document.createElement("tr");
      const hasRel = m.rel_diff_pct !== null;
      const gapWidth = hasRel ? Math.max(4, (Math.abs(m.rel_diff_pct) / maxGap) * 60) : 0;
      const st = matchStatus(m);
      const pillLabel = st === "decided" ? "Final" : st === "live" ? "Live" : "Not started";
      tr.innerHTML =
        `<td>${esc(m.player_a)}${m.decided && m.winner === m.player_a ? " ✓" : ""}</td>` +
        `<td>${esc(m.player_b)}${m.decided && m.winner === m.player_b ? " ✓" : ""}</td>` +
        `<td class="mono">${fmtPct(pct(m.model_prob_a))}</td>` +
        `<td class="mono">${m.market_prob_a !== null ? fmtPct(pct(m.market_prob_a)) : '<span class="no-mkt">no price</span>'}</td>` +
        `<td class="mono"><div class="gap-bar-wrap">${hasRel ? `<span>${m.rel_diff_pct > 0 ? "+" : ""}${m.rel_diff_pct.toFixed(1)}%</span><span class="gap-bar" style="width:${gapWidth}px"></span>` : "—"}</div></td>` +
        `<td><span class="status-pill ${st}">${st === "live" ? '<span class="live-dot"></span>' : ""}${pillLabel}</span></td>` +
        `<td class="mono score-text">${m.score ? esc(m.score) : "—"}</td>`;
      tr.addEventListener("click", () => {
        const pa = d.players.find((p) => p.player === m.player_a) || { player: m.player_a };
        const pb = d.players.find((p) => p.player === m.player_b) || { player: m.player_b };
        openDetail(pa, pb, m);
      });
      tbody.appendChild(tr);
    });
  }

  function updateSortHeaders() {
    document.querySelectorAll("table.matches th[data-key]").forEach((th) => {
      const active = th.dataset.key === state.sortKey;
      th.classList.toggle("sorted", active);
      th.innerHTML = th.textContent.replace(/\s*[▲▼]$/, "") + (active ? `<span class="arrow">${state.sortDir === "asc" ? "▲" : "▼"}</span>` : "");
    });
  }

  function openDetail(pa, pb, match) {
    const card = document.getElementById("detail-card");
    const modelA = match ? pct(match.model_prob_a) : (pa.current ? pct(pa.current.p_champ) : null);
    const mktA = match ? pct(match.market_prob_a) : null;
    const modelPct = modelA !== null ? modelA : 50;
    const mktPct = mktA !== null ? mktA : null;

    let compareHtml = "";
    if (match) {
      compareHtml = `
        <div class="compare-row">
          <div class="compare-label"><span class="who">${esc(pa.player)}</span><span>${esc(pb.player)}</span></div>
          <div class="compare-track">
            <div class="compare-fill model" style="width:${modelPct}%">${modelPct.toFixed(1)}%</div>
            <div class="compare-fill" style="width:${100 - modelPct}%;background:var(--surface-2);color:var(--ink-faint);justify-content:flex-end;padding-right:6px;">${(100 - modelPct).toFixed(1)}%</div>
          </div>
          <div style="font-size:10px;color:var(--ink-faint);margin-top:2px;">MODEL · pre-match, not updated live</div>
        </div>`;
      if (mktPct !== null) {
        compareHtml += `
        <div class="compare-row">
          <div class="compare-label"><span class="who">${esc(pa.player)}</span><span>${esc(pb.player)}</span></div>
          <div class="compare-track">
            <div class="compare-fill market" style="width:${mktPct}%">${mktPct.toFixed(1)}%</div>
            <div class="compare-fill" style="width:${100 - mktPct}%;background:var(--surface-2);color:var(--ink-faint);justify-content:flex-end;padding-right:6px;">${(100 - mktPct).toFixed(1)}%</div>
          </div>
          <div style="font-size:10px;color:var(--ink-faint);margin-top:2px;">MARKET · live price when unsettled</div>
        </div>`;
      } else {
        compareHtml += `<div class="detail-empty">No market price on record for this match (never captured pregame).</div>`;
      }
      if (match.live_state === "in") {
        compareHtml += `<div class="detail-empty" style="color:var(--elim);"><span class="live-dot"></span><b class="live-tag">LIVE NOW</b>${match.score ? " — " + esc(match.score) : ""} — model above is frozen from before the match started.</div>`;
      }
    } else {
      compareHtml = `<div class="detail-empty">No real match on record yet for this pairing — showing current tournament-win odds only.</div>`;
    }

    const stats = [];
    if (match) {
      const st = matchStatus(match);
      stats.push(["Diff vs market", match.rel_diff_pct !== null ? (match.rel_diff_pct > 0 ? "+" : "") + match.rel_diff_pct.toFixed(1) + "%" : "—"]);
      stats.push(["Status", st === "decided" ? "Final" : st === "live" ? "Live" : "Not started"]);
      if (match.score) stats.push(["Score", esc(match.score)]);
      stats.push(["Winner", match.winner ? esc(match.winner) : "—"]);
      stats.push(["Round context", esc(match.round_label || "—")]);
    }
    [[pa, "A"], [pb, "B"]].forEach(([p]) => {
      if (p.current) stats.push([`${p.player} — title odds now`, pct(p.current.p_champ).toFixed(1) + "%"]);
      if (p.pretournament) stats.push([`${p.player} — pre-tournament`, pct(p.pretournament.p_champ).toFixed(1) + "%"]);
    });

    card.innerHTML = `
      <div class="detail-head">
        <div>
          <h3>${esc(pa.player)} vs ${esc(pb.player)}</h3>
        </div>
        <button class="detail-close" id="detail-close" aria-label="Close">×</button>
      </div>
      <div class="detail-sub">${match ? esc(match.round_label || "") : "No match played yet"}</div>
      ${compareHtml}
      <div class="detail-stats">
        ${stats.map(([k, v]) => `<div class="detail-stat"><div class="k">${esc(k)}</div><div class="v num">${v}</div></div>`).join("")}
      </div>
    `;
    document.getElementById("backdrop").hidden = false;
    document.getElementById("detail-close").addEventListener("click", closeDetail);
  }

  function closeDetail() { document.getElementById("backdrop").hidden = true; }

  function renderAll() {
    renderMeta();
    if (state.view === "draw") renderDraw(); else { renderMatches(); updateSortHeaders(); }
  }

  document.getElementById("tab-atp").addEventListener("click", () => setTour("atp"));
  document.getElementById("tab-wta").addEventListener("click", () => setTour("wta"));
  document.getElementById("tab-draw").addEventListener("click", () => setView("draw"));
  document.getElementById("tab-matches").addEventListener("click", () => setView("matches"));

  function setTour(tour) {
    state.tour = tour;
    document.getElementById("tab-atp").setAttribute("aria-selected", String(tour === "atp"));
    document.getElementById("tab-wta").setAttribute("aria-selected", String(tour === "wta"));
    renderAll();
  }
  function setView(view) {
    state.view = view;
    document.getElementById("tab-draw").setAttribute("aria-selected", String(view === "draw"));
    document.getElementById("tab-matches").setAttribute("aria-selected", String(view === "matches"));
    document.getElementById("view-draw").classList.toggle("active", view === "draw");
    document.getElementById("view-matches").classList.toggle("active", view === "matches");
    document.getElementById("legend-draw").style.display = view === "draw" ? "flex" : "none";
    renderAll();
  }

  document.getElementById("match-search").addEventListener("input", (e) => { state.search = e.target.value; renderMatches(); });
  document.getElementById("status-filter").addEventListener("click", (e) => {
    const btn = e.target.closest("button[data-status]");
    if (!btn) return;
    state.status = btn.dataset.status;
    document.querySelectorAll("#status-filter button").forEach((b) => b.setAttribute("aria-pressed", String(b === btn)));
    renderMatches();
  });
  document.getElementById("mkt-filter").addEventListener("click", (e) => {
    const btn = e.target.closest("button[data-mkt]");
    if (!btn) return;
    state.mkt = btn.dataset.mkt;
    document.querySelectorAll("#mkt-filter button").forEach((b) => b.setAttribute("aria-pressed", String(b === btn)));
    renderMatches();
  });
  document.querySelectorAll("table.matches th[data-key]").forEach((th) => {
    th.addEventListener("click", () => {
      if (state.sortKey === th.dataset.key) state.sortDir = state.sortDir === "asc" ? "desc" : "asc";
      else { state.sortKey = th.dataset.key; state.sortDir = "desc"; }
      renderMatches();
      updateSortHeaders();
    });
  });
  document.getElementById("backdrop").addEventListener("click", (e) => { if (e.target.id === "backdrop") closeDetail(); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeDetail(); });

  renderAll();
})();
