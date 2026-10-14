// Side panel: live defensive dashboard, match report, Gemini chat.
const esc = (s) => String(s ?? '').replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
export function md(s) {
  // tiny markdown: **bold**, _italic_, `code`, lists, tables, line breaks
  const lines = esc(s).split('\n');
  let html = '', inList = false, inTable = false;
  for (let raw of lines) {
    let l = raw.replace(/\*\*(.+?)\*\*/g, '<b>$1</b>').replace(/(^|\W)_(.+?)_(\W|$)/g, '$1<i>$2</i>$3').replace(/`(.+?)`/g, '<code>$1</code>');
    if (/^\s*\|.*\|\s*$/.test(raw)) {
      if (/^\s*\|[\s:-|]+\|\s*$/.test(raw)) continue;
      if (!inTable) { html += '<table style="border-collapse:collapse;font-size:12px;margin:4px 0">'; inTable = true; }
      html += '<tr>' + l.trim().slice(1, -1).split('|').map((c) => `<td style="border:1px solid rgba(255,255,255,.12);padding:2px 5px">${c.trim()}</td>`).join('') + '</tr>';
      continue;
    } else if (inTable) { html += '</table>'; inTable = false; }
    if (/^\s*([-*]|\d+\.)\s+/.test(raw)) {
      if (!inList) { html += '<ul style="margin:4px 0;padding-left:18px">'; inList = true; }
      html += `<li>${l.replace(/^\s*([-*]|\d+\.)\s+/, '')}</li>`; continue;
    } else if (inList) { html += '</ul>'; inList = false; }
    if (/^#{1,4}\s/.test(raw)) { html += `<div style="font-weight:700;margin-top:6px">${l.replace(/^#+\s/, '')}</div>`; continue; }
    html += l.trim() ? `<div>${l}</div>` : '<div style="height:5px"></div>';
  }
  if (inList) html += '</ul>'; if (inTable) html += '</table>';
  return html;
}

const f0 = (v, d = 0) => (Number.isFinite(v) ? v.toFixed(d) : '–');

export function renderLive(el, m, ctx) {
  const { teamName, sel, selCard, baseline, dangerSpace } = ctx;
  if (!m || !m.inPlay) { el.innerHTML = '<div class="muted">Ball out of play or possession unknown at this instant.</div>' + (selCard || ''); return; }
  const base = baseline || {};
  const cmp = (v, b, higherBad) => {
    if (!Number.isFinite(v) || !Number.isFinite(b)) return '';
    const d = v - b; const bad = higherBad ? d > 0 : d < 0;
    return `<small>${d >= 0 ? '+' : ''}${d.toFixed(0)} vs median</small>${Math.abs(d) > Math.abs(b) * 0.2 ? (bad ? ' ⚠' : '') : ''}`;
  };
  const alerts = m.alerts.length ? m.alerts.map((a) => `<div class="alert ${a.lvl}">${esc(a.txt)}</div>`).join('')
    : '<div class="alert ok">No structural alarm at this instant</div>';
  el.innerHTML = `
    ${selCard || ''}
    <div><b>${esc(teamName(m.defTeam))}</b> <span class="muted">defending</span></div>
    <div class="kpis">
      <div class="kpi"><span>Back line from goal</span><b>${f0(m.lineHeight)} m</b>${cmp(m.lineHeight, base.line_height, false)}</div>
      <div class="kpi"><span>Block length × width</span><b>${f0(m.length)} × ${f0(m.width)} m</b></div>
      <div class="kpi"><span>Ball-zone compactness</span><b>${f0(m.ballZoneArea)} m²</b>${cmp(m.ballZoneArea, base.ball_zone_area, true)}</div>
      <div class="kpi"><span>Space between lines</span><b>${f0(m.lineGap, 1)} m</b>${cmp(m.lineGap, base.line_gap, true)}</div>
      <div class="kpi"><span>Behind the ball</span><b>${m.behindBall ?? '–'} / ${m.D.length}</b></div>
      <div class="kpi"><span>Pressure on ball</span><b>${f0(m.pressDist, 1)} m · ${m.pressN ?? 0} close</b></div>
      <div class="kpi"><span>Free attackers ≤35 m</span><b style="color:${m.freeAttackers ? '#f87171' : 'inherit'}">${m.freeAttackers} / ${m.attackersInZone}</b></div>
      <div class="kpi"><span>Space behind the line</span><b>${Number.isFinite(dangerSpace) ? f0(dangerSpace) + ' m²' : '<small>turn on Space control</small>'}</b></div>
    </div>
    <div class="muted" style="font-size:11px;margin-bottom:4px">Live alerts</div>
    ${alerts}
    <div style="margin-top:8px"><button onclick="window.__runSim({frame: Math.round(window.__viewer.t)})">🧪 Simulate better defending here</button></div>
    <div class="muted" style="font-size:11px;margin-top:8px">Medians = this team's organised-defence values in this match. Ball-zone compactness = hull of the 5 defenders nearest the ball (the compactness linked to regaining possession).</div>`;
}

export function selectedCard(p, team, live, prof) {
  if (!p) return '';
  const pill = (txt, cls) => `<span class="pill ${cls}">${esc(txt)}</span>`;
  const rows = [];
  if (live) {
    if (live.sees) rows.push(['Vision', `${live.sees.ball ? pill('sees ball', 'good') : pill('not on ball', 'mid')} ${live.sees.man == null ? '' : live.sees.man ? pill('sees his man', 'good') : pill('man outside view', 'bad')}`]);
    if (live.shape) rows.push(['Body vs own goal', pill(live.shape, live.shape === 'open' ? 'good' : live.shape === 'square' ? 'bad' : 'mid')]);
    if (live.man) rows.push(['Nearest attacker', `#${esc(live.man.number)} ${esc(live.man.name)} · ${f0(live.man.dist, 1)} m ${live.man.goalSide ? pill('goal-side', 'good') : pill('not goal-side', 'bad')}`]);
    if (Number.isFinite(live.speed)) rows.push(['Speed', `${f0(live.speed, 1)} m/s`]);
    if (!live.pose) rows.push(['Body pose', '<span class="muted">not visible on broadcast now</span>']);
  }
  if (prof) {
    const pc = (v) => (Number.isFinite(v) ? `${Math.round(v * 100)}%` : '–');
    rows.push(['Match profile', `<span class="muted">duels ${prof.duels ?? 0} · runs ${prof.runs_tracked ?? 0} · passes marked ${prof.passes_marked ?? 0}</span>`]);
    rows.push(['SEE', `dual vision ${pc(prof.dual_vision)} · checks/pass ${f0(prof.checks_per_pass, 2)} · blind-side runs ${pc(prof.blind_side)}`]);
    rows.push(['SHAPE / STANCE', `side-on ${pc(prof.side_on)} · square at runs ${pc(prof.square_at_run)} · loaded ${pc(prof.loaded)}`]);
    rows.push(['REACT', `reaction lag ${f0(prof.reaction_lag_s, 2)} s · beaten in behind ${pc(prof.beaten_in_behind)}`]);
  }
  return `<div class="card" style="border-color:#f59e0b"><h4>#${esc(p.number)} ${esc(p.name)} <span class="muted" style="font-weight:400">${esc(p.role || '')} · ${esc(team)}</span></h4>
    ${rows.map(([k, v]) => `<div class="row"><span>${k}</span><span style="text-align:right">${v}</span></div>`).join('')}</div>`;
}

export function renderReport(el, ins, events, clock, onJump, goals, onAsk) {
  if (!ins) { el.innerHTML = '<span class="muted">No report.</span>'; return; }
  let h = '';
  if (goals && goals.length) {
    h += '<h3>Goals — how they happened &amp; responsibility</h3>';
    goals.forEach((b, i) => {
      const g = b.goal, top = (b.responsibility || []).slice(0, 3);
      const seq = (b.sequence || []).slice(-5).map((s) => `${esc(s.player)}${s.to && typeof s.to === 'string' ? ' → ' + esc(s.to) : s.action === 'shot' ? ' <b>shot</b>' : ''}${s.line_break ? ' <span class="pill mid">breaks ' + esc(s.line_break) + '</span>' : ''}`).join(' · ');
      const at = b.at_shot && b.at_shot.defender ? `<div class="muted" style="font-size:11.5px">At the shot: ${esc(b.at_shot.defender)} ${b.at_shot.distance_m} m from the scorer, ${b.at_shot.defenders_in_shot_line} in the shot line</div>` : '';
      h += `<div class="card"><h4>⚽ ${esc(g.time)} ${esc(g.scorer)} <span class="muted" style="font-weight:400">(${esc(g.team)}) · ${esc(g.score_after)}</span></h4>
        <div style="font-size:12px;margin-bottom:4px">${seq}</div>${at}
        <div class="muted" style="font-size:11px;margin-top:4px">Responsibility indicators (${esc(g.conceding)})</div>
        ${top.map((r) => `<div class="row"><span><b style="color:#eef2f7">${esc(r.defender)}</b> · ${r.points} pts</span><span style="text-align:right;font-size:11.5px">${esc(r.reasons.map((x) => x.split(': ').slice(1).join(': ').replace(/ \(\+\d\)$/, '')).join('; '))}</span></div>`).join('')}
        <div style="display:flex;gap:6px;margin-top:6px;flex-wrap:wrap"><button data-goal="${i}" class="gwatch">▶ Watch build-up</button><button data-goal="${i}" class="gsim">🧪 How to stop it</button><button data-goal="${i}" class="gask">Ask Gemini</button></div></div>`;
    });
  }
  for (const t of ins.teams) {
    h += `<h3>${esc(t)} — defending</h3><ul>${(ins.bullets[t] || []).map((b) => `<li>${md(b).replace(/<\/?div>/g, '')}</li>`).join('')}</ul>`;
  }
  const kp = ins.kpis || [];
  if (kp.length) {
    const cols = [['ppda', 'PPDA'], ['counter_presses', 'Counter-presses'], ['regains', 'Regains'], ['line_breaks_conceded', 'Line breaks conceded'], ['opp_shots', 'Shots conceded']];
    h += `<h3>KPIs</h3><table style="width:100%;font-size:12px;border-collapse:collapse"><tr><td></td>${kp.map((k) => `<td style="text-align:right"><b>${esc(k.team)}</b></td>`).join('')}</tr>` +
      cols.map(([c, n]) => `<tr><td class="muted">${n}</td>${kp.map((k) => `<td style="text-align:right">${esc(k[c])}</td>`).join('')}</tr>`).join('') + '</table>';
  }
  const key = (events || []).filter((e) => e.key).slice(0, 400);
  const pick = key.filter((e) => /beaten in behind|not recovered|Beaten/.test(e.label)).slice(0, 12);
  if (pick.length) {
    h += `<h3>Moments to review</h3>` + pick.map((e) => `<div class="rev" data-f="${e.f}" data-d="${e.defender ?? ''}"><b>${clock(e.f)}</b> · ${esc(e.label)}</div>`).join('');
  }
  h += `<div class="muted" style="font-size:11px;margin-top:10px">Computed from SkillCorner tracking, Dynamic Events and body pose (no LLM). PPDA uses SkillCorner engagements ending in regain/disruption/foul, so it is not directly comparable with event-provider PPDA.</div>`;
  el.innerHTML = h;
  el.querySelectorAll('.rev').forEach((d) => (d.onclick = () => onJump(Number(d.dataset.f), d.dataset.d ? Number(d.dataset.d) : null)));
  el.querySelectorAll('.gwatch').forEach((b) => (b.onclick = () => {
    const g = goals[Number(b.dataset.goal)];
    const f = g.sequence && g.sequence.length ? Math.max(g.sequence[0].frame, g.goal.frame - 150) : g.goal.frame - 120;
    onJump(f + 30, null, true);
  }));
  el.querySelectorAll('.gsim').forEach((b) => (b.onclick = () => window.__runSim({ goal: goals[Number(b.dataset.goal)].goal.goal_number })));
  el.querySelectorAll('.gask').forEach((b) => (b.onclick = () => {
    const g = goals[Number(b.dataset.goal)].goal;
    onAsk(`How did goal ${g.goal_number} (${g.scorer}, ${g.time}) happen and which defenders are responsible?`);
  }));
}

export function addMsg(chat, role, html, extra = '') {
  const d = document.createElement('div');
  d.className = `msg ${role}`; d.innerHTML = html + extra;
  chat.appendChild(d); chat.parentElement.scrollTop = chat.parentElement.scrollHeight;
  return d;
}
