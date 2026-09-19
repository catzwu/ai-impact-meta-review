"""Shared visual theme for the Flask review app and the static GitHub Pages site.

Both front ends embed their HTML as Python strings. Each page's <head> carries a
`__THEME_HEAD__` placeholder (fonts + base stylesheet) and its body opens with a
`__MASTHEAD__` placeholder (site title + nav). `apply()` fills both in, so the two
front ends stay visually identical without a build step.

Page-specific <style> blocks come *after* the theme and hold only layout that is
unique to that page; colors there should use the CSS variables defined below.
"""
from __future__ import annotations

from html import escape

FONTS_LINK = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
    'family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400'
    '&family=Source+Sans+3:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">'
)

THEME_CSS = r"""
  :root {
    color-scheme: light;
    --paper:      #faf8f4;   /* page background: warm off-white */
    --surface:    #ffffff;
    --surface-2:  #f4f1ea;   /* inset panels, table heads */
    --ink:        #1d1f23;
    --ink-2:      #474b53;
    --muted:      #787c84;
    --rule:       #e2ddd2;
    --rule-soft:  #eeebe4;
    --accent:     #23406a;   /* oxford navy */
    --accent-2:   #1a3354;
    --accent-soft:#e9eef5;
    --link:       #2b5288;
    --pos:        #2e6b3f;
    --neg:        #9c2f2f;
    --warn-bg:    #fbf3dc;
    --warn-ink:   #6f5200;
    --warn-rule:  #e6d29a;
    --ok-bg:      #e9f3ea;
    --ok-ink:     #2e6b3f;
    --err-bg:     #f8e7e5;
    --err-ink:    #8e2a2a;
    --hover:      #f3f0e8;
    --speed:      #2f5d8a;  --speed-bg:   #e6eef6;
    --quality:    #8a4a6b;  --quality-bg: #f4e9ef;
    --serif: "Source Serif 4", "Iowan Old Style", Georgia, "Times New Roman", serif;
    --sans:  "Source Sans 3", -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
    --mono:  "IBM Plex Mono", ui-monospace, Menlo, monospace;
    --radius: 6px;
    --page-w: 1240px;
  }
  * { box-sizing: border-box; }
  html { -webkit-text-size-adjust: 100%; }
  body { margin:0; background:var(--paper); color:var(--ink); font-family:var(--sans);
         font-size:14px; line-height:1.5; -webkit-font-smoothing:antialiased; }
  a { color:var(--link); text-decoration:none; }
  a:hover { text-decoration:underline; text-underline-offset:2px; }
  code, kbd, pre, .mono { font-family:var(--mono); font-size:0.9em; }
  h1, h2, h3 { font-family:var(--serif); font-weight:600; color:var(--ink); letter-spacing:-0.005em; }
  b, strong { font-weight:600; }
  ::selection { background:#d9e2ee; }
  :focus-visible { outline:2px solid var(--accent); outline-offset:2px; }

  /* ---------- masthead ---------- */
  header.masthead { background:var(--surface); border-bottom:1px solid var(--rule); }
  header.masthead::before { content:""; display:block; height:3px; background:var(--accent); }
  .mh-inner { margin:0 auto; padding:10px 28px; display:flex; align-items:center; gap:28px; flex-wrap:wrap; }
  .mh-title { font-family:var(--serif); font-size:19px; font-weight:600; color:var(--ink); letter-spacing:-0.01em; white-space:nowrap; }
  .mh-title:hover { text-decoration:none; color:var(--accent); }
  .mh-title .mh-sub { display:block; font-family:var(--sans); font-size:11.5px; font-weight:400; color:var(--muted); letter-spacing:0.01em; margin-top:-1px; }
  nav.mh-nav { display:flex; gap:2px; margin-left:auto; flex-wrap:wrap; align-items:center; }
  nav.mh-nav a { color:var(--ink-2); font-size:14px; padding:6px 11px; border-radius:4px; }
  nav.mh-nav a:hover { background:var(--hover); color:var(--ink); text-decoration:none; }
  nav.mh-nav a.active { color:var(--accent); font-weight:600; box-shadow:inset 0 -2px 0 var(--accent); border-radius:0; }
  nav.mh-nav a.ext { color:var(--muted); }

  /* ---------- page layout ---------- */
  .wrap { max-width:var(--page-w); margin:0 auto; padding:0 28px; }
  .wrap.wide { max-width:none; }
  .table-scroll { overflow:auto; max-height:calc(100vh - 140px); border-top:1.5px solid var(--ink); border-bottom:1.5px solid var(--ink); background:var(--surface); }
  .table-scroll > table { border-top:0 !important; border-bottom:0 !important; }
  main.page { padding-bottom:48px; }
  .page-head { display:flex; gap:24px; align-items:flex-end; justify-content:space-between; flex-wrap:wrap;
               padding:26px 0 16px; }
  .page-head h1 { margin:0; font-size:28px; line-height:1.2; }
  .page-head .lede { margin:6px 0 0; color:var(--ink-2); max-width:72ch; font-size:15px; }
  .page-head .crumbs { font-size:13px; color:var(--muted); margin-bottom:4px; }
  .page-head .crumbs a { color:var(--muted); }
  .page-head .crumbs a:hover { color:var(--link); }
  .page-meta { color:var(--muted); font-size:13px; font-variant-numeric:tabular-nums; }
  .page-actions { display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
  footer.site { border-top:1px solid var(--rule); margin-top:32px; padding:18px 0 28px; color:var(--muted); font-size:13px; }

  /* ---------- cards & sections ---------- */
  .card { background:var(--surface); border:1px solid var(--rule); border-radius:var(--radius); padding:20px 24px; margin-bottom:18px; }
  .card > h2, .section-title { font-size:18px; margin:0 0 10px; text-transform:none; letter-spacing:0; color:var(--ink); }
  .card h3 { font-size:15px; margin:0 0 6px; }
  .help, .note { color:var(--muted); font-size:13px; line-height:1.5; }
  details > summary { cursor:pointer; color:var(--ink-2); }
  details.card > summary { font-family:var(--serif); font-size:18px; font-weight:600; color:var(--ink); list-style:none; }
  details.card > summary::-webkit-details-marker { display:none; }
  details.card > summary::before { content:"\25B8"; display:inline-block; width:18px; color:var(--muted); transition:transform .15s; }
  details.card[open] > summary::before { transform:rotate(90deg); }
  details.card[open] > summary { margin-bottom:12px; }

  /* ---------- controls ---------- */
  button, input, select, textarea { font-family:inherit; font-size:13.5px; color:var(--ink); }
  input[type=text], input[type=number], input[type=search], select, textarea {
    padding:6px 9px; border:1px solid #cfc9bc; border-radius:4px; background:var(--surface); }
  input[type=text]:focus, input[type=number]:focus, input[type=search]:focus, select:focus, textarea:focus {
    outline:none; border-color:var(--accent); box-shadow:0 0 0 3px var(--accent-soft); }
  input[type=checkbox] { accent-color:var(--accent); }
  .btn, button.primary, button.secondary, button.row-btn {
    display:inline-flex; align-items:center; gap:6px; padding:6px 13px; border-radius:4px; cursor:pointer;
    border:1px solid #cfc9bc; background:var(--surface); color:var(--ink); font-weight:500; line-height:1.3; white-space:nowrap; }
  .btn:hover, button.secondary:hover, button.row-btn:hover { background:var(--hover); text-decoration:none; }
  .btn-primary, button.primary { background:var(--accent); border-color:var(--accent); color:#fff; }
  .btn-primary:hover, button.primary:hover { background:var(--accent-2); }
  button.primary { padding:8px 20px; font-size:14px; }
  .btn:disabled, button:disabled { opacity:0.45; cursor:not-allowed; }
  .btn-danger, button.row-btn.danger { color:var(--neg); border-color:#e2c3bf; }
  .btn-danger:hover, button.row-btn.danger:hover { background:var(--err-bg); }
  .btn-quiet { border-color:transparent; background:transparent; color:var(--ink-2); }
  .btn-quiet:hover { background:var(--hover); }
  button.row-btn { padding:3px 10px; font-size:12px; }
  .btn-warn { background:var(--warn-bg); border-color:var(--warn-rule); color:var(--warn-ink); }

  /* segmented control */
  .seg { display:inline-flex; border:1px solid #cfc9bc; border-radius:5px; overflow:hidden; background:var(--surface); }
  .seg button { padding:6px 14px; background:transparent; border:0; border-left:1px solid var(--rule); cursor:pointer; color:var(--ink-2); font-size:13px; }
  .seg button:first-child { border-left:0; }
  .seg button:hover { background:var(--hover); }
  .seg button.active { background:var(--accent); color:#fff; }

  .toolbar { display:flex; gap:14px; align-items:center; flex-wrap:wrap; background:var(--surface);
             border:1px solid var(--rule); border-radius:var(--radius); padding:10px 14px; margin-bottom:14px; }
  .toolbar label { font-size:13px; color:var(--ink-2); display:inline-flex; gap:6px; align-items:center; }

  /* ---------- tables (booktabs-flavoured) ---------- */
  table { border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums; }
  table.data, .card table, main.page > table, .wrap > table {
    background:var(--surface); border-top:1.5px solid var(--ink); border-bottom:1.5px solid var(--ink); }
  th, td { padding:8px 10px; text-align:left; border-bottom:1px solid var(--rule-soft); vertical-align:top; font-size:13px; }
  th { background:var(--surface); color:var(--ink-2); font-weight:600; font-size:12px; letter-spacing:0.01em;
       text-transform:none; border-bottom:1px solid var(--ink); white-space:nowrap; }
  th[data-sort], th.sortable { cursor:pointer; user-select:none; }
  th[data-sort]:hover, th.sortable:hover { color:var(--accent); }
  tbody tr:nth-child(even) { background:transparent; }
  tbody tr:hover { background:var(--hover); }
  td.num, td.value { font-family:var(--mono); font-size:12.5px; text-align:right; white-space:nowrap; }
  .pos, td.num.pos, td.value.pos { color:var(--pos); }
  .neg, td.num.neg, td.value.neg { color:var(--neg); }

  /* ---------- pills & badges ---------- */
  .pill, .badge, .kind-speed, .kind-quality, .conf-pill, .conf-pill-cell, .tag {
    display:inline-block; padding:1px 8px; border-radius:999px; font-size:11.5px; font-weight:500;
    line-height:1.6; background:var(--surface-2); color:var(--ink-2); text-transform:none; letter-spacing:0; white-space:nowrap; }
  .kind-speed,   .pill.speed   { background:var(--speed-bg);   color:var(--speed); }
  .kind-quality, .pill.quality { background:var(--quality-bg); color:var(--quality); }
  .conf-pill.high,   .conf-pill-cell.high   { background:var(--ok-bg);   color:var(--ok-ink); }
  .conf-pill.medium, .conf-pill-cell.medium { background:var(--warn-bg); color:var(--warn-ink); }
  .conf-pill.low,    .conf-pill-cell.low    { background:var(--err-bg);  color:var(--err-ink); }
  .badge.observed { background:var(--accent-soft); color:var(--accent); }
  .badge.imputed  { background:var(--surface-2); color:var(--muted); }
  .tag.loo { background:var(--warn-bg); color:var(--warn-ink); font-weight:600; }

  /* ---------- callouts ---------- */
  .callout { border-left:3px solid var(--rule); background:var(--surface-2); padding:10px 14px; border-radius:0 4px 4px 0; font-size:13.5px; }
  .callout.warn { border-color:var(--warn-rule); background:var(--warn-bg); color:var(--warn-ink); }
  .callout.ok   { border-color:#a9cdb1; background:var(--ok-bg); color:var(--ok-ink); }
  .callout.err, .err { border:0; border-left:3px solid #d9a8a2; background:var(--err-bg); color:var(--err-ink);
                       padding:10px 14px; border-radius:0 4px 4px 0; font-size:13.5px; margin-top:10px; }

  /* ---------- stat tiles ---------- */
  .metric { background:var(--surface-2); border-radius:5px; padding:10px 14px; }
  .metric .k { color:var(--muted); font-size:12px; text-transform:none; letter-spacing:0; }
  .metric .v { font-family:var(--mono); font-size:18px; font-weight:500; color:var(--ink); margin-top:2px; }
  .metric .sub, .metric .p { color:var(--muted); font-size:11.5px; font-family:var(--sans); margin-top:2px; }

  /* ---------- modal & drawer ---------- */
  #backdrop { position:fixed; inset:0; background:rgba(29,31,35,0.28); opacity:0; pointer-events:none; transition:opacity .2s; z-index:90; }
  #backdrop.open { opacity:1; pointer-events:auto; }
  #drawer { position:fixed; top:0; right:0; width:min(900px, 62%); min-width:min(560px, 100%); height:100%; background:var(--surface);
            box-shadow:-8px 0 30px rgba(29,31,35,0.14); transform:translateX(100%); transition:transform .22s ease; overflow:auto; z-index:100; }
  #drawer.open { transform:translateX(0); }
  #drawer > header { position:sticky; top:0; z-index:2; background:var(--surface); border-bottom:1px solid var(--rule);
                     padding:10px 22px; display:flex; gap:12px; align-items:center; justify-content:space-between; }
  #drawer > header h1 { margin:0; font-family:var(--mono); font-size:12.5px; font-weight:400; color:var(--muted); }
  .modal-shade { position:fixed; inset:0; background:rgba(29,31,35,0.35); z-index:300; align-items:flex-start; justify-content:center; padding-top:8vh; }
  .modal { background:var(--surface); border-radius:8px; width:min(720px, 92vw); max-height:80vh; overflow:auto; box-shadow:0 18px 50px rgba(29,31,35,0.25); }
  .modal-head { padding:14px 22px; border-bottom:1px solid var(--rule); display:flex; justify-content:space-between; align-items:center; }
  .modal-head h2 { margin:0; font-size:18px; }
  .modal-body { padding:18px 22px 22px; }

  /* ---------- paper detail (drawer body) ---------- */
  .pd-meta { padding:20px 24px 16px; border-bottom:1px solid var(--rule); }
  .pd-meta h2 { margin:0 0 6px; font-size:22px; line-height:1.25; }
  .pd-meta .authors { color:var(--ink-2); font-size:14px; margin-bottom:8px; }
  .pd-meta .authors i { font-family:var(--serif); }
  .pd-meta .ids { color:var(--muted); font-size:12px; display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
  .pd-meta .ids .pill { font-family:var(--mono); font-size:11px; border-radius:3px; }
  .pd-effects { display:flex; gap:14px; padding:16px 24px; border-bottom:1px solid var(--rule); flex-wrap:wrap; }
  .pd-effect { flex:1 1 260px; border:1px solid var(--rule); border-radius:var(--radius); padding:12px 14px; background:var(--surface); }
  .pd-effect.empty { color:var(--muted); border-style:dashed; background:transparent; }
  .pd-effect h3 { margin:0 0 6px; font-size:15px; display:flex; justify-content:space-between; align-items:center; text-transform:capitalize; }
  .pd-effect .value { font-family:var(--mono); font-size:17px; margin-bottom:6px; }
  .pd-effect .row { display:flex; justify-content:space-between; gap:12px; font-size:12.5px; color:var(--ink-2); padding:3px 0; border-top:1px solid var(--rule-soft); }
  .pd-effect .row b { color:var(--muted); font-weight:500; }
  .pd-effect .notes { margin-top:8px; font-size:12.5px; color:var(--ink-2); line-height:1.5; }
  .pd-section { padding:6px 24px 16px; border-bottom:1px solid var(--rule-soft); }
  .pd-section h3 { margin:16px 0 8px; font-size:16px; text-transform:none; letter-spacing:0; color:var(--ink); }
  .pd-table { width:100%; border-collapse:collapse; font-size:13px; border:0 !important; }
  .pd-table th, .pd-table td { padding:5px 8px; border-bottom:1px solid var(--rule-soft); vertical-align:top; }
  .pd-table th { background:transparent; font-size:12px; color:var(--muted); border-bottom:1px solid var(--rule); }
  .pd-onet-rationale { background:var(--surface-2); border-left:3px solid var(--accent); padding:10px 14px; margin-top:8px; font-size:13.5px; line-height:1.55; border-radius:0 4px 4px 0; }
  .pd-onet-alt { font-size:12.5px; color:var(--ink-2); margin-top:8px; }
  .pd-onet-alt .pill { margin:3px 4px 0 0; font-family:var(--mono); font-size:11px; border-radius:3px; }
  .pd-quote { font-family:var(--serif); font-style:italic; border-left:2px solid var(--rule); padding:4px 12px; margin:6px 0; font-size:14px; line-height:1.55; color:var(--ink-2); background:transparent; }
  .pd-quotes-block { max-height:320px; overflow-y:auto; }
  .pd-quotes-block summary { padding:4px 0; text-transform:capitalize; }
  .pd-stat { border:1px solid var(--rule-soft); border-radius:4px; padding:8px 12px; margin:6px 0; font-size:12.5px; }
  .pd-stat .head { display:flex; justify-content:space-between; gap:12px; font-size:13px; margin-bottom:2px; }
  .pd-stat .head b { color:var(--ink); }
  .pd-stat .nums { font-family:var(--mono); font-size:11.5px; color:var(--ink-2); margin:2px 0; }
  .pd-stat .vq { font-family:var(--serif); font-size:13px; color:var(--muted); font-style:italic; }
  details.pd-raw { margin:12px 24px 24px; }
  details.pd-raw summary { font-size:13px; color:var(--muted); padding:4px 0; }
  details.pd-raw pre { background:var(--surface-2); padding:10px; font-size:11px; overflow:auto; max-height:320px; border-radius:4px; }

  /* ---------- charts ---------- */
  .chart-legend { display:flex; gap:18px; flex-wrap:wrap; font-size:12px; color:var(--ink-2); margin-top:8px; }
  .chart-legend .sw { display:inline-block; width:12px; height:10px; border-radius:2px; vertical-align:middle; margin-right:5px; }

  @media (max-width: 720px) {
    .mh-inner, .wrap { padding-left:16px; padding-right:16px; }
    nav.mh-nav { margin-left:0; }
    .page-head h1 { font-size:23px; }
    #drawer { width:100%; min-width:0; }
  }
"""


def masthead(items: list[tuple[str, str, str]], active: str, home_href: str,
             external: list[tuple[str, str]] | None = None) -> str:
    """items = [(href, label, key)]; external = [(href, label)] opened in a new tab."""
    links = []
    for href, label, key in items:
        cls = ' class="active" aria-current="page"' if key == active else ""
        links.append(f'<a href="{escape(href)}"{cls}>{escape(label)}</a>')
    for href, label in external or []:
        links.append(f'<a class="ext" href="{escape(href)}" target="_blank" rel="noopener">{escape(label)} &#8599;</a>')
    return f"""<header class="masthead"><div class="mh-inner">
  <a class="mh-title" href="{escape(home_href)}">AI Impact Meta-Review
    <span class="mh-sub">Effects of generative AI on work, mapped to O*NET</span></a>
  <nav class="mh-nav">{"".join(links)}</nav>
</div></header>"""


def apply(html: str, masthead_html: str) -> str:
    """Fill the __THEME_HEAD__ and __MASTHEAD__ placeholders in a page template."""
    head = f'<meta name="viewport" content="width=device-width, initial-scale=1">{FONTS_LINK}<style>{THEME_CSS}</style>'
    return html.replace("__THEME_HEAD__", head).replace("__MASTHEAD__", masthead_html)
