#!/usr/bin/env python3
"""Build the devlog and the blog from a notes-style source tree.

Source layout (default: temp/notes/devlog, override with DEVLOG_SRC):
    <project>/project.yml         name / published / order / blurb / thumbnail
    <project>/<YYYY-MM-DD>-<slug>.md   frontmatter: title, project, date, published[, time_spent, thumbnail]
    <project>/media/*             images + mp4

Output:
    content/devlog/index.html                unified feed, newest first
    content/devlog/<project>/index.html      per-project feed
    content/devlog/<project>/<slug>.html     entry page
    content/assets/devlog/<project>/*        copied media (skipped silently if absent)

The blog is one more project directory in the same vault
(default: <DEVLOG_SRC>/../inbox/blog, override with BLOG_SRC), with the same layout:
    project.yml                       optional; name / published / blurb
    <YYYY-MM-DD>-<slug>.md            frontmatter: title, date, published
    media/*

    content/blog/index.html                  blog feed, newest first
    content/blog/<slug>.html                 post page
    content/assets/devlog/_blog/*            copied media

A blog post is emitted when its own frontmatter is published: true, unless a
project.yml is present and says published: false.

An entry is emitted only when its project.yml and its own frontmatter are both published: true.
"""
import os
import re
import html
import shutil
from pathlib import Path

import markdown2

REPO_ROOT = Path(__file__).parent.parent
SRC = Path(os.environ.get("DEVLOG_SRC", REPO_ROOT / "temp" / "notes" / "devlog"))
TEMPLATE = (REPO_ROOT / "templates" / "devlog_base.html").read_text(encoding="utf-8")
OUT_ROOT = REPO_ROOT / "content" / "devlog"
ASSET_ROOT = REPO_ROOT / "content" / "assets" / "devlog"
BLOG_SRC = Path(os.environ.get("BLOG_SRC", SRC.parent / "inbox" / "blog"))
BLOG_ASSET_SLUG = "_blog"
BLOG_OUT = REPO_ROOT / "content" / "blog"

VIDEO_EXT = {".mp4", ".mov", ".avi", ".webm", ".mkv"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}


# --------------------------------------------------------------------------- io
def parse_frontmatter(text):
    """Minimal `key: value` frontmatter. Returns (meta dict, body str)."""
    if not text.startswith("---"):
        return {}, text
    _, fm, body = text.split("---", 2)
    meta = {}
    for line in fm.strip().splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        v = v.strip()
        low = v.lower()
        if low in ("true", "false"):
            v = low == "true"
        meta[k.strip()] = v
    return meta, body.lstrip("\n")


def load_project(pdir):
    meta = {"name": pdir.name, "published": False, "order": 999, "blurb": ""}
    cfg = pdir / "project.yml"
    if cfg.exists():
        m, _ = parse_frontmatter("---\n" + cfg.read_text(encoding="utf-8").strip() + "\n---\n")
        meta.update(m)
    try:
        meta["order"] = int(meta["order"])
    except (ValueError, TypeError):
        meta["order"] = 999
    return meta


# --------------------------------------------------------------- markdown build
def convert_youtube(md):
    patterns = [
        r'https://youtu\.be/([a-zA-Z0-9_-]+)(?:\?t=(\d+))?',
        r'https://www\.youtube\.com/watch\?v=([a-zA-Z0-9_-]+)(?:&t=(\d+)s?)?',
    ]
    for pat in patterns:
        def repl(m):
            vid, ts = m.group(1), m.group(2)
            url = f"https://www.youtube-nocookie.com/embed/{vid}"
            if ts:
                url += f"?start={ts}"
            return (f'\n\n<iframe src="{url}" title="YouTube video" frameborder="0" '
                    f'allow="accelerometer; autoplay; clipboard-write; encrypted-media; '
                    f'gyroscope; picture-in-picture; web-share" allowfullscreen '
                    f'class="devlog-youtube"></iframe>\n\n')
        md = re.sub(pat, repl, md)
    return md


def media_refs(md):
    """All `media/<file>` targets referenced by image syntax."""
    return re.findall(r'!\[[^\]]*\]\(\s*media/([^)\s]+)\s*\)', md)


def render_body(md, media_dir, asset_url_base):
    md = convert_youtube(md)
    html_out = markdown2.markdown(
        md,
        extras=["fenced-code-blocks", "tables", "code-friendly", "header-ids", "break-on-newline"],
    )

    def fix_img(m):
        attrs, src = m.group(0), m.group("src")
        name = src[len("media/"):]
        stem, ext = os.path.splitext(name)
        ext = ext.lower()
        alt = ""
        am = re.search(r'alt="([^"]*)"', attrs)
        if am:
            alt = am.group(1)
        if ext in VIDEO_EXT:
            web = f"{asset_url_base}/{stem}.mp4"
            if not (media_dir / name).exists() and not (media_dir / f"{stem}.mp4").exists():
                return f"<!-- media omitted: {name} -->"
            return (f'<video controls playsinline class="devlog-video">'
                    f'<source src="{web}" type="video/mp4"></video>')
        if not (media_dir / name).exists():
            return f"<!-- media omitted: {name} -->"
        return f'<img class="devlog-image" src="{asset_url_base}/{name}" alt="{html.escape(alt)}">'

    html_out = re.sub(r'<img\b[^>]*\bsrc="(?P<src>media/[^"]+)"[^>]*>', fix_img, html_out)
    return html_out


def excerpt(md, limit=180):
    text = re.sub(r'!\[[^\]]*\]\([^)]*\)', '', md)          # images
    text = re.sub(r'\[([^\]]+)\]\([^)]*\)', r'\1', text)    # links -> text
    text = re.sub(r'[*_`#>-]', '', text)
    text = re.sub(r'^\s*Goal:\s*', '', text, flags=re.I)    # drop the leading Goal: label
    text = re.sub(r'\s+', ' ', text).strip()
    return (text[:limit].rstrip() + "…") if len(text) > limit else text


def pick_thumb(meta, md, media_dir):
    cand = meta.get("thumbnail")
    names = ([cand] if cand else []) + media_refs(md)
    for n in names:
        n = n[len("media/"):] if n.startswith("media/") else n
        if os.path.splitext(n)[1].lower() in IMAGE_EXT and (media_dir / n).exists():
            return n
    return None


# --------------------------------------------------------------------- assemble
def page(title, body):
    return TEMPLATE.replace("{{TITLE}}", html.escape(title)).replace("{{BODY}}", body)


def chip_bar(projects, active):
    chips = [f'<a class="devlog-chip{" active" if active is None else ""}" href="/devlog/">all</a>']
    for p in projects:
        cls = " active" if p["slug"] == active else ""
        chips.append(f'<a class="devlog-chip{cls}" href="/devlog/{p["slug"]}/">{html.escape(p["name"])}</a>')
    return '<div class="devlog-chips">' + "".join(chips) + "</div>"


def feed_html(entries, show_project=True):
    rows = []
    for e in entries:
        thumb = (f'<img class="devlog-feed-thumb" src="/assets/devlog/{e["slug_proj"]}/{e["thumb"]}" alt="">'
                 if e["thumb"] else '')
        project = (f'<span class="devlog-feed-project">{html.escape(e["project_name"])}</span>'
                   if show_project else '')
        rows.append(
            f'<a class="devlog-feed-item" href="{e["url"]}">'
            f'{thumb}'
            f'<span class="devlog-feed-meta"><span class="devlog-feed-date">{e["date"]}</span>'
            f'{project}</span>'
            f'<span class="devlog-feed-title">{html.escape(e["title"])}</span>'
            f'<span class="devlog-feed-excerpt">{html.escape(e["excerpt"])}</span>'
            f'</a>'
        )
    return '<div class="devlog-feed">' + "".join(rows) + "</div>"


def build():
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    OUT_ROOT.mkdir(parents=True)
    if BLOG_OUT.exists():
        shutil.rmtree(BLOG_OUT)
    BLOG_OUT.mkdir(parents=True)
    if ASSET_ROOT.exists():
        shutil.rmtree(ASSET_ROOT)

    if not SRC.exists():
        raise SystemExit(f"devlog source not found: {SRC}")

    project_dirs = sorted(d for d in SRC.iterdir() if d.is_dir() and (d / "project.yml").exists())
    projects, all_entries, blog_entries = [], [], []
    blog_meta = {"name": "Blog", "blurb": ""}
    sources = [(d, False) for d in project_dirs]
    if BLOG_SRC.is_dir():
        sources.append((BLOG_SRC, True))

    for pdir, is_blog in sources:
        pmeta = load_project(pdir)
        slug = BLOG_ASSET_SLUG if is_blog else pdir.name
        if is_blog and not (pdir / "project.yml").exists():
            pmeta["published"] = True
        media_dir = pdir / "media"
        asset_base = f"/assets/devlog/{slug}"
        proj_pub = bool(pmeta["published"])
        if is_blog:
            blog_meta = pmeta
        proj = {"slug": slug, "name": pmeta["name"], "order": pmeta["order"],
                "blurb": pmeta["blurb"], "published": proj_pub, "entries": []}

        for md_file in sorted(pdir.glob("*.md")):
            meta, body = parse_frontmatter(md_file.read_text(encoding="utf-8"))
            if not (proj_pub and bool(meta.get("published", False))):
                continue
            date = str(meta.get("date", md_file.name[:10]))
            title = str(meta.get("title", md_file.stem))
            out_slug = md_file.stem
            entry = {
                "slug_proj": slug, "project_name": pmeta["name"], "project_slug": slug,
                "date": date, "title": title, "out_slug": out_slug,
                "is_blog": is_blog,
                "url": f"/blog/{out_slug}.html" if is_blog else f"/devlog/{slug}/{out_slug}.html",
                "excerpt": excerpt(body),
                "thumb": pick_thumb(meta, body, media_dir),
                "time_spent": meta.get("time_spent", ""),
                "body_md": body, "media_dir": media_dir, "asset_base": asset_base,
            }
            if is_blog:
                blog_entries.append(entry)
            else:
                proj["entries"].append(entry)
                all_entries.append(entry)

        if proj["entries"]:
            projects.append(proj)

    projects.sort(key=lambda p: (p["order"], p["name"].lower()))
    sort_key = lambda e: (e["date"], e["out_slug"])
    all_entries.sort(key=sort_key, reverse=True)
    blog_entries.sort(key=sort_key, reverse=True)

    # copy referenced media that exists
    copied = 0
    for e in all_entries + blog_entries:
        dest = ASSET_ROOT / e["slug_proj"]
        for name in set(media_refs(e["body_md"])) | ({e["thumb"]} if e["thumb"] else set()):
            src = e["media_dir"] / name
            if src.exists():
                dest.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest / name)
                copied += 1

    # entry pages
    for e in all_entries + blog_entries:
        content = render_body(e["body_md"], e["media_dir"], e["asset_base"])
        if e["is_blog"]:
            sub = e["date"]
            backlinks = '<a class="hoverable" href="/blog/">&larr; all posts</a>'
            out = BLOG_OUT / f'{e["out_slug"]}.html'
            section = "Blog"
        else:
            sub = f'<a class="hoverable" href="/devlog/{e["project_slug"]}/">{html.escape(e["project_name"])}</a> &middot; {e["date"]}'
            backlinks = (
                f'<a class="hoverable" href="/devlog/{e["project_slug"]}/">&larr; more {html.escape(e["project_name"])}</a>'
                f' &nbsp;&nbsp; <a class="hoverable" href="/devlog/">all devlogs</a>'
            )
            out = OUT_ROOT / e["project_slug"] / f'{e["out_slug"]}.html'
            section = "Devlog"
        if e["time_spent"]:
            sub += f' &middot; {html.escape(str(e["time_spent"]))}'
        body = (
            f'<div class="devlog-header"><h1>{html.escape(e["title"])}</h1>'
            f'<p class="devlog-subtitle">{sub}</p></div>\n'
            f'<div class="devlog-content">\n{content}\n</div>\n'
            f'<p class="devlog-backlinks">{backlinks}</p>'
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(page(f'{e["title"]} — {section}', body), encoding="utf-8")

    # blog feed
    blurb = f'<p class="devlog-subtitle">{html.escape(blog_meta["blurb"])}</p>' if blog_meta["blurb"] else ""
    posts = (feed_html(blog_entries, show_project=False) if blog_entries
             else '<p class="devlog-subtitle">No posts yet.</p>')
    body = f'<div class="devlog-header"><h1>Blog</h1>{blurb}</div>\n' + posts
    (BLOG_OUT / "index.html").write_text(page("Blog — Eryk Halicki", body), encoding="utf-8")

    # main feed
    body = (
        '<div class="devlog-header"><h1>Devlog</h1>'
        '<p class="devlog-subtitle">behind-the-scenes on my projects</p></div>\n'
        + chip_bar(projects, None) + "\n" + feed_html(all_entries)
    )
    (OUT_ROOT / "index.html").write_text(page("Devlog — Eryk Halicki", body), encoding="utf-8")

    # per-project feeds
    for p in projects:
        ents = sorted(p["entries"], key=sort_key, reverse=True)
        blurb = f'<p class="devlog-subtitle">{html.escape(p["blurb"])}</p>' if p["blurb"] else ""
        body = (
            f'<div class="devlog-header"><h1>{html.escape(p["name"])}</h1>{blurb}</div>\n'
            + chip_bar(projects, p["slug"]) + "\n" + feed_html(ents)
        )
        (OUT_ROOT / p["slug"] / "index.html").write_text(
            page(f'{p["name"]} — Devlog', body), encoding="utf-8")

    # legacy redirect
    (REPO_ROOT / "content" / "zima_devlog.html").write_text(
        '<!DOCTYPE html><meta charset="utf-8">'
        '<meta http-equiv="refresh" content="0; url=/devlog/zima/">'
        '<link rel="canonical" href="/devlog/zima/">', encoding="utf-8")

    print(f"projects: {[p['slug'] for p in projects]}")
    print(f"entries: {len(all_entries)}  blog posts: {len(blog_entries)}  media files copied: {copied}")


if __name__ == "__main__":
    build()
