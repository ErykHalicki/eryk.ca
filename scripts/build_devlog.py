#!/usr/bin/env python3
import markdown2
import re
import os
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
TEMPLATE_PATH = REPO_ROOT / 'templates' / 'devlog_template.html'
MARKDOWN_SOURCE = REPO_ROOT / 'temp' / 'zima_devlog' / 'Projects' / 'Zima' / 'Devlog.md'
OUTPUT_PATH = REPO_ROOT / 'content' / 'zima_devlog.html'
IMAGE_SOURCE_DIR = REPO_ROOT / 'temp' / 'zima_devlog' / 'Projects' / 'Zima'
IMAGE_DEST_DIR = REPO_ROOT / 'content' / 'assets' / 'images' / 'zima'

def strip_frontmatter(markdown_text):
    frontmatter_pattern = r'^---\n.*?\n---\n'
    return re.sub(frontmatter_pattern, '', markdown_text, flags=re.DOTALL)

def convert_youtube_links(markdown_text):
    youtube_patterns = [
        r'https://youtu\.be/([a-zA-Z0-9_-]+)(\?t=(\d+))?',
        r'https://www\.youtube\.com/watch\?v=([a-zA-Z0-9_-]+)(&t=(\d+)s?)?'
    ]

    for pattern in youtube_patterns:
        def youtube_replacer(match):
            video_id = match.group(1)
            timestamp = match.group(3) if len(match.groups()) >= 3 and match.group(3) else None

            embed_url = f'https://www.youtube-nocookie.com/embed/{video_id}'
            params = []
            if timestamp:
                params.append(f'start={timestamp}')

            if params:
                embed_url += '?' + '&'.join(params)

            return f'\n\n<iframe width="560" height="315" src="{embed_url}" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen class="devlog-youtube"></iframe>\n\n'

        markdown_text = re.sub(pattern, youtube_replacer, markdown_text)

    return markdown_text

def convert_obsidian_embeds(markdown_text, image_dest_path='/assets/images/zima/'):
    image_pattern = r'!\[\[([^\]]+\.(png|jpg|jpeg|gif|webp|svg|PNG|JPG|JPEG|GIF|WEBP|SVG))\]\]'

    def image_replacer(match):
        filename = match.group(1)
        clean_filename = os.path.basename(filename)
        alt_text = os.path.splitext(clean_filename)[0].replace('-', ' ').replace('_', ' ')
        return f'<img src="{image_dest_path}{clean_filename}" alt="{alt_text}" class="devlog-image">'

    markdown_text = re.sub(image_pattern, image_replacer, markdown_text, flags=re.IGNORECASE)

    video_pattern = r'!\[\[([^\]]+\.(mov|mp4|webm|avi|MOV|MP4|WEBM|AVI))\]\]'

    def video_replacer(match):
        filename = match.group(1)
        clean_filename = os.path.basename(filename)
        web_filename = clean_filename
        if clean_filename.lower().endswith('.mov'):
            web_filename = os.path.splitext(clean_filename)[0] + '.mp4'

        return f'''<video controls class="devlog-video">
    <source src="{image_dest_path}{web_filename}" type="video/mp4">
    Your browser does not support the video tag.
</video>'''

    markdown_text = re.sub(video_pattern, video_replacer, markdown_text, flags=re.IGNORECASE)

    return markdown_text

def convert_obsidian_links(markdown_text):
    link_pattern = r'(?<!!)\[\[([^\]|]+)(?:\|([^\]]+))?\]\]'

    def link_replacer(match):
        target = match.group(1)
        display = match.group(2) if match.group(2) else target
        return display

    return re.sub(link_pattern, link_replacer, markdown_text)

def copy_media_files(source_dir, dest_dir, markdown_text):
    dest_dir.mkdir(parents=True, exist_ok=True)

    media_pattern = r'!\[\[([^\]]+)\]\]'
    referenced_files = re.findall(media_pattern, markdown_text)

    copied_files = []
    for filename in referenced_files:
        clean_filename = os.path.basename(filename)

        if clean_filename.lower().endswith('.mov'):
            mp4_filename = os.path.splitext(clean_filename)[0] + '.mp4'
            search_paths = [
                source_dir / mp4_filename,
                source_dir / 'images' / mp4_filename,
                source_dir / clean_filename,
                source_dir / 'images' / clean_filename,
            ]
        else:
            search_paths = [
                source_dir / clean_filename,
                source_dir / 'images' / clean_filename,
            ]

        source_file = None
        actual_filename = clean_filename
        for path in search_paths:
            if path.exists():
                source_file = path
                actual_filename = path.name
                break

        if source_file:
            dest_file = dest_dir / actual_filename
            import shutil
            shutil.copy2(source_file, dest_file)
            copied_files.append(actual_filename)
            print(f"Copied: {actual_filename}")
        else:
            print(f"WARNING: Referenced file not found: {filename}")

    return copied_files

def build_devlog():
    print("Starting devlog build...")

    if not MARKDOWN_SOURCE.exists():
        raise FileNotFoundError(f"Markdown source not found: {MARKDOWN_SOURCE}")

    with open(MARKDOWN_SOURCE, 'r', encoding='utf-8') as f:
        markdown_content = f.read()

    print("\nCopying media files...")
    copy_media_files(IMAGE_SOURCE_DIR, IMAGE_DEST_DIR, markdown_content)

    print("\nConverting Obsidian syntax...")
    markdown_content = strip_frontmatter(markdown_content)
    markdown_content = convert_obsidian_links(markdown_content)
    markdown_content = convert_youtube_links(markdown_content)
    markdown_content = convert_obsidian_embeds(markdown_content)

    print("\nConverting markdown to HTML...")
    html_content = markdown2.markdown(
        markdown_content,
        extras=[
            'fenced-code-blocks',
            'tables',
            'code-friendly',
            'header-ids',
            'break-on-newline',
        ]
    )

    print("\nPost-processing HTML for any remaining Obsidian syntax...")
    html_content = convert_obsidian_embeds(html_content)

    print("\nAdding horizontal rules before headings...")
    html_content = re.sub(r'(<h1[^>]*>)', r'<hr class="heading-separator" />\n\1', html_content)
    html_content = re.sub(r'(<h2[^>]*>)', r'<hr class="heading-separator" />\n\1', html_content)
    html_content = re.sub(r'^<hr class="heading-separator" />\n', '', html_content)

    print("\nCleaning up HTML structure around iframes...")
    def convert_ul_to_br(html_text):
        html_text = html_text.replace('<ul>\n<li>', '<br />\n- ')
        html_text = html_text.replace('</li>\n<li>', '<br />\n- ')
        html_text = html_text.replace('</li>\n</ul>', '<br />')
        html_text = html_text.replace('<ul>', '<br />')
        html_text = html_text.replace('</ul>', '')
        html_text = html_text.replace('<li>', '- ')
        html_text = html_text.replace('</li>', '<br />')
        return html_text

    html_content = re.sub(
        r'(</iframe>)\s*<ul>.*?(?=<h[12]|<p><strong>|$)',
        lambda m: m.group(1) + '\n\n<p>' + convert_ul_to_br(m.group(0)[len(m.group(1)):]) + '</p>',
        html_content,
        flags=re.DOTALL
    )

    if not TEMPLATE_PATH.exists():
        raise FileNotFoundError(f"Template not found: {TEMPLATE_PATH}")

    with open(TEMPLATE_PATH, 'r', encoding='utf-8') as f:
        template = f.read()

    final_html = template.replace('{{DEVLOG_CONTENT}}', html_content)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        f.write(final_html)

    print(f"\nDevlog built successfully: {OUTPUT_PATH}")
    print(f"File size: {OUTPUT_PATH.stat().st_size} bytes")

if __name__ == '__main__':
    try:
        build_devlog()
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
