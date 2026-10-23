"""Scrape a public Instagram profile (counters, biography, posts, likes and comments) with Playwright."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

from dotenv import load_dotenv
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import Page, sync_playwright

BASE_URL = "https://www.instagram.com"
DATA_DIR = Path(__file__).parent / "data"
PROFILE_DIR = Path(__file__).parent / "browser_profile"
POST_SELECTOR = 'a[href*="/p/"], a[href*="/reel/"]'
FOLLOWERS = "followers|seguidores"
FOLLOWING = "following|seguidos|siguiendo"
POSTS = "posts|publicaciones"
LIKES = "likes?|me gusta"
COMMENTS = "comments?|comentarios"
COMMENT_ITEM_SELECTOR = "article ul li, main ul li"
LOAD_MORE_SELECTOR = 'svg[aria-label="Load more comments"], svg[aria-label="Cargar más comentarios"]'
MENTION = re.compile(r"@[\w.]+")
MAX_STALLED_SCROLLS = 6
MAX_LOAD_MORE_CLICKS = 5
MAX_COMMENTS = 20
DEFAULT_DETAILS = 10
LOGIN_TIMEOUT = 180
READ_LINKS_JS = """els => els.map(e => ({
    href: e.getAttribute('href'),
    alt: (e.querySelector('img') || {}).alt || ''
}))"""
READ_COMMENTS_JS = r"""selector => {
    const profileLink = /^\/[A-Za-z0-9._]+\/$/;
    const uiLabel = /^(reply|responder|view replies.*|ver respuestas.*|\d+\s*(likes?|me gusta)|\d+\s*[smhdw]|\d+\s*(sem|min|h|d)\.?)$/i;
    return [...document.querySelectorAll(selector)].flatMap(li => {
        const author = [...li.querySelectorAll('a')].find(a => profileLink.test(a.getAttribute('href') || ''));
        if (!author) return [];
        const name = author.innerText.trim();
        const text = [...li.querySelectorAll('span[dir="auto"]')]
            .filter(span => !span.closest('a, h3, button'))
            .map(span => span.innerText.trim())
            .find(t => t && t !== name && !uiLabel.test(t));
        return text ? [{author: name, text}] : [];
    });
}"""

log = logging.getLogger("scraper")


def build_session_cookie(session_id: str) -> dict:
    """Build the `sessionid` cookie that tells Instagram we are a logged-in user.

    `domain` (with the leading dot) makes the browser send it to every instagram.com
    subdomain, `path` scopes it to the whole site, `httpOnly` hides it from page
    JavaScript and `secure` restricts it to HTTPS. These are the same columns shown in
    the browser's DevTools > Application > Cookies table.
    """
    return {
        "name": "sessionid",
        "value": session_id,
        "domain": ".instagram.com",
        "path": "/",
        "httpOnly": True,
        "secure": True,
        "sameSite": "Lax",
    }


def human_pause(low: float = 1.5, high: float = 4.0) -> None:
    """Sleep for a random time so the request rhythm does not look automated."""
    time.sleep(random.uniform(low, high))


def wait_for_selector(page: Page, selector: str, timeout: float = 15) -> bool:
    """Poll the DOM every 0.5 s until `selector` exists; return False on timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if page.query_selector(selector):
            return True
        time.sleep(0.5)
    return False


def og_description(page: Page) -> str:
    """Return the `og:description` meta tag (the text Instagram uses for link previews)."""
    tag = page.query_selector('meta[property="og:description"]')
    return (tag.get_attribute("content") if tag else "") or ""


def parse_count(text: str | None) -> int | None:
    """Convert '1,234', '1.234', '12.4K' or '1,2 M' into an integer; None if it is not a count."""
    match = re.fullmatch(r"([\d.,]+)\s*([KMkm]?)", (text or "").strip())
    if not match:
        return None
    number, suffix = match.groups()
    try:
        if suffix:
            return round(float(number.replace(",", ".")) * {"k": 1_000, "m": 1_000_000}[suffix.lower()])
        return int(re.sub(r"[.,]", "", number))
    except ValueError:
        return None


def extract_count(text: str, labels: str) -> str | None:
    """Return the figure that precedes one of the `labels` (e.g. '1.2M' in '1.2M Followers')."""
    match = re.search(rf"([\d.,]+\s*[KMkm]?)\s*(?:{labels})", text, re.I)
    return match.group(1).strip() if match else None


def parse_profile(page: Page, username: str) -> dict:
    """Read followers, following, post count and biography from the profile header.

    Counters come from the `og:description` meta tag, which is stable and language-aware.
    The biography is whatever the header shows after the 'N following' line, so it may
    start with the display name.
    """
    meta = og_description(page)
    lines = [line.strip() for line in page.locator("header").first.inner_text().splitlines()]
    counter = re.compile(rf"[\d.,]+\s*[KMkm]?\s*(?:{FOLLOWING})", re.I)
    start = next((i for i, line in enumerate(lines) if counter.match(line)), None)
    biography = "\n".join(line for line in lines[start + 1:] if line) if start is not None else ""
    return {
        "username": username,
        "followers": extract_count(meta, FOLLOWERS),
        "following": extract_count(meta, FOLLOWING),
        "posts_count": extract_count(meta, POSTS),
        "biography": biography,
        "scraped_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def collect_posts(
    page: Page, target: int, on_progress: Callable[[str, int, int], None] | None = None
) -> list[dict]:
    """Scroll the grid and gather post links until `target` is reached or no new ones load.

    Instagram removes off-screen posts from the DOM, so links are read after every scroll
    and de-duplicated by URL instead of being collected once at the end.
    `on_progress(stage, done, total)` reports how many posts were collected so far.
    """
    posts: dict[str, dict] = {}
    stalled = 0
    while True:
        known = len(posts)
        for item in page.eval_on_selector_all(POST_SELECTOR, READ_LINKS_JS):
            url = urljoin(BASE_URL, item["href"])
            posts.setdefault(url, {
                "url": url,
                "type": "reel" if "/reel/" in url else "post",
                "description": item["alt"].strip(),
            })
        stalled = stalled + 1 if len(posts) == known else 0
        log.info("Collected %d/%d posts", len(posts), target)
        if on_progress:
            on_progress("posts", len(posts), target)
        if len(posts) >= target or stalled >= MAX_STALLED_SCROLLS:
            return list(posts.values())
        page.evaluate("window.scrollBy(0, window.innerHeight * 0.9)")
        human_pause(2.0, 4.5)


def parse_post_meta(meta: str) -> dict:
    """Extract likes, comment count and caption from a post's `og:description` text.

    The text looks like '1,234 likes, 56 comments - user on October 1, 2025: "caption"'.
    Counts are read only from the part before ' - ' so a caption cannot fake them, and
    @mentions inside the caption are replaced because they name third parties.
    """
    counters = meta.split(" - ", 1)[0]
    caption = re.search(r':\s*["“](.*)["”]\.?\s*$', meta, re.S)
    return {
        "likes": parse_count(extract_count(counters, LIKES)),
        "comments_count": parse_count(extract_count(counters, COMMENTS)),
        "caption": MENTION.sub("@user", caption.group(1).strip()) if caption else "",
    }


def pseudonymize(author: str, owner: str, aliases: dict[str, str]) -> str:
    """Map a username to 'owner' or a stable 'user_N' label so no third-party names are stored.

    `aliases` lives only in memory during one run: the mapping is never written to disk,
    so the labels cannot be reversed afterwards.
    """
    if author == owner.lower():
        return "owner"
    return aliases.setdefault(author, f"user_{len(aliases) + 1}")


def clean_comments(
    raw: list[dict], owner: str, caption: str, aliases: dict[str, str], limit: int
) -> list[dict]:
    """Drop duplicates and the caption entry, scrub @mentions and pseudonymize every author."""
    seen: set[tuple[str, str]] = set()
    comments = []
    for item in raw:
        author, text = item["author"].lower(), MENTION.sub("@user", item["text"])
        is_caption = author == owner.lower() and caption.startswith(text[:40])
        if is_caption or (author, text) in seen:
            continue
        seen.add((author, text))
        comments.append({"author": pseudonymize(author, owner, aliases), "text": text})
    return comments[:limit]


def read_comments(page: Page, limit: int) -> list[dict]:
    """Click 'load more' until more than `limit` entries are visible, then read author and text."""
    for _ in range(MAX_LOAD_MORE_CLICKS):
        button = page.query_selector(LOAD_MORE_SELECTOR)
        if not button or len(page.evaluate(READ_COMMENTS_JS, COMMENT_ITEM_SELECTOR)) > limit:
            break
        try:
            button.click()
        except PlaywrightError:
            break
        human_pause(1.0, 2.5)
    return page.evaluate(READ_COMMENTS_JS, COMMENT_ITEM_SELECTOR)


def scrape_post_details(
    page: Page, url: str, owner: str, aliases: dict[str, str], limit: int = MAX_COMMENTS
) -> dict:
    """Open one post and return its caption, likes, comment count, date and anonymized comments.

    Likes come from `og:description`, with the 'liked by' link as fallback. They stay None
    when the owner hides them. If the page reports comments but none could be read, the
    selectors need adjusting and a screenshot is saved to debug_comments.png.
    """
    page.goto(url, wait_until="domcontentloaded")
    human_pause(2, 4)
    details = parse_post_meta(og_description(page))

    likes_link = page.query_selector('a[href$="/liked_by/"]')
    if details["likes"] is None and likes_link:
        details["likes"] = parse_count(extract_count(likes_link.inner_text(), LIKES))
    time_tag = page.query_selector("time[datetime]")
    details["posted_at"] = time_tag.get_attribute("datetime") if time_tag else None

    details["comments"] = []
    if wait_for_selector(page, COMMENT_ITEM_SELECTOR, timeout=8):
        raw = read_comments(page, limit)
        details["comments"] = clean_comments(raw, owner, details["caption"], aliases, limit)
    if details["comments_count"] and not details["comments"]:
        log.warning("No comments could be read for %s (see debug_comments.png)", url)
        page.screenshot(path="debug_comments.png")
    return details


def read_details(
    page: Page,
    posts: list[dict],
    owner: str,
    count: int,
    on_progress: Callable[[str, int, int], None] | None = None,
) -> None:
    """Enrich the first `count` posts in place; a post that fails to load is skipped."""
    selected = posts[:count]
    aliases: dict[str, str] = {}
    if on_progress:
        on_progress("details", 0, len(selected))
    for index, post in enumerate(selected, start=1):
        try:
            post.update(scrape_post_details(page, post["url"], owner, aliases))
        except PlaywrightError as error:
            log.warning("Skipping %s: %s", post["url"], error)
        log.info("Read details %d/%d", index, len(selected))
        if on_progress:
            on_progress("details", index, len(selected))
        human_pause(3, 6)


def wait_for_login(page: Page, timeout: float = LOGIN_TIMEOUT) -> bool:
    """Wait while the user logs in by hand; return False if the login page is still open at timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if "accounts/login" not in page.url:
            return True
        time.sleep(1)
    return False


def open_profile(page: Page, username: str) -> bool:
    """Navigate to the profile; return False if Instagram redirected to the login page."""
    page.goto(f"{BASE_URL}/{username}/", wait_until="domcontentloaded")
    human_pause(2, 4)
    return "accounts/login" not in page.url


def scrape(
    username: str,
    target: int,
    headless: bool,
    details: int = DEFAULT_DETAILS,
    on_progress: Callable[[str, int, int], None] | None = None,
    on_status: Callable[[str], None] | None = None,
) -> tuple[dict, list[dict]]:
    """Open the profile in a persistent browser profile and return (profile, posts).

    The first `details` posts are opened one by one to read caption, likes and comments.

    The session lives in `browser_profile/`, so it survives between runs. The optional
    INSTAGRAM_SESSIONID cookie is injected first; if Instagram still asks for a login, a
    visible window lets the user sign in by hand and the scraper continues on its own.
    """
    def status(message: str) -> None:
        log.info(message)
        if on_status:
            on_status(message)

    session_id = os.environ.get("INSTAGRAM_SESSIONID")
    with sync_playwright() as playwright:
        context = playwright.chromium.launch_persistent_context(
            PROFILE_DIR, headless=headless, viewport={"width": 1280, "height": 900}, locale="en-US"
        )
        try:
            if session_id:
                context.add_cookies([build_session_cookie(session_id)])
            page = context.pages[0] if context.pages else context.new_page()

            if not open_profile(page, username):
                if headless:
                    raise RuntimeError("Instagram asks for a login. Run again with the browser window visible to sign in once.")
                status(f"Log in inside the browser window (you have {LOGIN_TIMEOUT // 60} minutes).")
                if not wait_for_login(page):
                    raise RuntimeError("Login timed out. Start the scraper again and sign in within 3 minutes.")
                status("Login detected. Opening the profile...")
                if not open_profile(page, username):
                    raise RuntimeError("Instagram still asks for a login. Check your credentials and try again.")

            if not (wait_for_selector(page, "header") and wait_for_selector(page, POST_SELECTOR)):
                page.screenshot(path="debug.png")
                raise RuntimeError("Profile did not load (private, wrong username or layout change). See debug.png.")

            profile = parse_profile(page, username)
            posts = collect_posts(page, target, on_progress)
            read_details(page, posts, username, details, on_progress)
            return profile, posts
        finally:
            context.close()


def save(profile: dict, posts: list[dict]) -> None:
    """Write the results to data/profile.json and data/posts.json."""
    DATA_DIR.mkdir(exist_ok=True)
    for filename, payload in (("profile.json", profile), ("posts.json", posts)):
        (DATA_DIR / filename).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def run(
    username: str,
    target: int = 50,
    headless: bool = False,
    details: int = DEFAULT_DETAILS,
    on_progress: Callable[[str, int, int], None] | None = None,
    on_status: Callable[[str], None] | None = None,
) -> tuple[dict, list[dict]]:
    """Scrape a profile and save the results; shared by the CLI and the dashboard."""
    profile, posts = scrape(username, target, headless, details, on_progress, on_status)
    save(profile, posts)
    return profile, posts


def main() -> None:
    """Parse CLI arguments, run the scraper and persist the output."""
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("username", help="public profile to scrape, without the @")
    parser.add_argument("-n", "--posts", type=int, default=50, help="minimum posts to collect (default: 50)")
    parser.add_argument("-d", "--details", type=int, default=DEFAULT_DETAILS,
                        help="posts to open for caption, likes and comments; 0 skips (default: 10)")
    parser.add_argument("--headless", action="store_true", help="run the browser without a window")
    args = parser.parse_args()

    try:
        profile, posts = run(args.username.lstrip("@"), args.posts, args.headless, args.details)
    except RuntimeError as error:
        sys.exit(f"error: {error}")
    log.info("Saved %d posts for @%s in %s", len(posts), profile["username"], DATA_DIR)


if __name__ == "__main__":
    main()
