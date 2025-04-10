import math
import pandas as pd

import aiohttp
from mwparserfromhtml import Article
from mwparserfromhtml.parse.utils import is_transcluded
from python.decorators import fetch_size_bytes


@fetch_size_bytes("articlequality")
async def get_article_html(lang, revid, protocol):
    """Get an article revision's HTML.

    NOTE: fetching HTML for old revisions can be slow as it's not always cached.
    Here we use exact revision IDs.

    See: https://www.mediawiki.org/wiki/RESTBase/service_migration#Parsoid_endpoints

    """
    base_url = f"{protocol}://{lang}.wikipedia.org/w/rest.php/v1/revision/{revid}/html"
    timeout = aiohttp.ClientTimeout(total=5)
    try:
        async with aiohttp.ClientSession(
            timeout=timeout, raise_for_status=True
        ) as session:
            async with session.get(
                base_url, headers={"User-Agent": "liftwing articlequality model"}
            ) as response:
                return await response.text()
    except Exception:
        return None


def _html_to_plaintext(article):
    """Convert Parsoid HTML to reasonable plaintext."""
    exclude_transcluded_paragraphs = True
    exclude_elements = {
        "Category",
        "Citation",
        "Comment",
        "Heading",
        "Infobox",
        "Math",
        "Media-audio",
        "Media-img",
        "Media-video",
        "Messagebox",
        "Navigational",
        "Note",
        "Reference",
        "TF-sup",  # superscript -- catches Citation-needed tags etc.
        "Table",
        "Wikitable",
    }
    exclude_para_context = {"pre-first-para", "between-paras", "post-last-para"}

    paragraphs = [
        paragraph.strip()
        for heading, paragraph in article.wikistew.get_plaintext(
            exclude_transcluded_paragraphs=exclude_transcluded_paragraphs,
            exclude_para_context=exclude_para_context,
            exclude_elements=exclude_elements,
        )
    ]

    return "\n".join(paragraphs) if paragraphs else None


def get_article_features(article_html):
    try:
        article = Article(article_html)
    except TypeError:
        print(f"Skipping article due to TypeError: {article_html}")
        return 0, 0, 0, 0, 0, 0, 0, 0, 0

    plaintext = _html_to_plaintext(article)
    page_length = len(plaintext) if plaintext else 0
    refs = len(article.wikistew.get_citations())
    wikilinks_objects = [
        w for w in article.wikistew.get_wikilinks() if not is_transcluded(w.html_tag)
    ]
    wikilinks = len(
        [
            wikilink.link.lstrip(".")
            for wikilink in wikilinks_objects
            if not wikilink.link.endswith("redlink=1")
        ]
    )
    categories = len(
        [1 for c in article.wikistew.get_categories() if not is_transcluded(c.html_tag)]
    )
    max_icon_pixel_area = 100 * 100  # 10000 pixels
    article_images = [
        image
        for image in article.wikistew.get_images()
        if image.height * image.width > max_icon_pixel_area
    ]
    article_videos = [video for video in article.wikistew.get_video()]
    article_audio = [audio for audio in article.wikistew.get_audio()]
    media = len(article_images) + len(article_videos) + len(article_audio)
    headings = len([h for h in article.wikistew.get_headings() if h.level <= 3])

    # new additions
    # unique sources
    sources = len(article.wikistew.get_references())

    # infoboxes, if len >= 1, then return true else false
    infoboxes = True if len(article.wikistew.get_infobox()) >= 1 else False

    # message boxes, return true if len >= 1 else false
    messageboxes = True if len(article.wikistew.get_message_boxes()) >= 1 else False

    return [
        page_length,
        refs,
        wikilinks,
        categories,
        media,
        headings,
        sources,
        infoboxes,
        messageboxes,
    ]


def normalize_features(
    MAX_QUAL_VALS,
    lang,
    page_length,
    num_refs,
    num_wikilinks,
    num_cats,
    num_media,
    num_headings,
    num_sources,
    num_infoboxes,
    num_messageboxes,
):
    """Convert raw count features into values between 0 and 1.

    Possible transformations:
    * square root: make initial gains in feature of more importance to model
                   e.g., going from 0 -> 16 wikilinks is the same as 16 -> 64
    * divide by page length: convert absolutes to an expectation per byte of content
                             e.g., total references -> references per paragraph
    """
    normed_page_length = math.sqrt(page_length)
    length_x = min(1, normed_page_length / MAX_QUAL_VALS[lang]["max_length"])
    refs_x = min(1, (num_refs / normed_page_length) / MAX_QUAL_VALS[lang]["max_refs"])
    wikilinks_x = min(
        1, (num_wikilinks / normed_page_length) / MAX_QUAL_VALS[lang]["max_links"]
    )  # The squareRoot dropped
    categories_x = min(1, num_cats / MAX_QUAL_VALS[lang]["max_cats"])
    media_x = min(1, num_media / MAX_QUAL_VALS[lang]["max_media"])
    headings_x = min(
        1, (num_headings / normed_page_length) / MAX_QUAL_VALS[lang]["max_headings"]
    )

    # new additions
    sources_x = min(1, num_sources / MAX_QUAL_VALS[lang]["max_srcs"])
    infoboxes_x = num_infoboxes
    messageboxes_x = num_messageboxes

    return (
        length_x,
        refs_x,
        wikilinks_x,
        categories_x,
        media_x,
        headings_x,
        sources_x,
        infoboxes_x,
        messageboxes_x,
    )


def load_quality_max_featurevalues(file_path):
    feature_names = [
        "max_length",
        "max_media",
        "max_cats",
        "max_refs",
        "max_headings",
        "max_links",
        "max_srcs",
    ]

    MIN_MAX_LEN = 45  # changed from 100 to 45
    MIN_MAX_MED = 2
    MIN_MAX_CAT = 5
    MIN_MAX_REF = 0.2  # changed from 0.15 to 0.2
    MIN_MAX_HEA = 0.1
    MIN_MAX_LIN = 0.45  # changed from 0.1 to 0.45
    MIN_MAX_UNIQUE_SOURCES = 5  # added sources

    df = pd.read_csv(file_path, sep="\t")
    df["lang"] = df["wiki_db"].apply(lambda x: x.replace("wiki", ""))
    df["max_length"] = df["max_length"].apply(lambda x: max(MIN_MAX_LEN, float(x)))
    df["max_media"] = df["max_media"].apply(lambda x: max(MIN_MAX_MED, float(x)))
    df["max_cats"] = df["max_cats"].apply(lambda x: max(MIN_MAX_CAT, float(x)))
    df["max_refs"] = df["max_refs"].apply(lambda x: max(MIN_MAX_REF, float(x)))
    df["max_headings"] = df["max_headings"].apply(lambda x: max(MIN_MAX_HEA, float(x)))
    df["max_links"] = df["max_links"].apply(lambda x: max(MIN_MAX_LIN, float(x)))
    df["max_srcs"] = df["max_srcs"].apply(
        lambda x: max(MIN_MAX_UNIQUE_SOURCES, float(x))
    )

    df = df.set_index("lang")
    return df[feature_names].T.to_dict()
