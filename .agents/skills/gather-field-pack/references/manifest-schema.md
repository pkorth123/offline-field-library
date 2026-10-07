# Knowledge-pack manifest

Create UTF-8 `manifest.json` with `schema_version` set to `1`. Paths are relative to the staging
directory and may not escape it.

```json
{
  "schema_version": 1,
  "slug": "honda-eu2200i",
  "title": "Honda EU2200i Field Reference",
  "description": "Operation, maintenance, troubleshooting, and repair references.",
  "brief": "Research US-market Honda EU2200i operation, scheduled maintenance, common symptoms, and owner-serviceable repair boundaries.",
  "report_file": "research-report.md",
  "sources": [
    {
      "key": "honda-owner-manual-2024",
      "title": "EU2200i Owner's Manual",
      "file": "sources/eu2200i-owner-manual.pdf",
      "url": "https://example.com/eu2200i-owner-manual.pdf",
      "publisher": "Honda",
      "author": null,
      "published_at": null,
      "retrieved_at": "2026-10-07",
      "source_class": "manufacturer_manual",
      "authority": "official",
      "applicability": ["EU2200i", "US market"],
      "license_note": "Cached locally for personal offline reference.",
      "notes": "Covers operation and scheduled maintenance."
    },
    {
      "key": "owner-surge-discussion",
      "title": "Owner discussion of surging symptoms",
      "url": "https://example.com/forum/surging",
      "retrieved_at": "2026-10-07",
      "source_class": "forum",
      "authority": "anecdotal",
      "applicability": ["EU2200i"],
      "notes": "URL-only because a durable public copy was not available."
    }
  ]
}
```

## Required manifest fields

- `schema_version`: integer `1`
- `slug`: 3–64 lowercase letters, digits, or hyphens; stable collection identifier
- `title`: user-facing collection title
- `sources`: non-empty source array

`description`, `brief`, and `report_file` are optional but expected for researched packs.

## Source fields

- `key`: unique stable identifier using lowercase letters, digits, dots, underscores, or hyphens
- `title`: source title
- `file`: optional cached file path relative to the staging directory
- `url`: optional original HTTP or HTTPS URL

Each source must have `file`, `url`, or both. Use both whenever a public source was cached.

Optional provenance fields are `publisher`, `author`, `published_at`, `retrieved_at`,
`source_class`, `authority`, `applicability`, `license_note`, and `notes`. `applicability` is an
array of equipment models, revisions, markets, or other scope qualifiers.

Allowed authority values:

- `official`: manufacturer or product owner
- `government`: government, regulator, or recall authority
- `professional`: attributable technical or service expertise
- `community`: established community reference or collaborative documentation
- `anecdotal`: individual owner reports, forums, reviews, or unverified experience
- `derived`: AI- or researcher-produced interpretation
- `unknown`: authority could not be established

Use specific source classes such as `manufacturer_manual`, `service_bulletin`, `parts_diagram`,
`government_recall`, `technical_article`, `forum`, `review`, `illustration`, or `video_transcript`.

## Cache and indexing behavior

- PDF, HTML, text, Markdown, and RST files are cached and indexed.
- PNG, JPEG, WebP, and other files are cached and listed but not text-indexed.
- PDF illustrations remain in the original PDF; retain page-aware citations.
- HTML image alt text is searchable. Preserve important standalone diagrams as separate sources.
- URL-only sources remain in provenance records but are unavailable during offline chat.
- Identical files are stored once by SHA-256 even when referenced by multiple packs.
- The canonical imported manifest records object paths, hashes, media types, and index status.

Do not make a generated image description the sole authority for technical values. Put derived
descriptions in notes or the research report and direct the user to the original illustration.

## Research report

Use concise sections for:

1. Scope and exact applicability
2. Sources acquired, organized by authority
3. Coverage by requested topic
4. Conflicting or anecdotal guidance
5. Safety-critical boundaries
6. Gaps, rejected sources, and offline limitations
7. Suggested questions for offline use
