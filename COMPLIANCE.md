# EU / GDPR compliance response

This release responds to the review *"cctv_zarr — EU legislation
and litigation breakdown"* (30 July 2026, 18 findings). Every
finding is addressed in code, configuration, or documentation
below. Two framing points first: software cannot *be* GDPR
compliant - only a deployment operated by an identified controller
can - so the goal here is that the defaults are safe and every
capability an operator legally needs actually exists. And none of
this is legal advice; deployment decisions (member state, context,
lawful basis, DPIA, works-council gates) need a lawyer with the
operator's concrete facts.

## Finding-by-finding

**1. Retention (Critical).** A `retention:` config section now
governs the whole system: movement and non-movement chunks expire
on separate clocks (defaults 30 days / 72 hours), enforced
chunk-by-chunk from the manifest timestamps by `cctv-zarr prune`
(module `retention.py`) - locally, in the archive
(`CloudArchive.delete_objects`), in the query cache, and over the
exports folder. A store whose every chunk expired is deleted
whole. Every deletion is journalled to `deletion_log.jsonl`, so
erasure is provable (the Croatian lesson: deletion *and* evidence
of it). The GCE deployment ships a daily prune systemd timer.

**2. Everything stored (Critical).** `retention.keep_non_movement:
false` makes the writer delete non-movement image chunks at ingest
- quiet footage is never stored at all (Art. 25 minimisation by
design). The manifest row, movement flag, and timestamps remain,
and reads of removed ranges return fill values, so queries degrade
cleanly. Stores record `non_movement_stored` so the posture is
auditable. Differential retention (finding 1) covers the softer
variant.

**3. Subject access / erasure (Critical).** `cctv-zarr
access-request <store> --start --end --out clip.mp4 --mask
x0,y0,x1,y1` produces the Art. 15 copy with third-party regions
blacked out (`mask_regions` on both export paths in `query.py`);
`cctv-zarr erase <store> --start --end [--archive] [--cache]`
honours Art. 17 across every copy: local chunks
(`zarr_io.delete_chunk`), archive objects, and the query cache
(`retention.invalidate_cache`) - journalled. Both are logged with
purpose in `access_log.jsonl`.

**4. Art. 32 security (High).** Client-side encryption of every
archive object (`security.encrypt_archive` + Fernet key;
`archive.build_cipher`) - the bucket never sees plaintext;
`archive.bucket_location` fails closed on region mismatch; an
append-only access log (`audit.py`) records query/preview/export
actions with client identity and purpose from both the CLI and
the panel.

**5. Chapter V (High).** `bucket_location` in `ArchiveConfig`,
verified against the real bucket at binding time. The mock archive
(footage stays on the operator's machine) remains the default and
is the documented mode for anything containing real people.

**6. Uncontrolled copies (High).** Exports and the query cache are
on retention clocks (`exports_max_age_hours`,
`cache_max_age_hours`), swept by `prune` and opportunistically on
every panel export. Every export is access-logged with its
destination. `--purpose` on `cctv-zarr query` lands in the log.

**7. Source files with audio (Medium).**
`retention.source_after_ingest: keep | quarantine | delete`. The
ingest CLI and the panel batch worker apply it; `keep` warns that
the source may carry an audio track the store does not.
Quarantined sources sit under `_ingested/` where the sweep can
expire them.

**8. Accountability metadata (Medium).** Every store's `cctv`
namespace now carries a `governance` block (controller, site,
purpose, legal basis, retention policy, DPO contact - from the
`governance:` config section) plus the retention clocks in force
and `biometric_source: false`. Ingest against a real archive with
an empty controller warns loudly.

**9. Per-object tracking overlays (Medium).**
`flow.record_events: false` stores the binary per-chunk movement
flag without per-object timelines - the minimising configuration
for workplace/vehicle contexts. The README states the deployment
contexts and the Art. 88-family procedural gates (works council /
union / inspectorate) as operator preconditions.

**10. Timestamp provenance (Low).** Stores record `time_source:
explicit | file_mtime`; the mtime fallback warns at ingest that
timestamps may not reflect recording time.

**11. Biometric enrichment guard (Low).** The LICENSE now carries
an acceptable-use clause forbidding biometric template extraction
and facial-recognition database building from stores (AI Act
Art. 5(1)(e); GDPR Art. 9), and every store self-declares
`biometric_source: false`.

**12. Panel authentication (Critical).** `security.auth_token`
protects every `/api/*` route (Bearer token; 401 otherwise); the
panel shows an access-code screen and remembers the code. The
server **refuses to start** on a non-loopback host unless a token
is set *and* `security.allow_remote: true` - so the unauthenticated
configuration physically cannot face a network.

**13. Filesystem confinement (High).** `/api/browse` and
`/api/ingest` now resolve every client-supplied path and require
it inside `video_directory` (the same discipline `_store_root`
already applied to stores). `..` past the video area returns 400.

**14. Silent rmtree (High).** Re-ingest never destroys footage:
the existing store is moved to `stores/_replaced/` (tombstone),
journalled with who-replaced-what, and hidden from listings. The
footage an outstanding access request needs survives.

**15. Exports folder (High).** Under the retention engine (finding
6), swept on every export and by `prune`; every clip's creation is
access-logged.

**16. Plain HTTP (Medium).** The non-loopback guard (finding 12)
plus documentation: remote serving requires TLS termination in
front; the GCE deployment's default access path is an SSH tunnel,
which is encrypted end to end with nothing public.

**17. Cross-user leakage (Low).** Substantially mitigated by
authentication (12); job ids and prefs remain per-client
best-effort within an authenticated group, as the review deemed
acceptable.

**18. Dataset testing (Medium).** `run_dataset_tests.py --full`
refuses a non-mock archive unless `--allow-real-archive` is
passed, and the docs state that dataset runs should use synthetic,
licensed, or consent-cleared footage.

## What remains the operator's job

Signage and Art. 13 information, the Art. 30 record, DPIA where
required, lawful basis selection, national procedural gates
(works councils, inspectorates), breach procedures (Arts. 33-34),
choosing retention numbers that match their member state's caps
(the defaults are conservative but not jurisdiction-specific), key
management for the archive cipher, and TLS in front of any
non-loopback panel. The `governance:` block, the two journals, and
the store metadata exist to make evidencing all of that easier.

Verification: 70 panel/compliance tests plus the pre-existing 57
pipeline tests run fully offline in CI; the static Power of 10
gate still applies to all new code.
