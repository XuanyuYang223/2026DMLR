# Subsequent documentation revisions

The publication integration revises the root README to introduce the complete
PermWorld and DMLR program. This is new editorial content, not a translation of
the old pilot-only page. `publication_integration.json` records the prior English
Git revision and the before/after hashes for that one file.

The original English translation ledger and all scientific manifests remain
unchanged. The English verifier first validates the original translation against
its historical Git bytes, then resolves the current README through this explicit
bridge. All other translated files still require their exact recorded English
hashes. The new page must independently pass the language scan.

This mechanism does not declare the revised page byte-identical to the old
document and does not rebind a frozen scientific protocol. The publication
manifest separately records the integrated package and verifies that original
experiment paths have no changes relative to the integration base revision.
