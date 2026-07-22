from typing import Dict, Any, Optional

class MetadataEnricher:
    """
    Handles the merging and enrichment of various metadata sources 
    (e.g., AECP, Chroma-specific data).
    """
    @staticmethod
    def enrich_row(record: Dict[str, Any], aecp_metadata: Optional[Dict[str, Any]], row_metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Enriches the core record with all available metadata.

        Invariant Check Improvement: If AECP is present, it must be included 
        in the output even if secondary sources are None/empty.
        """
        enriched_data = {
            "core_record": record,
            "metadata": {}
        }

        # 1. Always include base AECP metadata lineage data first
        if aecp_metadata:
            enriched_data["metadata"]["aecp"] = aecp_metadata
        else:
             # Critical check: Ensure that if the process requires it, 
             # we record its absence rather than simply omitting it.
            enriched_data["metadata"]["aecp"] = {}


        # 2. Handle row-specific metadata (where the bug occurred)
        if row_metadata:
            enriched_data["metadata"].update(row_metadata)

        # 3. Defensive update: Ensure no core lineage keys are lost if row_metadata is None.
        # If we attempt to merge row_metadata and it's None, we must fallback 
        # entirely onto AECP/core data.
        if not row_metadata and aecp_metadata:
             # Explicitly ensure key presence for integrity check compatibility
            enriched_data["metadata"]["lineage_source"] = "AECP_FALLBACK"

        return enriched_data


class MigrationService:
    """
    Manages the migration batch process, ensuring data consistency.
    """
    @staticmethod
    def process_batch(records: list[Dict], aecp_metadata: Dict[str, Any], row_metadata_batches: list[Optional[Dict[str, Any]]]) -> list[Dict[str, Any]]:
        """
        Processes an entire batch.
        Assumes records and row_metadata_batches are aligned one-to-one.
        """
        enriched_batch = []
        for record, rmd in zip(records, row_metadata_batches):
            # The fix ensures MetadataEnricher handles the None/Dict mix safely
            enriched = MetadataEnricher.enrich_row(record, aecp_metadata, rmd)
            enriched_batch.append(enriched)
        return enriched_batch
