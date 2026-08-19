import click
from loguru import logger

from utils import get_db
from const import DETECTION_CONFIDENCE_THRESHOLD


VIEW_NAME = "filtered_dataset"

VIEW_SQL = f"""
CREATE OR REPLACE VIEW {VIEW_NAME} AS
SELECT
    d.id_detection,
    pbi.id_pipeline_batch_item AS pipeline_batch_item_id,
    v.barcode,
    d.scan_filename,
    d.bbox_xyxy,
    d.bbox_xywh,
    d.bbox_conf,
    d.orientation_correction_gen,
    d.orientation_correction_confidence_gen,
    d.orientation_correction_probs_gen,
    c.pred_class,
    c.pred_conf AS classification_conf,
    c.probs AS classification_probs,
    cap.text AS caption_text,
    cap.lang AS caption_lang,
    cap.lang_detected AS caption_lang_detected,
    cap.linear_prob AS caption_linear_prob,
    cap.thesaurus_matches AS caption_thesaurus_matches,
    ih.image_hash,
    ie.embedding
FROM detection d
JOIN pipeline_batch_item pbi
    ON d.pipeline_batch_item_id = pbi.id_pipeline_batch_item
JOIN ib_volume v
    ON pbi.ib_volume_id = v.barcode
JOIN classification c
    ON c.detection_id = d.id_detection
LEFT JOIN caption cap
    ON cap.detection_id = d.id_detection
LEFT JOIN image_hash ih
    ON ih.detection_id = d.id_detection
LEFT JOIN image_embedding ie
    ON ie.detection_id = d.id_detection
JOIN (
    SELECT DISTINCT ON (dh.group_id, de.group_id) dh.detection_id
    FROM deduped_hash dh
    JOIN deduped_embedding de ON de.detection_id = dh.detection_id
    JOIN detection d2 ON d2.id_detection = dh.detection_id
    ORDER BY dh.group_id, de.group_id, d2.bbox_conf DESC, dh.detection_id ASC
) deduped ON deduped.detection_id = d.id_detection
WHERE d.bbox_conf >= {DETECTION_CONFIDENCE_THRESHOLD}
"""


@click.command("create-view")
@click.option(
    "--drop-existing",
    is_flag=True,
    help="Drop the existing view before recreating it",
)
def create_view(drop_existing):
    """
    Create the filtered_dataset PostgreSQL view.

    This view joins pipeline data (detections, classifications, captions,
    hashes, embeddings, deduplication groups) into a single queryable surface
    that export and post-processing commands read from.

    Filtering logic:
    - Detections below the detection confidence threshold are excluded
    - Only records present in both deduplication groups (hash-based and embedding-based) are included

    Classification reclassification (low-confidence -> "Other") is handled at
    export time via --classification-threshold, not in this view.
    """
    db = get_db()

    logger.info(f"Creating view '{VIEW_NAME}'...")
    logger.info(f"  Detection confidence threshold: {DETECTION_CONFIDENCE_THRESHOLD}")

    if drop_existing:
        logger.info(f"  Dropping existing view '{VIEW_NAME}'...")
        db.execute_sql(f"DROP VIEW IF EXISTS {VIEW_NAME} CASCADE")

    db.execute_sql(VIEW_SQL)
    logger.success(f"View '{VIEW_NAME}' created successfully.")
