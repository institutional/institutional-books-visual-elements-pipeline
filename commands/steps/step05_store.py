# NOTE:
# This is the last item of the batch-level steps series.
#
# Because every step after that is dataset-scale and will not have easy access to scans,
# the goal of this step would be to store intermediary objects to remote storage for easy access.
#
# In that case, we want to store crops on R2:
# - `bucket/crops/barcode/page-filename/crop_*.jpg`
# - OR -
# - `bucket/crops/barcode/page-filename/crops.tar.gz`
#
# We should keep track of these crops and their properties in the database so they're easy to retrieve and analyze.
