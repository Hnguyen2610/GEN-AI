ALTER TABLE assets
ALTER COLUMN status TYPE dataset_status
USING status::dataset_status;

ALTER TABLE assets
ALTER COLUMN status SET DEFAULT 'pending'::dataset_status;
