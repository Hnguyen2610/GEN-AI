--;;
-- 1. Create AssetKind Enum
DO $$ 
BEGIN 
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'asset_kind') THEN
        CREATE TYPE asset_kind AS ENUM ('dataset', 'knowledge');
    END IF;
END $$;
--;;

-- 1b. Create DatasetStatus Enum for unified asset status
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'dataset_status') THEN
        CREATE TYPE dataset_status AS ENUM ('pending', 'processing', 'ready', 'failed');
    END IF;
END $$;
--;;

-- 2. Create assets table
CREATE TABLE IF NOT EXISTS assets (
    id UUID PRIMARY KEY,
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    kind asset_kind NOT NULL,
    title VARCHAR(255) NOT NULL,
    original_filename VARCHAR(512) NOT NULL,
    mime_type VARCHAR(255) NOT NULL,
    status dataset_status NOT NULL DEFAULT 'pending'::dataset_status,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL
);
--;;

-- 3. Create asset_versions table
CREATE TABLE IF NOT EXISTS asset_versions (
    id UUID PRIMARY KEY,
    asset_id UUID NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
    version_number INTEGER NOT NULL,
    storage_backend VARCHAR(32) NOT NULL,
    storage_path VARCHAR(1024) NOT NULL,
    file_size_bytes BIGINT NOT NULL,
    checksum_sha256 VARCHAR(128) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    UNIQUE (asset_id, version_number)
);
--;;

-- 4. Initial Backfill from legacy tables
INSERT INTO assets (id, workspace_id, kind, title, original_filename, mime_type, status, created_at, updated_at)
SELECT id, workspace_id, 'dataset'::asset_kind, title, original_filename, mime_type, status::dataset_status, created_at, updated_at
FROM datasets
ON CONFLICT (id) DO NOTHING;
--;;

INSERT INTO assets (id, workspace_id, kind, title, original_filename, mime_type, status, created_at, updated_at)
SELECT id, workspace_id, 'knowledge'::asset_kind, title, original_filename, mime_type, status::dataset_status, created_at, updated_at
FROM knowledge_assets
ON CONFLICT (id) DO NOTHING;
--;;

INSERT INTO asset_versions (id, asset_id, version_number, storage_backend, storage_path, file_size_bytes, checksum_sha256, created_at)
SELECT id, dataset_id, version_number, storage_backend, storage_path, file_size_bytes, checksum_sha256, created_at
FROM dataset_versions
ON CONFLICT (id) DO NOTHING;
--;;

INSERT INTO asset_versions (id, asset_id, version_number, storage_backend, storage_path, file_size_bytes, checksum_sha256, created_at)
SELECT id, knowledge_asset_id, version_number, storage_backend, storage_path, file_size_bytes, checksum_sha256, created_at
FROM knowledge_versions
ON CONFLICT (id) DO NOTHING;
--;;

-- 5. Add generic FKs to ingestion_jobs
ALTER TABLE ingestion_jobs ADD COLUMN IF NOT EXISTS asset_id UUID REFERENCES assets(id) ON DELETE CASCADE;
--;;
ALTER TABLE ingestion_jobs ADD COLUMN IF NOT EXISTS asset_version_id UUID REFERENCES asset_versions(id) ON DELETE CASCADE;
--;;
ALTER TABLE ingestion_jobs ADD COLUMN IF NOT EXISTS asset_kind asset_kind;
--;;

-- Backfill jobs
UPDATE ingestion_jobs SET asset_id = dataset_id, asset_version_id = dataset_version_id, asset_kind = 'dataset'::asset_kind
WHERE dataset_id IS NOT NULL AND asset_id IS NULL;
--;;

UPDATE ingestion_jobs SET asset_id = knowledge_asset_id, asset_version_id = knowledge_version_id, asset_kind = 'knowledge'::asset_kind
WHERE knowledge_asset_id IS NOT NULL AND asset_id IS NULL;
--;;

-- 6. Add generic FKs to lane-specific tables
ALTER TABLE dataset_sheets ADD COLUMN IF NOT EXISTS asset_version_id UUID REFERENCES asset_versions(id) ON DELETE CASCADE;
--;;
UPDATE dataset_sheets SET asset_version_id = dataset_version_id WHERE asset_version_id IS NULL;
--;;

ALTER TABLE column_profiles ADD COLUMN IF NOT EXISTS asset_version_id UUID REFERENCES asset_versions(id) ON DELETE CASCADE;
--;;
UPDATE column_profiles SET asset_version_id = dataset_version_id WHERE asset_version_id IS NULL;
--;;

ALTER TABLE knowledge_chunks ADD COLUMN IF NOT EXISTS asset_version_id UUID REFERENCES asset_versions(id) ON DELETE CASCADE;
--;;
UPDATE knowledge_chunks SET asset_version_id = knowledge_version_id WHERE asset_version_id IS NULL;
--;;

-- 7. Add generic FKs to chunks (legacy lane)
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS asset_id UUID REFERENCES assets(id) ON DELETE CASCADE;
--;;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS asset_version_id UUID REFERENCES asset_versions(id) ON DELETE CASCADE;
--;;
UPDATE chunks SET asset_id = dataset_id, asset_version_id = dataset_version_id WHERE asset_id IS NULL;
--;;

-- 8. Indexes for performance
CREATE INDEX IF NOT EXISTS idx_assets_workspace_id ON assets(workspace_id);
--;;
CREATE INDEX IF NOT EXISTS idx_asset_versions_asset_id ON asset_versions(asset_id);
--;;
CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_asset_id ON ingestion_jobs(asset_id);
