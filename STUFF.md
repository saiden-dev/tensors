# Additional Schema Ideas

From `gen.py/SQL.md` - views, triggers, and FTS not yet in models-db.md.

## Views

```sql
-- Complete model info with latest version
CREATE VIEW v_models_with_latest AS
SELECT
    m.id,
    m.civitai_id,
    m.name,
    m.type,
    m.nsfw,
    c.username as creator,
    mv.name as latest_version,
    mv.base_model,
    m.download_count,
    m.thumbs_up_count
FROM models m
LEFT JOIN creators c ON m.creator_id = c.id
LEFT JOIN model_versions mv ON mv.model_id = m.id AND mv.version_index = 0;

-- Local files with CivitAI info
CREATE VIEW v_local_files_full AS
SELECT
    lf.file_path,
    lf.sha256,
    m.name as model_name,
    m.type as model_type,
    mv.name as version_name,
    mv.base_model,
    c.username as creator
FROM local_files lf
LEFT JOIN models m ON lf.civitai_model_id = m.civitai_id
LEFT JOIN model_versions mv ON lf.civitai_version_id = mv.civitai_id
LEFT JOIN creators c ON m.creator_id = c.id;

-- Search by hash
CREATE VIEW v_files_by_hash AS
SELECT
    fh.hash_type,
    fh.hash_value,
    vf.name as filename,
    m.name as model_name,
    mv.name as version_name,
    mv.base_model
FROM file_hashes fh
JOIN version_files vf ON fh.file_id = vf.id
JOIN model_versions mv ON vf.version_id = mv.id
JOIN models m ON mv.model_id = m.id;
```

## Triggers

```sql
-- Update timestamp on local_files
CREATE TRIGGER update_local_files_timestamp
AFTER UPDATE ON local_files
BEGIN
    UPDATE local_files SET updated_at = datetime('now') WHERE id = NEW.id;
END;
```

## FTS (Full-Text Search)

```sql
-- Full-text search on model names and descriptions
CREATE VIRTUAL TABLE models_fts USING fts5(
    name,
    description,
    content='models',
    content_rowid='id'
);

-- Triggers to keep FTS in sync
CREATE TRIGGER models_fts_insert AFTER INSERT ON models BEGIN
    INSERT INTO models_fts(rowid, name, description) VALUES (NEW.id, NEW.name, NEW.description);
END;

CREATE TRIGGER models_fts_delete AFTER DELETE ON models BEGIN
    INSERT INTO models_fts(models_fts, rowid, name, description) VALUES ('delete', OLD.id, OLD.name, OLD.description);
END;

CREATE TRIGGER models_fts_update AFTER UPDATE ON models BEGIN
    INSERT INTO models_fts(models_fts, rowid, name, description) VALUES ('delete', OLD.id, OLD.name, OLD.description);
    INSERT INTO models_fts(rowid, name, description) VALUES (NEW.id, NEW.name, NEW.description);
END;

-- Full-text search on prompts
CREATE VIRTUAL TABLE prompts_fts USING fts5(
    prompt,
    negative_prompt,
    content='image_generation_params',
    content_rowid='id'
);
```

## Example Queries

```sql
-- Search models by name
SELECT * FROM models WHERE id IN (
    SELECT rowid FROM models_fts WHERE models_fts MATCH 'dreamshaper'
);

-- Find models by hash
SELECT * FROM v_files_by_hash WHERE hash_value LIKE '08AA146%';
```
