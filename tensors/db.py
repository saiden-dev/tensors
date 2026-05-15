"""SQLModel database for local model metadata and CivitAI/HuggingFace cache."""

from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlmodel import Session, col, func, select

from tensors.config import DATA_DIR
from tensors.models import (
    Creator,
    FileHash,
    HFModel,
    HFModelTag,
    HFSafetensorFile,
    ImageGenerationParam,
    ImageResource,
    LocalFile,
    Model,
    ModelTag,
    ModelVersion,
    SafetensorMetadata,
    Tag,
    TrainedWord,
    VersionFile,
    VersionImage,
    create_tables,
    get_engine,
)
from tensors.safetensor import compute_sha256, read_safetensor_metadata

if TYPE_CHECKING:
    from pathlib import Path

    from rich.console import Console
    from sqlalchemy import Engine

# Database location
DB_PATH = DATA_DIR / "models.db"


class Database:
    """SQLModel database wrapper for models metadata."""

    def __init__(self, db_path: Path | None = None) -> None:
        """Initialize database connection."""
        self.db_path = db_path or DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._engine: Engine | None = None

    @property
    def engine(self) -> Engine:
        """Get or create database engine."""
        if self._engine is None:
            self._engine = get_engine(str(self.db_path))
        return self._engine

    def close(self) -> None:
        """Close database connection."""
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None

    def __enter__(self) -> Database:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def init_schema(self) -> None:
        """Initialize database schema."""
        create_tables(self.engine)

    def session(self) -> Session:
        """Create a new session."""
        return Session(self.engine)

    # =========================================================================
    # Local Files Operations
    # =========================================================================

    def scan_directory(
        self,
        directory: Path,
        console: Console | None = None,
    ) -> list[dict[str, Any]]:
        """Scan directory for safetensor files and add to database."""
        results: list[dict[str, Any]] = []
        safetensor_files = list(directory.rglob("*.safetensors"))

        for path in safetensor_files:
            if console:
                console.print(f"[dim]Scanning {path.name}...[/dim]")

            try:
                sha256 = compute_sha256(path)
                metadata = read_safetensor_metadata(path)

                with self.session() as session:
                    file_info = self._upsert_local_file(
                        session,
                        file_path=str(path.resolve()),
                        sha256=sha256,
                        header_size=metadata.get("header_size"),
                        tensor_count=metadata.get("tensor_count"),
                    )
                    self._store_safetensor_metadata(session, file_info.id, metadata.get("metadata", {}))
                    session.commit()
                    # Extract values before session closes
                    result = {"id": file_info.id, "file_path": file_info.file_path, "sha256": file_info.sha256}

                results.append(result)

            except Exception as e:
                if console:
                    console.print(f"[red]Error scanning {path.name}: {e}[/red]")

        return results

    def _upsert_local_file(
        self,
        session: Session,
        file_path: str,
        sha256: str,
        header_size: int | None = None,
        tensor_count: int | None = None,
    ) -> LocalFile:
        """Insert or update a local file record."""
        existing = session.exec(select(LocalFile).where(LocalFile.file_path == file_path)).first()

        if existing:
            existing.sha256 = sha256
            existing.header_size = header_size
            existing.tensor_count = tensor_count
            existing.updated_at = datetime.utcnow()
            session.add(existing)
            return existing

        local_file = LocalFile(
            file_path=file_path,
            sha256=sha256,
            header_size=header_size,
            tensor_count=tensor_count,
        )
        session.add(local_file)
        session.flush()
        return local_file

    def _store_safetensor_metadata(self, session: Session, local_file_id: int | None, metadata: dict[str, Any]) -> None:
        """Store safetensor header metadata."""
        if not local_file_id:
            return
        for key, value in metadata.items():
            str_value = json.dumps(value) if not isinstance(value, str) else value
            existing = session.exec(
                select(SafetensorMetadata).where(SafetensorMetadata.local_file_id == local_file_id, SafetensorMetadata.key == key)
            ).first()
            if existing:
                existing.value = str_value
                session.add(existing)
            else:
                session.add(SafetensorMetadata(local_file_id=local_file_id, key=key, value=str_value))

    def list_local_files(self) -> list[dict[str, Any]]:
        """List all local files with CivitAI info and trigger words."""
        with self.session() as session:
            files = session.exec(select(LocalFile)).all()
            results = []
            for f in files:
                model = None
                if f.civitai_model_id:
                    model = session.exec(select(Model).where(Model.civitai_id == f.civitai_model_id)).first()
                version = None
                if f.civitai_version_id:
                    version = session.exec(select(ModelVersion).where(ModelVersion.civitai_id == f.civitai_version_id)).first()
                creator = None
                if model and model.creator_id:
                    creator = session.exec(select(Creator).where(Creator.id == model.creator_id)).first()
                # Get trigger words for this version
                triggers: list[str] = []
                if version:
                    words = session.exec(
                        select(TrainedWord).where(TrainedWord.version_id == version.id).order_by(col(TrainedWord.position))
                    ).all()
                    triggers = [w.word for w in words]
                results.append(
                    {
                        "id": f.id,
                        "file_path": f.file_path,
                        "sha256": f.sha256,
                        "header_size": f.header_size,
                        "tensor_count": f.tensor_count,
                        "civitai_model_id": f.civitai_model_id,
                        "civitai_version_id": f.civitai_version_id,
                        "model_name": model.name if model else None,
                        "model_type": model.type if model else None,
                        "version_name": version.name if version else None,
                        "base_model": version.base_model if version else None,
                        "creator": creator.username if creator else None,
                        "triggers": triggers,
                    }
                )
            return results

    def get_local_file_by_path(self, file_path: str) -> dict[str, Any] | None:
        """Get local file by path."""
        with self.session() as session:
            f = session.exec(select(LocalFile).where(LocalFile.file_path == file_path)).first()
            if not f:
                return None
            model = None
            if f.civitai_model_id:
                model = session.exec(select(Model).where(Model.civitai_id == f.civitai_model_id)).first()
            version = None
            if f.civitai_version_id:
                version = session.exec(select(ModelVersion).where(ModelVersion.civitai_id == f.civitai_version_id)).first()
            creator = None
            if model and model.creator_id:
                creator = session.exec(select(Creator).where(Creator.id == model.creator_id)).first()
            return {
                "id": f.id,
                "file_path": f.file_path,
                "sha256": f.sha256,
                "civitai_model_id": f.civitai_model_id,
                "civitai_version_id": f.civitai_version_id,
                "model_name": model.name if model else None,
                "model_type": model.type if model else None,
                "version_name": version.name if version else None,
                "base_model": version.base_model if version else None,
                "creator": creator.username if creator else None,
            }

    def get_local_file_by_hash(self, sha256: str) -> dict[str, Any] | None:
        """Get local file by SHA256 hash."""
        with self.session() as session:
            f = session.exec(select(LocalFile).where(LocalFile.sha256 == sha256.upper())).first()
            if not f:
                return None
            return {"id": f.id, "file_path": f.file_path, "sha256": f.sha256}

    def get_unlinked_files(self) -> list[dict[str, Any]]:
        """Get local files not linked to CivitAI."""
        with self.session() as session:
            files = session.exec(select(LocalFile).where(LocalFile.civitai_model_id == None)).all()  # noqa: E711
            return [{"id": f.id, "file_path": f.file_path, "sha256": f.sha256} for f in files]

    def link_file_to_civitai(self, file_id: int, model_id: int, version_id: int) -> None:
        """Link a local file to CivitAI model/version."""
        with self.session() as session:
            f = session.get(LocalFile, file_id)
            if f:
                f.civitai_model_id = model_id
                f.civitai_version_id = version_id
                f.updated_at = datetime.utcnow()
                session.add(f)
                session.commit()

    def register_downloaded_file(
        self,
        dest_path: Path,
        version_info: dict[str, Any],
        api_key: str | None = None,
        console: Console | None = None,
    ) -> dict[str, Any]:
        """Register a freshly-downloaded file: hash, store metadata, link, and cache full model.

        Idempotent and shared by the CLI download flow and the FastAPI background worker so
        both paths produce identical DB state (local_file row + cached models/versions/tags
        so ``db list`` can resolve names, triggers, and base_model).

        Args:
            dest_path: Path to the downloaded safetensor file.
            version_info: CivitAI ``model-versions/{id}`` response (already fetched).
            api_key: Optional CivitAI API key for the model fetch.
            console: Optional Rich console for hash progress output.

        Returns:
            ``{"file_id": int, "sha256": str, "linked": bool, "cached": bool, "error": str | None}``
        """
        # Lazy import to avoid pulling httpx into modules that only need DB ops
        from tensors.api import fetch_civitai_model  # noqa: PLC0415

        result: dict[str, Any] = {"file_id": None, "sha256": None, "linked": False, "cached": False, "error": None}
        try:
            sha256 = compute_sha256(dest_path, console)
            metadata = read_safetensor_metadata(dest_path)

            civitai_version_id = version_info.get("id")
            civitai_model_id = version_info.get("modelId") or version_info.get("model", {}).get("id")

            with self.session() as session:
                local_file = self._upsert_local_file(
                    session,
                    file_path=str(dest_path.resolve()),
                    sha256=sha256,
                    header_size=metadata.get("header_size"),
                    tensor_count=metadata.get("tensor_count"),
                )
                self._store_safetensor_metadata(session, local_file.id, metadata.get("metadata", {}))

                if civitai_model_id and civitai_version_id:
                    local_file.civitai_model_id = civitai_model_id
                    local_file.civitai_version_id = civitai_version_id
                    session.add(local_file)
                    result["linked"] = True

                session.commit()
                result["file_id"] = local_file.id
                result["sha256"] = sha256

            # Cache full model metadata so db list can resolve names/triggers/base_model.
            # The version endpoint payload is too sparse for cache_model() (no creator, tags,
            # or full modelVersions list), so we fetch the model endpoint here.
            if civitai_model_id:
                model_data = fetch_civitai_model(civitai_model_id, api_key, console)
                if model_data:
                    self.cache_model(model_data)
                    result["cached"] = True
        except Exception as e:  # surface any failure to caller without crashing the download
            result["error"] = str(e)
        return result

    # =========================================================================
    # CivitAI Cache Operations
    # =========================================================================

    def get_version_by_hash(self, sha256: str) -> dict[str, Any] | None:
        """Find cached version by file hash."""
        with self.session() as session:
            fh = session.exec(select(FileHash).where(FileHash.hash_value == sha256.upper())).first()
            if not fh:
                return None
            vf = session.get(VersionFile, fh.file_id)
            if not vf:
                return None
            mv = session.get(ModelVersion, vf.version_id)
            if not mv:
                return None
            m = session.get(Model, mv.model_id)
            return {
                "version_id": mv.civitai_id,
                "model_id": m.civitai_id if m else None,
                "model_name": m.name if m else None,
                "version_name": mv.name,
            }

    def cache_model(self, data: dict[str, Any]) -> int:
        """Cache full model data from CivitAI API response."""
        with self.session() as session:
            creator_id = self._get_or_create_creator(session, data.get("creator"))
            civitai_id = data.get("id")
            existing = session.exec(select(Model).where(Model.civitai_id == civitai_id)).first()
            stats = data.get("stats", {})

            if existing:
                existing.name = data.get("name", existing.name)
                existing.description = data.get("description")
                existing.type = data.get("type", existing.type)
                existing.nsfw = bool(data.get("nsfw"))
                existing.download_count = stats.get("downloadCount", 0)
                existing.thumbs_up_count = stats.get("thumbsUpCount", 0)
                existing.updated_at = datetime.utcnow()
                session.add(existing)
                model_id = existing.id
            else:
                model = Model(
                    civitai_id=civitai_id,
                    name=data.get("name", ""),
                    description=data.get("description"),
                    type=data.get("type", ""),
                    nsfw=bool(data.get("nsfw")),
                    poi=bool(data.get("poi")),
                    minor=bool(data.get("minor")),
                    sfw_only=bool(data.get("sfwOnly")),
                    nsfw_level=data.get("nsfwLevel"),
                    availability=data.get("availability"),
                    allow_no_credit=bool(data.get("allowNoCredit")),
                    allow_commercial_use=str(data.get("allowCommercialUse", "")),
                    allow_derivatives=bool(data.get("allowDerivatives")),
                    allow_different_license=bool(data.get("allowDifferentLicense")),
                    supports_generation=bool(data.get("supportsGeneration")),
                    creator_id=creator_id,
                    download_count=stats.get("downloadCount", 0),
                    thumbs_up_count=stats.get("thumbsUpCount", 0),
                    thumbs_down_count=stats.get("thumbsDownCount", 0),
                    comment_count=stats.get("commentCount", 0),
                    tipped_amount_count=stats.get("tippedAmountCount", 0),
                )
                session.add(model)
                session.flush()
                model_id = model.id

            # Cache tags
            for tag_name in data.get("tags", []):
                tag_id = self._get_or_create_tag(session, tag_name)
                if model_id and tag_id:
                    existing_mt = session.exec(
                        select(ModelTag).where(ModelTag.model_id == model_id, ModelTag.tag_id == tag_id)
                    ).first()
                    if not existing_mt:
                        session.add(ModelTag(model_id=model_id, tag_id=tag_id))

            # Cache versions
            for idx, version in enumerate(data.get("modelVersions", [])):
                self._cache_version(session, model_id, version, idx)

            session.commit()
            return model_id or 0

    def _get_or_create_creator(self, session: Session, creator_data: dict[str, Any] | None) -> int | None:
        """Get or create a creator record."""
        if not creator_data:
            return None
        username = creator_data.get("username")
        if not username:
            return None

        existing = session.exec(select(Creator).where(Creator.username == username)).first()
        if existing:
            return existing.id

        creator = Creator(username=username, image_url=creator_data.get("image"))
        session.add(creator)
        session.flush()
        return creator.id

    def _get_or_create_tag(self, session: Session, tag_name: str) -> int | None:
        """Get or create a tag record."""
        existing = session.exec(select(Tag).where(Tag.name == tag_name)).first()
        if existing:
            return existing.id

        tag = Tag(name=tag_name)
        session.add(tag)
        session.flush()
        return tag.id

    def _cache_version(self, session: Session, model_id: int | None, version: dict[str, Any], index: int) -> int | None:
        """Cache a model version."""
        if not model_id:
            return None
        civitai_id = version.get("id")
        existing = session.exec(select(ModelVersion).where(ModelVersion.civitai_id == civitai_id)).first()
        stats = version.get("stats", {})

        if existing:
            version_id = existing.id
        else:
            mv = ModelVersion(
                civitai_id=civitai_id,
                model_id=model_id,
                name=version.get("name", ""),
                description=version.get("description"),
                base_model=version.get("baseModel"),
                base_model_type=version.get("baseModelType"),
                nsfw_level=version.get("nsfwLevel"),
                status=version.get("status"),
                availability=version.get("availability"),
                download_count=stats.get("downloadCount", 0),
                thumbs_up_count=stats.get("thumbsUpCount", 0),
                thumbs_down_count=stats.get("thumbsDownCount", 0),
                supports_generation=bool(version.get("supportsGeneration")),
                download_url=version.get("downloadUrl"),
                version_index=index,
            )
            session.add(mv)
            session.flush()
            version_id = mv.id

        # Cache trained words
        for pos, word in enumerate(version.get("trainedWords", [])):
            existing_tw = session.exec(
                select(TrainedWord).where(TrainedWord.version_id == version_id, TrainedWord.word == word)
            ).first()
            if not existing_tw:
                session.add(TrainedWord(version_id=version_id, word=word, position=pos))

        # Cache files
        for file_data in version.get("files", []):
            self._cache_file(session, version_id, file_data)

        # Cache images
        for image_data in version.get("images", []):
            self._cache_image(session, version_id, image_data)

        return version_id

    def _cache_file(self, session: Session, version_id: int | None, file_data: dict[str, Any]) -> int | None:
        """Cache a version file."""
        if not version_id:
            return None
        civitai_id = file_data.get("id")
        if not civitai_id:
            return None

        existing = session.exec(select(VersionFile).where(VersionFile.civitai_id == civitai_id)).first()
        if existing:
            return existing.id

        meta = file_data.get("metadata", {})
        vf = VersionFile(
            civitai_id=civitai_id,
            version_id=version_id,
            name=file_data.get("name", ""),
            type=file_data.get("type"),
            size_kb=file_data.get("sizeKB"),
            format=meta.get("format"),
            size_type=meta.get("size"),
            fp=meta.get("fp"),
            is_primary=bool(file_data.get("primary")),
            pickle_scan_result=file_data.get("pickleScanResult"),
            virus_scan_result=file_data.get("virusScanResult"),
            download_url=file_data.get("downloadUrl"),
        )
        session.add(vf)
        session.flush()
        file_id = vf.id

        # Cache hashes
        for hash_type, hash_value in file_data.get("hashes", {}).items():
            existing_fh = session.exec(
                select(FileHash).where(FileHash.file_id == file_id, FileHash.hash_type == hash_type)
            ).first()
            if not existing_fh:
                session.add(FileHash(file_id=file_id, hash_type=hash_type, hash_value=hash_value))

        return file_id

    def _cache_image(self, session: Session, version_id: int | None, image_data: dict[str, Any]) -> int | None:
        """Cache a version image."""
        if not version_id:
            return None
        url = image_data.get("url")
        if not url:
            return None

        existing = session.exec(select(VersionImage).where(VersionImage.url == url)).first()
        if existing:
            return existing.id

        vi = VersionImage(
            civitai_id=image_data.get("id"),
            version_id=version_id,
            url=url,
            type=image_data.get("type"),
            nsfw_level=image_data.get("nsfwLevel"),
            width=image_data.get("width"),
            height=image_data.get("height"),
            hash=image_data.get("hash"),
            has_meta=bool(image_data.get("hasMeta")),
            has_positive_prompt=bool(image_data.get("hasPositivePrompt")),
            on_site=bool(image_data.get("onSite")),
            minor=bool(image_data.get("minor")),
            poi=bool(image_data.get("poi")),
            availability=image_data.get("availability"),
        )
        session.add(vi)
        session.flush()
        image_id = vi.id

        # Cache generation params
        meta = image_data.get("meta", {})
        for key, value in meta.items():
            if key == "resources":
                continue
            str_value = str(value) if value is not None else None
            session.add(ImageGenerationParam(image_id=image_id, key=key, value=str_value))

        # Cache resources
        for res in meta.get("resources", []):
            session.add(
                ImageResource(
                    image_id=image_id,
                    name=res.get("name", ""),
                    type=res.get("type"),
                    hash=res.get("hash"),
                    weight=res.get("weight"),
                )
            )

        return image_id

    # =========================================================================
    # HuggingFace Cache Operations
    # =========================================================================

    def cache_hf_model(self, data: dict[str, Any]) -> int:
        """Cache HuggingFace model data."""
        repo_id = data.get("repo_id") or data.get("id") or data.get("modelId")
        if not repo_id:
            raise ValueError("repo_id is required")

        author = data.get("author")
        model_name = repo_id
        if "/" in repo_id:
            parts = repo_id.split("/", 1)
            author = author or parts[0]
            model_name = parts[1]

        with self.session() as session:
            existing = session.exec(select(HFModel).where(HFModel.repo_id == repo_id)).first()

            if existing:
                existing.author = author
                existing.model_name = model_name
                existing.pipeline_tag = data.get("pipeline_tag")
                existing.library_name = data.get("library_name")
                existing.downloads = data.get("downloads", 0)
                existing.likes = data.get("likes", 0)
                existing.trending_score = data.get("trending_score")
                existing.is_private = bool(data.get("private"))
                existing.is_gated = bool(data.get("gated"))
                existing.last_modified = data.get("last_modified") or data.get("lastModified")
                existing.updated_at = datetime.utcnow()
                session.add(existing)
                model_id = existing.id
            else:
                hf_model = HFModel(
                    repo_id=repo_id,
                    author=author,
                    model_name=model_name,
                    pipeline_tag=data.get("pipeline_tag"),
                    library_name=data.get("library_name"),
                    downloads=data.get("downloads", 0),
                    likes=data.get("likes", 0),
                    trending_score=data.get("trending_score"),
                    is_private=bool(data.get("private")),
                    is_gated=bool(data.get("gated")),
                    last_modified=data.get("last_modified") or data.get("lastModified"),
                    created_at=data.get("created_at") or data.get("createdAt"),
                )
                session.add(hf_model)
                session.flush()
                model_id = hf_model.id

            # Cache tags
            for tag in data.get("tags", []):
                existing_tag = session.exec(
                    select(HFModelTag).where(HFModelTag.hf_model_id == model_id, HFModelTag.tag == tag)
                ).first()
                if not existing_tag:
                    session.add(HFModelTag(hf_model_id=model_id, tag=tag))

            # Cache safetensor files
            for file_info in data.get("safetensor_files", []):
                if isinstance(file_info, str):
                    existing_sf = session.exec(
                        select(HFSafetensorFile).where(
                            HFSafetensorFile.hf_model_id == model_id, HFSafetensorFile.filename == file_info
                        )
                    ).first()
                    if not existing_sf:
                        session.add(HFSafetensorFile(hf_model_id=model_id, filename=file_info))
                elif isinstance(file_info, dict):
                    filename = file_info.get("filename")
                    if filename:
                        existing_sf = session.exec(
                            select(HFSafetensorFile).where(
                                HFSafetensorFile.hf_model_id == model_id, HFSafetensorFile.filename == filename
                            )
                        ).first()
                        if not existing_sf:
                            session.add(
                                HFSafetensorFile(hf_model_id=model_id, filename=filename, size_bytes=file_info.get("size"))
                            )

            session.commit()
            return model_id or 0

    def search_hf_models(
        self,
        query: str | None = None,
        author: str | None = None,
        pipeline_tag: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Search cached HuggingFace models."""
        with self.session() as session:
            stmt = select(HFModel)

            if query:
                stmt = stmt.where(col(HFModel.repo_id).contains(query) | col(HFModel.model_name).contains(query))
            if author:
                stmt = stmt.where(HFModel.author == author)
            if pipeline_tag:
                stmt = stmt.where(HFModel.pipeline_tag == pipeline_tag)

            stmt = stmt.order_by(col(HFModel.downloads).desc()).limit(limit)
            models = session.exec(stmt).all()

            return [
                {
                    "id": m.id,
                    "repo_id": m.repo_id,
                    "author": m.author,
                    "model_name": m.model_name,
                    "pipeline_tag": m.pipeline_tag,
                    "downloads": m.downloads,
                    "likes": m.likes,
                    "is_gated": m.is_gated,
                }
                for m in models
            ]

    def get_hf_model(self, repo_id: str) -> dict[str, Any] | None:
        """Get cached HF model by repo_id."""
        with self.session() as session:
            m = session.exec(select(HFModel).where(HFModel.repo_id == repo_id)).first()
            if not m:
                return None
            return {
                "id": m.id,
                "repo_id": m.repo_id,
                "author": m.author,
                "model_name": m.model_name,
                "pipeline_tag": m.pipeline_tag,
                "downloads": m.downloads,
                "likes": m.likes,
                "is_gated": m.is_gated,
            }

    def get_hf_safetensor_files(self, repo_id: str) -> list[dict[str, Any]]:
        """Get safetensor files for an HF model."""
        with self.session() as session:
            m = session.exec(select(HFModel).where(HFModel.repo_id == repo_id)).first()
            if not m:
                return []
            files = session.exec(select(HFSafetensorFile).where(HFSafetensorFile.hf_model_id == m.id)).all()
            return [{"filename": f.filename, "size_bytes": f.size_bytes} for f in files]

    # =========================================================================
    # Query Operations
    # =========================================================================

    def search_models(
        self,
        query: str | None = None,
        model_type: str | None = None,
        base_model: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Search cached CivitAI models."""
        with self.session() as session:
            stmt = select(Model)

            if query:
                stmt = stmt.where(col(Model.name).contains(query))
            if model_type:
                stmt = stmt.where(Model.type == model_type)

            stmt = stmt.order_by(col(Model.download_count).desc()).limit(limit)
            models = session.exec(stmt).all()

            results = []
            for m in models:
                # Get latest version
                latest = session.exec(
                    select(ModelVersion).where(ModelVersion.model_id == m.id, ModelVersion.version_index == 0)
                ).first()
                creator = session.get(Creator, m.creator_id) if m.creator_id else None

                # Filter by base_model if specified
                if base_model and latest and latest.base_model and base_model.lower() not in latest.base_model.lower():
                    continue

                results.append(
                    {
                        "id": m.id,
                        "civitai_id": m.civitai_id,
                        "name": m.name,
                        "type": m.type,
                        "nsfw": m.nsfw,
                        "creator": creator.username if creator else None,
                        "latest_version": latest.name if latest else None,
                        "base_model": latest.base_model if latest else None,
                        "download_count": m.download_count,
                        "thumbs_up_count": m.thumbs_up_count,
                    }
                )

            return results[:limit]

    def get_model(self, civitai_id: int) -> dict[str, Any] | None:
        """Get cached model by CivitAI ID."""
        with self.session() as session:
            m = session.exec(select(Model).where(Model.civitai_id == civitai_id)).first()
            if not m:
                return None
            latest = session.exec(
                select(ModelVersion).where(ModelVersion.model_id == m.id, ModelVersion.version_index == 0)
            ).first()
            creator = session.get(Creator, m.creator_id) if m.creator_id else None
            return {
                "id": m.id,
                "civitai_id": m.civitai_id,
                "name": m.name,
                "type": m.type,
                "creator": creator.username if creator else None,
                "latest_version": latest.name if latest else None,
                "base_model": latest.base_model if latest else None,
                "download_count": m.download_count,
            }

    def get_triggers(self, file_path: str) -> list[str]:
        """Get trigger words for a local file."""
        with self.session() as session:
            lf = session.exec(select(LocalFile).where(LocalFile.file_path == file_path)).first()
            if not lf or not lf.civitai_version_id:
                return []
            mv = session.exec(select(ModelVersion).where(ModelVersion.civitai_id == lf.civitai_version_id)).first()
            if not mv:
                return []
            words = session.exec(
                select(TrainedWord).where(TrainedWord.version_id == mv.id).order_by(col(TrainedWord.position))
            ).all()
            return [w.word for w in words]

    def get_triggers_by_version(self, version_id: int) -> list[str]:
        """Get trigger words for a version by CivitAI version ID."""
        with self.session() as session:
            mv = session.exec(select(ModelVersion).where(ModelVersion.civitai_id == version_id)).first()
            if not mv:
                return []
            words = session.exec(
                select(TrainedWord).where(TrainedWord.version_id == mv.id).order_by(col(TrainedWord.position))
            ).all()
            return [w.word for w in words]

    def get_trigger_words_by_filename(self, filename: str) -> list[str]:
        """Get trigger words for a LoRA by matching filename in version_files.

        Args:
            filename: The filename to search for (e.g., "spumcostyle.safetensors")

        Returns:
            List of trigger/trained words from CivitAI metadata
        """
        with self.session() as session:
            # Find version file by filename match
            vf = session.exec(select(VersionFile).where(VersionFile.name == filename)).first()
            if not vf:
                # Try partial match (without extension)
                base_name = filename.rsplit(".", 1)[0] if "." in filename else filename
                vf = session.exec(select(VersionFile).where(col(VersionFile.name).contains(base_name))).first()

            if not vf or not vf.version_id:
                return []

            words = session.exec(
                select(TrainedWord).where(TrainedWord.version_id == vf.version_id).order_by(col(TrainedWord.position))
            ).all()
            return [w.word for w in words]

    def get_base_model_by_filename(self, filename: str) -> str | None:
        """Get base_model for a checkpoint/LoRA by filename lookup.

        Args:
            filename: The filename to search for

        Returns:
            Base model string (e.g., "Pony", "SDXL 1.0") or None
        """
        with self.session() as session:
            # Find version file by filename match
            vf = session.exec(select(VersionFile).where(VersionFile.name == filename)).first()
            if not vf:
                # Try partial match (without extension)
                base_name = filename.rsplit(".", 1)[0] if "." in filename else filename
                vf = session.exec(select(VersionFile).where(col(VersionFile.name).contains(base_name))).first()

            if not vf or not vf.version_id:
                return None

            mv = session.get(ModelVersion, vf.version_id)
            return mv.base_model if mv else None

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self) -> dict[str, int]:
        """Get database statistics."""
        with self.session() as session:
            stats = {
                "local_files": session.exec(select(func.count(col(LocalFile.id)))).one(),
                "models": session.exec(select(func.count(col(Model.id)))).one(),
                "model_versions": session.exec(select(func.count(col(ModelVersion.id)))).one(),
                "version_files": session.exec(select(func.count(col(VersionFile.id)))).one(),
                "trained_words": session.exec(select(func.count(col(TrainedWord.id)))).one(),
                "creators": session.exec(select(func.count(col(Creator.id)))).one(),
                "tags": session.exec(select(func.count(col(Tag.id)))).one(),
                "hf_models": session.exec(select(func.count(col(HFModel.id)))).one(),
                "hf_safetensor_files": session.exec(select(func.count(col(HFSafetensorFile.id)))).one(),
            }
            return stats
