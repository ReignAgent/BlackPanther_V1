"""Experience Memory Store

SQLite-backed persistent storage for agent actions and system state
snapshots.  Enables post-hoc analysis, replay, and learning-from-
experience across penetration-testing campaigns.

All configuration (db path) comes from ``Settings`` — nothing hardcoded.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
from loguru import logger
from sqlalchemy import Column, Float, Integer, String, Text, create_engine, func
from sqlalchemy.orm import declarative_base, sessionmaker

from blackpanther.settings import Settings, get_settings

Base = declarative_base()


# ------------------------------------------------------------------
# ORM model
# ------------------------------------------------------------------

class ExperienceRow(Base):  # type: ignore[misc]
    __tablename__ = "experiences"

    id = Column(Integer, primary_key=True, autoincrement=True)
    agent_name = Column(String(64), nullable=False, index=True)
    target = Column(String(256), nullable=False, index=True)
    k_gain = Column(Float, default=0.0)
    s_inc = Column(Float, default=0.0)
    a_delta = Column(Float, default=0.0)
    success = Column(Integer, default=1)
    knowledge = Column(Float, default=0.0)
    suspicion_mean = Column(Float, default=0.0)
    access_global = Column(Float, default=0.0)
    episode = Column(Integer, default=0)
    timestamp = Column(Float, nullable=False)
    duration = Column(Float, default=0.0)
    raw_json = Column(Text, default="{}")


# ------------------------------------------------------------------
# Public dataclass
# ------------------------------------------------------------------

@dataclass
class Experience:
    """Single recorded agent action with system-state snapshot."""
    agent_name: str
    target: str
    k_gain: float = 0.0
    s_inc: float = 0.0
    a_delta: float = 0.0
    success: bool = True
    knowledge: float = 0.0
    suspicion_mean: float = 0.0
    access_global: float = 0.0
    episode: int = 0
    timestamp: float = field(default_factory=time.time)
    duration: float = 0.0
    raw_data: Dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------
# DRY helper — single place to build an ORM row from an Experience
# ------------------------------------------------------------------

def _experience_to_row(exp: Experience) -> ExperienceRow:
    return ExperienceRow(
        agent_name=exp.agent_name,
        target=exp.target,
        k_gain=exp.k_gain,
        s_inc=exp.s_inc,
        a_delta=exp.a_delta,
        success=int(exp.success),
        knowledge=exp.knowledge,
        suspicion_mean=exp.suspicion_mean,
        access_global=exp.access_global,
        episode=exp.episode,
        timestamp=exp.timestamp,
        duration=exp.duration,
        raw_json=json.dumps(exp.raw_data, default=str),
    )


def _row_to_experience(row: ExperienceRow) -> Experience:
    raw: Dict[str, Any] = {}
    try:
        raw = json.loads(row.raw_json) if row.raw_json else {}
    except json.JSONDecodeError:
        pass
    return Experience(
        agent_name=row.agent_name,
        target=row.target,
        k_gain=row.k_gain,
        s_inc=row.s_inc,
        a_delta=row.a_delta,
        success=bool(row.success),
        knowledge=row.knowledge,
        suspicion_mean=row.suspicion_mean,
        access_global=row.access_global,
        episode=row.episode,
        timestamp=row.timestamp,
        duration=row.duration,
        raw_data=raw,
    )


# ------------------------------------------------------------------
# Store
# ------------------------------------------------------------------

class MemoryStore:
    """SQLite-backed experience store.

    Args:
        db_path:  Path to the SQLite file.  ``":memory:"`` for tests.
        settings: Override settings (ignored when *db_path* is given).
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        cfg = settings or get_settings()
        resolved = db_path if db_path is not None else cfg.db_path

        if resolved != ":memory:":
            Path(resolved).parent.mkdir(parents=True, exist_ok=True)

        self._engine = create_engine(
            f"sqlite:///{resolved}",
            echo=False,
            future=True,
        )
        Base.metadata.create_all(self._engine)
        self._Session = sessionmaker(bind=self._engine)
        logger.info("MemoryStore ready  db={}", resolved)

    # ---- write ----

    def add(self, exp: Experience) -> int:
        """Persist an experience and return its row id."""
        row = _experience_to_row(exp)
        with self._Session() as session:
            session.add(row)
            session.commit()
            row_id: int = row.id  # type: ignore[assignment]
        return row_id

    def add_batch(self, exps: Sequence[Experience]) -> List[int]:
        """Persist many experiences in a single transaction."""
        with self._Session() as session:
            rows = [_experience_to_row(e) for e in exps]
            session.add_all(rows)
            session.commit()
            return [r.id for r in rows]  # type: ignore[union-attr]

    # ---- read ----

    def get_recent(self, n: int = 20) -> List[Experience]:
        with self._Session() as session:
            rows = (
                session.query(ExperienceRow)
                .order_by(ExperienceRow.id.desc())
                .limit(n)
                .all()
            )
        return [_row_to_experience(r) for r in reversed(rows)]

    def get_by_target(self, target: str) -> List[Experience]:
        with self._Session() as session:
            rows = (
                session.query(ExperienceRow)
                .filter(ExperienceRow.target == target)
                .order_by(ExperienceRow.id)
                .all()
            )
        return [_row_to_experience(r) for r in rows]

    def get_by_agent(self, agent_name: str) -> List[Experience]:
        with self._Session() as session:
            rows = (
                session.query(ExperienceRow)
                .filter(ExperienceRow.agent_name == agent_name)
                .order_by(ExperienceRow.id)
                .all()
            )
        return [_row_to_experience(r) for r in rows]

    def query(
        self,
        agent_name: Optional[str] = None,
        target: Optional[str] = None,
        min_k_gain: Optional[float] = None,
        success_only: bool = False,
        limit: int = 100,
    ) -> List[Experience]:
        with self._Session() as session:
            q = session.query(ExperienceRow)
            if agent_name:
                q = q.filter(ExperienceRow.agent_name == agent_name)
            if target:
                q = q.filter(ExperienceRow.target == target)
            if min_k_gain is not None:
                q = q.filter(ExperienceRow.k_gain >= min_k_gain)
            if success_only:
                q = q.filter(ExperienceRow.success == 1)
            rows = q.order_by(ExperienceRow.id.desc()).limit(limit).all()
        return [_row_to_experience(r) for r in reversed(rows)]

    @property
    def count(self) -> int:
        with self._Session() as session:
            return session.query(ExperienceRow).count()

    # ---- analytics ----

    def to_dataframe(self) -> pd.DataFrame:
        with self._Session() as session:
            rows = session.query(ExperienceRow).order_by(ExperienceRow.id).all()
        return pd.DataFrame([
            {
                "id": r.id, "agent_name": r.agent_name, "target": r.target,
                "k_gain": r.k_gain, "s_inc": r.s_inc, "a_delta": r.a_delta,
                "success": bool(r.success), "knowledge": r.knowledge,
                "suspicion_mean": r.suspicion_mean, "access_global": r.access_global,
                "episode": r.episode, "timestamp": r.timestamp, "duration": r.duration,
            }
            for r in rows
        ])

    def success_rate(self, agent_name: Optional[str] = None) -> float:
        with self._Session() as session:
            q = session.query(ExperienceRow)
            if agent_name:
                q = q.filter(ExperienceRow.agent_name == agent_name)
            total = q.count()
            if total == 0:
                return 0.0
            wins = q.filter(ExperienceRow.success == 1).count()
        return wins / total

    def avg_k_gain(self, agent_name: Optional[str] = None) -> float:
        with self._Session() as session:
            q = session.query(func.avg(ExperienceRow.k_gain))
            if agent_name:
                q = q.filter(ExperienceRow.agent_name == agent_name)
            val = q.scalar()
        return float(val) if val else 0.0

    def suspicion_trend(self, last_n: int = 50) -> List[float]:
        with self._Session() as session:
            rows = (
                session.query(ExperienceRow.suspicion_mean)
                .order_by(ExperienceRow.id.desc())
                .limit(last_n)
                .all()
            )
        return [r[0] for r in reversed(rows)]
