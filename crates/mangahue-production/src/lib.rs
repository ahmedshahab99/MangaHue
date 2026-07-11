use anyhow::{Context, Result};
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use chrono::Utc;
use rand::RngCore;
use rusqlite::{params, Connection, DatabaseName, OptionalExtension, TransactionBehavior};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};
use uuid::Uuid;

const SCHEMA_VERSION: i64 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UserTier {
    Free,
    Creator,
    Studio,
    Admin,
}

impl UserTier {
    fn as_str(self) -> &'static str {
        match self {
            Self::Free => "free",
            Self::Creator => "creator",
            Self::Studio => "studio",
            Self::Admin => "admin",
        }
    }

    fn parse(value: &str) -> Result<Self> {
        match value {
            "free" => Ok(Self::Free),
            "creator" => Ok(Self::Creator),
            "studio" => Ok(Self::Studio),
            "admin" => Ok(Self::Admin),
            _ => anyhow::bail!("unknown user tier {value}"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserAccount {
    pub id: String,
    pub email: String,
    pub display_name: String,
    pub tier: UserTier,
    pub active: bool,
    pub created_at: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthSession {
    pub token: String,
    pub token_id: String,
    pub user: UserAccount,
    pub expires_at: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuotaPolicy {
    pub daily_pages: u64,
    pub daily_jobs: u64,
    pub max_pages_per_job: u64,
    pub max_concurrent_jobs: u64,
    pub max_output_bytes: u64,
}

impl QuotaPolicy {
    pub fn for_tier(tier: UserTier) -> Self {
        match tier {
            UserTier::Free => Self {
                daily_pages: 20,
                daily_jobs: 5,
                max_pages_per_job: 8,
                max_concurrent_jobs: 1,
                max_output_bytes: 250 * 1024 * 1024,
            },
            UserTier::Creator => Self {
                daily_pages: 500,
                daily_jobs: 50,
                max_pages_per_job: 100,
                max_concurrent_jobs: 3,
                max_output_bytes: 10 * 1024 * 1024 * 1024,
            },
            UserTier::Studio => Self {
                daily_pages: 10_000,
                daily_jobs: 500,
                max_pages_per_job: 1_000,
                max_concurrent_jobs: 16,
                max_output_bytes: 500 * 1024 * 1024 * 1024,
            },
            UserTier::Admin => Self {
                daily_pages: u64::MAX,
                daily_jobs: u64::MAX,
                max_pages_per_job: u64::MAX,
                max_concurrent_jobs: u64::MAX,
                max_output_bytes: u64::MAX,
            },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuotaSnapshot {
    pub policy: QuotaPolicy,
    pub pages_used: u64,
    pub jobs_used: u64,
    pub active_jobs: u64,
    pub output_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobStatus {
    Queued,
    Running,
    Complete,
    Failed,
    Cancelled,
    DeadLetter,
}

impl JobStatus {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "queued" => Ok(Self::Queued),
            "running" => Ok(Self::Running),
            "complete" => Ok(Self::Complete),
            "failed" => Ok(Self::Failed),
            "cancelled" => Ok(Self::Cancelled),
            "dead_letter" => Ok(Self::DeadLetter),
            _ => anyhow::bail!("unknown job status {value}"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DurableJob {
    pub id: String,
    pub user_id: String,
    pub kind: String,
    pub payload: serde_json::Value,
    pub status: JobStatus,
    pub priority: i64,
    pub attempt: u32,
    pub max_attempts: u32,
    pub lease_owner: Option<String>,
    pub lease_expires_at: Option<i64>,
    pub progress: f32,
    pub result: Option<serde_json::Value>,
    pub error: Option<String>,
    pub created_at: i64,
    pub updated_at: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewJob {
    pub user_id: String,
    pub kind: String,
    pub payload: serde_json::Value,
    pub priority: i64,
    pub max_attempts: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactRecord {
    pub id: String,
    pub user_id: String,
    pub job_id: Option<String>,
    pub kind: String,
    pub object_key: String,
    pub sha256: String,
    pub bytes: u64,
    pub mime: String,
    pub created_at: i64,
    pub expires_at: Option<i64>,
}

#[derive(Debug, Clone)]
pub struct ProductionStore {
    connection: Arc<Mutex<Connection>>,
}

impl ProductionStore {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
        let connection = Connection::open(path)
            .with_context(|| format!("failed to open production database {}", path.display()))?;
        connection.pragma_update(None, "journal_mode", "WAL")?;
        connection.pragma_update(None, "foreign_keys", "ON")?;
        connection.busy_timeout(std::time::Duration::from_secs(10))?;
        let store = Self {
            connection: Arc::new(Mutex::new(connection)),
        };
        store.migrate()?;
        Ok(store)
    }

    pub fn open_memory() -> Result<Self> {
        let connection = Connection::open_in_memory()?;
        connection.pragma_update(None, "foreign_keys", "ON")?;
        let store = Self {
            connection: Arc::new(Mutex::new(connection)),
        };
        store.migrate()?;
        Ok(store)
    }

    fn lock(&self) -> Result<MutexGuard<'_, Connection>> {
        self.connection
            .lock()
            .map_err(|_| anyhow::anyhow!("production database lock poisoned"))
    }

    fn migrate(&self) -> Result<()> {
        let connection = self.lock()?;
        connection.execute_batch(
            "BEGIN;
             CREATE TABLE IF NOT EXISTS schema_meta(version INTEGER NOT NULL);
             INSERT INTO schema_meta(version) SELECT 0 WHERE NOT EXISTS(SELECT 1 FROM schema_meta);
             CREATE TABLE IF NOT EXISTS users(
               id TEXT PRIMARY KEY, email TEXT NOT NULL UNIQUE COLLATE NOCASE,
               display_name TEXT NOT NULL, tier TEXT NOT NULL, active INTEGER NOT NULL,
               created_at INTEGER NOT NULL
             );
             CREATE TABLE IF NOT EXISTS api_tokens(
               id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
               token_hash TEXT NOT NULL UNIQUE, label TEXT NOT NULL, expires_at INTEGER,
               revoked_at INTEGER, created_at INTEGER NOT NULL
             );
             CREATE INDEX IF NOT EXISTS idx_api_tokens_user ON api_tokens(user_id);
             CREATE TABLE IF NOT EXISTS subscriptions(
               id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
               provider TEXT NOT NULL, external_id TEXT, status TEXT NOT NULL, tier TEXT NOT NULL,
               current_period_end INTEGER, updated_at INTEGER NOT NULL,
               UNIQUE(provider, external_id)
             );
             CREATE TABLE IF NOT EXISTS jobs(
               id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id), kind TEXT NOT NULL,
               payload_json TEXT NOT NULL, status TEXT NOT NULL, priority INTEGER NOT NULL,
               attempt INTEGER NOT NULL, max_attempts INTEGER NOT NULL,
               lease_owner TEXT, lease_expires_at INTEGER, progress REAL NOT NULL,
               result_json TEXT, error TEXT, created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL
             );
             CREATE INDEX IF NOT EXISTS idx_jobs_claim ON jobs(status, priority DESC, created_at);
             CREATE INDEX IF NOT EXISTS idx_jobs_user ON jobs(user_id, created_at DESC);
             CREATE TABLE IF NOT EXISTS usage_ledger(
               id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id), metric TEXT NOT NULL,
               units INTEGER NOT NULL, job_id TEXT, created_at INTEGER NOT NULL
             );
             CREATE INDEX IF NOT EXISTS idx_usage_user_time ON usage_ledger(user_id, created_at);
             CREATE TABLE IF NOT EXISTS artifacts(
               id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id), job_id TEXT,
               kind TEXT NOT NULL, object_key TEXT NOT NULL UNIQUE, sha256 TEXT NOT NULL,
               bytes INTEGER NOT NULL, mime TEXT NOT NULL, created_at INTEGER NOT NULL,
               expires_at INTEGER
             );
             CREATE INDEX IF NOT EXISTS idx_artifacts_expiry ON artifacts(expires_at);
             CREATE TABLE IF NOT EXISTS moderation_events(
               id TEXT PRIMARY KEY, user_id TEXT, job_id TEXT, decision TEXT NOT NULL,
               reason TEXT NOT NULL, details_json TEXT NOT NULL, created_at INTEGER NOT NULL
             );
             UPDATE schema_meta SET version = 1;
             COMMIT;",
        )?;
        let version: i64 =
            connection.query_row("SELECT version FROM schema_meta", [], |row| row.get(0))?;
        anyhow::ensure!(
            version == SCHEMA_VERSION,
            "unsupported database schema {version}"
        );
        Ok(())
    }

    pub fn create_user(
        &self,
        email: impl Into<String>,
        display_name: impl Into<String>,
        tier: UserTier,
    ) -> Result<UserAccount> {
        let account = UserAccount {
            id: Uuid::new_v4().to_string(),
            email: email.into().trim().to_ascii_lowercase(),
            display_name: display_name.into().trim().to_string(),
            tier,
            active: true,
            created_at: now(),
        };
        anyhow::ensure!(account.email.contains('@'), "invalid email address");
        anyhow::ensure!(
            !account.display_name.is_empty(),
            "display name cannot be empty"
        );
        self.lock()?.execute(
            "INSERT INTO users(id,email,display_name,tier,active,created_at) VALUES(?1,?2,?3,?4,1,?5)",
            params![account.id, account.email, account.display_name, tier.as_str(), account.created_at],
        )?;
        Ok(account)
    }

    pub fn issue_token(
        &self,
        user_id: &str,
        label: &str,
        expires_at: Option<i64>,
    ) -> Result<AuthSession> {
        let user = self.user(user_id)?.context("user not found")?;
        anyhow::ensure!(user.active, "user is disabled");
        let mut secret = [0u8; 32];
        rand::thread_rng().fill_bytes(&mut secret);
        let token = format!("mh_{}", URL_SAFE_NO_PAD.encode(secret));
        let token_id = Uuid::new_v4().to_string();
        self.lock()?.execute(
            "INSERT INTO api_tokens(id,user_id,token_hash,label,expires_at,created_at) VALUES(?1,?2,?3,?4,?5,?6)",
            params![token_id, user_id, hash_token(&token), label, expires_at, now()],
        )?;
        Ok(AuthSession {
            token,
            token_id,
            user,
            expires_at,
        })
    }

    pub fn authenticate(&self, token: &str) -> Result<Option<UserAccount>> {
        let current = now();
        let connection = self.lock()?;
        let mut statement = connection.prepare(
            "SELECT u.id,u.email,u.display_name,u.tier,u.active,u.created_at
             FROM api_tokens t JOIN users u ON u.id=t.user_id
             WHERE t.token_hash=?1 AND t.revoked_at IS NULL
               AND (t.expires_at IS NULL OR t.expires_at>?2) AND u.active=1",
        )?;
        let row = statement
            .query_row(params![hash_token(token), current], user_from_row)
            .optional()?;
        row.map(parse_user_row).transpose()
    }

    pub fn revoke_token(&self, token_id: &str) -> Result<bool> {
        Ok(self.lock()?.execute(
            "UPDATE api_tokens SET revoked_at=?2 WHERE id=?1 AND revoked_at IS NULL",
            params![token_id, now()],
        )? > 0)
    }

    pub fn user(&self, id: &str) -> Result<Option<UserAccount>> {
        let connection = self.lock()?;
        let mut statement = connection.prepare(
            "SELECT id,email,display_name,tier,active,created_at FROM users WHERE id=?1",
        )?;
        statement
            .query_row([id], user_from_row)
            .optional()?
            .map(parse_user_row)
            .transpose()
    }

    pub fn quota_snapshot(&self, user_id: &str) -> Result<QuotaSnapshot> {
        let user = self.user(user_id)?.context("user not found")?;
        let since = now() - 86_400;
        let connection = self.lock()?;
        let metric = |name: &str| -> Result<u64> {
            let value: i64 = connection.query_row(
                "SELECT COALESCE(SUM(units),0) FROM usage_ledger WHERE user_id=?1 AND metric=?2 AND created_at>=?3",
                params![user_id, name, since],
                |row| row.get(0),
            )?;
            Ok(value.max(0) as u64)
        };
        let active_jobs: i64 = connection.query_row(
            "SELECT COUNT(*) FROM jobs WHERE user_id=?1 AND status IN ('queued','running')",
            [user_id],
            |row| row.get(0),
        )?;
        Ok(QuotaSnapshot {
            policy: QuotaPolicy::for_tier(user.tier),
            pages_used: metric("pages")?,
            jobs_used: metric("jobs")?,
            output_bytes: metric("output_bytes")?,
            active_jobs: active_jobs.max(0) as u64,
        })
    }

    pub fn enforce_job_quota(&self, user_id: &str, requested_pages: u64) -> Result<QuotaSnapshot> {
        let snapshot = self.quota_snapshot(user_id)?;
        anyhow::ensure!(requested_pages > 0, "job must contain at least one page");
        anyhow::ensure!(
            requested_pages <= snapshot.policy.max_pages_per_job,
            "job page count exceeds tier limit"
        );
        anyhow::ensure!(
            snapshot.jobs_used < snapshot.policy.daily_jobs,
            "daily job quota exhausted"
        );
        anyhow::ensure!(
            snapshot.pages_used.saturating_add(requested_pages) <= snapshot.policy.daily_pages,
            "daily page quota exhausted"
        );
        anyhow::ensure!(
            snapshot.active_jobs < snapshot.policy.max_concurrent_jobs,
            "concurrent job quota exhausted"
        );
        Ok(snapshot)
    }

    pub fn record_usage(
        &self,
        user_id: &str,
        metric: &str,
        units: u64,
        job_id: Option<&str>,
    ) -> Result<()> {
        anyhow::ensure!(units <= i64::MAX as u64, "usage value is too large");
        self.lock()?.execute(
            "INSERT INTO usage_ledger(id,user_id,metric,units,job_id,created_at) VALUES(?1,?2,?3,?4,?5,?6)",
            params![Uuid::new_v4().to_string(), user_id, metric, units as i64, job_id, now()],
        )?;
        Ok(())
    }

    pub fn enqueue(&self, new_job: NewJob) -> Result<DurableJob> {
        let requested_pages = new_job
            .payload
            .get("requested_pages")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(1);
        self.enforce_job_quota(&new_job.user_id, requested_pages)?;
        let timestamp = now();
        let job = DurableJob {
            id: Uuid::new_v4().to_string(),
            user_id: new_job.user_id,
            kind: new_job.kind,
            payload: new_job.payload,
            status: JobStatus::Queued,
            priority: new_job.priority,
            attempt: 0,
            max_attempts: new_job.max_attempts.max(1),
            lease_owner: None,
            lease_expires_at: None,
            progress: 0.0,
            result: None,
            error: None,
            created_at: timestamp,
            updated_at: timestamp,
        };
        let mut connection = self.lock()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        transaction.execute(
            "INSERT INTO jobs(id,user_id,kind,payload_json,status,priority,attempt,max_attempts,progress,created_at,updated_at)
             VALUES(?1,?2,?3,?4,'queued',?5,0,?6,0,?7,?7)",
            params![job.id, job.user_id, job.kind, serde_json::to_string(&job.payload)?, job.priority, job.max_attempts, timestamp],
        )?;
        transaction.execute(
            "INSERT INTO usage_ledger(id,user_id,metric,units,job_id,created_at) VALUES(?1,?2,'jobs',1,?3,?4)",
            params![Uuid::new_v4().to_string(), job.user_id, job.id, timestamp],
        )?;
        transaction.commit()?;
        Ok(job)
    }

    pub fn claim(&self, worker: &str, lease_seconds: i64) -> Result<Option<DurableJob>> {
        anyhow::ensure!(!worker.trim().is_empty(), "worker id cannot be empty");
        let current = now();
        let mut connection = self.lock()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        transaction.execute(
            "UPDATE jobs SET status=CASE WHEN attempt+1>=max_attempts THEN 'dead_letter' ELSE 'queued' END,
             lease_owner=NULL,lease_expires_at=NULL,error='worker lease expired',updated_at=?1
             WHERE status='running' AND lease_expires_at<=?1",
            [current],
        )?;
        let id: Option<String> = transaction
            .query_row(
                "SELECT id FROM jobs WHERE status='queued' ORDER BY priority DESC,created_at ASC LIMIT 1",
                [],
                |row| row.get(0),
            )
            .optional()?;
        let Some(id) = id else {
            transaction.commit()?;
            return Ok(None);
        };
        transaction.execute(
            "UPDATE jobs SET status='running',attempt=attempt+1,lease_owner=?2,
             lease_expires_at=?3,updated_at=?1 WHERE id=?4 AND status='queued'",
            params![current, worker, current + lease_seconds.max(5), id],
        )?;
        let job = load_job_from(&transaction, &id)?.context("claimed job disappeared")?;
        transaction.commit()?;
        Ok(Some(job))
    }

    pub fn heartbeat(
        &self,
        job_id: &str,
        worker: &str,
        lease_seconds: i64,
        progress: f32,
    ) -> Result<bool> {
        let current = now();
        Ok(self.lock()?.execute(
            "UPDATE jobs SET lease_expires_at=?1,progress=?2,updated_at=?3
             WHERE id=?4 AND status='running' AND lease_owner=?5",
            params![
                current + lease_seconds.max(5),
                progress.clamp(0.0, 100.0),
                current,
                job_id,
                worker
            ],
        )? > 0)
    }

    pub fn complete(&self, job_id: &str, worker: &str, result: &serde_json::Value) -> Result<bool> {
        Ok(self.lock()?.execute(
            "UPDATE jobs SET status='complete',progress=100,result_json=?1,error=NULL,
             lease_owner=NULL,lease_expires_at=NULL,updated_at=?2
             WHERE id=?3 AND status='running' AND lease_owner=?4",
            params![serde_json::to_string(result)?, now(), job_id, worker],
        )? > 0)
    }

    pub fn fail(&self, job_id: &str, worker: &str, error: &str, retryable: bool) -> Result<bool> {
        let connection = self.lock()?;
        let changed = connection.execute(
            "UPDATE jobs SET status=CASE
               WHEN ?1=1 AND attempt<max_attempts THEN 'queued'
               WHEN attempt>=max_attempts THEN 'dead_letter' ELSE 'failed' END,
             error=?2,lease_owner=NULL,lease_expires_at=NULL,updated_at=?3
             WHERE id=?4 AND status='running' AND lease_owner=?5",
            params![retryable as i32, error, now(), job_id, worker],
        )?;
        Ok(changed > 0)
    }

    pub fn cancel(&self, job_id: &str, user_id: &str) -> Result<bool> {
        Ok(self.lock()?.execute(
            "UPDATE jobs SET status='cancelled',lease_owner=NULL,lease_expires_at=NULL,updated_at=?1
             WHERE id=?2 AND user_id=?3 AND status IN ('queued','running')",
            params![now(), job_id, user_id],
        )? > 0)
    }

    pub fn job(&self, id: &str) -> Result<Option<DurableJob>> {
        load_job_from(&*self.lock()?, id)
    }

    pub fn jobs_for_user(&self, user_id: &str, limit: usize) -> Result<Vec<DurableJob>> {
        let ids = {
            let connection = self.lock()?;
            let mut statement = connection.prepare(
                "SELECT id FROM jobs WHERE user_id=?1 ORDER BY created_at DESC LIMIT ?2",
            )?;
            let rows = statement
                .query_map(params![user_id, limit.clamp(1, 500) as i64], |row| {
                    row.get::<_, String>(0)
                })?;
            rows.collect::<rusqlite::Result<Vec<_>>>()?
        };
        ids.into_iter()
            .map(|id| self.job(&id)?.context("job disappeared while listing"))
            .collect()
    }

    pub fn register_artifact(&self, artifact: &ArtifactRecord) -> Result<()> {
        self.lock()?.execute(
            "INSERT INTO artifacts(id,user_id,job_id,kind,object_key,sha256,bytes,mime,created_at,expires_at)
             VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10)",
            params![artifact.id, artifact.user_id, artifact.job_id, artifact.kind, artifact.object_key,
                artifact.sha256, artifact.bytes as i64, artifact.mime, artifact.created_at, artifact.expires_at],
        )?;
        Ok(())
    }

    pub fn expired_artifacts(&self, limit: usize) -> Result<Vec<ArtifactRecord>> {
        let connection = self.lock()?;
        let mut statement = connection.prepare(
            "SELECT id,user_id,job_id,kind,object_key,sha256,bytes,mime,created_at,expires_at
             FROM artifacts WHERE expires_at IS NOT NULL AND expires_at<=?1 ORDER BY expires_at LIMIT ?2",
        )?;
        let rows =
            statement.query_map(params![now(), limit.min(10_000) as i64], artifact_from_row)?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Into::into)
    }

    pub fn delete_artifact_record(&self, id: &str) -> Result<bool> {
        Ok(self
            .lock()?
            .execute("DELETE FROM artifacts WHERE id=?1", [id])?
            > 0)
    }

    pub fn enforce_rate_limit(
        &self,
        user_id: &str,
        limit: u64,
        window_seconds: i64,
    ) -> Result<u64> {
        let current = now();
        let mut connection = self.lock()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let used: i64 = transaction.query_row(
            "SELECT COALESCE(SUM(units),0) FROM usage_ledger
             WHERE user_id=?1 AND metric='api_requests' AND created_at>=?2",
            params![user_id, current - window_seconds.max(1)],
            |row| row.get(0),
        )?;
        anyhow::ensure!((used.max(0) as u64) < limit, "API rate limit exceeded");
        transaction.execute(
            "INSERT INTO usage_ledger(id,user_id,metric,units,created_at) VALUES(?1,?2,'api_requests',1,?3)",
            params![Uuid::new_v4().to_string(), user_id, current],
        )?;
        transaction.commit()?;
        Ok(limit.saturating_sub(used.max(0) as u64 + 1))
    }

    pub fn apply_billing_event(&self, event: &BillingEvent) -> Result<()> {
        let mut connection = self.lock()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        transaction.execute(
            "INSERT INTO subscriptions(id,user_id,provider,external_id,status,tier,current_period_end,updated_at)
             VALUES(?1,?2,?3,?4,?5,?6,?7,?8)
             ON CONFLICT(provider,external_id) DO UPDATE SET status=excluded.status,tier=excluded.tier,
             current_period_end=excluded.current_period_end,updated_at=excluded.updated_at",
            params![Uuid::new_v4().to_string(), event.user_id, event.provider, event.external_id,
                event.status, event.tier.as_str(), event.current_period_end, now()],
        )?;
        if matches!(event.status.as_str(), "active" | "trialing") {
            transaction.execute(
                "UPDATE users SET tier=?1 WHERE id=?2",
                params![event.tier.as_str(), event.user_id],
            )?;
        } else if matches!(event.status.as_str(), "cancelled" | "expired" | "unpaid") {
            transaction.execute("UPDATE users SET tier='free' WHERE id=?1", [&event.user_id])?;
        }
        transaction.commit()?;
        Ok(())
    }

    pub fn record_moderation(&self, event: &ModerationEvent) -> Result<()> {
        self.lock()?.execute(
            "INSERT INTO moderation_events(id,user_id,job_id,decision,reason,details_json,created_at)
             VALUES(?1,?2,?3,?4,?5,?6,?7)",
            params![event.id, event.user_id, event.job_id, event.decision, event.reason,
                serde_json::to_string(&event.details)?, event.created_at],
        )?;
        Ok(())
    }

    pub fn metrics(&self) -> Result<ServiceMetrics> {
        let connection = self.lock()?;
        let count = |query: &str| -> Result<u64> {
            let value: i64 = connection.query_row(query, [], |row| row.get(0))?;
            Ok(value.max(0) as u64)
        };
        let pages_24h: i64 = connection.query_row(
            "SELECT COALESCE(SUM(units),0) FROM usage_ledger WHERE metric='pages' AND created_at>=?1",
            [now() - 86_400],
            |row| row.get(0),
        )?;
        Ok(ServiceMetrics {
            users: count("SELECT COUNT(*) FROM users")?,
            queued_jobs: count("SELECT COUNT(*) FROM jobs WHERE status='queued'")?,
            running_jobs: count("SELECT COUNT(*) FROM jobs WHERE status='running'")?,
            completed_jobs: count("SELECT COUNT(*) FROM jobs WHERE status='complete'")?,
            failed_jobs: count("SELECT COUNT(*) FROM jobs WHERE status='failed'")?,
            dead_letter_jobs: count("SELECT COUNT(*) FROM jobs WHERE status='dead_letter'")?,
            artifacts: count("SELECT COUNT(*) FROM artifacts")?,
            artifact_bytes: count("SELECT COALESCE(SUM(bytes),0) FROM artifacts")?,
            pages_24h: pages_24h.max(0) as u64,
        })
    }

    pub fn backup(&self, destination: impl AsRef<Path>) -> Result<()> {
        let destination = destination.as_ref();
        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent)?;
        }
        self.lock()?
            .backup(DatabaseName::Main, destination, None)
            .with_context(|| format!("failed to back up database to {}", destination.display()))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionReport {
    pub inspected: usize,
    pub deleted_objects: usize,
    pub deleted_records: usize,
    pub failures: Vec<String>,
}

pub fn sweep_retention(
    store: &ProductionStore,
    objects: &dyn ObjectStore,
    limit: usize,
) -> Result<RetentionReport> {
    let expired = store.expired_artifacts(limit)?;
    let mut report = RetentionReport {
        inspected: expired.len(),
        deleted_objects: 0,
        deleted_records: 0,
        failures: Vec::new(),
    };
    for artifact in expired {
        match objects.delete(&artifact.object_key) {
            Ok(true) => report.deleted_objects += 1,
            Ok(false) => {}
            Err(error) => {
                report
                    .failures
                    .push(format!("{}: {error}", artifact.object_key));
                continue;
            }
        }
        if store.delete_artifact_record(&artifact.id)? {
            report.deleted_records += 1;
        }
    }
    Ok(report)
}

pub trait ObjectStore: Send + Sync {
    fn put(&self, namespace: &str, bytes: &[u8]) -> Result<StoredObject>;
    fn get(&self, key: &str) -> Result<Vec<u8>>;
    fn delete(&self, key: &str) -> Result<bool>;
    fn exists(&self, key: &str) -> Result<bool>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredObject {
    pub key: String,
    pub sha256: String,
    pub bytes: u64,
}

#[derive(Debug, Clone)]
pub struct LocalObjectStore {
    root: PathBuf,
}

impl LocalObjectStore {
    pub fn new(root: impl Into<PathBuf>) -> Result<Self> {
        let root = root.into();
        fs::create_dir_all(&root)
            .with_context(|| format!("failed to create object store {}", root.display()))?;
        Ok(Self { root })
    }

    fn resolve(&self, key: &str) -> Result<PathBuf> {
        anyhow::ensure!(
            !key.is_empty()
                && !key.starts_with('/')
                && !key.contains("..")
                && key
                    .chars()
                    .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '/' | '-' | '_' | '.')),
            "invalid object key"
        );
        Ok(self.root.join(key))
    }
}

impl ObjectStore for LocalObjectStore {
    fn put(&self, namespace: &str, bytes: &[u8]) -> Result<StoredObject> {
        let namespace = namespace.trim_matches('/');
        anyhow::ensure!(
            !namespace.is_empty() && !namespace.contains(".."),
            "invalid namespace"
        );
        let sha256 = hex_sha256(bytes);
        let key = format!("{}/{}/{}", namespace, &sha256[..2], sha256);
        let path = self.resolve(&key)?;
        if !path.exists() {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }
            let temporary = path.with_extension(format!("tmp-{}", Uuid::new_v4()));
            fs::write(&temporary, bytes)?;
            match fs::rename(&temporary, &path) {
                Ok(()) => {}
                Err(error) if path.exists() => {
                    let _ = fs::remove_file(&temporary);
                    let _ = error;
                }
                Err(error) => return Err(error.into()),
            }
        }
        Ok(StoredObject {
            key,
            sha256,
            bytes: bytes.len() as u64,
        })
    }

    fn get(&self, key: &str) -> Result<Vec<u8>> {
        let path = self.resolve(key)?;
        fs::read(&path).with_context(|| format!("failed to read object {}", path.display()))
    }

    fn delete(&self, key: &str) -> Result<bool> {
        let path = self.resolve(key)?;
        if !path.exists() {
            return Ok(false);
        }
        fs::remove_file(path)?;
        Ok(true)
    }

    fn exists(&self, key: &str) -> Result<bool> {
        Ok(self.resolve(key)?.is_file())
    }
}

pub trait BillingProvider: Send + Sync {
    fn checkout_url(&self, user: &UserAccount, tier: UserTier, return_url: &str) -> Result<String>;
    fn portal_url(&self, user: &UserAccount, return_url: &str) -> Result<String>;
    fn verify_webhook(&self, signature: &str, payload: &[u8]) -> Result<BillingEvent>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BillingEvent {
    pub provider: String,
    pub external_id: String,
    pub user_id: String,
    pub status: String,
    pub tier: UserTier,
    pub current_period_end: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModerationEvent {
    pub id: String,
    pub user_id: Option<String>,
    pub job_id: Option<String>,
    pub decision: String,
    pub reason: String,
    pub details: serde_json::Value,
    pub created_at: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceMetrics {
    pub users: u64,
    pub queued_jobs: u64,
    pub running_jobs: u64,
    pub completed_jobs: u64,
    pub failed_jobs: u64,
    pub dead_letter_jobs: u64,
    pub artifacts: u64,
    pub artifact_bytes: u64,
    pub pages_24h: u64,
}

#[derive(Debug, Clone)]
pub struct DisabledBillingProvider;

impl BillingProvider for DisabledBillingProvider {
    fn checkout_url(
        &self,
        _user: &UserAccount,
        _tier: UserTier,
        _return_url: &str,
    ) -> Result<String> {
        anyhow::bail!("billing provider is not configured")
    }

    fn portal_url(&self, _user: &UserAccount, _return_url: &str) -> Result<String> {
        anyhow::bail!("billing provider is not configured")
    }

    fn verify_webhook(&self, _signature: &str, _payload: &[u8]) -> Result<BillingEvent> {
        anyhow::bail!("billing provider is not configured")
    }
}

fn user_from_row(
    row: &rusqlite::Row<'_>,
) -> rusqlite::Result<(String, String, String, String, bool, i64)> {
    Ok((
        row.get(0)?,
        row.get(1)?,
        row.get(2)?,
        row.get(3)?,
        row.get(4)?,
        row.get(5)?,
    ))
}

fn parse_user_row(row: (String, String, String, String, bool, i64)) -> Result<UserAccount> {
    Ok(UserAccount {
        id: row.0,
        email: row.1,
        display_name: row.2,
        tier: UserTier::parse(&row.3)?,
        active: row.4,
        created_at: row.5,
    })
}

fn load_job_from(connection: &Connection, id: &str) -> Result<Option<DurableJob>> {
    let mut statement = connection.prepare(
        "SELECT id,user_id,kind,payload_json,status,priority,attempt,max_attempts,lease_owner,
         lease_expires_at,progress,result_json,error,created_at,updated_at FROM jobs WHERE id=?1",
    )?;
    let row = statement
        .query_row([id], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, i64>(5)?,
                row.get::<_, i64>(6)?,
                row.get::<_, i64>(7)?,
                row.get::<_, Option<String>>(8)?,
                row.get::<_, Option<i64>>(9)?,
                row.get::<_, f32>(10)?,
                row.get::<_, Option<String>>(11)?,
                row.get::<_, Option<String>>(12)?,
                row.get::<_, i64>(13)?,
                row.get::<_, i64>(14)?,
            ))
        })
        .optional()?;
    let Some(row) = row else {
        return Ok(None);
    };
    Ok(Some(DurableJob {
        id: row.0,
        user_id: row.1,
        kind: row.2,
        payload: serde_json::from_str(&row.3)?,
        status: JobStatus::parse(&row.4)?,
        priority: row.5,
        attempt: row.6.max(0) as u32,
        max_attempts: row.7.max(1) as u32,
        lease_owner: row.8,
        lease_expires_at: row.9,
        progress: row.10,
        result: row
            .11
            .map(|value| serde_json::from_str(&value))
            .transpose()?,
        error: row.12,
        created_at: row.13,
        updated_at: row.14,
    }))
}

fn artifact_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<ArtifactRecord> {
    Ok(ArtifactRecord {
        id: row.get(0)?,
        user_id: row.get(1)?,
        job_id: row.get(2)?,
        kind: row.get(3)?,
        object_key: row.get(4)?,
        sha256: row.get(5)?,
        bytes: row.get::<_, i64>(6)?.max(0) as u64,
        mime: row.get(7)?,
        created_at: row.get(8)?,
        expires_at: row.get(9)?,
    })
}

fn hash_token(token: &str) -> String {
    hex_sha256(token.as_bytes())
}

fn hex_sha256(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn now() -> i64 {
    Utc::now().timestamp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn authenticates_tokens_and_enforces_quota() {
        let store = ProductionStore::open_memory().unwrap();
        let user = store
            .create_user("artist@example.com", "Artist", UserTier::Free)
            .unwrap();
        let session = store
            .issue_token(&user.id, "test", Some(now() + 60))
            .unwrap();
        assert_eq!(
            store.authenticate(&session.token).unwrap().unwrap().id,
            user.id
        );
        store.record_usage(&user.id, "pages", 19, None).unwrap();
        assert!(store.enforce_job_quota(&user.id, 1).is_ok());
        assert!(store.enforce_job_quota(&user.id, 2).is_err());
        assert!(store.revoke_token(&session.token_id).unwrap());
        assert!(store.authenticate(&session.token).unwrap().is_none());
    }

    #[test]
    fn queue_claims_completes_and_recovers_jobs() {
        let store = ProductionStore::open_memory().unwrap();
        let user = store
            .create_user("worker@example.com", "Worker", UserTier::Studio)
            .unwrap();
        let queued = store
            .enqueue(NewJob {
                user_id: user.id.clone(),
                kind: "chapter".to_string(),
                payload: serde_json::json!({"requested_pages": 3}),
                priority: 10,
                max_attempts: 3,
            })
            .unwrap();
        let claimed = store.claim("gpu-1", 30).unwrap().unwrap();
        assert_eq!(claimed.id, queued.id);
        assert_eq!(claimed.status, JobStatus::Running);
        assert!(store.heartbeat(&claimed.id, "gpu-1", 30, 42.0).unwrap());
        assert!(store
            .complete(&claimed.id, "gpu-1", &serde_json::json!({"ok": true}))
            .unwrap());
        assert_eq!(
            store.job(&claimed.id).unwrap().unwrap().status,
            JobStatus::Complete
        );
    }

    #[test]
    fn object_store_is_content_addressed_and_path_safe() {
        let dir = tempfile::tempdir().unwrap();
        let store = LocalObjectStore::new(dir.path()).unwrap();
        let first = store.put("outputs", b"mangahue").unwrap();
        let second = store.put("outputs", b"mangahue").unwrap();
        assert_eq!(first.key, second.key);
        assert_eq!(store.get(&first.key).unwrap(), b"mangahue");
        assert!(store.get("../secret").is_err());
        assert!(store.delete(&first.key).unwrap());
    }
}
