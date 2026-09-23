// ============================================================================
// POSTGRESQL DATABASE CONNECTION POOL (frontend/lib/db.ts)
// ============================================================================
// This file manages the database connection pool for Next.js API routes.
// Why connection pooling:
// Opening and closing a new TCP database connection for every single HTTP request
// is very slow. A connection pool keeps a pool of reusable connections open,
// allowing fast concurrent queries.
// ============================================================================

import { Pool, PoolConfig } from "pg";

declare global {
  // In Next.js development mode, hot-reloading re-executes files frequently.
  // We attach the pool to globalThis so we do not create dozens of new pools.
  // eslint-disable-next-line no-var
  var __dbPool: Pool | undefined;
}

// Database connection configuration with sensible timeouts and limits
const poolConfig: PoolConfig = {
  connectionString:
    process.env.DATABASE_URL ||
    `postgresql://${process.env.PGUSER || "postgres"}:${process.env.PGPASSWORD || ""}@${
      process.env.PGHOST || "127.0.0.1"
    }:${process.env.PGPORT || "5432"}/${process.env.PGDATABASE || "voice_sales_copilot"}`,
  max: 10,                        // Maximum 10 simultaneous database connections
  idleTimeoutMillis: 30000,       // Close idle connections after 30 seconds
  connectionTimeoutMillis: 5000,  // Throw error if connection takes > 5 seconds
};

// Use existing cached pool in dev hot-reload, or initialize a new one
export const pool = globalThis.__dbPool || new Pool(poolConfig);

if (process.env.NODE_ENV !== "production") {
  globalThis.__dbPool = pool;
}

/**
 * Helper function to run a parameterized SQL query safely.
 * Automatically acquires a connection from the pool, executes the query,
 * and releases the client back to the pool even if an error occurs.
 */
export async function query<T = any>(text: string, params?: any[]): Promise<{ rows: T[]; rowCount: number | null }> {
  const client = await pool.connect();
  try {
    const res = await client.query(text, params);
    return { rows: res.rows, rowCount: res.rowCount };
  } finally {
    // CRITICAL: Always release the client back to the pool so other requests can use it
    client.release();
  }
}

/**
 * Ensures the 'users' authentication table exists in PostgreSQL.
 * Called automatically during registration or app startup.
 */
export async function ensureUsersTable(): Promise<void> {
  const createTableQuery = `
    CREATE TABLE IF NOT EXISTS users (
      id SERIAL PRIMARY KEY,
      name VARCHAR(255) NOT NULL,
      email VARCHAR(255) UNIQUE NOT NULL,
      company VARCHAR(255),
      role VARCHAR(100),
      password_hash VARCHAR(255) NOT NULL,
      created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
      updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );
  `;
  await query(createTableQuery);
}

