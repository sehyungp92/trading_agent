import pg from 'pg';

// Parse NUMERIC as float instead of string
pg.types.setTypeParser(pg.types.builtins.NUMERIC, parseFloat);
// Parse INT8 as int instead of string
pg.types.setTypeParser(pg.types.builtins.INT8, parseInt);

let pool: pg.Pool | undefined;

export function getPool(): pg.Pool {
  if (!pool) {
    pool = new pg.Pool({
      host: process.env.DB_HOST || 'localhost',
      port: parseInt(process.env.DB_PORT || '5432'),
      database: process.env.DB_NAME || 'trading',
      user: process.env.DB_USER || 'trading_reader',
      password: process.env.DB_PASSWORD || '',
      max: 10,
      idleTimeoutMillis: 30000,
      connectionTimeoutMillis: 5000,
    });
  }
  return pool;
}

export async function query<T = Record<string, unknown>>(
  sql: string,
  params?: unknown[]
): Promise<T[]> {
  const client = await getPool().connect();
  try {
    const result = await client.query(sql, params);
    return result.rows as T[];
  } finally {
    client.release();
  }
}
