import { Pool, types } from "pg";

// Parse NUMERIC as float (default is string)
types.setTypeParser(types.builtins.NUMERIC, parseFloat);
// Parse INT8 (bigint) as number
types.setTypeParser(types.builtins.INT8, Number);

const pool = new Pool({
  host: process.env.DB_HOST ?? "localhost",
  port: Number(process.env.DB_PORT ?? 5432),
  database: process.env.DB_NAME ?? "trading",
  user: process.env.DB_USER ?? "trading_reader",
  password: process.env.DB_PASSWORD ?? "",
  max: 5,
  idleTimeoutMillis: 30_000,
});

export default pool;
