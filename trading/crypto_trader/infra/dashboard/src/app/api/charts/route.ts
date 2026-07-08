import { NextResponse } from "next/server";
import pool from "@/lib/db";
import type { ChartBatchResponse } from "@/lib/types";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const [equityRes, pnlRes] = await Promise.all([
      pool.query("SELECT * FROM v_equity_curve_90d"),
      pool.query("SELECT * FROM v_daily_pnl_30d"),
    ]);

    const response: ChartBatchResponse = {
      equity_curve: equityRes.rows,
      daily_pnl: pnlRes.rows,
    };

    return NextResponse.json(response);
  } catch (err) {
    console.error("API /charts error:", err);
    return NextResponse.json({ error: "Database query failed" }, { status: 500 });
  }
}
