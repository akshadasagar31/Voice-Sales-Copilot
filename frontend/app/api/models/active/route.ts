import { NextRequest, NextResponse } from "next/server";

export async function GET(req: NextRequest) {
  try {
    const fastApiUrl =
      process.env.FASTAPI_BACKEND_URL ||
      process.env.NEXT_PUBLIC_FASTAPI_URL ||
      "http://127.0.0.1:8001";

    const targetUrl = `${fastApiUrl.replace(/\/+$/, "")}/api/models/active`;

    const backendRes = await fetch(targetUrl, {
      method: "GET",
      cache: "no-store",
    });

    const data = await backendRes.json();

    if (!backendRes.ok) {
      return NextResponse.json(
        {
          error: data.detail || data.error || "Failed to fetch active models.",
          status: backendRes.status,
        },
        { status: backendRes.status }
      );
    }

    return NextResponse.json(data, { status: 200 });
  } catch (error: any) {
    console.error("Error proxying request to FastAPI GET /api/models/active:", error);
    return NextResponse.json(
      {
        error:
          "Unable to connect to FastAPI backend service. Please ensure the backend is running at http://127.0.0.1:8001.",
        details: error.message,
      },
      { status: 503 }
    );
  }
}
