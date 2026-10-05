import { NextRequest, NextResponse } from "next/server";

export async function POST(req: NextRequest) {
  try {
    const fastApiUrl =
      process.env.FASTAPI_BACKEND_URL ||
      process.env.NEXT_PUBLIC_FASTAPI_URL ||
      "http://127.0.0.1:8001";

    const targetUrl = `${fastApiUrl.replace(/\/+$/, "")}/api/models/restore-defaults`;

    const backendRes = await fetch(targetUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
    });

    const data = await backendRes.json();

    if (!backendRes.ok) {
      return NextResponse.json(
        {
          error: data.detail || data.error || "Failed to restore default models.",
          status: backendRes.status,
        },
        { status: backendRes.status }
      );
    }

    return NextResponse.json(data, { status: 200 });
  } catch (error: any) {
    console.error("Error proxying request to FastAPI POST /api/models/restore-defaults:", error);
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
