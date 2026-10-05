import { NextRequest, NextResponse } from "next/server";

export async function GET(req: NextRequest) {
  try {
    const { searchParams } = new URL(req.url);
    const category = searchParams.get("category");

    const fastApiUrl =
      process.env.FASTAPI_BACKEND_URL ||
      process.env.NEXT_PUBLIC_FASTAPI_URL ||
      "http://127.0.0.1:8001";

    const targetUrl = new URL(`${fastApiUrl.replace(/\/+$/, "")}/api/models`);
    if (category) {
      targetUrl.searchParams.set("category", category);
    }

    const backendRes = await fetch(targetUrl.toString(), {
      method: "GET",
      cache: "no-store",
    });

    const data = await backendRes.json();

    if (!backendRes.ok) {
      return NextResponse.json(
        {
          error: data.detail || data.error || "Failed to fetch models.",
          status: backendRes.status,
        },
        { status: backendRes.status }
      );
    }

    return NextResponse.json(data, { status: 200 });
  } catch (error: any) {
    console.error("Error proxying request to FastAPI GET /api/models:", error);
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

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();

    const fastApiUrl =
      process.env.FASTAPI_BACKEND_URL ||
      process.env.NEXT_PUBLIC_FASTAPI_URL ||
      "http://127.0.0.1:8001";

    const targetUrl = `${fastApiUrl.replace(/\/+$/, "")}/api/models`;

    const backendRes = await fetch(targetUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    });

    const data = await backendRes.json();

    if (!backendRes.ok) {
      return NextResponse.json(
        {
          error: data.detail || data.error || "Failed to create model.",
          status: backendRes.status,
        },
        { status: backendRes.status }
      );
    }

    return NextResponse.json(data, { status: 201 });
  } catch (error: any) {
    console.error("Error proxying request to FastAPI POST /api/models:", error);
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
