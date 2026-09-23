import { NextRequest, NextResponse } from "next/server";

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();

    const fastApiUrl =
      process.env.FASTAPI_BACKEND_URL ||
      process.env.NEXT_PUBLIC_FASTAPI_URL ||
      "http://127.0.0.1:8001";

    const targetUrl = `${fastApiUrl.replace(/\/+$/, "")}/api/leads`;

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
          error: data.detail || data.error || "Failed to save lead to database.",
          status: backendRes.status,
        },
        { status: backendRes.status }
      );
    }

    return NextResponse.json(data, { status: 201 });
  } catch (error: any) {
    console.error("Error proxying request to FastAPI POST /api/leads:", error);
    return NextResponse.json(
      {
        error:
          "Unable to connect to FastAPI database service. Please ensure the backend is running at http://127.0.0.1:8001.",
        details: error.message,
      },
      { status: 503 }
    );
  }
}

export async function GET(req: NextRequest) {
  try {
    const { searchParams } = new URL(req.url);
    const limit = searchParams.get("limit") || "50";

    const fastApiUrl =
      process.env.FASTAPI_BACKEND_URL ||
      process.env.NEXT_PUBLIC_FASTAPI_URL ||
      "http://127.0.0.1:8001";

    const targetUrl = `${fastApiUrl.replace(/\/+$/, "")}/api/leads?limit=${limit}`;

    const backendRes = await fetch(targetUrl, {
      method: "GET",
    });

    const data = await backendRes.json();

    if (!backendRes.ok) {
      return NextResponse.json(
        {
          error: data.detail || data.error || "Failed to query leads from database.",
          status: backendRes.status,
        },
        { status: backendRes.status }
      );
    }

    return NextResponse.json(data, { status: 200 });
  } catch (error: any) {
    console.error("Error proxying request to FastAPI GET /api/leads:", error);
    return NextResponse.json(
      {
        error:
          "Unable to connect to FastAPI database service. Please ensure the backend is running at http://127.0.0.1:8001.",
        details: error.message,
      },
      { status: 503 }
    );
  }
}

export async function PUT(req: NextRequest) {
  try {
    const body = await req.json();
    const leadId = body?.id;

    if (!leadId) {
      return NextResponse.json(
        { error: "Lead 'id' is required to update an existing lead." },
        { status: 400 }
      );
    }

    const fastApiUrl =
      process.env.FASTAPI_BACKEND_URL ||
      process.env.NEXT_PUBLIC_FASTAPI_URL ||
      "http://127.0.0.1:8001";

    const targetUrl = `${fastApiUrl.replace(/\/+$/, "")}/api/leads/${leadId}`;

    const backendRes = await fetch(targetUrl, {
      method: "PUT",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    });

    const data = await backendRes.json();

    if (!backendRes.ok) {
      return NextResponse.json(
        {
          error: data.detail || data.error || `Failed to update lead #${leadId}.`,
          status: backendRes.status,
        },
        { status: backendRes.status }
      );
    }

    return NextResponse.json(data, { status: 200 });
  } catch (error: any) {
    console.error("Error proxying request to FastAPI PUT /api/leads:", error);
    return NextResponse.json(
      {
        error:
          "Unable to connect to FastAPI database service. Please ensure the backend is running at http://127.0.0.1:8001.",
        details: error.message,
      },
      { status: 503 }
    );
  }
}

