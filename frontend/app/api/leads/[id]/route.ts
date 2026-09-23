import { NextRequest, NextResponse } from "next/server";

export async function PUT(
  req: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;
    const body = await req.json();

    if (!id) {
      return NextResponse.json(
        { error: "Lead 'id' is required in route parameter." },
        { status: 400 }
      );
    }

    const fastApiUrl =
      process.env.FASTAPI_BACKEND_URL ||
      process.env.NEXT_PUBLIC_FASTAPI_URL ||
      "http://127.0.0.1:8001";

    const targetUrl = `${fastApiUrl.replace(/\/+$/, "")}/api/leads/${id}`;

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
          error: data.detail || data.error || `Failed to update lead #${id}.`,
          status: backendRes.status,
        },
        { status: backendRes.status }
      );
    }

    return NextResponse.json(data, { status: 200 });
  } catch (error: any) {
    console.error("Error proxying request to FastAPI PUT /api/leads/[id]:", error);
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
