import { NextRequest, NextResponse } from "next/server";

export async function POST(req: NextRequest) {
  try {
    const formData = await req.formData();
    const file = formData.get("file") as File | null;
    const namespace = (formData.get("namespace") as string) || "";
    const upsertParam = formData.get("upsert_to_pinecone") ?? formData.get("upsert");
    const upsertToPinecone = upsertParam === null ? true : upsertParam === "true";

    if (!file) {
      return NextResponse.json(
        { error: "No PDF file was provided in the upload request." },
        {
          status: 400,
          headers: { "Content-Type": "application/json" },
        }
      );
    }

    if (!file.name.toLowerCase().endsWith(".pdf")) {
      return NextResponse.json(
        { error: `Invalid file type '${file.name}'. Only .pdf files are supported.` },
        {
          status: 400,
          headers: { "Content-Type": "application/json" },
        }
      );
    }

    const fastApiUrl =
      process.env.FASTAPI_BACKEND_URL ||
      process.env.NEXT_PUBLIC_FASTAPI_URL ||
      "http://127.0.0.1:8001";

    const queryParams = new URLSearchParams();
    queryParams.set("upsert_to_pinecone", upsertToPinecone ? "true" : "false");
    if (namespace && namespace.trim()) {
      queryParams.set("namespace", namespace.trim());
    }

    const targetUrl = `${fastApiUrl.replace(/\/+$/, "")}/api/extract-pdf?${queryParams.toString()}`;

    // Reconstruct FormData for FastAPI upload
    const backendFormData = new FormData();
    backendFormData.append("file", file, file.name);

    let backendRes: Response;
    try {
      backendRes = await fetch(targetUrl, {
        method: "POST",
        body: backendFormData,
      });
    } catch (fetchErr: any) {
      console.error("Failed to connect to FastAPI /api/extract-pdf:", fetchErr);
      return NextResponse.json(
        {
          error:
            "Unable to connect to FastAPI PDF processing service. Please ensure the backend is running at http://127.0.0.1:8001.",
          details: fetchErr?.message || String(fetchErr),
        },
        {
          status: 503,
          headers: { "Content-Type": "application/json" },
        }
      );
    }

    // Safely parse JSON or text response without crashing
    let data: any = null;
    const contentType = backendRes.headers.get("content-type") || "";

    if (contentType.includes("application/json")) {
      try {
        data = await backendRes.json();
      } catch (jsonErr: any) {
        console.warn("Failed to parse JSON despite application/json header:", jsonErr);
        data = null;
      }
    }

    if (!data) {
      const rawText = await backendRes.text().catch(() => "");
      try {
        data = JSON.parse(rawText);
      } catch {
        data = {
          error: rawText || `FastAPI returned HTTP ${backendRes.status}`,
          status: backendRes.status,
        };
      }
    }

    if (!backendRes.ok) {
      const errorMsg =
        data?.detail ||
        data?.error ||
        `FastAPI PDF processing pipeline returned an error (HTTP ${backendRes.status}).`;

      return NextResponse.json(
        {
          error: errorMsg,
          status: backendRes.status,
        },
        {
          status: backendRes.status,
          headers: { "Content-Type": "application/json" },
        }
      );
    }

    return NextResponse.json(data, {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  } catch (error: any) {
    console.error("Error in Next.js /api/upload-pdf route:", error);
    return NextResponse.json(
      {
        error: "An unexpected error occurred while handling the PDF upload.",
        details: error?.message || String(error),
      },
      {
        status: 500,
        headers: { "Content-Type": "application/json" },
      }
    );
  }
}
