import { NextRequest, NextResponse } from "next/server";
import { query, ensureUsersTable } from "@/lib/db";
import { verifyPassword, hashPassword, signJwtToken, AUTH_COOKIE_NAME } from "@/lib/auth";

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const isDemo = Boolean(body.isDemo);
    let email = (body.email || "").trim().toLowerCase();
    const password = body.password || "";

    await ensureUsersTable();

    // Handle One-Click Demo Mode
    if (isDemo) {
      const demoEmail = "sales.rep@enterprise.ai";
      let userRes = await query<{
        id: number;
        name: string;
        email: string;
        company: string;
        role: string;
        password_hash: string;
      }>("SELECT id, name, email, company, role, password_hash FROM users WHERE LOWER(email) = $1", [
        demoEmail,
      ]);

      if (userRes.rows.length === 0) {
        // Seed demo user into PostgreSQL
        const defaultHash = await hashPassword("demo123456");
        userRes = await query(
          `INSERT INTO users (name, email, company, role, password_hash)
           VALUES ($1, $2, $3, $4, $5)
           RETURNING id, name, email, company, role, password_hash`,
          [
            "Alex Morgan",
            demoEmail,
            "Acme Cloud Technologies",
            "Senior Account Executive",
            defaultHash,
          ]
        );
      }

      const demoUser = userRes.rows[0];
      const token = await signJwtToken({
        id: demoUser.id,
        name: demoUser.name,
        email: demoUser.email,
        company: demoUser.company,
        role: demoUser.role,
      });

      const response = NextResponse.json({
        success: true,
        message: "Demo session initiated",
        user: {
          id: demoUser.id,
          name: demoUser.name,
          email: demoUser.email,
          company: demoUser.company,
          role: demoUser.role,
        },
      });

      response.cookies.set({
        name: AUTH_COOKIE_NAME,
        value: token,
        httpOnly: true,
        secure: process.env.NODE_ENV === "production",
        sameSite: "lax",
        path: "/",
        maxAge: 7 * 24 * 60 * 60,
      });

      return response;
    }

    // Standard credential validation
    if (!email || !password) {
      return NextResponse.json(
        { error: "Please provide both email and password." },
        { status: 400 }
      );
    }

    // Retrieve user from PostgreSQL
    const result = await query<{
      id: number;
      name: string;
      email: string;
      company: string;
      role: string;
      password_hash: string;
    }>(
      "SELECT id, name, email, company, role, password_hash FROM users WHERE LOWER(email) = $1",
      [email]
    );

    if (result.rows.length === 0) {
      return NextResponse.json(
        { error: "Invalid email or password." },
        { status: 401 }
      );
    }

    const user = result.rows[0];

    // Securely verify password hash
    const isMatch = await verifyPassword(password, user.password_hash);
    if (!isMatch) {
      return NextResponse.json(
        { error: "Invalid email or password." },
        { status: 401 }
      );
    }

    // Generate JWT token
    const token = await signJwtToken({
      id: user.id,
      name: user.name,
      email: user.email,
      company: user.company,
      role: user.role,
    });

    const response = NextResponse.json({
      success: true,
      message: "Authentication successful",
      user: {
        id: user.id,
        name: user.name,
        email: user.email,
        company: user.company,
        role: user.role,
      },
    });

    response.cookies.set({
      name: AUTH_COOKIE_NAME,
      value: token,
      httpOnly: true,
      secure: process.env.NODE_ENV === "production",
      sameSite: "lax",
      path: "/",
      maxAge: 7 * 24 * 60 * 60,
    });

    return response;
  } catch (err: any) {
    console.error("Login error:", err);
    return NextResponse.json(
      { error: err.message || "An unexpected error occurred during login." },
      { status: 500 }
    );
  }
}
