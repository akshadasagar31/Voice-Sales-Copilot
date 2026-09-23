import { NextRequest, NextResponse } from "next/server";
import { query, ensureUsersTable } from "@/lib/db";
import { hashPassword, signJwtToken, AUTH_COOKIE_NAME } from "@/lib/auth";

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const name = (body.fullName || body.name || "").trim();
    const email = (body.email || "").trim().toLowerCase();
    const company = (body.company || "").trim();
    const role = (body.role || "").trim();
    const password = body.password || "";

    if (!name) {
      return NextResponse.json(
        { error: "Full name is required." },
        { status: 400 }
      );
    }

    if (!email || !email.includes("@")) {
      return NextResponse.json(
        { error: "A valid email address is required." },
        { status: 400 }
      );
    }

    if (!password || password.length < 6) {
      return NextResponse.json(
        { error: "Password must be at least 6 characters long." },
        { status: 400 }
      );
    }

    // Ensure database table exists
    await ensureUsersTable();

    // Check if user already exists
    const existingUser = await query("SELECT id FROM users WHERE LOWER(email) = $1", [email]);
    if (existingUser.rows.length > 0) {
      return NextResponse.json(
        { error: "An account with this email address already exists." },
        { status: 409 }
      );
    }

    // Hash password with bcrypt
    const passwordHash = await hashPassword(password);

    // Insert user into PostgreSQL
    const insertResult = await query<{
      id: number;
      name: string;
      email: string;
      company: string;
      role: string;
    }>(
      `INSERT INTO users (name, email, company, role, password_hash)
       VALUES ($1, $2, $3, $4, $5)
       RETURNING id, name, email, company, role`,
      [name, email, company || null, role || null, passwordHash]
    );

    const newUser = insertResult.rows[0];

    // Generate JWT session
    const token = await signJwtToken({
      id: newUser.id,
      name: newUser.name,
      email: newUser.email,
      company: newUser.company,
      role: newUser.role,
    });

    const response = NextResponse.json(
      {
        success: true,
        message: "Account registered successfully",
        user: newUser,
      },
      { status: 201 }
    );

    // Set secure HTTP-only session cookie
    response.cookies.set({
      name: AUTH_COOKIE_NAME,
      value: token,
      httpOnly: true,
      secure: process.env.NODE_ENV === "production",
      sameSite: "lax",
      path: "/",
      maxAge: 7 * 24 * 60 * 60, // 7 days
    });

    return response;
  } catch (err: any) {
    console.error("Registration error:", err);
    return NextResponse.json(
      { error: err.message || "An unexpected error occurred during registration." },
      { status: 500 }
    );
  }
}
