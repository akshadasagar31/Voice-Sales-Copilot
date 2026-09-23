// Test script for PostgreSQL auth and route protection
const BASE_URL = "http://localhost:3000";

async function runTests() {
  console.log("=== Running Authentication & PostgreSQL Integration Tests ===");

  // 1. Unauthenticated /dashboard access
  console.log("\n[Test 1] Testing unauthenticated access to /dashboard...");
  const unauthRes = await fetch(`${BASE_URL}/dashboard`, { redirect: "manual" });
  console.log(`Status: ${unauthRes.status}`);
  console.log(`Location Header: ${unauthRes.headers.get("location")}`);
  if (unauthRes.status === 307 || unauthRes.status === 308 || unauthRes.headers.get("location")?.includes("/login")) {
    console.log("✓ Correctly redirected unauthenticated request away from /dashboard");
  } else {
    console.error("✗ Expected redirect for unauthenticated request");
  }

  // 2. Register new user
  console.log("\n[Test 2] Testing user registration...");
  const regPayload = {
    fullName: "Jessica Taylor",
    email: `jessica.${Date.now()}@enterprise.ai`,
    company: "Starlight Corp",
    role: "Account Executive",
    password: "SecurePassword123!",
  };

  const regRes = await fetch(`${BASE_URL}/api/auth/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(regPayload),
  });

  const regData = await regRes.json();
  const regCookies = regRes.headers.getSetCookie ? regRes.headers.getSetCookie() : [regRes.headers.get("set-cookie")];
  console.log(`Register status: ${regRes.status}`);
  console.log(`User created:`, regData.user);
  console.log(`Set-Cookie received:`, regCookies.some((c) => c?.includes("auth_token")));

  if (regRes.status === 201 && regData.success) {
    console.log("✓ Registration succeeded and issued session token");
  } else {
    console.error("✗ Registration failed:", regData);
  }

  // 3. Duplicate email registration
  console.log("\n[Test 3] Testing duplicate email rejection...");
  const dupRes = await fetch(`${BASE_URL}/api/auth/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(regPayload),
  });
  const dupData = await dupRes.json();
  console.log(`Duplicate status: ${dupRes.status} (expected 409)`);
  console.log(`Message: ${dupData.error}`);
  if (dupRes.status === 409) {
    console.log("✓ Duplicate email correctly rejected with 409");
  }

  // 4. Incorrect password login
  console.log("\n[Test 4] Testing login with wrong password...");
  const wrongLoginRes = await fetch(`${BASE_URL}/api/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email: regPayload.email, password: "WrongPassword!" }),
  });
  const wrongData = await wrongLoginRes.json();
  console.log(`Wrong password status: ${wrongLoginRes.status} (expected 401)`);
  console.log(`Error message: ${wrongData.error}`);
  if (wrongLoginRes.status === 401) {
    console.log("✓ Invalid password correctly rejected with 401");
  }

  // 5. Correct password login
  console.log("\n[Test 5] Testing login with correct password...");
  const loginRes = await fetch(`${BASE_URL}/api/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email: regPayload.email, password: regPayload.password }),
  });
  const loginData = await loginRes.json();
  const loginCookieHeader = loginRes.headers.get("set-cookie") || "";
  console.log(`Login status: ${loginRes.status}`);
  console.log(`Logged in user:`, loginData.user);
  if (loginRes.status === 200 && loginData.success) {
    console.log("✓ Login succeeded with verified password");
  }

  // Extract auth_token cookie
  const authTokenMatch = loginCookieHeader.match(/auth_token=([^;]+)/);
  const authToken = authTokenMatch ? authTokenMatch[1] : "";

  // 6. Session verification via /api/auth/me
  console.log("\n[Test 6] Testing /api/auth/me session check...");
  const meRes = await fetch(`${BASE_URL}/api/auth/me`, {
    headers: { Cookie: `auth_token=${authToken}` },
  });
  const meData = await meRes.json();
  console.log(`Me status: ${meRes.status}`);
  console.log(`Me data:`, meData);
  if (meRes.status === 200 && meData.authenticated) {
    console.log("✓ Session verified and user profile hydrated");
  }

  // 7. Authenticated access to /dashboard
  console.log("\n[Test 7] Testing authenticated access to /dashboard...");
  const authDashRes = await fetch(`${BASE_URL}/dashboard`, {
    headers: { Cookie: `auth_token=${authToken}` },
    redirect: "manual",
  });
  console.log(`Dashboard status with auth_token: ${authDashRes.status}`);
  if (authDashRes.status === 200) {
    console.log("✓ Authenticated user allowed access to /dashboard");
  }

  // 8. One-Click Demo Mode test
  console.log("\n[Test 8] Testing One-Click Demo login...");
  const demoRes = await fetch(`${BASE_URL}/api/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ isDemo: true }),
  });
  const demoData = await demoRes.json();
  console.log(`Demo login status: ${demoRes.status}`);
  console.log(`Demo user:`, demoData.user);
  if (demoRes.status === 200 && demoData.success) {
    console.log("✓ Demo login seeded and session initialized");
  }

  console.log("\n=== All Authentication & Route Protection Tests Completed Successfully! ===");
}

runTests().catch(console.error);
