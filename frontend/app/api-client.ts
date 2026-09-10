export const DEV_USER_HEADER = "X-Dev-User";

export async function fetchForUser(
  path: string,
  devUser: string,
  options?: RequestInit,
): Promise<Response> {
  const headers = new Headers(options?.headers);
  headers.set(DEV_USER_HEADER, devUser);

  return fetch(path, {
    ...options,
    credentials: "same-origin",
    headers,
  });
}

export async function fetchJsonForUser<T>(
  path: string,
  devUser: string,
  options?: RequestInit,
): Promise<T> {
  const headers = new Headers(options?.headers);
  if (!headers.has("Content-Type")) headers.set("Content-Type", "application/json");

  const response = await fetchForUser(path, devUser, { ...options, headers });
  const payload = await response.json();
  if (!response.ok) {
    const detail = payload.detail;
    const message = Array.isArray(detail)
      ? detail.map((item: { msg?: string }) => item.msg || "字段格式不正确").join("；")
      : typeof detail === "string" ? detail : "请求失败";
    throw new Error(message);
  }
  return payload as T;
}
