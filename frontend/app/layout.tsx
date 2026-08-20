import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "DataAgent",
  description: "面向企业数据的只读分析 Agent",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="zh-CN"><body>{children}</body></html>;
}
